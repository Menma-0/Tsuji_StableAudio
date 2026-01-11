"""
学習スクリプト

差分予測モデル（Δf → Δg）を学習する。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import DeltaPairDataset, create_dataloader
from training.cache_latents import load_cache_metadata
from models.delta_predictor import DeltaPredictor, DeltaPredictorLoss
from config import Config, default_config


class Trainer:
    """
    差分予測モデルのトレーナー
    """

    def __init__(
        self,
        config: Config = default_config,
        experiment_name: str = None,
    ):
        """
        Args:
            config: 設定
            experiment_name: 実験名（チェックポイントの保存先）
        """
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.checkpoint_dir = config.data.checkpoint_dir / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # メタデータ読み込み
        self.metadata = load_cache_metadata(config)
        if not self.metadata:
            raise ValueError("Cache metadata not found. Run cache_latents.py first.")

        # モデル初期化
        self._init_model()

        # データセット
        self._init_datasets()

        # 学習設定
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.model.num_epochs,
        )
        self.criterion = DeltaPredictorLoss(
            norm_weight=config.model.delta_norm_weight,
        )

        # ログ
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
        }

    def _init_model(self):
        """モデルを初期化"""
        latent_shape = self.metadata['latent_shape']  # [batch, channels, time] = [1, 64, 32]
        feature_dim = self.metadata['feature_dim']

        # latent_shape: [batch, channels, time] -> latent_dim=channels, latent_length=time
        latent_dim = latent_shape[1] if len(latent_shape) == 3 else latent_shape[0]
        latent_length = latent_shape[2] if len(latent_shape) == 3 else latent_shape[1]

        self.model = DeltaPredictor(
            input_dim=feature_dim,
            latent_dim=latent_dim,
            latent_length=latent_length,
            hidden_dims=self.config.model.hidden_dims,
            dropout=self.config.model.dropout,
            use_residual=self.config.model.use_residual,
        ).to(self.device)

        print(f"Model initialized:")
        print(f"  Input dim: {feature_dim}")
        print(f"  Latent dim: {latent_dim}, Latent length: {latent_length}")
        print(f"  Output dim: {latent_dim * latent_length}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _init_datasets(self):
        """データセットを初期化"""
        sample_ids = self.metadata['sample_ids']
        categories = self.metadata['categories']

        # データ分割
        n = len(sample_ids)
        n_train = int(n * self.config.data.train_ratio)
        n_val = int(n * self.config.data.val_ratio)

        # シャッフル
        import random
        random.seed(42)
        shuffled_ids = sample_ids.copy()
        random.shuffle(shuffled_ids)

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val:]

        print(f"\nData split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        # カテゴリをサンプルIDでフィルタ
        def filter_categories(ids):
            result = {}
            id_set = set(ids)
            for cat, cat_ids in categories.items():
                filtered = [i for i in cat_ids if i in id_set]
                if filtered:
                    result[cat] = filtered
            return result

        # データセット作成
        self.train_dataset = DeltaPairDataset(
            cache_dir=self.config.data.latent_cache_dir,
            sample_ids=train_ids,
            pairs_per_epoch=10000,
            same_category_ratio=0.5,
            categories=filter_categories(train_ids),
        )

        self.val_dataset = DeltaPairDataset(
            cache_dir=self.config.data.latent_cache_dir,
            sample_ids=val_ids,
            pairs_per_epoch=1000,
            same_category_ratio=0.5,
            categories=filter_categories(val_ids),
        )

        # DataLoader
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
        )

    def train_epoch(self) -> dict:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_norm = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            delta_f = batch['delta_f'].to(self.device)
            delta_g = batch['delta_g'].to(self.device)

            self.optimizer.zero_grad()

            # 予測
            delta_g_pred = self.model(delta_f)

            # 損失計算
            losses = self.criterion(delta_g_pred, delta_g)

            # バックプロパゲーション
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_norm += losses['norm'].item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'norm': total_norm / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """検証"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        num_batches = 0

        for batch in self.val_loader:
            delta_f = batch['delta_f'].to(self.device)
            delta_g = batch['delta_g'].to(self.device)

            delta_g_pred = self.model(delta_f)
            losses = self.criterion(delta_g_pred, delta_g)

            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'latent_length': self.model.latent_length,
            },
            # 正規化パラメータ（推論時に必要）
            'feature_std': self.train_dataset.feature_std,
        }

        # 最新のチェックポイント
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # ベストモデル
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch+1}.pt")

    def load_checkpoint(self, path: Path):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']

    def train(self, num_epochs: int = None):
        """学習を実行"""
        if num_epochs is None:
            num_epochs = self.config.model.num_epochs

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # ペアを再生成
            self.train_dataset.regenerate_pairs()

            # 学習
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['loss']:.6f}, MSE: {train_metrics['mse']:.6f}")

            # 検証
            val_metrics = self.validate()
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}")

            # 学習率更新
            self.scheduler.step()
            print(f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # 履歴更新
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])

            # ベストモデルの判定
            is_best = val_metrics['loss'] < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_metrics['loss']
                print("★ New best model!")

            # チェックポイント保存
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.history['best_val_loss']:.6f}")

        # 履歴をJSONで保存
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train delta predictor")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # 設定を更新
    config = default_config
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.lr
    config.model.num_epochs = args.epochs

    # トレーナー作成
    trainer = Trainer(config, experiment_name=args.name)

    # チェックポイントから再開
    if args.resume:
        start_epoch = trainer.load_checkpoint(Path(args.resume))
        print(f"Resumed from epoch {start_epoch + 1}")

    # 学習
    trainer.train()


if __name__ == "__main__":
    main()
