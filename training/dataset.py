"""
PyTorchデータセット

差分予測モデルの学習用データセット。
(l1, z1, l2, z2) のペアから (Δf, Δg) を生成する。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.rwcp_loader import RWCPSample


class DeltaPairDataset(Dataset):
    """
    差分ペアデータセット

    キャッシュ済みのlatentとオノマトペ特徴量から、
    ランダムペアを生成して (Δf, Δg) を返す。

    学習サンプル:
    - 入力: Δf = f(z2) - f(z1)
    - 出力: Δg = g(l2) - g(l1)
    """

    def __init__(
        self,
        cache_dir: Path,
        sample_ids: List[str],
        pairs_per_epoch: int = 10000,
        same_category_ratio: float = 0.5,
        categories: Optional[Dict[str, List[str]]] = None,
        normalize: bool = True,
    ):
        """
        Args:
            cache_dir: キャッシュディレクトリ（latentとfeatureが保存されている）
            sample_ids: 使用するサンプルIDのリスト
            pairs_per_epoch: 1エポックあたりのペア数
            same_category_ratio: 同カテゴリペアの割合
            categories: カテゴリ→サンプルIDリストの辞書
            normalize: 正規化を行うか
        """
        self.cache_dir = Path(cache_dir)
        self.sample_ids = sample_ids
        self.pairs_per_epoch = pairs_per_epoch
        self.same_category_ratio = same_category_ratio
        self.categories = categories or {}
        self.normalize = normalize

        # キャッシュを読み込み
        self.cache: Dict[str, dict] = {}
        self._load_cache()

        # カテゴリ逆引き
        self.sample_to_category: Dict[str, str] = {}
        for category, ids in self.categories.items():
            for sid in ids:
                self.sample_to_category[sid] = category

        # 正規化パラメータ（学習データから計算）
        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_std: Optional[torch.Tensor] = None
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        if normalize:
            self._compute_normalization_params()

        # エポックごとにペアを再生成
        self.pairs: List[Tuple[str, str]] = []
        self.regenerate_pairs()

    def _load_cache(self):
        """キャッシュファイルを読み込み"""
        for sample_id in self.sample_ids:
            cache_path = self.cache_dir / f"{sample_id}.pt"
            if cache_path.exists():
                data = torch.load(cache_path)
                self.cache[sample_id] = data
            else:
                print(f"Warning: Cache not found for {sample_id}")

        print(f"Loaded {len(self.cache)} samples from cache")

    def _compute_normalization_params(self):
        """正規化パラメータを計算"""
        latents = []
        features = []

        for data in self.cache.values():
            latent = data['latent'].squeeze(0) if data['latent'].dim() == 3 else data['latent']
            latents.append(latent.float().flatten())
            features.append(data['feature'])

        if latents:
            all_latents = torch.stack(latents)
            self.latent_mean = all_latents.mean(dim=0)
            self.latent_std = all_latents.std(dim=0).clamp(min=1e-6)

        if features:
            all_features = torch.stack(features)
            self.feature_mean = all_features.mean(dim=0)
            self.feature_std = all_features.std(dim=0).clamp(min=1e-6)

    def regenerate_pairs(self):
        """ペアを再生成（エポックごとに呼ぶ）"""
        self.pairs = []
        sample_ids = list(self.cache.keys())

        if len(sample_ids) < 2:
            return

        for _ in range(self.pairs_per_epoch):
            # 同カテゴリペアか異カテゴリペアかを決定
            if random.random() < self.same_category_ratio and self.categories:
                # 同カテゴリペア
                # ランダムにカテゴリを選択
                valid_categories = [
                    cat for cat, ids in self.categories.items()
                    if len([i for i in ids if i in self.cache]) >= 2
                ]
                if valid_categories:
                    category = random.choice(valid_categories)
                    valid_ids = [i for i in self.categories[category] if i in self.cache]
                    id1, id2 = random.sample(valid_ids, 2)
                else:
                    # 同カテゴリペアが作れない場合はランダム
                    id1, id2 = random.sample(sample_ids, 2)
            else:
                # ランダムペア
                id1, id2 = random.sample(sample_ids, 2)

            self.pairs.append((id1, id2))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        ペアデータを取得

        Returns:
            {
                'delta_f': オノマトペ特徴量の差分,
                'delta_g': latentの差分,
                'latent1': 元latent,
                'latent2': 目標latent,
            }
        """
        id1, id2 = self.pairs[idx]

        data1 = self.cache[id1]
        data2 = self.cache[id2]

        # latentは[1, 64, 32]でキャッシュされているのでsqueeze
        # また、float16でキャッシュされている場合があるのでfloat32に変換
        latent1 = data1['latent'].squeeze(0) if data1['latent'].dim() == 3 else data1['latent']
        latent2 = data2['latent'].squeeze(0) if data2['latent'].dim() == 3 else data2['latent']
        latent1 = latent1.float()
        latent2 = latent2.float()
        feature1 = data1['feature']
        feature2 = data2['feature']

        # 差分を計算
        delta_g = latent2 - latent1

        # 正規化: fを正規化してから差分を取る（= 差分をσで割る）
        # これにより平均μがキャンセルされ、各特徴量のスケールが揃う
        if self.normalize and self.feature_std is not None:
            delta_f = (feature2 - feature1) / self.feature_std
        else:
            delta_f = feature2 - feature1

        return {
            'delta_f': delta_f,
            'delta_g': delta_g,
            'latent1': latent1,
            'latent2': latent2,
        }


class CachedDataset(Dataset):
    """
    キャッシュ済みデータを読み込むシンプルなデータセット

    キャッシュ生成時に使用する。
    """

    def __init__(self, samples: List[RWCPSample]):
        """
        Args:
            samples: RWCPSampleのリスト
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RWCPSample:
        return self.samples[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoaderを作成"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # テスト
    print("Dataset test")

    # ダミーキャッシュを作成
    cache_dir = Path("./test_cache")
    cache_dir.mkdir(exist_ok=True)

    latent_dim = 64
    latent_length = 32
    feature_dim = 1280

    sample_ids = [f"sample_{i}" for i in range(10)]
    categories = {
        "cat_a": sample_ids[:5],
        "cat_b": sample_ids[5:],
    }

    # ダミーデータ保存
    for sid in sample_ids:
        data = {
            'latent': torch.randn(latent_dim, latent_length),
            'feature': torch.randn(feature_dim),
            'onomatopoeia': 'k o q',
        }
        torch.save(data, cache_dir / f"{sid}.pt")

    # データセット作成
    dataset = DeltaPairDataset(
        cache_dir=cache_dir,
        sample_ids=sample_ids,
        pairs_per_epoch=100,
        categories=categories,
    )

    print(f"Dataset size: {len(dataset)}")

    # サンプル取得
    sample = dataset[0]
    print(f"Delta F shape: {sample['delta_f'].shape}")
    print(f"Delta G shape: {sample['delta_g'].shape}")

    # DataLoader
    loader = create_dataloader(dataset, batch_size=8)
    batch = next(iter(loader))
    print(f"\nBatch delta_f shape: {batch['delta_f'].shape}")
    print(f"Batch delta_g shape: {batch['delta_g'].shape}")

    # クリーンアップ
    import shutil
    shutil.rmtree(cache_dir)
