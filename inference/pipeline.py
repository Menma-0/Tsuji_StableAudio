"""
推論パイプライン

オノマトペの差分指示に基づいて、元音声を編集した音声を生成する。

フロー:
1. 入力: 元音声(l1), 元オノマトペ(z1), 目標オノマトペ(z2)
2. l1を前処理 → VAEエンコード → g(l1)
3. Δf = f(z2) - f(z1) を計算
4. Δg' = Model(Δf) を予測
5. g(l2)' = g(l1) + α * Δg' を計算
6. g(l2)'をVAEデコード → 出力音声(l2')
"""
import torch
import soundfile as sf
from pathlib import Path
from typing import Optional, Union
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import AudioPreprocessor
from models.vae_wrapper import VAEWrapper
from models.onomatopoeia_encoder import OnomatopoeiaEncoder
from models.delta_predictor import DeltaPredictor
from config import Config, default_config


class InferencePipeline:
    """
    オノマトペベースの音声編集パイプライン

    使用例:
    ```python
    pipeline = InferencePipeline()
    pipeline.load_model("checkpoints/best.pt")

    # 音声編集
    output = pipeline.edit_audio(
        audio_path="input.wav",
        source_onomatopoeia="k o q",
        target_onomatopoeia="g a sh a N",
    )
    sf.write("output.wav", output, 44100)
    ```
    """

    def __init__(
        self,
        config: Config = default_config,
        device: str = None,
    ):
        """
        Args:
            config: 設定
            device: 使用デバイス（Noneなら自動選択）
        """
        self.config = config
        self.device = device or (config.device if torch.cuda.is_available() else "cpu")

        # 前処理器
        self.preprocessor = AudioPreprocessor(
            target_sr=config.audio.sample_rate,
            target_channels=config.audio.audio_channels,
            sample_size=config.audio.sample_size,
        )

        # モデル（遅延ロード）
        self.vae: Optional[VAEWrapper] = None
        self.onomatopoeia_encoder: Optional[OnomatopoeiaEncoder] = None
        self.delta_predictor: Optional[DeltaPredictor] = None

        # 正規化パラメータ（σのみ使用、μは不要）
        self.feature_std: Optional[torch.Tensor] = None

    def load_models(self, checkpoint_path: Union[str, Path] = None):
        """
        モデルをロード

        Args:
            checkpoint_path: 差分予測モデルのチェックポイントパス
        """
        print(f"Loading models on {self.device}...")

        # VAE
        print("  Loading VAE...")
        self.vae = VAEWrapper(
            model_id=self.config.hf_model_id,
            device=self.device,
        )
        self.vae.load()

        # オノマトペエンコーダ（38次元版）
        print("  Loading onomatopoeia encoder...")
        self.onomatopoeia_encoder = OnomatopoeiaEncoder()

        # 差分予測モデル
        if checkpoint_path:
            print(f"  Loading delta predictor from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            model_config = checkpoint.get('config', {})
            self.delta_predictor = DeltaPredictor(
                input_dim=model_config.get('input_dim', self.onomatopoeia_encoder.feature_dim),
                latent_dim=model_config.get('latent_dim', self.config.audio.latent_dim),
                latent_length=model_config.get('latent_length', self.config.audio.latent_length),
                hidden_dims=self.config.model.hidden_dims,
                dropout=0.0,  # 推論時はドロップアウト無効
            ).to(self.device)

            self.delta_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.delta_predictor.eval()

            # 正規化パラメータ（σのみ、μは不要）
            if 'feature_std' in checkpoint:
                self.feature_std = checkpoint['feature_std'].to(self.device)
                print(f"  Loaded feature_std for normalization")

        print("Models loaded successfully!")

    def ensure_loaded(self):
        """モデルがロードされていなければロード"""
        if self.vae is None:
            self.load_models()

    @torch.no_grad()
    def edit_audio(
        self,
        audio_path: Union[str, Path],
        source_onomatopoeia: str,
        target_onomatopoeia: str,
        alpha: float = 1.0,
        output_path: Union[str, Path] = None,
    ) -> torch.Tensor:
        """
        オノマトペ指示に基づいて音声を編集

        Args:
            audio_path: 入力音声ファイルパス
            source_onomatopoeia: 元のオノマトペ（例: "k o q"）
            target_onomatopoeia: 目標のオノマトペ（例: "g a sh a N"）
            alpha: 編集の強度（1.0がデフォルト）
            output_path: 出力ファイルパス（Noneなら保存しない）

        Returns:
            編集後の音声波形, shape (channels, samples)
        """
        self.ensure_loaded()

        print(f"\nEditing audio:")
        print(f"  Source: {source_onomatopoeia}")
        print(f"  Target: {target_onomatopoeia}")
        print(f"  Alpha: {alpha}")

        # 1. 音声の前処理とエンコード
        audio = self.preprocessor.process(Path(audio_path))
        audio = audio.unsqueeze(0).to(self.device)  # (1, channels, samples)

        latent1 = self.vae.encode(audio)  # (latent_dim, latent_length)
        print(f"  Latent1 shape: {latent1.shape}")

        # 2. オノマトペ特徴量の差分を計算
        f1 = self.onomatopoeia_encoder.encode_single(source_onomatopoeia)
        f2 = self.onomatopoeia_encoder.encode_single(target_onomatopoeia)
        delta_f = (f2 - f1).to(self.device)

        # 正規化: 差分をσで割る（fを正規化してから差分を取るのと等価）
        if self.feature_std is not None:
            delta_f = delta_f / self.feature_std

        delta_f = delta_f.unsqueeze(0).float()  # (1, feature_dim)
        print(f"  Delta F norm: {delta_f.norm():.4f}")

        # 3. latent差分を予測
        if self.delta_predictor is not None:
            delta_g = self.delta_predictor(delta_f)  # (1, latent_dim, latent_length)
            delta_g = delta_g.squeeze(0)  # (latent_dim, latent_length)
            print(f"  Delta G norm: {delta_g.norm():.4f}")

            # 4. 目標latentを計算
            latent2 = latent1 + alpha * delta_g
        else:
            print("  Warning: Delta predictor not loaded, returning original audio")
            latent2 = latent1

        # 5. 音声をデコード
        output_audio = self.vae.decode(latent2)  # (batch, channels, samples) or (channels, samples)
        output_audio = output_audio.cpu().float()  # float32に変換

        # バッチ次元があれば削除
        if output_audio.dim() == 3:
            output_audio = output_audio.squeeze(0)  # (channels, samples)

        # 6. 音量セーフティ（ピーク正規化）
        max_val = torch.abs(output_audio).max()
        if max_val > 0.95:
            output_audio = output_audio / max_val * 0.95
            print(f"  Safety: Peak normalized from {max_val:.2f} to 0.95")

        # 7. 出力を保存
        if output_path:
            output_np = output_audio.numpy().T  # (samples, channels)
            sf.write(str(output_path), output_np, self.config.audio.sample_rate)
            print(f"  Saved to: {output_path}")

        return output_audio

    @torch.no_grad()
    def reconstruct_audio(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path] = None,
    ) -> torch.Tensor:
        """
        音声をVAEでエンコード・デコードして再構成

        VAEの品質確認用

        Args:
            audio_path: 入力音声ファイルパス
            output_path: 出力ファイルパス

        Returns:
            再構成音声
        """
        self.ensure_loaded()

        audio = self.preprocessor.process(Path(audio_path))
        audio = audio.unsqueeze(0).to(self.device)

        latent = self.vae.encode(audio)
        reconstructed = self.vae.decode(latent)
        reconstructed = reconstructed.cpu().float()

        # バッチ次元があれば削除
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.squeeze(0)

        if output_path:
            output_np = reconstructed.numpy().T
            sf.write(str(output_path), output_np, self.config.audio.sample_rate)
            print(f"  Saved to: {output_path}")

        return reconstructed

    @torch.no_grad()
    def get_onomatopoeia_similarity(
        self,
        onomatopoeia1: str,
        onomatopoeia2: str,
    ) -> float:
        """
        2つのオノマトペ間の類似度を計算

        Args:
            onomatopoeia1: オノマトペ1
            onomatopoeia2: オノマトペ2

        Returns:
            コサイン類似度 (-1 to 1)
        """
        self.ensure_loaded()

        f1 = self.onomatopoeia_encoder.encode_single(onomatopoeia1)
        f2 = self.onomatopoeia_encoder.encode_single(onomatopoeia2)

        similarity = torch.nn.functional.cosine_similarity(
            f1.unsqueeze(0), f2.unsqueeze(0)
        ).item()

        return similarity


def main():
    import argparse

    # デフォルトのチェックポイントパス
    default_checkpoint = Path(__file__).parent.parent / "checkpoints" / "experiment_38dim_v2" / "best.pt"

    parser = argparse.ArgumentParser(
        description="オノマトペで音声を編集する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 日本語オノマトペで指定（推奨）
  python -m inference.pipeline --input sound.wav --source "コッ" --target "ガシャン"

  # 音素表記でも指定可能
  python -m inference.pipeline --input sound.wav --source "k o q" --target "g a sh a N"

  # 出力ファイル名を指定
  python -m inference.pipeline --input sound.wav --source "コッ" --target "ドン" --output edited.wav

  # 編集強度を調整 (0.5=弱め, 1.0=通常, 2.0=強め)
  python -m inference.pipeline --input sound.wav --source "スタスタ" --target "ドスドス" --alpha 1.5

オノマトペ例:
  コッ, ガシャン, ドン, カーン, パタパタ, ゴロゴロ, スタスタ, ドスドス
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="入力音声ファイル（相対パス可）")
    parser.add_argument("--source", "-s", type=str, required=True, help="元のオノマトペ (例: 'コッ', 'k o q')")
    parser.add_argument("--target", "-t", type=str, required=True, help="目標のオノマトペ (例: 'ガシャン', 'g a sh a N')")
    parser.add_argument("--output", "-o", type=str, default=None, help="出力ファイル (デフォルト: input_edited.wav)")
    parser.add_argument("--alpha", "-a", type=float, default=1.0, help="編集強度 (デフォルト: 1.0)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="モデルチェックポイント")
    args = parser.parse_args()

    # 入力パスを解決（相対パス対応）
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    # 出力ファイル名のデフォルト設定
    if args.output is None:
        args.output = str(input_path.parent / f"{input_path.stem}_edited{input_path.suffix}")
    else:
        args.output = str(Path(args.output).resolve())

    # チェックポイントのデフォルト設定
    checkpoint = args.checkpoint if args.checkpoint else default_checkpoint
    if not Path(checkpoint).exists():
        print(f"Error: チェックポイントが見つかりません: {checkpoint}")
        print("先に学習を実行してください: python training/train.py")
        sys.exit(1)

    pipeline = InferencePipeline()
    pipeline.load_models(checkpoint)

    pipeline.edit_audio(
        audio_path=input_path,
        source_onomatopoeia=args.source,
        target_onomatopoeia=args.target,
        alpha=args.alpha,
        output_path=args.output,
    )

    print(f"\n完了！出力: {args.output}")


if __name__ == "__main__":
    main()
