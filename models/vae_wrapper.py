"""
VAEラッパー（AutoencoderOobleck）

Stable Audio OpenのVAE（AutoencoderOobleck）をラップして、
音声のエンコード・デコードを簡単に行えるようにする。
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from diffusers import AutoencoderOobleck


class VAEWrapper:
    """
    AutoencoderOobleckのラッパークラス

    主な機能:
    - 音声 → latent (encode)
    - latent → 音声 (decode)
    - latent差分の計算
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_id: Hugging FaceのモデルID
            device: 使用デバイス
            dtype: データ型（GPU使用時はfloat16推奨）
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32

        self.vae: Optional[AutoencoderOobleck] = None
        self.scaling_factor: float = 1.0

    def load(self):
        """VAEモデルをロード"""
        print(f"Loading VAE from {self.model_id}...")

        # diffusers経由でStable Audio Openのパイプラインからロード
        from diffusers import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )

        self.vae = pipe.vae.to(self.device)
        self.vae.eval()

        # scaling factorを取得
        if hasattr(self.vae.config, 'scaling_factor'):
            self.scaling_factor = self.vae.config.scaling_factor
        else:
            self.scaling_factor = 1.0

        # パイプラインの他の部分は解放
        del pipe

        print(f"VAE loaded. Sampling rate: {self.vae.sampling_rate}")
        print(f"Scaling factor: {self.scaling_factor}")

    def ensure_loaded(self):
        """VAEがロードされていなければロード"""
        if self.vae is None:
            self.load()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        音声をlatent空間にエンコード

        Args:
            audio: shape (batch, channels, samples) または (channels, samples)
            use_mean: Trueなら潜在分布の平均を使用、Falseならサンプリング

        Returns:
            latent: shape (batch, latent_dim, time) または (latent_dim, time)
        """
        self.ensure_loaded()

        # バッチ次元の追加
        squeeze_batch = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
            squeeze_batch = True

        # デバイスと型の変換
        audio = audio.to(device=self.device, dtype=self.dtype)

        # エンコード
        latent_dist = self.vae.encode(audio)

        if use_mean:
            latent = latent_dist.latent_dist.mean
        else:
            latent = latent_dist.latent_dist.sample()

        # スケーリング
        latent = latent * self.scaling_factor

        if squeeze_batch:
            latent = latent.squeeze(0)

        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latentから音声をデコード

        Args:
            latent: shape (batch, latent_dim, time) または (latent_dim, time)

        Returns:
            audio: shape (batch, channels, samples) または (channels, samples)
        """
        self.ensure_loaded()

        # バッチ次元の追加
        squeeze_batch = False
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            squeeze_batch = True

        # デバイスと型の変換
        latent = latent.to(device=self.device, dtype=self.dtype)

        # スケーリング解除
        latent = latent / self.scaling_factor

        # デコード
        audio = self.vae.decode(latent).sample

        if squeeze_batch:
            audio = audio.squeeze(0)

        return audio

    @torch.no_grad()
    def encode_decode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        音声をエンコード→デコードして再構成

        Args:
            audio: 入力音声

        Returns:
            再構成音声
        """
        latent = self.encode(audio)
        return self.decode(latent)

    def compute_latent_delta(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
    ) -> torch.Tensor:
        """
        latent差分 Δg = g(l2) - g(l1) を計算

        Args:
            latent1: 編集前のlatent
            latent2: 編集後のlatent

        Returns:
            差分latent
        """
        return latent2 - latent1

    def apply_latent_delta(
        self,
        latent1: torch.Tensor,
        delta_g: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        latent差分を適用: g(l2)' = g(l1) + α * Δg

        Args:
            latent1: 元のlatent
            delta_g: 差分latent
            alpha: スケール係数

        Returns:
            編集後のlatent
        """
        return latent1 + alpha * delta_g

    @property
    def latent_dim(self) -> int:
        """latentの次元（チャンネル数）"""
        self.ensure_loaded()
        return self.vae.config.decoder_input_channels

    @property
    def sampling_rate(self) -> int:
        """サンプルレート"""
        self.ensure_loaded()
        return self.vae.sampling_rate

    @property
    def downsampling_ratio(self) -> int:
        """ダウンサンプリング比率（時間方向の圧縮率）"""
        # [2, 4, 4, 8, 8] = 2048
        self.ensure_loaded()
        ratios = self.vae.config.downsampling_ratios
        ratio = 1
        for r in ratios:
            ratio *= r
        return ratio


class LatentCache:
    """
    latentのキャッシュ管理

    VAEのエンコード結果をファイルに保存/読み込みすることで、
    学習時の計算コストを削減する。
    """

    def __init__(self, cache_dir: Path):
        """
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, sample_id: str) -> Path:
        """サンプルIDからキャッシュパスを生成"""
        return self.cache_dir / f"{sample_id}.pt"

    def exists(self, sample_id: str) -> bool:
        """キャッシュが存在するか"""
        return self.get_cache_path(sample_id).exists()

    def save(self, sample_id: str, latent: torch.Tensor):
        """latentをキャッシュに保存"""
        path = self.get_cache_path(sample_id)
        torch.save(latent.cpu(), path)

    def load(self, sample_id: str) -> torch.Tensor:
        """キャッシュからlatentを読み込み"""
        path = self.get_cache_path(sample_id)
        return torch.load(path)

    def save_with_metadata(
        self,
        sample_id: str,
        latent: torch.Tensor,
        onomatopoeia: str,
        audio_path: str,
    ):
        """latentとメタデータをまとめて保存"""
        path = self.get_cache_path(sample_id)
        data = {
            'latent': latent.cpu(),
            'onomatopoeia': onomatopoeia,
            'audio_path': audio_path,
        }
        torch.save(data, path)

    def load_with_metadata(self, sample_id: str) -> dict:
        """latentとメタデータを読み込み"""
        path = self.get_cache_path(sample_id)
        return torch.load(path)


if __name__ == "__main__":
    # テスト（実行にはHugging Faceログインが必要）
    import sys

    print("VAEWrapper test")
    print("Note: Requires Hugging Face login and model access")

    # VAEのロードテスト
    try:
        vae = VAEWrapper(device="cuda" if torch.cuda.is_available() else "cpu")
        vae.load()

        print(f"\nVAE Info:")
        print(f"  Latent dim: {vae.latent_dim}")
        print(f"  Sampling rate: {vae.sampling_rate}")
        print(f"  Downsampling ratio: {vae.downsampling_ratio}")

        # ダミー音声でテスト
        sample_size = 65536  # 約1.5秒
        dummy_audio = torch.randn(1, 2, sample_size)  # (batch, channels, samples)
        dummy_audio = dummy_audio.to(vae.device)

        print(f"\nInput shape: {dummy_audio.shape}")

        latent = vae.encode(dummy_audio)
        print(f"Latent shape: {latent.shape}")

        reconstructed = vae.decode(latent)
        print(f"Reconstructed shape: {reconstructed.shape}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you are logged in to Hugging Face and have accepted the model license.")
