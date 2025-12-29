"""
システム設定パラメータ
オノマトペ差分から音声差分を予測するシステムの設定
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AudioConfig:
    """音声処理の設定"""
    sample_rate: int = 44100  # AutoencoderOobleckのサンプルレート
    audio_channels: int = 2   # ステレオ
    # Stable Audio Openの固定長（約47秒 = 2097152サンプル）
    # 短い音声用に短縮版も用意
    sample_size: int = 65536  # 約1.5秒（短い効果音向け）
    # sample_size: int = 2097152  # フル版（約47秒）

    # latent空間の設定
    latent_dim: int = 64  # decoder_input_channels
    downsampling_ratio: int = 2048  # 2×4×4×8×8

    @property
    def latent_length(self) -> int:
        """latentの時間長"""
        return self.sample_size // self.downsampling_ratio


@dataclass
class OnomatopoeiaConfig:
    """オノマトペ特徴量の設定（38次元版）

    特徴量グループ:
    - グループA: 全体構造・繰り返し (6次元)
    - グループB: 長さ・アクセント (4次元)
    - グループC: 母音ヒストグラム (5次元)
    - グループD: 子音カテゴリ・ヒストグラム (6次元)
    - グループE: 子音比率のサマリ (3次元)
    - グループF: 位置情報 (14次元)
    """
    # 特徴量の次元（固定）
    feature_dim: int = 38


@dataclass
class ModelConfig:
    """差分予測モデル（MLP）の設定"""
    # 入力：オノマトペ特徴量差分の次元（38次元）
    # 出力：latent差分の次元（64 × latent_length = 64 × 32 = 2048）
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    dropout: float = 0.1
    use_residual: bool = True

    # 正則化
    delta_norm_weight: float = 0.01  # ||Δg_pred||の正則化重み

    # 学習設定
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100

    # 推論時のスケール係数
    alpha: float = 1.0  # g(l2)' = g(l1) + α * Δg_pred


@dataclass
class DataConfig:
    """データパスの設定"""
    # RWCPデータセット
    rwcp_root: Path = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia")
    rwcp_audio_root: Path = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files")
    training_csv: Path = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\training_data_en_utf8bom.csv")

    # キャッシュディレクトリ
    cache_dir: Path = Path(r"C:\Users\A6000-2\Documents\Tsuji_StableAudio - コピー\cache")
    latent_cache_dir: Path = Path(r"C:\Users\A6000-2\Documents\Tsuji_StableAudio - コピー\cache\latents")
    feature_cache_dir: Path = Path(r"C:\Users\A6000-2\Documents\Tsuji_StableAudio - コピー\cache\features")

    # モデル保存
    checkpoint_dir: Path = Path(r"C:\Users\A6000-2\Documents\Tsuji_StableAudio - コピー\checkpoints")

    # データ分割
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # フィルタリング
    min_confidence: int = 4  # confidence >= 4 のデータのみ使用
    min_acceptability: float = 4.0  # avg_acceptability >= 4.0 のデータのみ使用


@dataclass
class Config:
    """全体設定"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    onomatopoeia: OnomatopoeiaConfig = field(default_factory=OnomatopoeiaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # デバイス
    device: str = "cuda"

    # Hugging Face
    hf_model_id: str = "stabilityai/stable-audio-open-1.0"

    def __post_init__(self):
        """ディレクトリの作成"""
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.latent_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# デフォルト設定のインスタンス
default_config = Config()
