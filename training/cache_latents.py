"""
Latentキャッシュ生成

全音声ファイルをVAEでエンコードし、latentをキャッシュとして保存する。
同時にオノマトペ特徴量も保存する。

これにより学習時のVAE計算を省略でき、高速化できる。
"""
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.rwcp_loader import RWCPLoader, RWCPSample
from data.preprocessing import AudioPreprocessor
from models.vae_wrapper import VAEWrapper, LatentCache
from models.onomatopoeia_encoder import OnomatopoeiaEncoder
from config import Config, default_config


def create_latent_cache(
    config: Config = default_config,
    force_recreate: bool = False,
) -> dict:
    """
    全データのlatentとオノマトペ特徴量をキャッシュとして保存

    Args:
        config: 設定
        force_recreate: Trueなら既存キャッシュを上書き

    Returns:
        {
            'sample_ids': キャッシュ済みサンプルIDのリスト,
            'categories': カテゴリ→サンプルIDリストの辞書,
            'latent_shape': latentの形状,
            'feature_dim': オノマトペ特徴量の次元,
        }
    """
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # データローダー
    print("\n[1/4] Loading RWCP dataset...")
    loader = RWCPLoader(
        csv_path=config.data.training_csv,
        audio_root=config.data.rwcp_audio_root,
        min_confidence=config.data.min_confidence,
        min_acceptability=config.data.min_acceptability,
    )

    if len(loader) == 0:
        print("Error: No samples loaded")
        return {}

    # 前処理器
    print("\n[2/4] Initializing preprocessor and models...")
    preprocessor = AudioPreprocessor(
        target_sr=config.audio.sample_rate,
        target_channels=config.audio.audio_channels,
        sample_size=config.audio.sample_size,
    )

    # VAE
    vae = VAEWrapper(
        model_id=config.hf_model_id,
        device=device,
    )
    vae.load()

    # オノマトペエンコーダ（38次元版）
    onomatopoeia_encoder = OnomatopoeiaEncoder()

    # キャッシュ
    cache = LatentCache(config.data.latent_cache_dir)

    # 処理
    print(f"\n[3/4] Processing {len(loader)} samples...")
    sample_ids = []
    categories = {}
    latent_shape = None

    for sample in tqdm(loader.samples):
        sample_id = sample.sample_id

        # 既存キャッシュをスキップ
        if not force_recreate and cache.exists(sample_id):
            sample_ids.append(sample_id)
            if sample.category not in categories:
                categories[sample.category] = []
            categories[sample.category].append(sample_id)
            continue

        try:
            # 音声の前処理
            audio = preprocessor.process(sample.audio_path)
            audio = audio.unsqueeze(0).to(device)  # (1, channels, samples)

            # VAEエンコード
            with torch.no_grad():
                latent = vae.encode(audio)  # (latent_dim, latent_length)

            if latent_shape is None:
                latent_shape = latent.shape

            # オノマトペ特徴量
            with torch.no_grad():
                feature = onomatopoeia_encoder.encode_single(sample.onomatopoeia)

            # キャッシュに保存
            cache_data = {
                'latent': latent.cpu(),
                'feature': feature.cpu(),
                'onomatopoeia': sample.onomatopoeia,
                'audio_path': str(sample.audio_path),
                'category': sample.category,
            }
            torch.save(cache_data, cache.get_cache_path(sample_id))

            sample_ids.append(sample_id)
            if sample.category not in categories:
                categories[sample.category] = []
            categories[sample.category].append(sample_id)

        except Exception as e:
            print(f"\nError processing {sample_id}: {e}")
            continue

    # 結果サマリ
    print(f"\n[4/4] Cache creation complete!")
    print(f"  Samples cached: {len(sample_ids)}")
    print(f"  Categories: {len(categories)}")
    if latent_shape:
        print(f"  Latent shape: {latent_shape}")
    print(f"  Feature dim: {onomatopoeia_encoder.feature_dim}")

    # メタデータを保存
    metadata = {
        'sample_ids': sample_ids,
        'categories': categories,
        'latent_shape': list(latent_shape) if latent_shape else None,
        'feature_dim': onomatopoeia_encoder.feature_dim,
        'config': {
            'sample_rate': config.audio.sample_rate,
            'sample_size': config.audio.sample_size,
            'latent_dim': config.audio.latent_dim,
        }
    }
    torch.save(metadata, config.data.latent_cache_dir / "metadata.pt")

    return metadata


def load_cache_metadata(config: Config = default_config) -> dict:
    """キャッシュのメタデータを読み込み"""
    metadata_path = config.data.latent_cache_dir / "metadata.pt"
    if metadata_path.exists():
        return torch.load(metadata_path)
    return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create latent cache")
    parser.add_argument("--force", action="store_true", help="Force recreate cache")
    args = parser.parse_args()

    metadata = create_latent_cache(force_recreate=args.force)

    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            if key == 'sample_ids':
                print(f"  {key}: {len(value)} samples")
            elif key == 'categories':
                print(f"  {key}: {len(value)} categories")
            else:
                print(f"  {key}: {value}")
