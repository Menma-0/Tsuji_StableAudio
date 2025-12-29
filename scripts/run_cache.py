"""
キャッシュ生成スクリプト

全音声ファイルをVAEでエンコードし、latentをキャッシュとして保存する。
学習前に必ず実行すること。

使用方法:
    python scripts/run_cache.py
    python scripts/run_cache.py --force  # 既存キャッシュを上書き
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.cache_latents import create_latent_cache
from config import default_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create latent cache for training")
    parser.add_argument("--force", action="store_true", help="Force recreate all cache")
    args = parser.parse_args()

    print("=" * 60)
    print("Latent Cache Generation")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Audio root: {default_config.data.rwcp_audio_root}")
    print(f"  Training CSV: {default_config.data.training_csv}")
    print(f"  Cache dir: {default_config.data.latent_cache_dir}")
    print(f"  Sample rate: {default_config.audio.sample_rate}")
    print(f"  Sample size: {default_config.audio.sample_size}")

    metadata = create_latent_cache(force_recreate=args.force)

    if metadata:
        print("\n" + "=" * 60)
        print("Cache Summary")
        print("=" * 60)
        print(f"  Total samples: {len(metadata.get('sample_ids', []))}")
        print(f"  Categories: {len(metadata.get('categories', {}))}")
        print(f"  Latent shape: {metadata.get('latent_shape')}")
        print(f"  Feature dim: {metadata.get('feature_dim')}")


if __name__ == "__main__":
    main()
