"""
学習実行スクリプト

差分予測モデル（Δf → Δg）を学習する。
事前にrun_cache.pyでキャッシュを生成しておくこと。

使用方法:
    python scripts/run_train.py
    python scripts/run_train.py --epochs 200 --batch_size 64
    python scripts/run_train.py --resume checkpoints/latest.pt
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import Trainer, main as train_main


if __name__ == "__main__":
    train_main()
