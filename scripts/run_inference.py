"""
推論実行スクリプト

学習済みモデルを使用して、オノマトペ指示に基づく音声編集を行う。

使用方法:
    python scripts/run_inference.py --input audio.wav --source "k o q" --target "g a sh a N"
    python scripts/run_inference.py --input audio.wav --source "k o q" --target "g a sh a N" --alpha 0.8
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline import InferencePipeline, main as inference_main


if __name__ == "__main__":
    inference_main()
