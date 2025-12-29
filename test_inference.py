"""推論テストスクリプト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.pipeline import InferencePipeline

# パイプライン初期化
pipeline = InferencePipeline()
checkpoint_path = Path(__file__).parent / "checkpoints" / "experiment_38dim" / "best.pt"
pipeline.load_models(checkpoint_path)

# 音声編集テスト
input_audio = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files\a1\cherry1\043.wav")
output_audio = Path(__file__).parent / "output_test.wav"

pipeline.edit_audio(
    audio_path=input_audio,
    source_onomatopoeia="k o q",
    target_onomatopoeia="g a sh a N",
    alpha=1.0,
    output_path=output_audio,
)

print("\nInference complete!")
print(f"Output saved to: {output_audio}")
