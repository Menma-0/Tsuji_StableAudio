"""推論テストスクリプト（v2モデル用）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.pipeline import InferencePipeline

# パイプライン初期化
pipeline = InferencePipeline()
checkpoint_path = Path(__file__).parent / "checkpoints" / "experiment_38dim_v2" / "best.pt"
pipeline.load_models(checkpoint_path)

# テスト1: RWCPデータで編集テスト (koq -> gashaN)
input_audio = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files\a1\cherry1\043.wav")
output_audio = Path(__file__).parent / "output_test_v2_gashaN.wav"

print("=" * 60)
print("Test 1: koq -> gashaN")
print("=" * 60)
pipeline.edit_audio(
    audio_path=input_audio,
    source_onomatopoeia="k o q",
    target_onomatopoeia="g a sh a N",
    alpha=1.0,
    output_path=output_audio,
)

# テスト2: 別のオノマトペ変換 (koq -> kaN)
output_audio2 = Path(__file__).parent / "output_test_v2_kaN.wav"
print()
print("=" * 60)
print("Test 2: koq -> kaN")
print("=" * 60)
pipeline.edit_audio(
    audio_path=input_audio,
    source_onomatopoeia="k o q",
    target_onomatopoeia="k a N",
    alpha=1.0,
    output_path=output_audio2,
)

# テスト3: オノマトペ類似度テスト
print()
print("=" * 60)
print("Onomatopoeia Similarity Test")
print("=" * 60)
pairs = [
    ("k o q", "g a sh a N"),
    ("k o q", "k a N"),
    ("k o q", "k o q k o q"),
    ("p a t a p a t a", "b a t a b a t a"),
]
for z1, z2 in pairs:
    sim = pipeline.get_onomatopoeia_similarity(z1, z2)
    print(f"  '{z1}' vs '{z2}': {sim:.4f}")

print()
print("=" * 60)
print("Inference tests complete!")
print(f"Output 1: {output_audio}")
print(f"Output 2: {output_audio2}")
