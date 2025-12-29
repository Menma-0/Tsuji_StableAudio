"""複数条件での推論テスト"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.pipeline import InferencePipeline

# パイプライン初期化
pipeline = InferencePipeline()
checkpoint_path = Path(__file__).parent / "checkpoints" / "experiment_38dim" / "best.pt"
pipeline.load_models(checkpoint_path)

input_audio = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files\a1\cherry1\043.wav")
output_dir = Path(__file__).parent

# テスト1: 元音声の再構成（VAE品質確認）
print("\n=== Test 1: VAE Reconstruction ===")
pipeline.reconstruct_audio(
    audio_path=input_audio,
    output_path=output_dir / "output_reconstructed.wav",
)

# テスト2: 異なるalpha値
print("\n=== Test 2: Different alpha values ===")
for alpha in [0.5, 1.0, 2.0]:
    output_path = output_dir / f"output_alpha_{alpha}.wav"
    pipeline.edit_audio(
        audio_path=input_audio,
        source_onomatopoeia="k o q",
        target_onomatopoeia="g a sh a N",
        alpha=alpha,
        output_path=output_path,
    )

# テスト3: 異なるオノマトペペア
print("\n=== Test 3: Different onomatopoeia pairs ===")
pairs = [
    ("k o q", "k a N"),      # コツ → カン（軽→やや重）
    ("k o q", "g o N"),      # コツ → ゴン（軽→重）
    ("k o q", "p o q"),      # コツ → ポツ（硬→柔）
]
for i, (src, tgt) in enumerate(pairs):
    output_path = output_dir / f"output_pair_{i+1}.wav"
    print(f"\nEditing: {src} -> {tgt}")
    pipeline.edit_audio(
        audio_path=input_audio,
        source_onomatopoeia=src,
        target_onomatopoeia=tgt,
        alpha=1.0,
        output_path=output_path,
    )

print("\n=== All tests complete! ===")
print("Generated files:")
print("  - output_reconstructed.wav (VAE reconstruction)")
print("  - output_alpha_0.5.wav, output_alpha_1.0.wav, output_alpha_2.0.wav")
print("  - output_pair_1.wav (k o q -> k a N)")
print("  - output_pair_2.wav (k o q -> g o N)")
print("  - output_pair_3.wav (k o q -> p o q)")
