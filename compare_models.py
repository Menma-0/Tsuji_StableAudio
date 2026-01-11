"""正規化修正前後のモデル比較スクリプト"""
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from inference.pipeline import InferencePipeline
from models.onomatopoeia_encoder import OnomatopoeiaEncoder

# テスト用の入力
input_audio = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files\a1\cherry1\043.wav")
test_cases = [
    ("コッ", "ガシャン"),
    ("コッ", "カン"),
    ("コッ", "ドン"),
]

print("=" * 70)
print("正規化修正前後のモデル比較")
print("=" * 70)

# 旧モデル（正規化修正前）
print("\n【旧モデル: experiment_38dim】")
print("-" * 70)
old_checkpoint = Path("checkpoints/experiment_38dim/best.pt")
if old_checkpoint.exists():
    pipeline_old = InferencePipeline()
    pipeline_old.load_models(old_checkpoint)

    print(f"\nfeature_std loaded: {pipeline_old.feature_std is not None}")

    for source, target in test_cases:
        output_path = f"compare_old_{source}_{target}.wav"
        print(f"\n{source} → {target}:")
        pipeline_old.edit_audio(
            audio_path=input_audio,
            source_onomatopoeia=source,
            target_onomatopoeia=target,
            alpha=1.0,
            output_path=output_path,
        )
else:
    print("旧モデルが見つかりません")

# 新モデル（正規化修正後）
print("\n" + "=" * 70)
print("【新モデル: experiment_38dim_v2】")
print("-" * 70)
new_checkpoint = Path("checkpoints/experiment_38dim_v2/best.pt")
pipeline_new = InferencePipeline()
pipeline_new.load_models(new_checkpoint)

print(f"\nfeature_std loaded: {pipeline_new.feature_std is not None}")
if pipeline_new.feature_std is not None:
    print(f"feature_std shape: {pipeline_new.feature_std.shape}")
    print(f"feature_std min/max: {pipeline_new.feature_std.min():.4f} / {pipeline_new.feature_std.max():.4f}")

for source, target in test_cases:
    output_path = f"compare_new_{source}_{target}.wav"
    print(f"\n{source} → {target}:")
    pipeline_new.edit_audio(
        audio_path=input_audio,
        source_onomatopoeia=source,
        target_onomatopoeia=target,
        alpha=1.0,
        output_path=output_path,
    )

# Delta F の比較（正規化あり/なし）
print("\n" + "=" * 70)
print("【Delta F の正規化比較】")
print("-" * 70)

encoder = OnomatopoeiaEncoder()
feature_std = pipeline_new.feature_std.cpu()

for source, target in test_cases:
    f1 = encoder.encode_single(source)
    f2 = encoder.encode_single(target)
    delta_f_raw = f2 - f1
    delta_f_normalized = delta_f_raw / feature_std

    print(f"\n{source} → {target}:")
    print(f"  Delta F (正規化なし): norm={delta_f_raw.norm():.4f}, min={delta_f_raw.min():.4f}, max={delta_f_raw.max():.4f}")
    print(f"  Delta F (正規化あり): norm={delta_f_normalized.norm():.4f}, min={delta_f_normalized.min():.4f}, max={delta_f_normalized.max():.4f}")

# 出力ファイルの比較
print("\n" + "=" * 70)
print("【出力ファイル比較】")
print("-" * 70)

import soundfile as sf

for source, target in test_cases:
    old_file = f"compare_old_{source}_{target}.wav"
    new_file = f"compare_new_{source}_{target}.wav"

    if Path(old_file).exists() and Path(new_file).exists():
        old_audio, sr = sf.read(old_file)
        new_audio, _ = sf.read(new_file)

        # 統計比較
        old_rms = np.sqrt(np.mean(old_audio ** 2))
        new_rms = np.sqrt(np.mean(new_audio ** 2))

        # 差分
        diff = old_audio - new_audio
        diff_rms = np.sqrt(np.mean(diff ** 2))

        print(f"\n{source} → {target}:")
        print(f"  旧モデル RMS: {old_rms:.6f}")
        print(f"  新モデル RMS: {new_rms:.6f}")
        print(f"  差分 RMS: {diff_rms:.6f}")
        print(f"  相関係数: {np.corrcoef(old_audio.flatten(), new_audio.flatten())[0,1]:.6f}")

print("\n" + "=" * 70)
print("比較完了！")
print("出力ファイル:")
for source, target in test_cases:
    print(f"  compare_old_{source}_{target}.wav (旧モデル)")
    print(f"  compare_new_{source}_{target}.wav (新モデル)")
