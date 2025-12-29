"""音声長分布の分析"""
import soundfile as sf
from pathlib import Path
import numpy as np

audio_dir = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files")
files = list(audio_dir.glob("**/*.wav"))

durations = []
for f in files:
    try:
        info = sf.info(str(f))
        durations.append(info.duration)
    except:
        pass

durations = np.array(durations)
print(f"Analyzed: {len(durations)} files")
print(f"\nDuration statistics:")
print(f"  Min:    {durations.min():.3f}s")
print(f"  Max:    {durations.max():.3f}s")
print(f"  Mean:   {durations.mean():.3f}s")
print(f"  Median: {np.median(durations):.3f}s")
print(f"  Std:    {durations.std():.3f}s")
print(f"\nCoverage with 1.5s window (65536 samples @ 44.1kHz):")
print(f"  <0.5s: {(durations < 0.5).sum()} files ({(durations < 0.5).mean()*100:.1f}%)")
print(f"  <1.0s: {(durations < 1.0).sum()} files ({(durations < 1.0).mean()*100:.1f}%)")
print(f"  <1.5s: {(durations < 1.5).sum()} files ({(durations < 1.5).mean()*100:.1f}%)")
print(f"  <2.0s: {(durations < 2.0).sum()} files ({(durations < 2.0).mean()*100:.1f}%)")
print(f"  >=1.5s (truncated): {(durations >= 1.5).sum()} files ({(durations >= 1.5).mean()*100:.1f}%)")
