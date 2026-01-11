import soundfile as sf
import numpy as np

files = ['test_alpha_1.0.wav', 'test_alpha_2.0.wav', 'test_alpha_3.0.wav']

print('Alpha値による出力の違い:')
print('-' * 50)

base_audio = None
for f in files:
    audio, sr = sf.read(f)
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.abs(audio).max()

    if base_audio is None:
        base_audio = audio
        diff_rms = 0
    else:
        diff = audio - base_audio
        diff_rms = np.sqrt(np.mean(diff ** 2))

    print(f'{f}:')
    print(f'  RMS: {rms:.6f}')
    print(f'  Peak: {peak:.6f}')
    print(f'  Diff from alpha=1.0: {diff_rms:.6f}')
    print()
