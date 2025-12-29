# Stable Audio Open 1.0 音声生成環境

Stable Audio Open 1.0を使用したText-to-AudioおよびAudio-to-Audio変換のための環境です。

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install torch torchaudio diffusers transformers accelerate soundfile scipy
```

### 2. Hugging Faceへのログイン（初回のみ）

モデルを使用するにはHugging Faceアカウントが必要です。
https://huggingface.co/stabilityai/stable-audio-open-1.0 でライセンスに同意してください。

```bash
huggingface-cli login
```

## 使い方

### Text-to-Audio（テキストから音声生成）

```bash
python generate_audio_diffusers.py --prompt "A dog barking in a park" --duration 10 --output dog_barking.wav
```

オプション:
- `--prompt`: 生成したい音声の説明（必須）
- `--negative_prompt`: 避けたい特徴（デフォルト: "Low quality"）
- `--duration`: 音声の長さ（秒）（デフォルト: 10.0）
- `--steps`: 推論ステップ数（デフォルト: 100）
- `--cfg_scale`: CFGスケール（デフォルト: 7.0）
- `--output`: 出力ファイル名（デフォルト: output.wav）
- `--seed`: 乱数シード

### Audio-to-Audio（音声スタイル変換）

```bash
python audio_to_audio_diffusers.py --input input.wav --prompt "Jazz music with saxophone" --output jazz_version.wav
```

オプション:
- `--input`: 入力音声ファイル（必須）
- `--prompt`: 変換先のスタイル（必須）
- `--negative_prompt`: 避けたい特徴
- `--strength`: 変換強度 0.0-1.0（デフォルト: 0.7、高いほど元音声から離れる）
- `--steps`: 推論ステップ数（デフォルト: 100）
- `--cfg_scale`: CFGスケール（デフォルト: 7.0）
- `--output`: 出力ファイル名（デフォルト: output_a2a.wav）
- `--seed`: 乱数シード

## プロンプト例

### 音楽
- "Acoustic guitar melody with soft drums"
- "Electronic dance music with heavy bass"
- "Piano jazz improvisation"
- "Orchestral dramatic soundtrack"

### 効果音
- "Thunder and rain storm"
- "Footsteps on gravel"
- "Car engine starting"
- "Glass breaking"

### 環境音
- "Birds singing in a forest"
- "Ocean waves on a beach"
- "Busy city street with traffic"
- "Quiet library with occasional page turning"

## 注意事項

- GPUを推奨（CUDAが利用可能な場合は自動的に使用されます）
- 初回実行時にモデルがダウンロードされます（約2GB）
- 生成される音声は最大47秒まで
- モデルはCC-BY-NC-4.0ライセンスです（非商用利用のみ）

## ファイル構成

```
Tsuji_StableAudio/
├── README.md                    # このファイル
├── requirements.txt             # 依存パッケージ
├── generate_audio_diffusers.py  # Text-to-Audio（推奨）
├── audio_to_audio_diffusers.py  # Audio-to-Audio（推奨）
├── generate_audio.py            # Text-to-Audio（stable-audio-tools版）
└── audio_to_audio.py            # Audio-to-Audio（stable-audio-tools版）
```
