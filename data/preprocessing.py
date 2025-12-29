"""
音声前処理モジュール

AutoencoderOobleckに適した形式に音声を変換する。
- サンプルレート: 44100Hz
- チャンネル: 2ch（ステレオ）
- 固定長: sample_size
- オンセット揃え（無音トリム）
"""
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


class AudioPreprocessor:
    """
    音声前処理クラス

    RWCP音声（48kHz, mono/stereo, 可変長）を
    AutoencoderOobleck用（44.1kHz, stereo, 固定長）に変換
    """

    def __init__(
        self,
        target_sr: int = 44100,
        target_channels: int = 2,
        sample_size: int = 65536,  # 約1.5秒
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
        normalize: bool = True,
    ):
        """
        Args:
            target_sr: 目標サンプルレート
            target_channels: 目標チャンネル数
            sample_size: 固定長サンプル数
            trim_silence: 無音トリムを行うか
            silence_threshold_db: 無音と判定するしきい値（dB）
            normalize: 正規化を行うか
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.sample_size = sample_size
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db
        self.normalize = normalize

    def load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        音声ファイルを読み込み

        Returns:
            audio: shape (channels, samples)
            sr: サンプルレート
        """
        audio, sr = torchaudio.load(str(audio_path))
        return audio, sr

    def resample(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        サンプルレートを変換

        Args:
            audio: shape (channels, samples)
            orig_sr: 元のサンプルレート

        Returns:
            リサンプル済み音声
        """
        if orig_sr == self.target_sr:
            return audio

        resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return resampler(audio)

    def convert_channels(self, audio: torch.Tensor) -> torch.Tensor:
        """
        チャンネル数を変換

        Args:
            audio: shape (channels, samples)

        Returns:
            shape (target_channels, samples)
        """
        current_channels = audio.shape[0]

        if current_channels == self.target_channels:
            return audio
        elif current_channels == 1 and self.target_channels == 2:
            # Mono to Stereo: 複製
            return audio.repeat(2, 1)
        elif current_channels == 2 and self.target_channels == 1:
            # Stereo to Mono: 平均
            return audio.mean(dim=0, keepdim=True)
        elif current_channels > self.target_channels:
            # チャンネル数削減
            return audio[:self.target_channels]
        else:
            # チャンネル数増加: 最初のチャンネルを複製
            result = audio[0:1].repeat(self.target_channels, 1)
            result[:current_channels] = audio
            return result

    def find_onset(self, audio: torch.Tensor) -> int:
        """
        音声のオンセット（開始位置）を検出

        RMS energyがしきい値を超える最初のフレームを見つける

        Args:
            audio: shape (channels, samples)

        Returns:
            オンセットのサンプルインデックス
        """
        # Monoに変換して計算
        if audio.shape[0] > 1:
            mono = audio.mean(dim=0)
        else:
            mono = audio[0]

        # dBからリニアに変換
        threshold = 10 ** (self.silence_threshold_db / 20)

        # 短いフレームでRMSを計算
        frame_size = 256
        hop_size = 64

        for i in range(0, len(mono) - frame_size, hop_size):
            frame = mono[i:i + frame_size]
            rms = torch.sqrt(torch.mean(frame ** 2))
            if rms > threshold:
                return max(0, i - hop_size * 2)  # 少し手前から

        return 0  # オンセットが見つからない場合は先頭

    def trim_leading_silence(self, audio: torch.Tensor) -> torch.Tensor:
        """
        先頭の無音を除去

        Args:
            audio: shape (channels, samples)

        Returns:
            トリム済み音声
        """
        onset = self.find_onset(audio)
        return audio[:, onset:]

    def pad_or_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """
        固定長にパディングまたはクロップ

        Args:
            audio: shape (channels, samples)

        Returns:
            shape (channels, sample_size)
        """
        current_length = audio.shape[1]

        if current_length == self.sample_size:
            return audio
        elif current_length < self.sample_size:
            # パディング（末尾にゼロ）
            padding = self.sample_size - current_length
            return torch.nn.functional.pad(audio, (0, padding), mode='constant', value=0)
        else:
            # クロップ（先頭から固定長を取得）
            return audio[:, :self.sample_size]

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        音声を正規化（ピーク正規化）

        Args:
            audio: shape (channels, samples)

        Returns:
            正規化済み音声
        """
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # 少しマージンを持たせる
        return audio

    def process(self, audio_path: Path) -> torch.Tensor:
        """
        音声ファイルを前処理

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            前処理済み音声, shape (target_channels, sample_size)
        """
        # 読み込み
        audio, sr = self.load_audio(audio_path)

        # リサンプル
        audio = self.resample(audio, sr)

        # チャンネル変換
        audio = self.convert_channels(audio)

        # 正規化（無音検出の前に行う）
        if self.normalize:
            audio = self.normalize_audio(audio)

        # 無音トリム
        if self.trim_silence:
            audio = self.trim_leading_silence(audio)

        # 固定長化
        audio = self.pad_or_crop(audio)

        return audio

    def process_batch(self, audio_paths: list) -> torch.Tensor:
        """
        複数の音声ファイルをバッチ処理

        Args:
            audio_paths: 音声ファイルパスのリスト

        Returns:
            shape (batch_size, target_channels, sample_size)
        """
        audios = [self.process(p) for p in audio_paths]
        return torch.stack(audios)


if __name__ == "__main__":
    # テスト
    from pathlib import Path

    preprocessor = AudioPreprocessor()

    # テスト用の音声ファイルを探す
    test_dir = Path(r"C:\Users\A6000-2\RWCPSSD_Onomatopoeia\selected_files\a1")
    if test_dir.exists():
        wav_files = list(test_dir.glob("*.wav"))
        if wav_files:
            test_file = wav_files[0]
            print(f"Testing with: {test_file}")

            audio = preprocessor.process(test_file)
            print(f"Output shape: {audio.shape}")
            print(f"Sample rate: {preprocessor.target_sr}")
            print(f"Duration: {audio.shape[1] / preprocessor.target_sr:.3f} seconds")
            print(f"Max value: {audio.abs().max():.4f}")
