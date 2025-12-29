"""
Stable Audio Open 1.0 - Audio-to-Audio 変換スクリプト
入力音声をベースにしてプロンプトに従った変換を行います
"""
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open 1.0 Audio-to-Audio")
    parser.add_argument("--input", type=str, required=True, help="入力音声ファイル")
    parser.add_argument("--prompt", type=str, required=True, help="変換先のスタイルを指定するプロンプト")
    parser.add_argument("--negative_prompt", type=str, default=None, help="ネガティブプロンプト")
    parser.add_argument("--strength", type=float, default=0.7, help="変換強度 (0.0-1.0, 高いほど元音声から離れる)")
    parser.add_argument("--steps", type=int, default=100, help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFGスケール")
    parser.add_argument("--output", type=str, default="output_a2a.wav", help="出力ファイル名")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルをロード
    print("Loading Stable Audio Open 1.0 model...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    # 入力音声を読み込み
    print(f"Loading input audio: {args.input}")
    audio, sr = torchaudio.load(args.input)

    # サンプルレートを変換
    if sr != sample_rate:
        print(f"Resampling from {sr}Hz to {sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)

    # モノラルに変換
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # 長さを調整
    audio_length = audio.shape[-1]
    duration = audio_length / sample_rate
    print(f"Input audio duration: {duration:.2f}s")

    # sample_sizeに合わせてパディングまたはトリミング
    if audio_length > sample_size:
        audio = audio[..., :sample_size]
        print(f"Audio trimmed to {sample_size / sample_rate:.2f}s")
    elif audio_length < sample_size:
        padding = sample_size - audio_length
        audio = torch.nn.functional.pad(audio, (0, padding))
        print(f"Audio padded to {sample_size / sample_rate:.2f}s")

    # バッチ次元を追加
    audio = audio.unsqueeze(0).to(device)

    # シード設定
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # conditioning設定
    audio_duration = sample_size / sample_rate
    conditioning = [{
        "prompt": args.prompt,
        "seconds_start": 0,
        "seconds_total": audio_duration
    }]

    # 入力音声をVAEでエンコード
    print("Encoding input audio...")
    with torch.no_grad():
        # Pretrained modelからVAEを使って潜在空間にエンコード
        if hasattr(model, 'pretransform'):
            init_audio = model.pretransform.encode(audio)
        else:
            init_audio = audio

    # 生成（init_audioを使ったimg2img風の変換）
    print(f"Transforming audio with prompt: '{args.prompt}'")
    print(f"Strength: {args.strength}, Steps: {args.steps}, CFG Scale: {args.cfg_scale}")

    # strengthに基づいてノイズを追加するステップを計算
    init_noise_level = args.strength

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            negative_conditioning=[{"prompt": args.negative_prompt, "seconds_start": 0, "seconds_total": audio_duration}] if args.negative_prompt else None,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device,
            init_audio=init_audio,
            init_noise_level=init_noise_level
        )

    # 出力を正規化して保存
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # 元の長さにトリミング（パディングした場合）
    if audio_length < sample_size:
        output = output[..., :audio_length]

    torchaudio.save(args.output, output, sample_rate)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
