"""
Stable Audio Open 1.0 - Text-to-Audio (diffusersベース)
よりシンプルで依存関係が少ない実装
"""
import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open 1.0 Text-to-Audio (diffusers)")
    parser.add_argument("--prompt", type=str, required=True, help="生成する音声のテキストプロンプト")
    parser.add_argument("--negative_prompt", type=str, default="Low quality", help="ネガティブプロンプト")
    parser.add_argument("--duration", type=float, default=10.0, help="生成する音声の長さ（秒）")
    parser.add_argument("--steps", type=int, default=100, help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFGスケール")
    parser.add_argument("--output", type=str, default="output.wav", help="出力ファイル名")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # パイプラインをロード
    print("Loading Stable Audio Open 1.0 pipeline...")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # ジェネレータの設定
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # 音声を生成
    print(f"Generating audio: '{args.prompt}'")
    print(f"Duration: {args.duration}s, Steps: {args.steps}, CFG Scale: {args.cfg_scale}")

    audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        audio_end_in_s=args.duration,
        guidance_scale=args.cfg_scale,
        generator=generator
    ).audios[0]

    # 保存
    audio_np = audio.cpu().float().numpy() if torch.is_tensor(audio) else audio
    sf.write(args.output, audio_np.T, pipe.vae.sampling_rate)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
