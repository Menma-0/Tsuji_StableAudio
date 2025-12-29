"""
Stable Audio Open 1.0 - Text-to-Audio 生成スクリプト
"""
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open 1.0 Text-to-Audio Generation")
    parser.add_argument("--prompt", type=str, required=True, help="生成する音声のテキストプロンプト")
    parser.add_argument("--negative_prompt", type=str, default=None, help="ネガティブプロンプト")
    parser.add_argument("--duration", type=float, default=10.0, help="生成する音声の長さ（秒）")
    parser.add_argument("--steps", type=int, default=100, help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFGスケール")
    parser.add_argument("--output", type=str, default="output.wav", help="出力ファイル名")
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

    # シード設定
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # conditioning設定
    conditioning = [{
        "prompt": args.prompt,
        "seconds_start": 0,
        "seconds_total": args.duration
    }]

    # 生成
    print(f"Generating audio: '{args.prompt}'")
    print(f"Duration: {args.duration}s, Steps: {args.steps}, CFG Scale: {args.cfg_scale}")

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            negative_conditioning=[{"prompt": args.negative_prompt, "seconds_start": 0, "seconds_total": args.duration}] if args.negative_prompt else None,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

    # 出力を正規化して保存
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save(args.output, output, sample_rate)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
