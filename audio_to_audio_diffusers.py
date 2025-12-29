"""
Stable Audio Open 1.0 - Audio-to-Audio (diffusersベース)
入力音声をベースにしてプロンプトに従った変換を行います
"""
import torch
import torchaudio
import soundfile as sf
from diffusers import StableAudioPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stable Audio Open 1.0 Audio-to-Audio (diffusers)")
    parser.add_argument("--input", type=str, required=True, help="入力音声ファイル")
    parser.add_argument("--prompt", type=str, required=True, help="変換先のスタイルを指定するプロンプト")
    parser.add_argument("--negative_prompt", type=str, default="Low quality", help="ネガティブプロンプト")
    parser.add_argument("--strength", type=float, default=0.7, help="変換強度 (0.0-1.0, 高いほど元音声から離れる)")
    parser.add_argument("--steps", type=int, default=100, help="推論ステップ数")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFGスケール")
    parser.add_argument("--output", type=str, default="output_a2a.wav", help="出力ファイル名")
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

    # 入力音声を読み込み
    print(f"Loading input audio: {args.input}")
    audio, sr = torchaudio.load(args.input)

    # サンプルレートを変換
    target_sr = pipe.vae.sampling_rate
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    # モノラルに変換
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # バッチ次元を追加してfloat型に変換
    audio = audio.unsqueeze(0).to(device)
    if device == "cuda":
        audio = audio.half()

    # 音声の長さを計算
    duration = audio.shape[-1] / target_sr
    print(f"Input audio duration: {duration:.2f}s")

    # ジェネレータの設定
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Audio-to-Audio変換
    print(f"Transforming audio with prompt: '{args.prompt}'")
    print(f"Strength: {args.strength}, Steps: {args.steps}, CFG Scale: {args.cfg_scale}")

    # VAEで入力音声をエンコード
    with torch.no_grad():
        latents = pipe.vae.encode(audio).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # ノイズを追加（strengthに基づく）
    noise = torch.randn_like(latents)
    timesteps = pipe.scheduler.timesteps
    num_inference_steps = int(len(timesteps) * args.strength)
    if num_inference_steps < 1:
        num_inference_steps = 1

    # init_timestepの計算
    init_timestep = min(int(args.steps * args.strength), args.steps)
    t_start = max(args.steps - init_timestep, 0)

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps[t_start:]

    # 初期ノイズを追加
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])

    # プロンプトエンコーディング
    prompt_embeds = pipe._encode_prompt(
        args.prompt,
        device,
        1,  # num_images_per_prompt
        True,  # do_classifier_free_guidance
        args.negative_prompt
    )

    # デノイジングループ
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # ノイズ予測
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds
        ).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)

        # デノイズステップ
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # デコード
    with torch.no_grad():
        audio_output = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    audio_output = audio_output.squeeze(0).cpu().float().numpy()

    # 保存
    sf.write(args.output, audio_output.T, target_sr)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
