"""Generate / edit images with Qwen-Image-2.1 on Apple Silicon (MPS).

Usage:
    uv run python -m qwen_image_2_1.generate "A neon shop sign that reads QWEN"
    uv run python -m qwen_image_2_1.generate "Edit: change background to sunset" --input photo.png
    uv run python -m qwen_image_2_1.generate "RGBA sticker of a dragon" --transparent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def load_pipe(device: str = "mps", dtype: torch.dtype = torch.bfloat16):
    from diffusers import QwenImage21Pipeline  # type: ignore[import-not-found]

    return QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1", torch_dtype=dtype
    ).to(device)


ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 generation on MPS")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--input", help="Input image(s) for editing, comma-separated")
    parser.add_argument("--ratio", choices=ASPECT_RATIOS, default="1:1")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transparent", action="store_true",
                        help="Generate an RGBA transparent image")
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    width, height = ASPECT_RATIOS[args.ratio]
    prompt = args.prompt
    if args.transparent:
        prompt = (
            "This is an RGBA image with transparency. "
            f"{prompt} The image has alpha channel and the background is transparent."
        )

    images = None
    if args.input:
        images = [Image.open(p) for p in args.input.split(",")]

    # bf16 on MPS: run denoising in fp16/fp32 fallback if a layer errors
    pipe = load_pipe(device="mps", dtype=torch.bfloat16)

    out = pipe(
        prompt=prompt,
        image=images,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]

    out_path = Path(args.output)
    out.save(out_path)
    print(f"Saved {out_path.resolve()} ({out.size[0]}x{out.size[1]}, mode={out.mode})")


if __name__ == "__main__":
    main()
