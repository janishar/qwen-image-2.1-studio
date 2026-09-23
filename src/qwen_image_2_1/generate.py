"""Generate / edit images with Qwen-Image-2.1 on Apple Silicon (MPS), CUDA or CPU.

Usage:
    uv run python -m qwen_image_2_1.generate "A neon shop sign that reads QWEN"
    uv run python -m qwen_image_2_1.generate "Edit: sunset" --input photo.png
    uv run python -m qwen_image_2_1.generate "Dragon sticker" --transparent
    uv run qwen-image-2-1 "Prompt" -m ~/models/Qwen-Image-2.1
    QWEN_IMAGE_21_PATH=~/models/Qwen-Image-2.1 uv run qwen-image-2-1 "Prompt"
    uv run qwen-image-2-1 "a corgi playing guitar in the rain" --enhance
    uv run qwen-image-2-1 "make the sky sunset" --input photo.png --enhance
    uv run qwen-image-2-1 "Prompt" --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image


def patch_mps_empty_cache() -> None:
    # torch 2.14: mps.empty_cache() waits on the GPU while holding the GIL, but finishing the
    # non_blocking weight copies device_map makes needs the GIL -> deadlock. synchronize() releases it.
    # ponytail: global monkeypatch, drop once torch's empty_cache releases the GIL itself
    empty_cache = torch.mps.empty_cache

    def safe_empty_cache() -> None:
        torch.mps.synchronize()
        empty_cache()

    torch.mps.empty_cache = safe_empty_cache


DEVICES = ("auto", "mps", "cuda", "cpu")


def pick_device(requested: str) -> str:
    """The torch device to run on: the one asked for, or with "auto" the first of MPS, CUDA, CPU present."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(d for d, ok in available.items() if ok)
    if not available[requested]:
        raise ValueError(f"{requested.upper()} is not available in this torch build")
    return requested


def empty_cache(device: str) -> None:
    """Hand freed memory back to the device, so the next model has room."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def noise_seed(seed: int, images: list[Image.Image] | None) -> int:
    """The seed for the starting noise: `seed` itself, or for an edit, `seed` mixed with the inputs' pixels.

    Editing an image with the seed and size it was generated with would start from the exact noise that made it,
    and the model then over-sharpens it into a crunchy, HDR-like texture instead of editing. Mixing in the inputs
    gives an edit its own noise, while the same seed and inputs still reproduce the same take.
    """
    if not images:
        return seed
    digest = hashlib.sha256(str(seed).encode())
    for img in images:
        digest.update(f"{img.mode}{img.size}".encode())
        digest.update(img.tobytes())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


def load_pipe(model: str, device: str = "mps", dtype: torch.dtype = torch.bfloat16):
    from diffusers import QwenImage21Pipeline  # type: ignore[import-not-found]

    # device_map loads each weight straight to the GPU instead of building a full CPU copy first
    return QwenImage21Pipeline.from_pretrained(model, dtype=dtype, device_map=device)


def stage(name: str) -> None:
    # machine-readable marker for the studio UI: enhance, load, denoise, decode, save
    print(f"@stage {name}", flush=True)


def on_step_end(pipe, i: int, t, kwargs: dict) -> dict:
    n = pipe._num_timesteps
    print(f"@step {i + 1}/{n}", flush=True)
    if i + 1 == n:  # the pipeline decodes the latents right after the last step
        stage("decode")
    return kwargs


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
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 generation")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--input", help="Input image(s) for editing, comma-separated")
    parser.add_argument(
        "--ratio",
        choices=ASPECT_RATIOS,
        help="Output aspect ratio (default: 1:1, or the input image's aspect with --input)",
    )
    parser.add_argument("--width", type=int, help="Override width (multiple of 32)")
    parser.add_argument("--height", type=int, help="Override height (multiple of 32)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transparent", action="store_true", help="Generate an RGBA transparent image"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=os.path.expanduser,
        default=os.environ.get("QWEN_IMAGE_21_PATH") or "Qwen/Qwen-Image-2.1",
        help="Local path or HF repo ID (default: %(default)s, set via QWEN_IMAGE_21_PATH)",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Rewrite the prompt (and pick the aspect ratio) with Qwen-Image-2.1-PE-T2I / PE-I2I first",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="With --enhance: let the prompt enhancer reason before answering (slower)",
    )
    parser.add_argument(
        "--pe-model",
        type=os.path.expanduser,
        help="Prompt enhancer path or HF repo ID (default: QWEN_IMAGE_21_PE_T2I_PATH / "
        "QWEN_IMAGE_21_PE_I2I_PATH, else Qwen/Qwen-Image-2.1-PE-T2I / -PE-I2I)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=os.environ.get("QWEN_IMAGE_21_DEVICE") or "auto",
        help="Where to run (default: %(default)s = MPS, else CUDA, else CPU; set via QWEN_IMAGE_21_DEVICE)",
    )
    parser.add_argument("--output", default="output/output.png")
    args = parser.parse_args()

    kind = "I2I" if args.input else "T2I"
    pe_model = args.pe_model or os.path.expanduser(
        os.environ.get(f"QWEN_IMAGE_21_PE_{kind}_PATH")
        or f"Qwen/Qwen-Image-2.1-PE-{kind}"
    )

    # ponytail: a relative "dir/name" path looks like a Hub ID, so only absolute, dotted or multi-segment paths are checked
    for model in [args.model, pe_model] if args.enhance else [args.model]:
        if not os.path.isdir(model) and (
            model.startswith((".", "/")) or model.count("/") != 1
        ):
            parser.error(f"model directory not found: {model}")
    try:
        device = pick_device(args.device)
    except ValueError as e:
        parser.error(str(e))
    print(f"Device: {device}", flush=True)

    images: list[Image.Image] | None = None
    if args.input:
        images = [Image.open(p.strip()) for p in args.input.split(",")]
        if len(images) > 10:
            parser.error("Qwen-Image-2.1 supports at most 10 input images")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "mps":
        patch_mps_empty_cache()

    prompt, pe_ratio = args.prompt, None
    if args.enhance:
        from qwen_image_2_1.enhance import enhance

        stage("enhance")
        prompt, pe_ratio = enhance(
            prompt, images, pe_model, args.seed, args.think, device
        )
        print(f"Enhanced prompt ({pe_ratio or 'ratio from input'}):\n{prompt}\n")
        print(
            f"@enhanced {json.dumps({'prompt': prompt, 'ratio': pe_ratio})}", flush=True
        )

    ratio = args.ratio or pe_ratio or (None if args.input else "1:1")
    width, height = ASPECT_RATIOS[ratio] if ratio else (None, None)
    width, height = args.width or width, args.height or height
    if args.transparent:
        prompt = (
            "This is an RGBA image with transparency. "
            f"{prompt} The image has alpha channel and the background is transparent."
        )

    stage("load")
    pipe = load_pipe(args.model, device)

    stage("denoise")

    out = pipe(
        prompt=prompt,
        image=images,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(
            noise_seed(args.seed, images)
        ),
        callback_on_step_end=on_step_end,
    ).images[
        0
    ]  # pyright: ignore[reportAttributeAccessIssue]

    stage("save")
    try:
        out.save(out_path)
    except (
        OSError,
        ValueError,
    ) as e:  # e.g. RGBA as JPEG, unknown extension: keep the render as PNG
        out_path = out_path.with_suffix(".png")
        print(f"Could not save {args.output} ({e}); saving PNG instead")
        out.save(out_path)
    print(f"Saved {out_path.resolve()} ({out.size[0]}x{out.size[1]}, mode={out.mode})")


if __name__ == "__main__":
    main()
