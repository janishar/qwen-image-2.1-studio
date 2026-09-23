"""Rewrite prompts with Qwen-Image-2.1-PE-T2I / PE-I2I before generation."""

from __future__ import annotations

import gc
import json
import math
import os
import re
from pathlib import Path

import torch
from PIL import Image

from qwen_image_2_1.generate import ASPECT_RATIOS, empty_cache


def nearest_ratio(w: float, h: float) -> str:
    return min(
        ASPECT_RATIOS,
        key=lambda r: abs(
            math.log(w / h) - math.log(ASPECT_RATIOS[r][0] / ASPECT_RATIOS[r][1])
        ),
    )


def enhance(
    prompt: str,
    images: list[Image.Image] | None,
    model_path: str,
    seed: int,
    think: bool = False,
    device: str = "mps",
) -> tuple[str, str | None]:
    """Return (rewritten prompt, ASPECT_RATIOS key or None). Loads the PE on `device` and frees it before returning."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        TextStreamer,
    )

    if os.path.isdir(model_path):
        system_prompt = (Path(model_path) / "system_prompt.txt").read_text().strip()
    else:
        from huggingface_hub import hf_hub_download

        system_prompt = (
            Path(hf_hub_download(model_path, "system_prompt.txt")).read_text().strip()
        )

    torch.manual_seed(seed)
    if images is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        ).eval()
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        max_new_tokens = 16256
    else:
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        ).eval()
        content = [{"type": "image", "image": img.convert("RGB")} for img in images]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [*content, {"type": "text", "text": prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=think,
        ).to(model.device)
        max_new_tokens = 24000

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            # print the thinking + answer live, the rewrite takes minutes
            streamer=TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            ),
        )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # the 9B PE and the image pipeline don't fit in memory together
    del model, inputs, out
    gc.collect()
    empty_cache(device)

    try:
        result = json.loads(gen.rpartition("</think>")[2].strip())
    except json.JSONDecodeError:
        raise SystemExit(f"Prompt enhancer did not return JSON:\n{gen}") from None

    ratio = None
    if m := re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", result.get("wh_ratio") or ""):
        ratio = nearest_ratio(int(m[1]), int(m[2]))
    if (
        (m := re.fullmatch(r"<image(\d+)>", result.get("ratio_follow") or ""))
        and images
        and 1 <= int(m[1]) <= len(images)
    ):
        ratio = nearest_ratio(*images[int(m[1]) - 1].size)
    return result["rewritten_prompt"], ratio


if __name__ == "__main__":
    assert nearest_ratio(1, 1) == "1:1"
    assert nearest_ratio(16, 9) == "16:9"
    assert nearest_ratio(1920, 1080) == "16:9"
    assert nearest_ratio(3000, 4000) == "3:4"
    assert nearest_ratio(21, 9) == "16:9"
    print("ok")
