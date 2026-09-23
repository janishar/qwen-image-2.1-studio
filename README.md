# QwenImage2.1

Download the model
```bash
hf download Qwen/Qwen-Image-2.1 \
  --include "model_index.json" "processor/*" "scheduler/*" "text_encoder/*" "transformer/*" "vae/*" \
  --local-dir ~/models/Qwen-Image-2.1
```

Run with the local copy
```bash
uv run qwen-image-2-1 "A neon shop sign that reads QWEN" -m ~/models/Qwen-Image-2.1
```
Or `export QWEN_IMAGE_21_PATH=~/models/Qwen-Image-2.1` once. Without either, the model is downloaded from the Hub into the HF cache.

## Prompt enhancer (optional)

`--enhance` first rewrites the prompt with a 9B prompt enhancer and lets it pick the aspect ratio. It uses PE-T2I for text-to-image and PE-I2I when `--input` is given. `--ratio`/`--width`/`--height` still win.
```bash
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir ~/models/Qwen-Image-2.1-PE-T2I
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir ~/models/Qwen-Image-2.1-PE-I2I
export QWEN_IMAGE_21_PE_T2I_PATH=~/models/Qwen-Image-2.1-PE-T2I
export QWEN_IMAGE_21_PE_I2I_PATH=~/models/Qwen-Image-2.1-PE-I2I
```
```bash
uv run qwen-image-2-1 "a corgi playing guitar in the rain" --enhance
uv run qwen-image-2-1 "make the sky sunset" --input photo.png --enhance
```
The enhancer is freed before the image model loads. Pass `--pe-model` to point at a specific enhancer instead.

## Studio (web UI)

`web/` is a helmstudio studio: every CLI option in a page, with sessions, reference images, a render queue, live progress and takes kept in helmstudio's gallery. Install it from helmstudio with `helmstudio.yaml`, or run it on its own under `helm dev` (data in `./.helm`):
```bash
QWEN_MODELS=~/models bash web/run.sh
```
`QWEN_MODELS` is the directory holding `Qwen-Image-2.1`, `Qwen-Image-2.1-PE-T2I` and `Qwen-Image-2.1-PE-I2I`. The studio drives the CLI through its `@stage`, `@step` and `@enhanced` output lines.
