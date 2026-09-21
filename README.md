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
