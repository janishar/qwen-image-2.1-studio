# QwenImage2.1

Download the model
```bash
hf download Qwen/Qwen-Image-2.1 \
  --include "model_index.json" "processor/*" "scheduler/*" "text_encoder/*" "transformer/*" "vae/*" \
  --local-dir ~/models/Qwen-Image-2.1
```