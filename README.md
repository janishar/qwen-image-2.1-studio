# qwen image studio

**A command line and a local web studio for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)** —
text-to-image and multi-image editing on Apple Silicon, with Qwen's PE prompt
enhancers.

`qwen-image-2-1` is a small PyTorch CLI around diffusers' `QwenImage21Pipeline`:
it loads the weights straight onto the GPU, rewrites the prompt with the 9B
enhancer when asked, and frees the enhancer before the image model loads. The
studio in `web/` is a standard-library Python server that drives that CLI — it
builds the arguments, queues renders and streams their progress, in a browser
tab, nothing sent off your machine. The studio runs as a
[helmstudio][helmstudio] studio and only that way: helmstudio installs it,
launches it, keeps its sessions, and takes every finished take into the library
it shares with the other studios. The CLI needs none of that.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%28Apple%20Silicon%29-lightgrey?logo=apple)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

## Table of contents

- [Requirements](#requirements) · [Weights](#weights)
- [The studio](#the-studio) — [Path A: the launcher](#path-a-install-from-the-launcher-recommended) ·
  [Path B: a checkout](#path-b-run-from-a-checkout) ·
  [Using it](#using-the-studio) · [What helmstudio adds](#what-helmstudio-adds) ·
  [Sessions and state](#sessions-and-state)
- [Command line](#command-line) — [Prompt enhancer](#prompt-enhancer) ·
  [Editing](#editing-with-reference-images) · [Device](#device) ·
  [Flags](#cli-reference) · [Environment variables](#environment-variables)
- [Security](#security) · [Limits](#limits) · [Contributing](#contributing) ·
  [License](#license)

## Requirements

### Hardware and OS

macOS on Apple Silicon with **64 GB** of unified memory is what this is built
and tested on. The image model is ~30 GB of bf16 weights (transformer 13 GB,
text encoder 16 GB, VAE 1.3 GB); the 9B enhancer is freed before it loads, so
the two never sit in memory together. `--device cuda` and `--device cpu` exist
for other machines, untested here — see [Device](#device).

### Toolchain

| | Why |
| --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | The environment. `uv sync` installs exactly `uv.lock`. |
| `git` | `uv.lock` pins diffusers to its git `main`, where `QwenImage21Pipeline` lives. |
| [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) | Downloading the weights, if helmstudio is not doing it. |
| **helmstudio's `helm`** | The studio only. It does not start without it — see [The studio](#the-studio). |

```bash
uv sync
```

## Weights

Three repositories. Only the image model is needed; the two enhancers are for
`--enhance`.

| Weight | Hugging Face | Disk | Used for |
| --- | --- | --- | --- |
| Image model | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) | ~30 GB | every render |
| PE-T2I | [`Qwen/Qwen-Image-2.1-PE-T2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) | ~18 GB | `--enhance` without reference images |
| PE-I2I | [`Qwen/Qwen-Image-2.1-PE-I2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) | ~18 GB | `--enhance` with reference images |

```bash
hf download Qwen/Qwen-Image-2.1 \
  --include "model_index.json" "processor/*" "scheduler/*" "text_encoder/*" "transformer/*" "vae/*" \
  --local-dir ~/models/Qwen-Image-2.1
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir ~/models/Qwen-Image-2.1-PE-T2I
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir ~/models/Qwen-Image-2.1-PE-I2I
```

Without a local copy the CLI downloads from the Hub into the HF cache. Installed
through the launcher, helmstudio downloads all three itself.

## The studio

Two ways in — **the studio never starts on its own**, since whatever launches it
also gives it the platform that keeps its sessions and takes.

| | **A — Install it** | **B — Run from a checkout** |
| --- | --- | --- |
| For | using the studio | changing the studio |
| Needs | the **launcher**, helmstudio's daemon and web UI | the **`helm` CLI**; `helm dev` runs one studio, no daemon |
| Install | one click in its library | `git clone`, `uv sync`, `bash web/run.sh` |
| Weights | it downloads or links them | you point `QWEN_MODELS` at them |

### Path A: Install from the launcher (recommended)

**1. Install the launcher** — helmstudio itself, a daemon and a web UI: the Mac
app from [its releases][helm-releases], or a clone:

```bash
git clone https://github.com/janishar/helmstudio && cd helmstudio && make build && ./bin/helmstudio
```

Then open **http://127.0.0.1:8700**; everything lives in `~/.helmstudio`.

**2. Install qwen image studio from its library.** It is in helmstudio's
registry, so it is already listed. Install shows every command it will run —
`uv sync --locked` — and the three weights it will fetch, about 67 GB.

**3. Start it.** The launcher gives the studio a port and the weights' paths,
then opens its page → [Using the studio](#using-the-studio).

### Path B: Run from a checkout

The developer's path: a checkout run against helmstudio's platform API under
`helm dev`, which keeps what the studio stores in `./.helm`.

```bash
# 1. helm, the CLI that runs this checkout
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/janishar/helmstudio/main/installer/install.sh)"

# 2. the checkout and its environment
git clone https://github.com/janishar/qwen-image-2.1-studio && cd qwen-image-2.1-studio && uv sync

# 3. start, or restart; then `stop` to end it
QWEN_MODELS=~/models bash web/run.sh
bash web/run.sh stop
```

`web/run.sh` checks that `helm` and the runtime SDK are there, links the three
weights from `QWEN_MODELS` (default `~/models`), and starts the studio under
`helm dev` on **http://127.0.0.1:8730**. Running it again stops the one it
started before, so it is also a restart. Only one `helm dev` can hold `./.helm`
at a time; a second one fails with *already locked by another holder*.

### Using the studio

Write a prompt, press **Generate**. The left pane is every CLI flag:

- **Model** and **Device** — the image model's path (change it under
  **⚙ Paths**, with the two enhancers') and where to run.
- **Mode** — *Text → image*, or *Edit* as soon as there is a reference image.
- **Prompt** — type `@` to pick a reference image, or click one on the right; it
  is written as `@name` and sent as `<imageN>` by the image's place in the list,
  so removing another image never breaks it.
- **Prompt enhancer** — **Enhance prompt** and **Let it think**.
- **Aspect ratio** — *Auto* or one of the seven, with the size it will render;
  **Width** and **Height** override it.
- **Steps**, **Seed** (⚄ for a random one) and **Transparent background**.
- Under **Generate**, the command it will run.

The centre shows the take with its seed, steps and size, the five stages of the
render as they happen (Enhance → Load → Denoise → Decode → Save), the enhanced
prompt beside the original — short, scrollable, **expand** for all of it — and the
session's takes. On a take: **Use as reference**, **Reuse settings**, **Open**, or
delete. The right pane holds the reference images (up to 10), the queue, and the
terminal, pinned to the bottom.

### What helmstudio adds

| | |
| --- | --- |
| **Gallery** | helmstudio's own grid over this studio's takes, live. |
| **References from anywhere** | **from gallery** picks an image out of that grid, another studio's included, as a reference. |
| **Render log** | A second Terminal tab streaming the render as helmstudio keeps it; it reconnects after a dropped stream. |
| **The launcher** | A render appears there as a job with its progress and its log, and can be cancelled from there. |

All of it arrives through the same-origin `/helm/` proxy the server mounts, as
do the theme and this studio's teal, so the page holds no token of helmstudio's.
There is no timeline: a helmstudio sequence is an edit of video clips, and this
studio makes stills.

### Sessions and state

The studio keeps nothing of its own; helmstudio keeps it all:

| What | Where in helmstudio |
| --- | --- |
| A session's settings and reference images | the session's state |
| Reference images and takes | assets |
| A take with every setting that made it, its enhanced prompt and its inputs | a gallery item |
| A render and its log | a job |
| The paths chosen under **⚙ Paths** | kv |

For an installed studio that is `~/.helmstudio`; for a checkout under `helm dev`
it is `./.helm`. Nothing is written into the repository.

## Command line

```bash
uv run qwen-image-2-1 "A neon shop sign that reads QWEN" -m ~/models/Qwen-Image-2.1
```

Or `export QWEN_IMAGE_21_PATH=~/models/Qwen-Image-2.1` once and drop `-m`. The
take is written to `output/output.png`; `--output` puts it elsewhere, and a
format that cannot hold the image (RGBA as JPEG) falls back to PNG.

```bash
uv run qwen-image-2-1 "Dragon sticker, die-cut" --transparent       # RGBA, transparent background
uv run qwen-image-2-1 "Swiss poster, 'QWEN 2.1'" --ratio 3:4 --seed 7
uv run qwen-image-2-1 "A bookshelf in warm oak" --width 1024 --height 768 --steps 30
```

Sizes come from the aspect ratio, at about 4 MP: `1:1` is 2048×2048, `16:9`
2752×1536, and so on through `4:3`, `3:4`, `3:2`, `2:3` and `9:16`. `--width` and
`--height` override either side and must be multiples of 32.

### Prompt enhancer

`--enhance` first rewrites the prompt with a 9B enhancer and lets it choose the
aspect ratio: PE-T2I for text-to-image, PE-I2I when `--input` is given.
`--ratio`, `--width` and `--height` still win. `--think` lets it reason first,
which is slower, often by minutes.

```bash
export QWEN_IMAGE_21_PE_T2I_PATH=~/models/Qwen-Image-2.1-PE-T2I
export QWEN_IMAGE_21_PE_I2I_PATH=~/models/Qwen-Image-2.1-PE-I2I
uv run qwen-image-2-1 "a corgi playing guitar in the rain" --enhance
```

The enhancer is freed before the image model loads. `--pe-model` points at a
specific enhancer instead of the environment's.

### Editing with reference images

`--input` takes up to 10 images, comma-separated, and makes the render an edit.
The prompt names them `<image1>`, `<image2>`, … in that order. With no ratio set,
the output follows the **last** image's aspect at about 1 MP.

```bash
uv run qwen-image-2-1 "put the mug from <image1> under the neon sign from <image2>" \
  --input mug.jpg,neon-sign.png --enhance
```

### Device

`--device auto` (the default) runs on MPS, else CUDA, else CPU. `mps`, `cuda` and
`cpu` pick one, and asking for one this torch build lacks fails at once. On MPS
the weights load straight onto the GPU, and the MPS cache is emptied after the
enhancer with a workaround for a torch 2.14 deadlock. CPU works but is very slow
for a model this size.

```bash
uv run qwen-image-2-1 "A neon shop sign that reads QWEN" --device cuda
```

### CLI reference

| Flag | Default | Description |
| --- | --- | --- |
| `prompt` | *(required)* | The text prompt. |
| `--input` | *(none)* | Reference images for editing, comma-separated, at most 10. |
| `--ratio` | `1:1`, or the last input's aspect | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`. |
| `--width`, `--height` | from the ratio | Override one side; a multiple of 32. |
| `--steps` | `40` | Denoising steps. |
| `--seed` | `42` | Seed for the CPU generator, so a seed repeats across devices. |
| `--transparent` | off | Ask for an RGBA image with a transparent background. |
| `-m`, `--model` | `$QWEN_IMAGE_21_PATH`, else `Qwen/Qwen-Image-2.1` | Local directory or Hub id. |
| `--enhance` | off | Rewrite the prompt, and choose the ratio, with PE-T2I / PE-I2I first. |
| `--think` | off | With `--enhance`: let the enhancer reason before answering. |
| `--pe-model` | `$QWEN_IMAGE_21_PE_{T2I,I2I}_PATH`, else the Hub | Enhancer directory or Hub id. |
| `--device` | `$QWEN_IMAGE_21_DEVICE`, else `auto` | `auto`, `mps`, `cuda` or `cpu`. |
| `--output` | `output/output.png` | Where the take is written. |

Besides its human-readable output the CLI prints one line per event for the
studio to read: `@stage enhance|load|denoise|decode|save`, `@step 12/40`, and
`@enhanced {"prompt": …, "ratio": …}`.

### Environment variables

| Variable | Used as |
| --- | --- |
| `QWEN_IMAGE_21_PATH` | `--model` |
| `QWEN_IMAGE_21_PE_T2I_PATH`, `QWEN_IMAGE_21_PE_I2I_PATH` | `--pe-model`, by mode |
| `QWEN_IMAGE_21_DEVICE` | `--device` |
| `QWEN_MODELS` | `web/run.sh`: the directory holding the three weights |

## Security

The studio has **no authentication**: anyone who can reach its port can run
renders. It binds to `127.0.0.1`, refuses a `Host` it does not recognise (which
blocks DNS rebinding), refuses a cross-origin write, and accepts only JSON for
anything that changes state. `--allow-host` adds a name to accept; binding to
anything but loopback is on you. The terminal is output only, never a shell.

## Limits

- One render at a time, deliberately; the rest wait in the queue.
- Each render loads the model afresh, which is seconds with a warm page cache
  and minutes the first time.
- The pipeline returns RGBA whether or not `--transparent` is set, so every take
  is saved as a 4-channel PNG; the RGBA badge follows the setting, not the file.
- **Load** has no progress of its own: its bar jumps from nothing to done.
- A load has once hung in safetensors' parallel loader, all threads idle; the
  same command went through on retry. Cancel and render again.
- `peak_ram_gb` in `helmstudio.yaml` is an estimate, not a measurement.
- CUDA and CPU are untested; `helmstudio.yaml` still requires an Apple Silicon
  Mac, so only the CLI runs elsewhere.

## Contributing

Bug reports, feature requests and pull requests are welcome. Two conventions:
no Python dependencies for the studio beyond helmstudio's runtime SDK, and no
front-end build step — `web/static/` is served as written.

## License

MIT — see [LICENSE](LICENSE).

[helmstudio]: https://github.com/janishar/helmstudio
[helm-releases]: https://github.com/janishar/helmstudio/releases
