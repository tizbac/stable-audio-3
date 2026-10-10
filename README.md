---
title: Stable Audio 3
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: true
license: other
short_description: Text-to-audio with SA3 Medium / Small Music / Small SFX.
suggested_hardware: zero-a10g
models:
- stabilityai/stable-audio-3-medium
- stabilityai/stable-audio-3-small-music
- stabilityai/stable-audio-3-small-sfx
---

# Stable Audio 3

ZeroGPU demo of the [Stable Audio 3](https://huggingface.co/stabilityai) family. Three variants preloaded at module load; switch between them with a radio button.

- [`stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) — general audio (largest).
- [`stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music) — 0.6B, music-focused.
- [`stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) — 0.6B, sound effects.

## Docker deployment

Runs the same app on a self-hosted GPU box (NVIDIA driver + `nvidia-container-toolkit` required).

```bash
HF_TOKEN=hf_... ./deploy.sh build   # build the image, baking in all 3 checkpoints (~17 GB)
./deploy.sh up --wait               # start: GPU, port 7860, waits until healthy
./deploy.sh logs -f                 # follow startup / inference logs
./deploy.sh status                  # container, health, URL, GPU usage
./deploy.sh down                    # stop
./deploy.sh doctor                  # diagnose docker / GPU / disk issues
./deploy.sh clean --all             # remove image (and any leftover cache volume)
```

Notes:

- **Weights are baked into the image** (medium 10.5 GB + 2 × 3.5 GB) at
  `/cache/huggingface`, so the container starts with no downloads. Expect
  the image to take ~40 GB on disk.
- **By default only `medium` is loaded into VRAM** (`SA3_VARIANTS=medium`) —
  all three checkpoints leave too little activation headroom on a 12 GB
  card. Every checkpoint is still baked into the image; pick others with
  e.g. `SA3_VARIANTS=small-music ./deploy.sh up`, `-e SA3_VARIANTS=all`,
  or `SA3_VARIANTS=medium,small-sfx`.
- **The three SA3 repos are gated**: accept the license at
  <https://huggingface.co/stabilityai> first. The build reads `$HF_TOKEN`
  (falling back to `~/.cache/huggingface/token`) and passes it as a
  BuildKit secret, so the token never ends up in the image layers. Without
  a token, build with `PRELOAD_MODELS=0 ./deploy.sh build` and let the app
  download on first boot instead.
- `@spaces.GPU` is a no-op outside ZeroGPU — inference runs on the local GPU.
- Override via env: `PORT=8080 IMAGE=sa3:v2 ./deploy.sh up`, or add
  `-e HF_HUB_OFFLINE=0` to the container if you build with `PRELOAD_MODELS=0`.