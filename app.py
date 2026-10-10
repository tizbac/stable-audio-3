"""ZeroGPU Gradio demo for Stable Audio 3 — Medium, Small Music, Small SFX.

Two tabs:

* **Simple** — prompt + duration with a slim Advanced accordion (steps/CFG/seed
  /sampler). Mirrors the original tiny UI.
* **Advanced** — replicates the reference repo's
  ``stable_audio_3/interface/diffusion_cond.py`` controls: negative prompt,
  sampler params (sigma_max, APG, duration padding), init audio + noise level,
  inpainting with mask start/end, spectrogram gallery, send-to-init /
  send-to-inpaint buttons.
"""

from __future__ import annotations

import spaces  # noqa: F401

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

def _ensure_stable_audio_tools() -> None:
    try:
        import stable_audio_tools  # noqa: F401
        return
    except ImportError:
        pass
    # stable-audio-tools 0.0.20 strict-pins torch==2.7.1 / torchaudio==2.7.1,
    # which lack sm_120 (Blackwell) kernels. Install with --no-deps; the
    # transitive deps are listed in requirements.txt and resolved against the
    # sm_120-capable torch at build time.
    print("[startup] installing stable-audio-tools (--no-deps) …", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
         "stable-audio-tools"],
    )
    import stable_audio_tools  # noqa: F401
    print("[startup] stable-audio-tools installed.", flush=True)

_ensure_stable_audio_tools()


import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    key: str
    repo: str
    label: str
    default_duration: int
    placeholder: str


VARIANTS: list[Variant] = [
    Variant(
        key="medium",
        repo="stabilityai/stable-audio-3-medium",
        label="Medium — general audio (largest)",
        default_duration=60,
        placeholder="A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie 120 BPM",
    ),
    Variant(
        key="small-music",
        repo="stabilityai/stable-audio-3-small-music",
        label="Small Music — 0.6B, music-focused",
        default_duration=60,
        placeholder="Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM",
    ),
    Variant(
        key="small-sfx",
        repo="stabilityai/stable-audio-3-small-sfx",
        label="Small SFX — 0.6B, sound effects",
        default_duration=7,
        placeholder="Chugging train coming into station with horn",
    ),
]

# Which variants to preload: SA3_VARIANTS is a comma-separated list of keys
# (e.g. "medium" or "small-music,small-sfx"); unset loads everything. The
# Docker deployment sets SA3_VARIANTS=medium so a 12 GB card doesn't have to
# hold all three checkpoints at once.
_variant_keys_env = os.environ.get("SA3_VARIANTS", "").strip()
if _variant_keys_env:
    _known = {v.key: v for v in VARIANTS}
    _wanted = [k.strip() for k in _variant_keys_env.split(",") if k.strip()]
    _unknown = [k for k in _wanted if k not in _known]
    if _unknown:
        sys.exit(
            f"SA3_VARIANTS: unknown variant(s): {', '.join(_unknown)} "
            f"(valid keys: {', '.join(_known)})"
        )
    VARIANTS = [_known[k] for k in _wanted]


# ---------------------------------------------------------------------------
# Preload all variants at module level (ZeroGPU CUDA emulation accepts it)
# ---------------------------------------------------------------------------

@dataclass
class LoadedVariant:
    variant: Variant
    model: object
    sample_rate: int
    sample_size: int
    max_seconds: int


LOADED: dict[str, LoadedVariant] = {}
for v in VARIANTS:
    print(f"[startup] loading {v.repo} …", flush=True)
    t0 = time.time()
    model, config = get_pretrained_model(v.repo)
    sr = int(config["sample_rate"])
    ss = int(config["sample_size"])
    # Cast to fp16 *before* moving to the GPU: a 12 GB card cannot hold the
    # fp32 copy of the medium model (peak ~10.4 GB) that the reversed order
    # requires. Same final state, half the transfer.
    model = model.to(torch.float16).to("cuda")
    LOADED[v.key] = LoadedVariant(
        variant=v,
        model=model,
        sample_rate=sr,
        sample_size=ss,
        max_seconds=ss // sr,
    )
    print(
        f"[startup] {v.key} ready in {time.time() - t0:.1f}s · "
        f"sr={sr} · sample_size={ss} (~{ss // sr}s max)",
        flush=True,
    )

VARIANT_CHOICES = [(v.label, v.key) for v in VARIANTS]
# Samplers valid for rf_denoiser diffusion objective (the SA3 family).
SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]


# ---------------------------------------------------------------------------
# Spectrogram helper (Mel; adapted from the reference repo's aeiou.py)
# ---------------------------------------------------------------------------


def _power_to_db(spec: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(amin, spec))


def audio_spectrogram_image(
    waveform: torch.Tensor,
    sample_rate: int,
    db_range=(35, 120),
    figsize=(5, 4),
) -> Image.Image:
    """Render a Mel spectrogram (left channel) as a PIL image."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    n_fft = 1024
    hop_length = n_fft // 2
    mel_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=None,
        hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
        norm="slaney", onesided=True, n_mels=128, mel_scale="htk",
    )
    melspec = mel_op(waveform.float())[0]  # left channel
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(_power_to_db(melspec.numpy()), origin="lower", aspect="auto",
              vmin=db_range[0], vmax=db_range[1])
    ax.set_ylabel("mel bins (log freq)")
    ax.set_xlabel("frame")
    ax.set_title("MelSpectrogram")
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba()))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _gradio_audio_to_tensor(
    audio_in: Optional[Tuple[int, np.ndarray]],
) -> Optional[Tuple[int, torch.Tensor]]:
    """Convert a gr.Audio (numpy) value to the (sr, torch.Tensor[C,N]) tuple
    that ``generate_diffusion_cond_inpaint`` expects. Accepts mono or stereo."""
    if audio_in is None:
        return None
    sr, arr = audio_in
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return None
    arr = np.asarray(arr)
    if arr.dtype.kind in ("i", "u"):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]                       # (1, N)
    else:
        # gr.Audio returns (N, C); transpose to (C, N)
        arr = arr.T if arr.shape[0] > arr.shape[1] else arr
    return int(sr), torch.from_numpy(arr)


def _tensor_to_wav(
    output: torch.Tensor,
    sample_rate: int,
    duration_seconds: Optional[int],
    out_dir: Optional[str] = None,
) -> Tuple[str, torch.Tensor]:
    """Pack a (B, C, N) generation tensor to int16, optionally cut to duration,
    write to disk, and return (path, int16-tensor)."""
    output = rearrange(output, "b d n -> d (b n)")
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)).clamp(min=1e-9))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    if duration_seconds is not None:
        output = output[:, : int(duration_seconds) * sample_rate]
    out_dir = out_dir or tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "sa3.wav")
    sf.write(out_path, output.numpy().T, sample_rate, subtype="PCM_16")
    return out_path, output


def _run_inference(
    variant_key: str,
    prompt: str,
    negative_prompt: str = "",
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    sigma_max: float = 1.0,
    apg_scale: float = 1.0,
    duration_padding_sec: float = 6.0,
    cut_to_seconds_total: bool = True,
    init_audio: Optional[Tuple[int, np.ndarray]] = None,
    init_noise_level: float = 0.9,
    inpaint_audio: Optional[Tuple[int, np.ndarray]] = None,
    mask_start_sec: float = 0.0,
    mask_end_sec: float = 0.0,
    preview_every: int = 0,
    return_spectrogram: bool = True,
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured generation. Returns (audio_path, [spectrogram_img, *previews])
    when ``return_spectrogram`` is True, else just ``audio_path``."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if variant_key not in LOADED:
        raise gr.Error(f"Unknown variant {variant_key!r}.")
    lv = LOADED[variant_key]
    duration = max(1, min(int(duration), lv.max_seconds))

    progress(0.05, desc=f"[{variant_key}] preparing conditioning")
    conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]
    negative_conditioning = None
    neg = (negative_prompt or "").strip()
    if neg:
        negative_conditioning = [{"prompt": neg, "seconds_total": int(duration)}]

    # The pretransform encoder is fp16 (we cast the whole model at startup),
    # but prepare_audio's torchaudio Resample uses an fp32 kernel. Pre-resample
    # in fp32 here so prepare_audio's resample is a no-op, then cast to the
    # model dtype so the encoder doesn't see a dtype mismatch.
    model_dtype = next(lv.model.parameters()).dtype

    def _prep(tup):
        if tup is None:
            return None
        sr, t = tup
        t = t.float()
        if sr != lv.sample_rate:
            t = torchaudio.functional.resample(t, sr, lv.sample_rate)
        return lv.sample_rate, t.to(model_dtype)

    init_audio_t = _prep(_gradio_audio_to_tensor(init_audio))
    inpaint_audio_t = _prep(_gradio_audio_to_tensor(inpaint_audio))

    # Inpaint mask: only enable if mask_end > mask_start AND we have either
    # inpaint_audio or init_audio (otherwise the mask wraps zero content).
    mask_start = max(0.0, float(mask_start_sec))
    mask_end = min(float(duration), float(mask_end_sec))
    use_mask = (
        inpaint_audio_t is not None
        and mask_end > mask_start
    )

    seed_val = int(seed) if seed and int(seed) > 0 else -1

    preview_images: list = []
    callback = None
    if preview_every and int(preview_every) > 0:
        every = int(preview_every)

        def _cb(info):
            i = info["i"]
            if i % every != 0:
                return
            denoised = info["denoised"]
            try:
                if lv.model.pretransform is not None:
                    denoised = lv.model.pretransform.decode(denoised)
                d = rearrange(denoised, "b d n -> d (b n)")
                d = d.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                img = audio_spectrogram_image(d, sample_rate=lv.sample_rate)
                preview_images.append((img, f"Step {i + 1}"))
            except Exception as e:
                print(f"[preview] skipped step {i}: {e}", flush=True)
        callback = _cb

    gen_kwargs: dict = dict(
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=lv.sample_size,
        sampler_type=sampler_type,
        seed=seed_val,
        device="cuda",
        sigma_max=float(sigma_max),
        apg_scale=float(apg_scale),
        duration_padding_sec=float(duration_padding_sec),
    )
    if init_audio_t is not None:
        gen_kwargs["init_audio"] = init_audio_t
        gen_kwargs["init_noise_level"] = float(init_noise_level)
    if inpaint_audio_t is not None:
        gen_kwargs["inpaint_audio"] = inpaint_audio_t
    if use_mask:
        gen_kwargs["inpaint_mask_start_seconds"] = mask_start
        gen_kwargs["inpaint_mask_end_seconds"] = mask_end
    if callback is not None:
        gen_kwargs["callback"] = callback

    progress(0.25, desc=f"[{variant_key}] sampling {steps} steps with {sampler_type}")
    t0 = time.time()
    output = generate_diffusion_cond_inpaint(lv.model, **gen_kwargs)
    print(f"[infer/{variant_key}] sampling done in {time.time() - t0:.1f}s", flush=True)

    progress(0.92, desc="Normalising & saving")
    cut_dur = int(duration) if cut_to_seconds_total else None
    out_path, int16_audio = _tensor_to_wav(output, lv.sample_rate, cut_dur)

    if not return_spectrogram:
        return out_path

    spec_img = audio_spectrogram_image(int16_audio, sample_rate=lv.sample_rate)
    return out_path, [spec_img, *preview_images]


@spaces.GPU
def infer(
    variant_key: str,
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    progress: gr.Progress = gr.Progress(),
):
    """Slim handler used by the Simple tab and the Examples cache."""
    return _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        return_spectrogram=False,
        progress=progress,
    )


@spaces.GPU
def infer_advanced(
    variant_key: str,
    prompt: str,
    negative_prompt: str,
    duration: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    seed: int,
    sigma_max: float,
    apg_scale: float,
    duration_padding_sec: float,
    cut_to_seconds_total: bool,
    init_audio: Optional[Tuple[int, np.ndarray]],
    init_noise_level: float,
    inpaint_audio: Optional[Tuple[int, np.ndarray]],
    mask_start_sec: float,
    mask_end_sec: float,
    preview_every: int,
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured handler used by the Advanced tab."""
    return _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        sigma_max=sigma_max,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        cut_to_seconds_total=cut_to_seconds_total,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        inpaint_audio=inpaint_audio,
        mask_start_sec=mask_start_sec,
        mask_end_sec=mask_end_sec,
        preview_every=preview_every,
        return_spectrogram=True,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎵 Stable Audio 3

Text-to-audio generation with <a href="https://huggingface.co/collections/stabilityai/stable-audio-3" target="_blank" rel="noopener noreferrer">Stable Audio 3</a>. Pick a variant, write a prompt, hit Generate. Switch to **Advanced** for the full sampler / init-audio / inpainting controls.
"""

EXAMPLES = [
    ["medium",      "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM", 60],
    ["small-music", "Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM", 45],
    ["small-music", "Driving techno track with rolling 16th-note hats, deep sub bass, acid arpeggios building tension 132 BPM", 60],
    ["small-sfx",   "Chugging train coming into station with horn", 7],
    ["small-sfx",   "Heavy rain on a tin roof with distant thunder rolls", 10],
    ["medium",      "Rainy night, lo-fi hip-hop beat with vinyl crackle, mellow piano chords, soft kick and snare 80 BPM", 30],
]


def _variant_change_simple(variant_key: str):
    lv = LOADED[variant_key]
    return (
        gr.update(maximum=lv.max_seconds, value=min(lv.variant.default_duration, lv.max_seconds),
                  label=f"Duration (s) · model max {lv.max_seconds}s"),
        gr.update(placeholder=lv.variant.placeholder),
    )


def _variant_change_advanced(variant_key: str):
    lv = LOADED[variant_key]
    dur = min(lv.variant.default_duration, lv.max_seconds)
    return (
        gr.update(maximum=lv.max_seconds, value=dur,
                  label=f"Seconds total · model max {lv.max_seconds}s"),
        gr.update(placeholder=lv.variant.placeholder),
        gr.update(maximum=float(lv.max_seconds), value=0.0),
        gr.update(maximum=float(lv.max_seconds), value=float(dur)),
    )


with gr.Blocks(theme=gr.themes.Citrus(), title="Stable Audio 3") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # -----------------------------------------------------------------
        # Simple tab
        # -----------------------------------------------------------------
        with gr.Tab("Simple"):
            variant = gr.Radio(
                choices=VARIANT_CHOICES,
                value=VARIANTS[0].key,
                label="Model",
            )

            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder=VARIANTS[0].placeholder,
                        lines=3,
                    )
                    duration = gr.Slider(
                        1, LOADED[VARIANTS[0].key].max_seconds,
                        value=VARIANTS[0].default_duration, step=1,
                        label=f"Duration (s) · model max {LOADED[VARIANTS[0].key].max_seconds}s",
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                        cfg_scale = gr.Slider(0.5, 8.0, value=1.0, step=0.1, label="CFG scale")
                        sampler_type = gr.Dropdown(SAMPLERS, value="pingpong", label="Sampler")
                        seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                    run_btn = gr.Button("🎼 Generate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)

            gr.Examples(
                examples=EXAMPLES,
                inputs=[variant, prompt, duration],
                outputs=[audio_out],
                fn=infer,
                cache_examples=True,
                cache_mode="lazy",
                label="Examples (lazy-cached on first click)",
            )

            variant.change(
                fn=_variant_change_simple,
                inputs=[variant],
                outputs=[duration, prompt],
            )

            run_btn.click(
                fn=infer,
                inputs=[variant, prompt, duration, steps, cfg_scale, sampler_type, seed],
                outputs=[audio_out],
            )

        # -----------------------------------------------------------------
        # Advanced tab — mirrors stable_audio_3/interface/diffusion_cond.py
        # -----------------------------------------------------------------
        with gr.Tab("Advanced"):
            adv_variant = gr.Radio(
                choices=VARIANT_CHOICES,
                value=VARIANTS[0].key,
                label="Model",
            )

            with gr.Row():
                with gr.Column(scale=6):
                    adv_prompt = gr.Textbox(
                        show_label=False,
                        placeholder=VARIANTS[0].placeholder,
                    )
                    adv_negative = gr.Textbox(
                        show_label=False, placeholder="Negative prompt"
                    )
                adv_generate = gr.Button("Generate", variant="primary", scale=1)

            with gr.Row(equal_height=False):
                with gr.Column():
                    adv_seconds_total = gr.Slider(
                        minimum=1,
                        maximum=LOADED[VARIANTS[0].key].max_seconds,
                        step=1,
                        value=VARIANTS[0].default_duration,
                        label=f"Seconds total · model max {LOADED[VARIANTS[0].key].max_seconds}s",
                    )

                    with gr.Row():
                        adv_steps = gr.Slider(
                            minimum=1, maximum=500, step=1, value=8, label="Steps"
                        )
                        adv_cfg = gr.Slider(
                            minimum=0.0, maximum=25.0, step=0.1, value=1.0,
                            label="CFG scale",
                        )

                    with gr.Accordion("Sampler params", open=False):
                        with gr.Row():
                            adv_seed = gr.Number(
                                label="Seed (set to -1 for random seed)",
                                value=-1, precision=0,
                            )
                            adv_sampler = gr.Dropdown(
                                SAMPLERS, label="Sampler type", value="pingpong",
                            )
                            adv_sigma_max = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                                label="Sigma max",
                            )
                        with gr.Row():
                            adv_apg = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.1, value=1.0,
                                label="APG scale", info="1.0=full APG, 0.0=vanilla CFG",
                            )
                            adv_dur_padding = gr.Slider(
                                minimum=0.0, maximum=30.0, step=0.5, value=6.0,
                                label="Duration padding (sec)",
                            )

                    with gr.Accordion("Output params", open=False):
                        with gr.Row():
                            adv_preview_every = gr.Slider(
                                minimum=0, maximum=100, step=1, value=0,
                                label="Spec preview every N steps (0 = off)",
                            )
                            adv_cut_to_total = gr.Checkbox(
                                label="Cut to seconds total", value=True,
                            )

                    with gr.Accordion("Init audio", open=False):
                        adv_init_audio = gr.Audio(
                            label="Init audio",
                            type="numpy",
                        )
                        adv_init_noise = gr.Slider(
                            minimum=0.01, maximum=1.0, step=0.01, value=0.9,
                            label="Init noise level",
                        )

                    with gr.Accordion("Inpainting", open=False):
                        adv_inpaint_audio = gr.Audio(
                            label="Inpaint audio",
                            type="numpy",
                        )
                        adv_mask_start = gr.Slider(
                            minimum=0.0,
                            maximum=float(LOADED[VARIANTS[0].key].max_seconds),
                            step=0.1, value=0.0, label="Mask start (sec)",
                        )
                        adv_mask_end = gr.Slider(
                            minimum=0.0,
                            maximum=float(LOADED[VARIANTS[0].key].max_seconds),
                            step=0.1, value=0.0, label="Mask end (sec)",
                        )

                with gr.Column():
                    adv_audio_out = gr.Audio(
                        label="Output audio", type="filepath", autoplay=False,
                        sources=[],
                    )
                    adv_spec_gallery = gr.Gallery(
                        label="Output spectrogram", show_label=True, columns=2,
                    )
                    send_to_init_btn = gr.Button("Send to init audio")
                    send_to_inpaint_btn = gr.Button("Send to inpaint audio")

            send_to_init_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_init_audio]
            )
            send_to_inpaint_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_inpaint_audio]
            )

            # Keep the inpaint mask bounded by the current duration.
            def _update_mask_max(seconds_total):
                m = max(float(seconds_total), 1.0)
                return (
                    gr.update(maximum=m),
                    gr.update(maximum=m, value=m),
                )
            adv_seconds_total.change(
                _update_mask_max,
                inputs=[adv_seconds_total],
                outputs=[adv_mask_start, adv_mask_end],
            )

            adv_variant.change(
                fn=_variant_change_advanced,
                inputs=[adv_variant],
                outputs=[adv_seconds_total, adv_prompt, adv_mask_start, adv_mask_end],
            )

            adv_generate.click(
                fn=infer_advanced,
                inputs=[
                    adv_variant,
                    adv_prompt,
                    adv_negative,
                    adv_seconds_total,
                    adv_steps,
                    adv_cfg,
                    adv_sampler,
                    adv_seed,
                    adv_sigma_max,
                    adv_apg,
                    adv_dur_padding,
                    adv_cut_to_total,
                    adv_init_audio,
                    adv_init_noise,
                    adv_inpaint_audio,
                    adv_mask_start,
                    adv_mask_end,
                    adv_preview_every,
                ],
                outputs=[adv_audio_out, adv_spec_gallery],
            )


if __name__ == "__main__":
    demo.launch()
