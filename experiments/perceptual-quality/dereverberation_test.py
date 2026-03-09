"""
dereverberation_test.py

Evaluate a trained dereverberation model on a folder of audio files.

Usage:
    # Without clean references (wavs + spectrograms only)
    python dereverberation_test.py simple checkpoints/my.ckpt --input_dir data/test_clips

    # With clean references for SI-SNR metrics
    python dereverberation_test.py simple checkpoints/my.ckpt \
        --input_dir data/test_reverb --reference_dir data/test_clean

Output layout:
    test_output/<checkpoint_stem>/
        wavs/
            <stem>_reverb.wav
            <stem>_output.wav
            <stem>_clean.wav        # only if --reference_dir provided
        spectrograms/
            <stem>.png
        metrics.csv
        metrics_summary.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from tqdm import tqdm

from model.dereverberation_simple import DereverberationModel
from utils.seed import seed


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


# ---------------------------------------------------------------------------
# Config / model loading (mirrors existing test script)
# ---------------------------------------------------------------------------

def _load_config(config_key: str) -> dict:
    with open("config.yaml", "r") as f:
        all_configs = yaml.safe_load(f)

    if config_key not in all_configs:
        available = ", ".join(sorted(all_configs.keys()))
        raise KeyError(
            f"Config key '{config_key}' not found in config.yaml. "
            f"Available keys: {available}"
        )

    cfg = all_configs[config_key]
    use_cuda = torch.cuda.is_available()
    model_cfg = dict(cfg.get("model", {}))
    model_cfg = {k: v for k, v in model_cfg.items() if v is not None}
    if "gradient_checkpointing" in cfg:
        model_cfg.setdefault("gradient_checkpointing", cfg["gradient_checkpointing"])

    return {
        "device": "cuda" if use_cuda else "cpu",
        "segment_length": cfg.get("segment_length", 44100 * 4),
        "sample_rate": cfg.get("sample_rate", 44100),
        "model": model_cfg,
    }


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as first_error:
        for prefix in ("model.", "module."):
            if any(str(k).startswith(prefix) for k in state_dict.keys()):
                remapped = {
                    str(k)[len(prefix):]: v
                    for k, v in state_dict.items()
                    if str(k).startswith(prefix)
                }
                if not remapped:
                    continue
                try:
                    model.load_state_dict(remapped)
                    return
                except RuntimeError:
                    pass
        raise RuntimeError(
            "Could not load checkpoint into DereverberationModel. "
            "Tried original keys and remapping prefixes: model., module."
        ) from first_error


def load_model(checkpoint_path: Path, config: dict) -> DereverberationModel:
    model_kwargs = dict(config["model"])
    model_kwargs.setdefault("sr", config["sample_rate"])
    model = DereverberationModel(**model_kwargs)

    checkpoint_obj = torch.load(checkpoint_path, map_location=config["device"])
    state_dict = _extract_state_dict(checkpoint_obj)
    _load_model_state_dict(model, state_dict)

    model = model.to(config["device"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load, resample if needed, mix to mono. Returns (samples,)."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav.squeeze(0)


def save_wav(path: Path, audio: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.cpu().numpy(), sr)


# ---------------------------------------------------------------------------
# Inference (overlap-add for long files)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(
    model: DereverberationModel,
    audio: torch.Tensor,
    device: torch.device,
    segment_length: int,
) -> torch.Tensor:
    audio = audio.to(device)

    if segment_length <= 0 or audio.shape[0] <= segment_length:
        return model(audio.unsqueeze(0)).squeeze(0).cpu()

    # Overlap-add with 50 % overlap and Hann crossfade
    hop = segment_length // 2
    n = audio.shape[0]
    output = torch.zeros(n, device=device)
    weight = torch.zeros(n, device=device)
    window = torch.hann_window(segment_length, device=device)

    start = 0
    while start < n:
        end = min(start + segment_length, n)
        chunk = audio[start:end]
        pad = segment_length - chunk.shape[0]
        if pad > 0:
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        out_chunk = model(chunk.unsqueeze(0)).squeeze(0)
        w = window if pad == 0 else window[: end - start]
        output[start:end] += out_chunk[: end - start] * w
        weight[start:end] += w
        start += hop

    return (output / weight.clamp(min=1e-8)).cpu()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def si_snr_val(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """SI-SNR in dB (higher = better)."""
    y_hat = y_hat - y_hat.mean()
    y = y - y.mean()
    s_target = (y_hat @ y) / (y.norm() ** 2 + 1e-8) * y
    e_noise = y_hat - s_target
    val = 10 * torch.log10(
        (s_target.norm() ** 2 + 1e-8) / (e_noise.norm() ** 2 + 1e-8)
    )
    return val.item()


# ---------------------------------------------------------------------------
# Spectrograms via pyfar
# ---------------------------------------------------------------------------

def plot_spectrograms(
    path: Path,
    sr: int,
    reverb: torch.Tensor,
    output: torch.Tensor,
    clean: Optional[torch.Tensor],
    stem: str,
    si_snr_in: Optional[float],
    si_snr_out: Optional[float],
):
    """Plot spectrograms using pyfar.plot.spectrogram, one panel per signal."""
    panels = [("Reverberant Input", reverb), ("Model Output", output)]
    if clean is not None:
        panels.append(("Clean Reference", clean))

    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 4), constrained_layout=True)
    if cols == 1:
        axes = [axes]

    title = stem
    if si_snr_in is not None and si_snr_out is not None:
        title += (
            f"   SI-SNR: {si_snr_in:+.1f} → {si_snr_out:+.1f} dB"
            f"  (Δ {si_snr_out - si_snr_in:+.1f} dB)"
        )
    fig.suptitle(title, fontsize=10)

    for ax, (label, wav) in zip(axes, panels):
        signal = pf.Signal(wav.cpu().numpy(), sr)
        pf.plot.spectrogram(
            signal,
            dB=True,
            freq_scale="linear",
            unit="s",
            window="hann",
            window_length=2048,
            window_overlap_fct=0.5,
            colorbar=True,
            ax=ax,
            style="light",
        )
        ax.set_title(label, fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test dereverberation model on a folder of audio files"
    )
    parser.add_argument("config_key", help="Key in config.yaml")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Folder of reverberant audio files",
    )
    parser.add_argument(
        "--reference_dir", type=Path, default=None,
        help="Optional folder of clean references (matched by filename stem)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Where to write results (default: test_output/<checkpoint_stem>)",
    )
    parser.add_argument(
        "--segment_length", type=int, default=0,
        help="Override chunk size in samples for long files (0 = use config value)",
    )
    args = parser.parse_args()

    seed(42)

    config = _load_config(args.config_key)
    device = torch.device(config["device"])
    sr = config["sample_rate"]
    seg_len = args.segment_length if args.segment_length > 0 else config["segment_length"]

    out_dir  = args.output_dir or Path("test_output") / args.checkpoint.stem
    wav_dir  = out_dir / "wavs"
    spec_dir = out_dir / "spectrograms"
    for d in (wav_dir, spec_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Input dir   : {args.input_dir}")
    print(f"Reference   : {args.reference_dir or '(none)'}")
    print(f"Output dir  : {out_dir}")
    print(f"Device      : {device}\n")

    model = load_model(args.checkpoint, config)

    input_files = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not input_files:
        raise RuntimeError(f"No audio files found in {args.input_dir}")
    print(f"Files found : {len(input_files)}\n")

    csv_rows = []

    for audio_path in tqdm(input_files, desc="Processing"):
        stem = audio_path.stem

        reverb = load_audio(audio_path, sr)
        output = run_inference(model, reverb, device, seg_len)
        output = output[: reverb.shape[0]]

        # Match clean reference by stem
        clean: Optional[torch.Tensor] = None
        if args.reference_dir is not None:
            for ext in AUDIO_EXTS:
                ref = args.reference_dir / (stem + ext)
                if ref.exists():
                    clean = load_audio(ref, sr)[: reverb.shape[0]]
                    break

        # Wavs
        save_wav(wav_dir / f"{stem}_reverb.wav", reverb, sr)
        save_wav(wav_dir / f"{stem}_output.wav", output, sr)
        if clean is not None:
            save_wav(wav_dir / f"{stem}_clean.wav", clean, sr)

        # Metrics
        si_snr_in = si_snr_out = None
        row: dict = {"file": stem}
        if clean is not None:
            si_snr_in  = si_snr_val(reverb, clean)
            si_snr_out = si_snr_val(output, clean)
            row["si_snr_input_db"]       = round(si_snr_in,  3)
            row["si_snr_output_db"]      = round(si_snr_out, 3)
            row["si_snr_improvement_db"] = round(si_snr_out - si_snr_in, 3)

        # Spectrogram
        plot_spectrograms(
            spec_dir / f"{stem}.png", sr,
            reverb, output, clean, stem,
            si_snr_in, si_snr_out,
        )

        csv_rows.append(row)

    # CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        metrics_path = out_dir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nMetrics  -> {metrics_path}")

    # Summary
    summary_path = out_dir / "metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Input dir  : {args.input_dir}\n")
        f.write(f"Files      : {len(csv_rows)}\n")
        if csv_rows and "si_snr_output_db" in csv_rows[0]:
            outs   = [r["si_snr_output_db"]      for r in csv_rows]
            deltas = [r["si_snr_improvement_db"]  for r in csv_rows]
            f.write(
                f"\nSI-SNR output (dB) : "
                f"mean={np.mean(outs):+.2f}  std={np.std(outs):.2f}  "
                f"min={np.min(outs):+.2f}  max={np.max(outs):+.2f}\n"
            )
            f.write(
                f"SI-SNR Δ (dB)      : "
                f"mean={np.mean(deltas):+.2f}  std={np.std(deltas):.2f}  "
                f"min={np.min(deltas):+.2f}  max={np.max(deltas):+.2f}\n"
            )
        else:
            f.write("\n(No reference files — SI-SNR not computed.)\n")

    print(f"Summary  -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()