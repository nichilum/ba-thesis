from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from pesq import pesq
from tqdm import tqdm

from model.dereverberation_simple import DereverberationModel
from utils.seed import seed


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

    tcn_channels = model_cfg.get("tcn_channels")
    dilations = model_cfg.get("dilations")
    if isinstance(tcn_channels, list) and tcn_channels:
        if not isinstance(dilations, list) or not dilations:
            model_cfg["dilations"] = [2**i for i in range(len(tcn_channels))]
        elif len(dilations) != len(tcn_channels):
            if len(dilations) > len(tcn_channels):
                model_cfg["dilations"] = dilations[: len(tcn_channels)]
            else:
                start = len(dilations)
                extension = [2**i for i in range(start, len(tcn_channels))]
                model_cfg["dilations"] = dilations + extension

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
    model_state = model.state_dict()

    def _filter_compatible(sd: dict) -> dict:
        filtered = {}
        for k, v in sd.items():
            key = str(k)
            if key in model_state and model_state[key].shape == v.shape:
                filtered[key] = v
        return filtered

    candidates: list[dict] = [state_dict]
    for prefix in ("model.", "module."):
        if any(str(k).startswith(prefix) for k in state_dict.keys()):
            remapped = {
                str(k)[len(prefix) :]: v
                for k, v in state_dict.items()
                if str(k).startswith(prefix)
            }
            if remapped:
                candidates.append(remapped)

    last_error: Exception | None = None
    for candidate in candidates:
        filtered = _filter_compatible(candidate)
        if not filtered:
            continue
        try:
            model.load_state_dict(filtered, strict=False)
            return
        except RuntimeError as e:
            last_error = e

    if last_error is None:
        raise RuntimeError(
            "Could not load checkpoint into DereverberationModel. "
            "No compatible keys were found after remapping."
        )
    raise RuntimeError(
        "Could not load checkpoint into DereverberationModel after key filtering."
    ) from last_error


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


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav.squeeze(0)


def resolve_audio_path(raw_path: str, metadata_path: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    rel_to_meta = (metadata_path.parent / p).resolve()
    if rel_to_meta.exists():
        return rel_to_meta

    rel_to_cwd = p.resolve()
    if rel_to_cwd.exists():
        return rel_to_cwd

    return rel_to_meta


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


def si_snr_val(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat = y_hat - y_hat.mean()
    y = y - y.mean()
    s_target = (y_hat @ y) / (y.norm() ** 2 + 1e-8) * y
    e_noise = y_hat - s_target
    val = 10 * torch.log10((s_target.norm() ** 2 + 1e-8) / (e_noise.norm() ** 2 + 1e-8))
    return val.item()


def mse_val(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


def pesq_wb(clean: torch.Tensor, degraded: torch.Tensor, sr: int) -> float:
    clean_16k = torchaudio.functional.resample(clean.unsqueeze(0), sr, 16000).squeeze(0)
    degraded_16k = torchaudio.functional.resample(degraded.unsqueeze(0), sr, 16000).squeeze(0)

    n = min(clean_16k.shape[0], degraded_16k.shape[0])
    if n < 320:
        return float("nan")

    clean_np = clean_16k[:n].cpu().numpy().astype(np.float32)
    degraded_np = degraded_16k[:n].cpu().numpy().astype(np.float32)

    try:
        return float(pesq(16000, clean_np, degraded_np, "wb"))
    except Exception:
        return float("nan")


@dataclass
class PeaqResult:
    odg: float
    di: float


class PeaqWorker:
    def __init__(self, enabled: bool, script_path: Path):
        self.enabled = enabled
        self.script_path = script_path
        self.process: subprocess.Popen[str] | None = None
        self._warning_printed = False

    def start(self):
        if not self.enabled:
            return

        self.process = subprocess.Popen(
            ["/usr/bin/python", str(self.script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _disable_with_warning(self, message: str):
        compact = message.strip()
        if "\n" in compact:
            parts = [p.strip() for p in compact.splitlines() if p.strip()]
            compact = parts[-1] if parts else compact
        if not self._warning_printed:
            print(f"[WARN] PEAQ disabled: {compact}")
            self._warning_printed = True
        self.enabled = False

    def eval_pair(self, ref_path: Path, test_path: Path) -> PeaqResult:
        if not self.enabled:
            return PeaqResult(float("nan"), float("nan"))

        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            return PeaqResult(float("nan"), float("nan"))

        if self.process.poll() is not None:
            err = ""
            if self.process.stderr is not None:
                try:
                    err = self.process.stderr.read().strip()
                except Exception:
                    err = ""
            self._disable_with_warning(err or "worker process exited unexpectedly")
            return PeaqResult(float("nan"), float("nan"))

        try:
            self.process.stdin.write(f"{ref_path}\t{test_path}\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                return PeaqResult(float("nan"), float("nan"))
            odg_s, di_s = line.strip().split("\t")
            return PeaqResult(float(odg_s), float(di_s))
        except Exception:
            self._disable_with_warning("failed to communicate with worker")
            return PeaqResult(float("nan"), float("nan"))

    def close(self):
        if not self.enabled or self.process is None:
            return

        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
        except Exception:
            pass
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()
            if self.process.stdin is not None:
                self.process.stdin.close()


def _mean_std_min_max(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _format_stats(name: str, stats: dict[str, float]) -> str:
    if stats["count"] == 0:
        return f"{name:<28} count=0"
    return (
        f"{name:<28} count={stats['count']:>6} "
        f"mean={stats['mean']:+.4f} std={stats['std']:.4f} "
        f"min={stats['min']:+.4f} max={stats['max']:+.4f}"
    )


def parse_metadata(metadata_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with open(metadata_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if split and item.get("split") != split:
                continue
            rows.append(item)
    return rows


def write_outputs(out_dir: Path, rows: list[dict], summary: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "metrics_summary.txt"
    summary_json_path = out_dir / "metrics_summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_path, "w") as f:
        f.write(f"files_requested={summary['files_requested']}\n")
        f.write(f"files_processed={summary['files_processed']}\n")
        f.write(f"files_skipped={summary['files_skipped']}\n")
        f.write(f"files_failed={summary['files_failed']}\n\n")

        for metric_name, stats in summary["stats"].items():
            f.write(_format_stats(metric_name, stats) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dereverberation checkpoint on metadata.jsonl pairs"
    )
    parser.add_argument("config_key", help="Key in config.yaml")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata.jsonl"),
        help="Path to metadata jsonl with original_path and reverberant_path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split filter from metadata (default: test)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (default: test_output/<checkpoint_stem>_eval)",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=0,
        help="Override chunk size in samples for long files (0 = config value)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Optional limit on number of samples to process",
    )
    parser.add_argument(
        "--use_peaq",
        action="store_true",
        help="Compute PEAQ ODG/DI using utils/peaq.py worker",
    )
    parser.add_argument(
        "--peaq_script",
        type=Path,
        default=Path("utils/peaq.py"),
        help="Path to PEAQ worker script",
    )
    parser.add_argument(
        "--fail_on_missing_clean",
        action="store_true",
        help="Fail instead of skip when clean reference is missing",
    )
    args = parser.parse_args()

    seed(42)

    metadata_path = args.metadata.resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    config = _load_config(args.config_key)
    device = torch.device(config["device"])
    sr = int(config["sample_rate"])
    seg_len = args.segment_length if args.segment_length > 0 else int(config["segment_length"])

    out_dir = args.output_dir or (Path("test_output") / f"{args.checkpoint.stem}_eval")
    out_dir = out_dir.resolve()

    model = load_model(args.checkpoint, config)

    data = parse_metadata(metadata_path, args.split)
    if args.num_samples > 0:
        data = data[: args.num_samples]

    if not data:
        raise RuntimeError("No metadata rows matched the requested split/filters")

    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Metadata        : {metadata_path}")
    print(f"Split           : {args.split}")
    print(f"Sample rate     : {sr}")
    print(f"Segment length  : {seg_len}")
    print(f"Use PEAQ        : {args.use_peaq}")
    print(f"Output dir      : {out_dir}")
    print(f"Device          : {device}")
    print(f"Rows selected   : {len(data)}")

    peaq_worker = PeaqWorker(enabled=args.use_peaq, script_path=args.peaq_script)
    peaq_worker.start()

    csv_rows: list[dict] = []
    failed = 0
    skipped = 0

    metric_values: dict[str, list[float]] = {
        "mse_baseline": [],
        "mse_enhanced": [],
        "mse_delta": [],
        "sisnr_baseline_db": [],
        "sisnr_enhanced_db": [],
        "sisnr_delta_db": [],
        "pesq_baseline": [],
        "pesq_enhanced": [],
        "pesq_delta": [],
        "peaq_odg_baseline": [],
        "peaq_odg_enhanced": [],
        "peaq_odg_delta": [],
        "peaq_di_baseline": [],
        "peaq_di_enhanced": [],
        "peaq_di_delta": [],
        "inference_seconds": [],
        "seconds_per_sample": [],
        "realtime_factor": [],
    }

    with tempfile.TemporaryDirectory(prefix="dereverb_eval_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for idx, item in enumerate(tqdm(data, desc="Evaluating"), start=1):
            try:
                if "original_path" not in item or "reverberant_path" not in item:
                    skipped += 1
                    continue

                clean_path = resolve_audio_path(item["original_path"], metadata_path)
                reverb_path = resolve_audio_path(item["reverberant_path"], metadata_path)

                if not clean_path.exists():
                    msg = f"Clean reference not found: {clean_path}"
                    if args.fail_on_missing_clean:
                        raise FileNotFoundError(msg)
                    skipped += 1
                    continue

                if not reverb_path.exists():
                    skipped += 1
                    continue

                clean = load_audio(clean_path, sr)
                reverb = load_audio(reverb_path, sr)

                n = min(clean.shape[0], reverb.shape[0])
                if n < 2:
                    skipped += 1
                    continue

                clean = clean[:n]
                reverb = reverb[:n]

                t0 = time.perf_counter()
                enhanced = run_inference(model, reverb, device, seg_len)
                inference_seconds = time.perf_counter() - t0

                enhanced = enhanced[:n]
                n_out = min(clean.shape[0], enhanced.shape[0])
                clean = clean[:n_out]
                reverb = reverb[:n_out]
                enhanced = enhanced[:n_out]

                if n_out < 2:
                    skipped += 1
                    continue

                mse_baseline = mse_val(clean, reverb)
                mse_enhanced = mse_val(clean, enhanced)
                mse_delta = mse_baseline - mse_enhanced

                sisnr_baseline = si_snr_val(reverb, clean)
                sisnr_enhanced = si_snr_val(enhanced, clean)
                sisnr_delta = sisnr_enhanced - sisnr_baseline

                pesq_baseline = pesq_wb(clean, reverb, sr)
                pesq_enhanced = pesq_wb(clean, enhanced, sr)
                pesq_delta = (
                    pesq_enhanced - pesq_baseline
                    if np.isfinite(pesq_baseline) and np.isfinite(pesq_enhanced)
                    else float("nan")
                )

                if args.use_peaq:
                    peaq_baseline = peaq_worker.eval_pair(clean_path, reverb_path)
                    enhanced_tmp_path = tmp_dir_path / f"enhanced_{idx}.wav"
                    sf.write(
                        str(enhanced_tmp_path),
                        enhanced.cpu().numpy().astype(np.float32),
                        sr,
                    )
                    peaq_enhanced = peaq_worker.eval_pair(clean_path, enhanced_tmp_path)
                else:
                    peaq_baseline = PeaqResult(float("nan"), float("nan"))
                    peaq_enhanced = PeaqResult(float("nan"), float("nan"))

                odg_delta = (
                    peaq_enhanced.odg - peaq_baseline.odg
                    if np.isfinite(peaq_baseline.odg) and np.isfinite(peaq_enhanced.odg)
                    else float("nan")
                )
                di_delta = (
                    peaq_enhanced.di - peaq_baseline.di
                    if np.isfinite(peaq_baseline.di) and np.isfinite(peaq_enhanced.di)
                    else float("nan")
                )

                seconds_per_sample = inference_seconds / float(n_out)
                duration_seconds = n_out / float(sr)
                realtime_factor = inference_seconds / duration_seconds

                row = {
                    "file_id": Path(item["reverberant_path"]).stem,
                    "split": item.get("split", ""),
                    "clean_path": str(clean_path),
                    "reverberant_path": str(reverb_path),
                    "num_samples": n_out,
                    "duration_seconds": duration_seconds,
                    "mse_baseline": mse_baseline,
                    "mse_enhanced": mse_enhanced,
                    "mse_delta": mse_delta,
                    "sisnr_baseline_db": sisnr_baseline,
                    "sisnr_enhanced_db": sisnr_enhanced,
                    "sisnr_delta_db": sisnr_delta,
                    "pesq_baseline": pesq_baseline,
                    "pesq_enhanced": pesq_enhanced,
                    "pesq_delta": pesq_delta,
                    "peaq_odg_baseline": peaq_baseline.odg,
                    "peaq_odg_enhanced": peaq_enhanced.odg,
                    "peaq_odg_delta": odg_delta,
                    "peaq_di_baseline": peaq_baseline.di,
                    "peaq_di_enhanced": peaq_enhanced.di,
                    "peaq_di_delta": di_delta,
                    "inference_seconds": inference_seconds,
                    "seconds_per_sample": seconds_per_sample,
                    "realtime_factor": realtime_factor,
                }

                csv_rows.append(row)

                for key in metric_values:
                    metric_values[key].append(float(row[key]))

            except Exception as e:
                failed += 1
                print(f"[WARN] Row {idx} failed: {e}")

    peaq_worker.close()

    summary = {
        "files_requested": len(data),
        "files_processed": len(csv_rows),
        "files_skipped": skipped,
        "files_failed": failed,
        "stats": {k: _mean_std_min_max(v) for k, v in metric_values.items()},
    }

    write_outputs(out_dir, csv_rows, summary)

    print(f"\nMetrics written to: {out_dir / 'metrics.csv'}")
    print(f"Summary written to: {out_dir / 'metrics_summary.txt'}")
    print(f"Summary JSON     : {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
