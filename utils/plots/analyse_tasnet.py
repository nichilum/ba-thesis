"""
Evaluate dereverberation quality metrics for baseline/Conv-TasNet/output/.

File naming convention: {number}-dry.wav, {number}-wet.wav, {number}-out.wav

Metrics computed per sample (out vs dry):
  - SI-SNR   (scale-invariant signal-to-noise ratio)
  - SI-SDR   (scale-invariant signal-to-distortion ratio)
  - SI-SAR   (scale-invariant signal-to-artifacts ratio)
  - PESQ     (via torchaudio, wideband 16 kHz)
  - WV-MOS   (UTMOS/WV-MOS predictor via speechmos)

Then reports mean ± std across all samples.
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.pipelines import SQUIM_OBJECTIVE  # PESQ, STOI, SI-SDR from torchaudio
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
OUTPUT_DIR = ROOT / "baseline" / "Conv-TasNet" / "output"
TARGET_SR = 8_000   # Conv-TasNet output is 8 kHz

# ── helpers ───────────────────────────────────────────────────────────────────

def load_mono(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load audio, mix to mono, resample → (1, T) float32 tensor."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav  # (1, T)


def trim_to_shortest(a: torch.Tensor, b: torch.Tensor):
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def align_signals(est: torch.Tensor, ref: torch.Tensor, max_lag_ms: float = 100.0):
    """
    Align est to ref by finding the cross-correlation peak within ±max_lag_ms.
    Returns (est_aligned, ref_trimmed) of equal length.
    """
    max_lag = int(max_lag_ms / 1000 * TARGET_SR)
    e = est.squeeze().numpy()
    r = ref.squeeze().numpy()
    n = min(len(e), len(r), 10 * TARGET_SR)  # use up to 10 s for lag estimation
    e_norm = e[:n] / (np.linalg.norm(e[:n]) + 1e-9)
    r_norm = r[:n] / (np.linalg.norm(r[:n]) + 1e-9)
    corr = np.correlate(r_norm, e_norm, mode="full")
    mid = len(corr) // 2
    window = corr[mid - max_lag: mid + max_lag + 1]
    lag = int(np.argmax(np.abs(window)) - max_lag)  # positive → est is ahead of ref

    # Trim both signals to the overlapping region
    total = min(est.shape[-1], ref.shape[-1])
    if lag > 0:
        # est leads ref: drop first `lag` samples of est
        est = est[..., lag:]
        n_out = min(est.shape[-1], ref.shape[-1])
    elif lag < 0:
        # ref leads est: drop first `|lag|` samples of ref
        ref = ref[..., -lag:]
        n_out = min(est.shape[-1], ref.shape[-1])
    else:
        n_out = total

    return est[..., :n_out], ref[..., :n_out], lag


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> float:
    metric = ScaleInvariantSignalNoiseRatio()
    return -metric(est, ref).item()


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    metric = ScaleInvariantSignalDistortionRatio()
    return -metric(est, ref).item()


def si_sar(est: torch.Tensor, ref: torch.Tensor) -> float:
    """
    SI-SAR: SI-SDR of the artifacts component.
    e_artifacts = est - (projection of est onto ref)
    SI-SAR = SI-SDR(est, e_artifacts)  [how well est resembles its own artifact]

    Standard BSS-eval definition: SI-SAR measures artifacts relative to the
    target projection, i.e. how clean the estimate is from spurious content.
    """
    ref_f = ref.double().squeeze()
    est_f = est.double().squeeze()

    # target projection
    alpha = (est_f @ ref_f) / (ref_f @ ref_f + 1e-9)
    s_target = alpha * ref_f

    # artifacts = estimate minus the target projection
    e_artif = est_f - s_target

    if e_artif.norm() < 1e-9:
        return float("inf")

    # SI-SAR: how large is the target signal relative to artifacts
    scale = (s_target @ e_artif) / (e_artif @ e_artif + 1e-9)
    num = (scale * e_artif).norm() ** 2
    den = (s_target - scale * e_artif).norm() ** 2 + 1e-9
    return float(10 * torch.log10(num / den + 1e-9))


def compute_pesq(est: torch.Tensor, ref: torch.Tensor, model=None) -> float:
    """
    PESQ via torchaudio SQUIM objective model (non-intrusive proxy),
    falling back to torchmetrics PESQ if available.
    SQUIM returns (STOI, PESQ, SI-SDR); we take index 1 (PESQ).
    """
    try:
        if model is None:
            from torchaudio.pipelines import SQUIM_OBJECTIVE
            model = SQUIM_OBJECTIVE.get_model()
            model.eval()
        # SQUIM expects 16 kHz, shape (batch, time)
        est_16k = T.Resample(TARGET_SR, 16000)(est) if TARGET_SR != 16000 else est
        with torch.no_grad():
            stoi, pesq, _ = model(est_16k)  # non-intrusive (no ref needed)
        return pesq.item()
    except Exception:
        pass

    # Fallback: torchmetrics intrusive PESQ
    try:
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        pesq_metric = PerceptualEvaluationSpeechQuality(TARGET_SR, "wb")
        return pesq_metric(est, ref).item()
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        return float("nan")


def compute_wvmos(est: torch.Tensor, model=None) -> float:
    """WV-MOS / UTMOS via pre-loaded torch.hub model."""
    try:
        if model is None:
            import sys, io
            # suppress "Using cache found in ..." noise
            old_stdout = sys.stdout; sys.stdout = io.StringIO()
            model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            sys.stdout = old_stdout
            model.eval()
        with torch.no_grad():
            score = model(est, TARGET_SR)
        return score.item()
    except Exception as e:
        warnings.warn(f"WV-MOS computation failed: {e}")
        return float("nan")


# ── discover sample IDs ───────────────────────────────────────────────────────

def find_samples(output_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns {sample_id: {"dry": Path, "wet": Path, "out": Path}}
    for every complete triple found in output_dir.
    """
    pattern = re.compile(r"^(.+)-(dry|wet|out)\.wav$", re.IGNORECASE)
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    for f in sorted(output_dir.glob("*.wav")):
        m = pattern.match(f.name)
        if m:
            sample_id, role = m.group(1), m.group(2).lower()
            groups[sample_id][role] = f

    complete = {
        sid: roles
        for sid, roles in groups.items()
        if {"dry", "out"}.issubset(roles)
    }
    missing = len(groups) - len(complete)
    if missing:
        warnings.warn(f"{missing} sample(s) skipped (missing dry or out file).")
    return complete


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    samples = find_samples(OUTPUT_DIR)
    if not samples:
        raise RuntimeError(f"No complete sample pairs found in {OUTPUT_DIR}")

    print(f"Found {len(samples)} sample pairs in {OUTPUT_DIR}\n")

    # Pre-load SQUIM model so its download progress shows before the loop
    print("Loading SQUIM model (downloads on first run)...")
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    _squim_model = SQUIM_OBJECTIVE.get_model()
    _squim_model.eval()
    print("SQUIM model ready.")

    # Pre-load WV-MOS / UTMOS model
    import sys, io
    print("Loading WV-MOS model (downloads on first run)...")
    _wvmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    _wvmos_model.eval()
    print("WV-MOS model ready.\n")

    results: dict[str, list[float]] = defaultdict(list)
    metric_names = ["SI-SNR", "SI-SDR", "PESQ", "WV-MOS"]

    for i, (sid, roles) in enumerate(tqdm(sorted(samples.items()), desc='Evaluating', unit='sample'), 1):
        out = load_mono(roles["out"])
        dry = load_mono(roles["dry"])
        out, dry, _lag = align_signals(out, dry)

        scores = {
            "SI-SNR":  si_snr(out, dry),
            "SI-SDR":  si_sdr(out, dry),
            "PESQ":    compute_pesq(out, dry, model=_squim_model),
            "WV-MOS":  compute_wvmos(out, model=_wvmos_model),
        }

        row = "  ".join(f"{k}: {v:7.3f}" for k, v in scores.items())
        tqdm.write(f"{sid:40s}  {row}")

        for k, v in scores.items():
            if np.isfinite(v):
                results[k].append(v)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'N':>5}")
    print("-" * 70)
    for name in metric_names:
        vals = np.array(results[name])
        if len(vals) == 0:
            print(f"{name:<12}  {'N/A':>8}")
            continue
        print(
            f"{name:<12}  {vals.mean():8.3f}  {vals.std():8.3f}"
            f"  {vals.min():8.3f}  {vals.max():8.3f}  {len(vals):5d}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()