"""
Plot: Conv-TasNet training loss comparison
SI-SNR loss (unstable/diverging) vs. MSE loss (converging)
Output: thesis/figures/conv_tasnet_loss_comparison.svg
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = Path(__file__).parents[2] / "thesis" / "data" / "conv-tasnet"
OUT_DIR = Path(__file__).parents[2] / "thesis" / "figures"

SISNR_TRAIN = DATA_DIR / "tasnet_version_16_train_loss.csv"
SISNR_VAL   = DATA_DIR / "tasnet_version_16_val_loss.csv"
MSE_TRAIN   = DATA_DIR / "tasnet_mse_version_1_train_loss.csv"
MSE_VAL     = DATA_DIR / "tasnet_mse_version_1_val_loss.csv"


def load_csv(path: Path) -> tuple[list[int], list[float]]:
    steps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["Step"]))
            values.append(float(row["Value"]))
    return steps, values


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Exponential moving average (same weight convention as TensorBoard smoothing)."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


def steps_to_epochs(steps: list[int], steps_per_epoch: int) -> list[float]:
    return [s / steps_per_epoch for s in steps]


# Both runs log every 10 steps; one epoch ≈ 1782 steps (first val checkpoint step)
STEPS_PER_EPOCH = 1782

sisnr_train_steps, sisnr_train_vals = load_csv(SISNR_TRAIN)
sisnr_val_steps,   sisnr_val_vals   = load_csv(SISNR_VAL)
mse_train_steps,   mse_train_vals   = load_csv(MSE_TRAIN)
mse_val_steps,     mse_val_vals     = load_csv(MSE_VAL)

sisnr_train_epochs = steps_to_epochs(sisnr_train_steps, STEPS_PER_EPOCH)
sisnr_val_epochs   = steps_to_epochs(sisnr_val_steps,   STEPS_PER_EPOCH)
mse_train_epochs   = steps_to_epochs(mse_train_steps,   STEPS_PER_EPOCH)
mse_val_epochs     = steps_to_epochs(mse_val_steps,     STEPS_PER_EPOCH)

sisnr_train_smooth = ema(sisnr_train_vals)
mse_train_smooth   = ema(mse_train_vals)

# ── figure ────────────────────────────────────────────────────────────────────

TRAIN_RAW_COLOR  = "#aaaaaa"
TRAIN_SMOOTH_COLOR = "#1f77b4"
VAL_COLOR        = "#d62728"

fig, (ax_sisnr, ax_mse) = plt.subplots(
    1, 2,
    figsize=(9, 3.6),
    constrained_layout=True,
)

# ── left panel: SI-SNR loss ───────────────────────────────────────────────────
ax_sisnr.plot(
    sisnr_train_epochs, sisnr_train_vals,
    color=TRAIN_RAW_COLOR, linewidth=0.6, alpha=0.55, zorder=1,
)
ax_sisnr.plot(
    sisnr_train_epochs, sisnr_train_smooth,
    color=TRAIN_SMOOTH_COLOR, linewidth=1.8, label="Train (smoothed)", zorder=3,
)
ax_sisnr.plot(
    sisnr_val_epochs, sisnr_val_vals,
    color=VAL_COLOR, linewidth=1.8, linestyle="--", label="Validation", zorder=2,
)
ax_sisnr.set_xlabel("Epoch")
ax_sisnr.set_ylabel("SI-SNR loss (dB)")
# ax_sisnr.set_title("SI-SNR loss")
ax_sisnr.legend(fontsize=8)
ax_sisnr.grid(True, alpha=0.3, linewidth=0.5)
ax_sisnr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

# ── right panel: MSE loss ─────────────────────────────────────────────────────
ax_mse.plot(
    mse_train_epochs, mse_train_vals,
    color=TRAIN_RAW_COLOR, linewidth=0.6, alpha=0.55, zorder=1,
)
ax_mse.plot(
    mse_train_epochs, mse_train_smooth,
    color=TRAIN_SMOOTH_COLOR, linewidth=1.8, label="Train (smoothed)", zorder=3,
)
ax_mse.plot(
    mse_val_epochs, mse_val_vals,
    color=VAL_COLOR, linewidth=1.8, linestyle="--", label="Validation", zorder=2,
)
ax_mse.set_xlabel("Epoch")
ax_mse.set_ylabel("MSE loss")
# ax_mse.set_title("MSE loss")
ax_mse.legend(fontsize=8)
ax_mse.grid(True, alpha=0.3, linewidth=0.5)
ax_mse.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

# ── save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "conv_tasnet_loss_comparison.svg"
fig.savefig(out_path, format="svg")
print(f"Saved → {out_path}")
plt.show()
