"""
Plot: Conv-TasNet MSS (Multi-Scale Spectral) training loss
Output: thesis/figures/conv_tasnet_mss_loss.svg
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = Path(__file__).parents[2] / "thesis" / "data" / "conv-tasnet"
OUT_DIR = Path(__file__).parents[2] / "thesis" / "figures"

MSS_TRAIN = DATA_DIR / "tasnet_mss_version_17_train_loss.csv"
MSS_VAL   = DATA_DIR / "tasnet_mss_version_17_val_loss.csv"


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


# One epoch ≈ 1782 steps (first val checkpoint step)
STEPS_PER_EPOCH = 1782

mss_train_steps, mss_train_vals = load_csv(MSS_TRAIN)
mss_val_steps,   mss_val_vals   = load_csv(MSS_VAL)

mss_train_epochs = steps_to_epochs(mss_train_steps, STEPS_PER_EPOCH)
mss_val_epochs   = steps_to_epochs(mss_val_steps,   STEPS_PER_EPOCH)

mss_train_smooth = ema(mss_train_vals)

# ── figure ────────────────────────────────────────────────────────────────────

TRAIN_RAW_COLOR    = "#aaaaaa"
TRAIN_SMOOTH_COLOR = "#1f77b4"
VAL_COLOR          = "#d62728"

fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)

ax.plot(
    mss_train_epochs, mss_train_vals,
    color=TRAIN_RAW_COLOR, linewidth=0.6, alpha=0.55, zorder=1,
)
ax.plot(
    mss_train_epochs, mss_train_smooth,
    color=TRAIN_SMOOTH_COLOR, linewidth=1.8, label="Train (smoothed)", zorder=3,
)
ax.plot(
    mss_val_epochs, mss_val_vals,
    color=VAL_COLOR, linewidth=1.8, linestyle="--", label="Validation", zorder=2,
)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSS loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

# ── save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "conv_tasnet_mss_loss.svg"
fig.savefig(out_path, format="svg")
print(f"Saved → {out_path}")
plt.show()
