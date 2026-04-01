"""
Plot: Single model training loss vs validation loss
Configurable for any model with train/val CSV logs
Output: thesis/figures/[model_name]_loss.svg
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────
# Update these paths to point to your model's loss CSV files
MODEL_NAME = "derevnet-derev_1_overfit_version_22"
TRAIN_CSV = Path(__file__).parents[2] / "utils" / "plots" / "data" / "derevnet-derev_1_overfit_version_22_train_loss.csv"
VAL_CSV   = Path(__file__).parents[2] / "utils" / "plots" / "data" / "derevnet-derev_1_overfit_version_22_val_loss.csv"
OUT_DIR   = Path(__file__).parents[2] / "thesis" / "figures"

# Smoothing parameter for EMA (0.05 matches TensorBoard default)
SMOOTHING_ALPHA = 0.05

# Steps per epoch (adjust based on your dataset)
STEPS_PER_EPOCH = 10


def load_csv(path: Path) -> tuple[list[int], list[float]]:
    """Load Step and Value columns from CSV."""
    steps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["Step"]))
            values.append(float(row["Value"]))
    return steps, values


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Exponential moving average (same convention as TensorBoard)."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


def steps_to_epochs(steps: list[int], steps_per_epoch: int) -> list[float]:
    """Convert training steps to epoch numbers."""
    return [s / steps_per_epoch for s in steps]


# ── Load data ─────────────────────────────────────────────────────────────────
train_steps, train_vals = load_csv(TRAIN_CSV)
val_steps, val_vals     = load_csv(VAL_CSV)

train_epochs = steps_to_epochs(train_steps, STEPS_PER_EPOCH)
val_epochs   = steps_to_epochs(val_steps, STEPS_PER_EPOCH)

train_smooth = ema(train_vals, SMOOTHING_ALPHA)

# ── Plot ──────────────────────────────────────────────────────────────────────
TRAIN_RAW_COLOR    = "#aaaaaa"
TRAIN_SMOOTH_COLOR = "#1f77b4"
VAL_COLOR          = "#d62728"

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

ax.plot(
    train_epochs, train_vals,
    color=TRAIN_RAW_COLOR, linewidth=0.6, alpha=0.55, zorder=1,
)
ax.plot(
    train_epochs, train_smooth,
    color=TRAIN_SMOOTH_COLOR, linewidth=1.8, label="Train (smoothed)", zorder=3,
)
ax.plot(
    val_epochs, val_vals,
    color=VAL_COLOR, linewidth=1.8, linestyle="--", label="Validation", zorder=2,
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linewidth=0.5)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"{MODEL_NAME}_loss.svg"
fig.savefig(out_path, format="svg")
print(f"Saved → {out_path}")
plt.show()
