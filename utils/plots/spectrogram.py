"""
Plot: Spectrogram comparison – reverberant input / model output / dry reference
Rows: speech (195-*.wav) and music (-0a4gn2ob_E guitar)
Output: thesis/figures/spectrogram_comparison.svg
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyfar as pf
import pyfar.plot as pfplot

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
PLOTS = Path(__file__).parent

SPEECH_WET = PLOTS / "data" / "195-wet.wav"
SPEECH_OUT = PLOTS / "data" / "195-out.wav"
SPEECH_DRY = PLOTS / "data" / "195-dry.wav"

MUSIC_ID = "-0a4gn2ob_E ('Guitar', 'Music', 'Musical instrument').wav"
MUSIC_DRY = ROOT / "analysis" / "refs"  / MUSIC_ID
MUSIC_OUT = ROOT / "analysis" / "tests" / MUSIC_ID
MUSIC_WET = ROOT / "experiments" / "perceptual-quality" / "data" / "audio" / "33" / MUSIC_ID

OUT_PATH = ROOT / "thesis" / "figures" / "spectrogram_comparison.png"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_mono_8k(path: Path) -> pf.Signal:
    """Load a WAV, mix down to mono, and resample to 8 kHz."""
    sig = pf.io.read_audio(str(path))
    if sig.cshape[0] > 1:
        # mix down to mono by averaging channels
        data = sig.time.mean(axis=0, keepdims=True)
        sig = pf.Signal(data, sig.sampling_rate)
    if sig.sampling_rate != 8000:
        sig = pf.dsp.resample(sig, 8000)
    return sig


def trim_to_shortest(*signals: pf.Signal) -> list[pf.Signal]:
    n = min(s.n_samples for s in signals)
    return [pf.Signal(s.time[..., :n], s.sampling_rate) for s in signals]


# ── load & prepare ────────────────────────────────────────────────────────────

speech_wet, speech_out, speech_dry = (
    pf.io.read_audio(str(p)) for p in [SPEECH_WET, SPEECH_OUT, SPEECH_DRY]
)
speech_wet, speech_out, speech_dry = trim_to_shortest(speech_wet, speech_out, speech_dry)

music_wet  = load_mono_8k(MUSIC_WET)
music_out  = load_mono_8k(MUSIC_OUT)
music_dry  = load_mono_8k(MUSIC_DRY)
music_wet, music_out, music_dry = trim_to_shortest(music_wet, music_out, music_dry)

# ── figure layout ─────────────────────────────────────────────────────────────
# 2 rows × (3 panels + 1 colorbar)

COLS = ["Reverberant input", "Model output", "Dry reference"]
ROWS = ["Speech", "Music"]

WIN_LEN = 512
OVERLAP = 0.75

fig = plt.figure(figsize=(11, 5.5))
gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    width_ratios=[1, 1, 1, 0.035],
    hspace=0.42,
    wspace=0.28,
)

panels = [
    # (row, signals in order: wet, out, dry)
    (0, speech_wet, speech_out, speech_dry),
    (1, music_wet,  music_out,  music_dry),
]

for row_idx, wet, out, dry in panels:
    sigs   = [wet, out, dry]
    axes   = [fig.add_subplot(gs[row_idx, col]) for col in range(3)]
    meshes = []

    for col_idx, (ax, sig) in enumerate(zip(axes, sigs)):
        _, qmesh, _ = pfplot.spectrogram(
            sig,
            dB=True,
            log_prefix=20,
            log_reference=1,
            freq_scale="linear",
            window="hann",
            window_length=WIN_LEN,
            window_overlap_fct=OVERLAP,
            colorbar=False,
            ax=ax,
        )
        meshes.append(qmesh)

        if row_idx == 0:
            ax.set_title(COLS[col_idx], fontsize=9, pad=4)
        if col_idx == 0:
            ax.set_ylabel(f"{ROWS[row_idx]}\nFrequency (Hz)", fontsize=8)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_xlabel("Time (s)" if row_idx == 1 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    # harmonise colour limits across the row
    all_data = np.concatenate([m.get_array() for m in meshes])
    finite   = all_data[np.isfinite(all_data)]
    vmin     = float(np.percentile(finite, 2))
    vmax     = float(np.percentile(finite, 99.5))
    for m in meshes:
        m.set_clim(vmin, vmax)

    # shared colorbar in the 4th column
    cax = fig.add_subplot(gs[row_idx, 3])
    fig.colorbar(meshes[-1], cax=cax, label="dB")
    cax.tick_params(labelsize=7)
    cax.yaxis.label.set_size(8)

# ── save ──────────────────────────────────────────────────────────────────────
OUT_PATH = OUT_PATH.with_suffix(".png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, format="png", dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
