"""
Plot: Spectrogram comparison – reverberant input / model output / dry reference
Rows: speech (4481-17498-0004) and music (-0a4gn2ob_E guitar)
Output: thesis/figures/spectrogram_comparison.png
"""

import json
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

SPEECH_OUT_NAME = "4481-17498-0004_79_t60=0.80.wav"
SPEECH_OUT = ROOT / "baseline" / "storm" / "output" / SPEECH_OUT_NAME

# Extract the base stem used to match metadata: "4481-17498-0004"
SPEECH_STEM = SPEECH_OUT_NAME.split("_")[0]  # "4481-17498-0004"

METADATA = ROOT / "experiments" / "perceptual-quality" / "data" / "metadata.jsonl"

MUSIC_ID = "-0a4gn2ob_E ('Guitar', 'Music', 'Musical instrument').wav"
MUSIC_DRY = ROOT / "analysis" / "refs"       / MUSIC_ID
MUSIC_OUT_ID = "-0a4gn2ob_E ('Guitar', 'Music', 'Musical instrument')_with_air_stairway_0_1_2_90_mls.wav"
MUSIC_OUT = ROOT / "analysis" / "tests_storm" / MUSIC_OUT_ID
MUSIC_WET = ROOT / "experiments" / "perceptual-quality" / "data" / "audio" / "33" / MUSIC_ID

OUT_PATH = ROOT / "thesis" / "figures" / "spectrogram_comparison_storm.png"

# ── resolve speech wet + dry from metadata ────────────────────────────────────

LIBRISPEECH_SPLITS = [
    "dev-clean", "dev-other",
    "test-clean", "test-other",
    "train-clean-100", "train-clean-360", "train-other-500",
]

def find_speech_paths(metadata_path: Path, stem: str):
    """
    Search metadata.jsonl for the entry whose reverberant_path filename
    starts with the given stem (e.g. '4481-17498-0004').
    Returns (wet_path, dry_path) as Path objects.

    For the dry path, first tries the path as stored in the metadata. If that
    file doesn't exist, searches all known LibriSpeech splits under the same
    datasets root so the script works regardless of which split was recorded.
    """
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rev_path = Path(entry["reverberant_path"])
            if not rev_path.stem.startswith(stem):
                continue

            wet_path = rev_path

            # ── resolve dry path ──────────────────────────────────────────
            orig = entry["original_path"]
            dry_path = Path(orig) if Path(orig).is_absolute() \
                       else (metadata_path.parent / orig).resolve()

            if dry_path.exists():
                return wet_path, dry_path

            # Re-anchor to correct datasets root (metadata relative path is wrong)
            ls_root = ROOT / "datasets" / "LibriMix" / "data" / "LibriSpeech"
            filename = dry_path.name
            spk_chap = Path(*dry_path.parts[-3:-1])  # e.g. 4481/17498

            for split in LIBRISPEECH_SPLITS:
                candidate = ls_root / split / spk_chap / filename
                if candidate.exists():
                    print(f"  (dry path remapped: {split}/{spk_chap}/{filename})")
                    return wet_path, candidate

            # Last resort: recursive glob from ls_root
            matches = list(ls_root.rglob(filename))
            if matches:
                print(f"  (dry path found via glob: {matches[0]})")
                return wet_path, matches[0]

            raise FileNotFoundError(
                f"Dry reference '{filename}' not found under {ls_root}. "
                f"Tried splits: {LIBRISPEECH_SPLITS}"
            )

    raise FileNotFoundError(
        f"No metadata entry found with reverberant_path stem starting with '{stem}'"
    )


speech_wet_path, speech_dry_path = find_speech_paths(METADATA, SPEECH_STEM)

print(f"Speech wet : {speech_wet_path}")
print(f"Speech dry : {speech_dry_path}")
print(f"Speech out : {SPEECH_OUT}")

# ── helpers ───────────────────────────────────────────────────────────────────

def load_mono_8k(path: Path) -> pf.Signal:
    """Load a WAV/FLAC, mix down to mono, and resample to 8 kHz."""
    sig = pf.io.read_audio(str(path))
    if sig.cshape[0] > 1:
        data = sig.time.mean(axis=0, keepdims=True)
        sig = pf.Signal(data, sig.sampling_rate)
    if sig.sampling_rate != 16000:
        sig = pf.dsp.resample(sig, 16000)
    return sig


def trim_to_shortest(*signals: pf.Signal) -> list[pf.Signal]:
    n = min(s.n_samples for s in signals)
    return [pf.Signal(s.time[..., :n], s.sampling_rate) for s in signals]


# ── load & prepare ────────────────────────────────────────────────────────────

speech_wet = pf.io.read_audio(str(speech_wet_path))
speech_out = pf.io.read_audio(str(SPEECH_OUT))
speech_dry = pf.io.read_audio(str(speech_dry_path))

# mix down to mono if needed (speech files may already be mono)
for name, sig in [("wet", speech_wet), ("out", speech_out), ("dry", speech_dry)]:
    if sig.cshape[0] > 1:
        data = sig.time.mean(axis=0, keepdims=True)
        sig = pf.Signal(data, sig.sampling_rate)

speech_wet, speech_out, speech_dry = trim_to_shortest(speech_wet, speech_out, speech_dry)

music_wet = load_mono_8k(MUSIC_WET)
music_out = load_mono_8k(MUSIC_OUT)
music_dry = load_mono_8k(MUSIC_DRY)
music_wet, music_out, music_dry = trim_to_shortest(music_wet, music_out, music_dry)

# ── figure layout ─────────────────────────────────────────────────────────────

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
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, format="png", dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")