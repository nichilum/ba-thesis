import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import pyroomacoustics as pra
from pedalboard import Reverb
from tqdm import tqdm

from load_data import load_data

DEFAULT_METADATA = "../../experiments/perceptual-quality/data/metadata.jsonl"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(DEFAULT_METADATA),
        help="Path to metadata JSONL file.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Subset of records to process.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for synthetic impulse response generation.",
    )
    parser.add_argument(
        "--ir-duration",
        type=float,
        default=3.0,
        help="Duration (seconds) of synthetic impulse response.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of records to process.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize Pedalboard Reverb impulse responses from metadata fields "
            "size and wetness, then measure RT60 or generate pyfar plots."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser("measure", help="Measure RT60 summary statistics.")
    add_common_args(measure_parser)
    measure_parser.add_argument(
        "--dist-plot",
        type=Path,
        default=None,
        help="Optional path to save an RT60 distribution plot (PNG).",
    )
    measure_parser.add_argument(
        "--dist-bins",
        type=int,
        default=40,
        help="Number of histogram bins for RT60 distribution plot.",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate pyfar plots of impulse responses.")
    add_common_args(plot_parser)
    plot_parser.add_argument(
        "--num-plots",
        type=int,
        default=6,
        help="Number of individual impulse responses to plot.",
    )
    plot_parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for saved plot images.",
    )
    plot_parser.add_argument(
        "--plot-style",
        choices=["light", "dark"],
        default="light",
        help="pyfar plot style for generated figures.",
    )

    return parser.parse_args()


def get_records(metadata_path: Path, split: str) -> list[dict]:
    data = load_data(metadata_path)
    if split == "train":
        return data.train_files
    if split == "val":
        return data.val_files
    if split == "test":
        return data.test_files
    return data.train_files + data.val_files + data.test_files


def synthesize_reverb_ir(size: float, wetness: float, fs: int, duration_s: float) -> np.ndarray:
    n_samples = max(1, int(fs * duration_s))
    impulse = np.zeros(n_samples, dtype=np.float32)
    impulse[0] = 1.0

    effect = Reverb(
        room_size=float(size),
        wet_level=float(wetness),
        dry_level=0.0,
    )
    ir = effect(impulse, fs, reset=True)

    if getattr(ir, "ndim", 1) > 1:
        ir = np.mean(ir, axis=1)
    return np.asarray(ir, dtype=np.float64)


def measure_rt60(ir: np.ndarray, fs: int) -> float:
    # Compatibility fallback in case pyroomacoustics changes keyword support.
    try:
        value = pra.experimental.rt60.measure_rt60(ir, fs=fs, decay_db=60, energy_thres=1.0)
    except TypeError:
        value = pra.experimental.rt60.measure_rt60(ir, fs=fs)
    return float(value)


def print_summary(rt60_values: list[float], processed: int, failed: int, clipped: int) -> None:
    print(f"processed={processed}")
    print(f"succeeded={len(rt60_values)}")
    print(f"failed={failed}")
    print(f"clipped_parameters={clipped}")

    if not rt60_values:
        print("No valid RT60 values were measured.")
        return

    arr = np.asarray(rt60_values, dtype=np.float64)
    print(f"mean={arr.mean():.6f}")
    print(f"median={np.median(arr):.6f}")
    print(f"std={arr.std(ddof=0):.6f}")
    print(f"min={arr.min():.6f}")
    print(f"max={arr.max():.6f}")


def sanitize_name(text: str) -> str:
    safe = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in text]
    return "".join(safe)


def select_records_for_plots(records: list[dict], num_plots: int) -> list[dict]:
    if num_plots <= 0 or not records:
        return []
    if len(records) <= num_plots:
        return records

    candidates: list[tuple[float, dict]] = []
    for record in records:
        try:
            size = float(record["size"])
            wetness = float(record["wetness"])
        except (KeyError, TypeError, ValueError):
            continue
        size_clipped = float(np.clip(size, 0.0, 1.0))
        wetness_clipped = float(np.clip(wetness, 0.0, 1.0))
        score = size_clipped + wetness_clipped
        candidates.append((score, record))

    if not candidates:
        return records[:num_plots]

    candidates.sort(key=lambda x: x[0])
    idxs = np.linspace(0, len(candidates) - 1, num=num_plots, dtype=int)
    return [candidates[i][1] for i in idxs]


def finalize_figure(fig: plt.Figure, title: str | None = None) -> None:
    if title is not None:
        fig.suptitle(title, fontsize=12)


def collect_rt60_values(records: list[dict], sample_rate: int, ir_duration: float) -> tuple[list[float], int, int]:
    rt60_values: list[float] = []
    failed = 0
    clipped = 0

    for record in tqdm(records, desc="Measuring RT60"):
        try:
            size = float(record["size"])
            wetness = float(record["wetness"])
        except (KeyError, TypeError, ValueError):
            failed += 1
            continue

        size_clipped = float(np.clip(size, 0.0, 1.0))
        wetness_clipped = float(np.clip(wetness, 0.0, 1.0))
        if size_clipped != size or wetness_clipped != wetness:
            clipped += 1

        try:
            ir = synthesize_reverb_ir(
                size=size_clipped,
                wetness=wetness_clipped,
                fs=sample_rate,
                duration_s=ir_duration,
            )
            rt60_values.append(measure_rt60(ir=ir, fs=sample_rate))
        except Exception:
            failed += 1

    return rt60_values, failed, clipped


def save_rt60_distribution_plot(
    rt60_values: list[float],
    output_path: Path,
    bins: int,
    split: str,
) -> None:
    if not rt60_values:
        print("Skipping distribution plot because there are no valid RT60 values.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rt60_values, dtype=np.float64)
    mean = float(arr.mean())
    median = float(np.median(arr))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].hist(arr, bins=max(5, bins), color="#1f77b4", alpha=0.85, edgecolor="white")
    axes[0].axvline(mean, color="#d62728", linestyle="--", linewidth=2, label=f"mean={mean:.3f}s")
    axes[0].axvline(median, color="#2ca02c", linestyle=":", linewidth=2, label=f"median={median:.3f}s")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("RT60 [s]")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    parts = axes[1].violinplot(arr, showmeans=True, showmedians=True, vert=True)
    for body in parts["bodies"]:
        body.set_facecolor("#17becf")
        body.set_edgecolor("#1f3b4d")
        body.set_alpha(0.65)
    axes[1].boxplot(arr, widths=0.18, vert=True)
    axes[1].set_title("Violin + Box")
    axes[1].set_ylabel("RT60 [s]")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["All samples"])
    axes[1].grid(alpha=0.2, axis="y")

    fig.suptitle(f"RT60 Distribution ({split}) - n={len(arr)}", fontsize=13)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RT60 distribution plot to {output_path}")


def run_measure(args: argparse.Namespace) -> None:
    records = get_records(args.metadata, args.split)
    if args.max_items is not None:
        records = records[: args.max_items]

    rt60_values, failed, clipped = collect_rt60_values(
        records=records,
        sample_rate=args.sample_rate,
        ir_duration=args.ir_duration,
    )

    print_summary(
        rt60_values=rt60_values,
        processed=len(records),
        failed=failed,
        clipped=clipped,
    )
    if args.dist_plot is not None:
        save_rt60_distribution_plot(
            rt60_values=rt60_values,
            output_path=args.dist_plot,
            bins=args.dist_bins,
            split=args.split,
        )


def run_plot(args: argparse.Namespace) -> None:
    records = get_records(args.metadata, args.split)
    if args.max_items is not None:
        records = records[: args.max_items]
    selected = select_records_for_plots(records, args.num_plots)

    args.plot_dir.mkdir(parents=True, exist_ok=True)

    ir_bank: list[np.ndarray] = []
    labels: list[str] = []

    for idx, record in enumerate(tqdm(selected, desc="Plotting impulses"), start=1):
        try:
            size = float(record["size"])
            wetness = float(record["wetness"])
        except (KeyError, TypeError, ValueError):
            continue

        size_clipped = float(np.clip(size, 0.0, 1.0))
        wetness_clipped = float(np.clip(wetness, 0.0, 1.0))
        ir = synthesize_reverb_ir(
            size=size_clipped,
            wetness=wetness_clipped,
            fs=args.sample_rate,
            duration_s=args.ir_duration,
        )

        sig = pf.Signal(ir, args.sample_rate)
        rt60 = measure_rt60(ir=ir, fs=args.sample_rate)

        with pf.plot.context(args.plot_style):
            fig, _ = plt.subplots(2, 1, figsize=(10, 7))
            pf.plot.time_freq(
                sig,
                dB_time=True,
                dB_freq=True,
                unit="ms",
                freq_scale="log",
                style=args.plot_style,
            )
            fig = plt.gcf()
            finalize_figure(
                fig,
                title=(
                    f"IR {idx}: size={size_clipped:.3f}, "
                    f"wetness={wetness_clipped:.3f}, RT60={rt60:.3f}s"
                ),
            )
            out_name = sanitize_name(
                f"ir_{idx:02d}_size_{size_clipped:.3f}_wet_{wetness_clipped:.3f}.png"
            )
            fig.savefig(args.plot_dir / out_name, dpi=180, bbox_inches="tight")
            plt.close(fig)

        ir_bank.append(ir)
        labels.append(f"{idx}: s={size_clipped:.2f}, w={wetness_clipped:.2f}")

    if not ir_bank:
        print("No valid records available for plotting.")
        return

    extra_plots = 1
    if len(ir_bank) >= 2:
        stacked = np.stack(ir_bank, axis=0)
        sig_bank = pf.Signal(stacked, args.sample_rate)

        with pf.plot.context(args.plot_style):
            fig, _ = plt.subplots(figsize=(10, 5))
            pf.plot.time_2d(
                sig_bank,
                dB=True,
                unit="ms",
                indices=np.arange(1, len(ir_bank) + 1),
                colorbar=True,
                style=args.plot_style,
            )
            fig = plt.gcf()
            finalize_figure(fig, title="Impulse Bank (2D Time Map)")
            fig.savefig(args.plot_dir / "impulse_bank_time2d.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        extra_plots += 1

    with pf.plot.context(args.plot_style):
        fig, _ = plt.subplots(figsize=(10, 5))
        pf.plot.spectrogram(
            pf.Signal(ir_bank[0], args.sample_rate),
            dB=True,
            freq_scale="log",
            unit="ms",
            style=args.plot_style,
        )
        fig = plt.gcf()
        finalize_figure(fig, title="Representative IR Spectrogram")
        fig.savefig(args.plot_dir / "impulse_representative_spectrogram.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(ir_bank) + extra_plots} plots to {args.plot_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "measure":
        run_measure(args)
        return
    if args.command == "plot":
        run_plot(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
