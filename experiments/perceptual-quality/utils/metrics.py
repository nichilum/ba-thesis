import torch


def mse_mae_corr(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # Pearson correlation
    pred_mean = predictions.mean()
    target_mean = targets.mean()

    numerator = ((predictions - pred_mean) * (targets - target_mean)).sum()
    denominator = torch.sqrt(
        ((predictions - pred_mean) ** 2).sum() * ((targets - target_mean) ** 2).sum()
    )

    correlation = (numerator / (denominator + 1e-8)).item()

    return {"mse": mse, "mae": mae, "correlation": correlation}


def si_snr(y_hat: torch.Tensor, y: torch.Tensor):
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss."""
    y_hat = y_hat - y_hat.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    s_target = (
        (y_hat * y).sum(dim=-1, keepdim=True)
        / (y.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
        * y
    )
    e_noise = y_hat - s_target

    si_snr = 10 * torch.log10(
        (s_target.norm(dim=-1) ** 2 + 1e-8) / (e_noise.norm(dim=-1) ** 2 + 1e-8)
    )
    return -si_snr.mean()


if __name__ == "__main__":
    from load_data import load_data
    from pathlib import Path
    import soundfile as sf
    import torchaudio
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    sample_rate = 44100

    def load_audio(path):
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()

        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)

        if sr != sample_rate:
            waveform = waveform.unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            waveform = waveform.squeeze(0)

        return waveform

    data = load_data(Path("./data/metadata.jsonl"))

    def mapper(map_obj):
        ref_audio = load_audio(map_obj["original_path"])
        test_audio = load_audio(map_obj["reverberant_path"])
        metrics = mse_mae_corr(test_audio, ref_audio)
        wetness = map_obj["wetness"]
        size = map_obj["size"]

        odg_normalized = np.clip((map_obj["odg"] + 4.0) / 4.0, 0, 1)

        quality = odg_normalized * (1 - wetness * 0.4) * (1 - size * 0.3)
        quality = np.clip(quality, 0, 1)

        return {
            "size": size,
            "wetness": wetness,
            "odg": odg_normalized,
            "di": map_obj["di"],
            "sisnr": si_snr(test_audio, ref_audio),
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "correlation": metrics["correlation"],
            "quality": quality,
        }

    results = list(map(mapper, tqdm(data.train_files)))
    metrics = ["odg", "di", "sisnr", "mse", "mae", "correlation", "quality"]
    metric_labels = {
        "odg": "ODG Norm",
        "di": "DI",
        "sisnr": "SI-SNR",
        "mse": "MSE",
        "mae": "MAE",
        "correlation": "Correlation",
        "quality": "Quality",
    }

    x_axes = [
        ("size", "Size", lambda d: sorted(d, key=lambda r: r["size"])),
        ("wetness", "Wetness", lambda d: sorted(d, key=lambda r: r["wetness"])),
        (
            "combined",
            "Size \u22c5 Wetness",
            lambda d: sorted(d, key=lambda r: r["size"] * r["wetness"]),
        ),
    ]

    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(x_axes), figsize=(16, 18))

    for col_idx, (x_key, x_label, sort_fn) in enumerate(x_axes):
        sorted_data = sort_fn(results)

        if x_key == "combined":
            x_vals = [r["size"] * r["wetness"] for r in sorted_data]
        else:
            x_vals = [r[x_key] for r in sorted_data]

        axes[0, col_idx].set_title(f"{x_label}", fontsize=9, pad=12)

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            y_vals = [r[metric] for r in sorted_data]

            ax.tick_params(labelsize=8)
            ax.grid(linewidth=0.7, zorder=0)

            ax.plot(x_vals, y_vals, linewidth=2, zorder=3)
            ax.scatter(
                x_vals,
                y_vals,
                s=25,
                zorder=4,
                linewidths=0.8,
            )

            if col_idx == 0:
                ax.set_ylabel(metric_labels[metric], fontsize=9, labelpad=6)

            if row_idx == len(metrics) - 1:
                ax.set_xlabel(x_label, fontsize=9, labelpad=6)

            ax.set_xticks(np.linspace(min(x_vals), max(x_vals), 5))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("results_plots.png", dpi=150, bbox_inches="tight")
