from utils.load_data import load_data
from utils.metrics import mse_mae_corr, si_snr
import torch
from pathlib import Path
import soundfile as sf
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from model.perceptual_qualitynet import PerceptualQualityNet

if __name__ == "__main__":
    sample_rate = 44100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_net = PerceptualQualityNet(sample_rate=sample_rate)
    quality_net.load_state_dict(
        torch.load(
            "checkpoints/epoch_195-quality-perceptual_net_best.pth",
            map_location=device,
            weights_only=True,
        )
    )
    quality_net.eval()
    quality_net.to(device)

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

    def predict_quality(waveform):
        with torch.no_grad():
            audio = waveform.unsqueeze(0).to(device)
            preds = quality_net(audio, return_all=True)
            return {k: v.squeeze().cpu().item() for k, v in preds.items()}

    data = load_data(Path(sys.argv[1]))

    def mapper(map_obj):
        ref_audio = load_audio(map_obj["original_path"])
        test_audio = load_audio(map_obj["reverberant_path"])
        metrics = mse_mae_corr(test_audio, ref_audio)
        wetness = map_obj["wetness"]
        size = map_obj["size"]

        odg_normalized = np.clip((map_obj["odg"] + 4.0) / 4.0, 0, 1)

        quality = odg_normalized * (1 - wetness * 0.4) * (1 - size * 0.3)
        quality = np.clip(quality, 0, 1)

        net_preds = predict_quality(test_audio)

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
            "net_quality": net_preds["quality"],
        }

    results = list(map(mapper, tqdm(data.test_files)))
    metrics = [
        "odg",
        "di",
        "sisnr",
        "mse",
        "mae",
        "correlation",
        "quality",
        "net_quality",
    ]
    metric_labels = {
        "odg": "ODG Norm",
        "di": "DI",
        "sisnr": "SI-SNR",
        "mse": "MSE",
        "mae": "MAE",
        "correlation": "Correlation",
        "quality": "Quality",
        "net_quality": "NN Quality",
    }

    x_axes = [
        ("size", "Size", lambda d: sorted(d, key=lambda r: r["size"])),
        ("wetness", "Wetness", lambda d: sorted(d, key=lambda r: r["wetness"])),
    ]

    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(x_axes), figsize=(10, 18))
    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)

    for col_idx, (x_key, x_label, sort_fn) in enumerate(x_axes):
        sorted_data = sort_fn(results)

        x_vals = [r[x_key] for r in sorted_data]

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            y_vals = [float(r[metric]) for r in sorted_data]

            ax.grid(linewidth=0.7)

            sns.kdeplot(
                x=x_vals,
                y=y_vals,
                cmap=cmap,
                fill=True,
                clip=(
                    (0, 1),
                    (np.percentile(y_vals, 15), np.percentile(y_vals, 85)),
                ),
                cut=5,
                thresh=0,
                levels=15,
                ax=ax,
            )

            sns.regplot(
                x=x_vals,
                y=y_vals,
                scatter=False,
                ax=ax,
                line_kws={"linestyle": "--", "linewidth": 2},
            )

            if col_idx == 0:
                ax.set_ylabel(metric_labels[metric], fontsize=9, labelpad=6)

            if row_idx == len(metrics) - 1:
                ax.set_xlabel(x_label, fontsize=9, labelpad=6)

            ax.set_xticks(np.linspace(min(x_vals), max(x_vals), 5))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        "plots/data_metrics.svg",
    )


def analyze_mask():
    sample_rate = 44100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_net = PerceptualQualityNet(sample_rate=sample_rate)
    quality_net.load_state_dict(
        torch.load(
            "checkpoints/epoch_195-quality-perceptual_net_best.pth",
            map_location=device,
            weights_only=True,
        )
    )
    quality_net.eval()
    quality_net.to(device)

    def predict_quality(waveform):
        with torch.no_grad():
            audio = waveform.unsqueeze(0).to(device)
            preds = quality_net(audio, return_all=True)
            return {k: v.squeeze().cpu().item() for k, v in preds.items()}

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

    path_reverberant = Path(
        "/home/jojo/ba-thesis/experiments/perceptual-quality/data/audio/e4/5456-62043-0005.wav"
    )
    path_dry = Path(
        "/home/jojo/ba-thesis/datasets/LibriMix/data/LibriSpeech/train-clean-100/5456/62043/5456-62043-0005.flac"
    )

    # plt.plot(load_audio(path_reverberant).cpu().numpy())
    # plt.plot(load_audio(path_dry).cpu().numpy())
    # plt.show()

    dry_data = load_audio(path_dry)
    rev_data = load_audio(path_reverberant)

    print(len(dry_data))

    quality = []

    for pct in range(100):
        cutoff = int(len(dry_data) * pct / 100)
        modified = dry_data.clone()
        modified[:cutoff] = 0
        quality.append(predict_quality(modified)["quality"])

    plt.plot(quality)
    plt.show()
