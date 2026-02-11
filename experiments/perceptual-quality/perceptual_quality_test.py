import torch
from torch.utils.data import DataLoader

from model.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDataset
from utils.metrics import mse_msa_corr
from utils.load_data import load_data
import argparse
from tqdm import tqdm
from utils.seed import seed
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

import numpy as np
import seaborn as sns


def test_perceptual_net():
    config = {
        "data_split_file": Path("./data/metadata.jsonl"),
        "batch_size": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "segment_length": 44100 * 4,
        "sample_rate": 44100,
    }

    data = load_data(config["data_split_file"])

    test_dataset = PerceptualDataset(
        data.test_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model = PerceptualQualityNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location=config["device"]))
    model = model.to(config["device"])
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            reverb_audio = batch["reverb_audio"].to(config["device"])

            all_targets.append(
                {
                    "quality": batch["quality"].to(config["device"]).unsqueeze(1),
                    "odg": batch["odg"].to(config["device"]).unsqueeze(1),
                    "size": batch["size"].to(config["device"]).unsqueeze(1),
                    "wetness": batch["wetness"].to(config["device"]).unsqueeze(1),
                }
            )

            preds = model(reverb_audio, return_all=True)
            all_predictions.append(preds)

    fig, axs = plt.subplots(2, 2)
    sns.set_theme(style="white")

    for i, key in [
        ((0, 0), "quality"),
        ((0, 1), "odg"),
        ((1, 0), "size"),
        ((1, 1), "wetness"),
    ]:
        preds = torch.cat(list(map(lambda dict: dict[key], all_predictions)), dim=0)
        targets = torch.cat(list(map(lambda dict: dict[key], all_targets)), dim=0)
        metrics = mse_msa_corr(preds, targets)
        print(f"{key} MSE: {metrics['mse']:.4f}")
        print(f"{key} MAE: {metrics['mae']:.4f}")
        print(f"{key} Correlation: {metrics['correlation']:.4f} \n")

        x = preds.squeeze(1).cpu().numpy()
        y = targets.squeeze(1).cpu().numpy()

        cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)

        sns.kdeplot(
            x=x,
            y=y,
            cmap=cmap,
            fill=True,
            clip=(-5, 5),
            cut=10,
            thresh=0,
            levels=15,
            ax=axs[i],
        )

        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        axs[i].scatter(x, y, s=1)
        axs[i].plot(x, poly1d_fn(x), "--")
        axs[i].add_line(mlines.Line2D([0, 1], [0, 1], color="red"))
        axs[i].set(xlabel=f"{key}-pred", ylabel=f"{key}-target")
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Test",
        description="Test checkpoints",
    )
    parser.add_argument(
        "checkpoint",
        default="checkpoints/perceptual_net_best.pth",
    )
    args = parser.parse_args()

    seed(42)
    test_perceptual_net()
