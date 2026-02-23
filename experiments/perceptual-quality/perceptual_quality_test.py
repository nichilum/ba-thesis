import torch
from torch.utils.data import DataLoader

from model.perceptual_qualitynet import PerceptualQualityNet

# from model.old.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDataset
from utils.metrics import mse_mae_corr
from utils.load_data import load_data
import argparse
from tqdm import tqdm
from utils.seed import seed
import matplotlib.pyplot as plt
from pathlib import Path
import csv

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

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    sns.set_theme(style="white")

    with open(f"plots/{Path(args.checkpoint).stem}.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Type", "MSE", "MAE", "Correlation"])

        for i, key in [
            ((0, 0), "quality"),
            ((0, 1), "odg"),
            ((1, 0), "size"),
            ((1, 1), "wetness"),
        ]:
            preds = torch.cat(list(map(lambda dict: dict[key], all_predictions)), dim=0)
            targets = torch.cat(list(map(lambda dict: dict[key], all_targets)), dim=0)
            metrics = mse_mae_corr(preds, targets)
            print(f"{key} MSE: {metrics['mse']:.4f}")
            print(f"{key} MAE: {metrics['mae']:.4f}")
            print(f"{key} Correlation: {metrics['correlation']:.4f} \n")

            csv_writer.writerow(
                [
                    f"{key}",
                    f"{metrics['mse']}",
                    f"{metrics['mae']}",
                    f"{metrics['correlation']}",
                ]
            )

            x = preds.squeeze(1).cpu().numpy()
            y = targets.squeeze(1).cpu().numpy()

            cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)

            sns.kdeplot(
                x=x,
                y=y,
                cmap=cmap,
                fill=True,
                clip=(0, 1),
                cut=5,
                thresh=0,
                levels=15,
                ax=axs[i],
            )

            sns.regplot(
                x=x,
                y=y,
                scatter=False,
                ax=axs[i],
                line_kws={"linestyle": "--", "linewidth": 2},
            )

            axs[i].plot([0, 1], [0, 1], color="#bc6cbf", alpha=0.5, linewidth=2)
            axs[i].set(xlabel=f"{key}-pred", ylabel=f"{key}-target")
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].set_aspect("equal", adjustable="box")

    # plt.show()
    plt.savefig(f"plots/{Path(args.checkpoint).stem}.svg", bbox_inches="tight")


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
