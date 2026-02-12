import torch
from torch.utils.data import DataLoader
import os

from model.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDataset
from trainer.perceptual_quality_trainer import PerceptualNetTrainer

from utils.seed import seed
from pathlib import Path
from utils.load_data import load_data
import matplotlib.pyplot as plt
import yaml
import sys


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)[sys.argv[1]]
    config = {
        "data_split_file": Path(cfg.get("data_split_file", "./data/metadata.jsonl")),
        "batch_size": cfg.get("batch_size", 16),
        "num_workers": cfg.get("num_workers", 4),
        "epochs": cfg.get("epochs", 100),
        "lr": cfg.get("lr", 1e-3),
        "earlystopping": cfg.get("earlystopping", False),
        "patience": cfg.get("patience", 10),
        "delta": cfg.get("delta", 1e-5),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": cfg.get("save_dir", "checkpoints"),
        "segment_length": cfg.get("segment_length", 44100 * 4),
        "sample_rate": cfg.get("sample_rate", 44100),
    }

    os.makedirs(config["save_dir"], exist_ok=True)

    data = load_data(config["data_split_file"])

    train_dataset = PerceptualDataset(
        data.train_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    val_dataset = PerceptualDataset(
        data.val_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = PerceptualQualityNet()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    trainer = PerceptualNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["lr"],
        es=config["earlystopping"],
        patience=config["patience"],
        delta=config["delta"],
        device=config["device"],
        save_path=lambda loss_type, epoch: os.path.join(
            config["save_dir"], f"epoch_{epoch}-{loss_type}-perceptual_net_best.pth"
        ),
    )

    print(f"Training on {config['device']}")
    trainer.train(epochs=config["epochs"])

    plt.figure(figsize=(10, 6))

    plt.plot(trainer.plots["train_loss_full"], label="Train Loss (Total)", lw=2)
    plt.plot(trainer.plots["train_loss_quality"], label="Train Loss (Quality)")
    plt.plot(trainer.plots["train_loss_size"], label="Train Loss (Size)")
    plt.plot(trainer.plots["train_loss_odg"], label="Train Loss (ODG)")
    plt.plot(trainer.plots["train_loss_wetness"], label="Train Loss (Wetness)")
    plt.plot(
        trainer.plots["val_loss_full"], label="Val Loss (Total)", linestyle="--", lw=2
    )
    plt.plot(
        trainer.plots["val_loss_quality"], label="Val Loss (Quality)", linestyle="--"
    )
    plt.plot(trainer.plots["val_loss_size"], label="Val Loss (Size)", linestyle="--")
    plt.plot(trainer.plots["val_loss_odg"], label="Val Loss (ODG)", linestyle="--")
    plt.plot(
        trainer.plots["val_loss_wetness"], label="Val Loss (Wetness)", linestyle="--"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Losses")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    svg_out = Path("plots").resolve()
    svg_out.mkdir(exist_ok=True)
    plt.savefig(svg_out / "losses.svg")


if __name__ == "__main__":
    seed(42)
    main()
