import torch
from torch.utils.data import DataLoader
import os

from model.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDereverberationDataset
from trainer.perceptual_quality_trainer import PerceptualNetTrainer

from utils.seed import seed
from pathlib import Path
from utils.load_data import load_data


def main():
    config = {
        "data_split_file": Path("./data/metadata.jsonl"),
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 50,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "checkpoints",
        "segment_length": 44100 * 4,
        "sample_rate": 44100,
    }

    os.makedirs(config["save_dir"], exist_ok=True)

    data = load_data(config["data_split_file"])

    train_dataset = PerceptualDereverberationDataset(
        data.train_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    val_dataset = PerceptualDereverberationDataset(
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
        device=config["device"],
        save_path=os.path.join(config["save_dir"], "perceptual_net_best.pth"),
    )

    print(f"Training on {config['device']}")
    trainer.train(epochs=config["epochs"])

    print("Training complete!")


if __name__ == "__main__":
    seed(42)
    main()
