import torch
from torch.utils.data import DataLoader
import os

from model.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDereverberationDataset
from trainer.perceptual_quality_trainer import PerceptualNetTrainer

import json
from seed import seed


def main():
    config = {
        "data_split_file": "./data/metadata.jsonl",
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

    train_files = []
    val_files = []

    with open(config["data_split_file"]) as f:
        for line in f:
            line = json.loads(line)
            if line["split"] == "train":
                train_files.append(line)
            if line["split"] == "val":
                val_files.append(line)

    print(train_files)

    train_dataset = PerceptualDereverberationDataset(
        train_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    val_dataset = PerceptualDereverberationDataset(
        val_files,
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
