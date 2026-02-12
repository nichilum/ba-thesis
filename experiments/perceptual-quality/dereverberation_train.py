from data_loader.dereverberation_dataset import DereverberationDataset
from model.dereverberation_simple import DereverberationLightningModule
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from utils.load_data import load_data
from pathlib import Path

from utils.seed import seed
import yaml
import sys


def train():
    seed(42)
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)[sys.argv[1]]
    config = {
        "data_split_file": Path(cfg.get("data_split_file", "./data/metadata.jsonl")),
        "batch_size": cfg.get("batch_size", 16),
        "num_workers": cfg.get("num_workers", 4),
        "epochs": cfg.get("epochs", 100),
        "lr": cfg.get("lr", 1e-3),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": cfg.get("save_dir", "checkpoints"),
        "model_out": cfg.get("model_out", "output/derevnet.pt"),
        "segment_length": cfg.get("segment_length", 44100 * 4),
        "sample_rate": cfg.get("sample_rate", 44100),
        "perceptual_loss_model_path": cfg.get(
            "perceptual_loss_model_path",
            "checkpoints/7358_100ep_perceptual_net_best.pth",
        ),
    }

    os.makedirs(os.path.dirname(config["model_out"]), exist_ok=True)

    data = load_data(config["data_split_file"])

    train_dataset = DereverberationDataset(
        data.train_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )
    val_dataset = DereverberationDataset(
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

    model = DereverberationLightningModule(
        loss="perceptual",
        perceptual_loss_model_path=config["perceptual_loss_model_path"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        # monitor="val_loss",
        # mode="min",
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", patience=5, mode="min"
    # )

    logger = TensorBoardLogger("logs", name="derevnet")

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",  # Automatically use GPU if available
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), config["model_out"])
    print(f"model saved to {config['model_out']}")


if __name__ == "__main__":
    train()
