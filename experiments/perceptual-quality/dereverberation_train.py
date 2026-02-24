from data_loader.dereverberation_dataset import DereverberationDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from utils.load_data import load_data
from pathlib import Path
from importlib import import_module

from utils.seed import seed
import yaml
import sys


def _get_dereverb_classes(model_variant: str):
    variant = (model_variant or "simple").lower()
    if variant == "simple":
        module_name = "model.dereverberation_simple"
    elif variant == "mha":
        module_name = "model.dereverberation_mha"
    else:
        raise ValueError(
            f"Unsupported model_variant '{model_variant}'. Use 'simple' or 'mha'."
        )

    module = import_module(module_name)
    return module.DereverberationModel, module.DereverberationLightningModule, variant


def train():
    seed(42)
    torch.set_float32_matmul_precision("high")

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)[sys.argv[1]]
    use_cuda = torch.cuda.is_available()
    default_precision = "16-mixed" if use_cuda else "32-true"
    config = {
        "data_split_file": Path(cfg.get("data_split_file", "./data/metadata.jsonl")),
        "batch_size": cfg.get("batch_size", 16),
        "val_batch_size": cfg.get("val_batch_size", cfg.get("batch_size", 16)),
        "num_workers": cfg.get("num_workers", 4),
        "epochs": cfg.get("epochs", 100),
        "lr": cfg.get("lr", 1e-3),
        "device": "cuda" if use_cuda else "cpu",
        "save_dir": cfg.get("save_dir", "checkpoints"),
        "model_out": cfg.get("model_out", "output/derevnet.pt"),
        "segment_length": cfg.get("segment_length", 44100 * 4),
        "sample_rate": cfg.get("sample_rate", 44100),
        "loss": cfg.get("loss", "l1"),
        "precision": cfg.get("precision", default_precision),
        "accumulate_grad_batches": cfg.get("accumulate_grad_batches", 1),
        "detect_anomaly": cfg.get("detect_anomaly", False),
        "perceptual_loss_model_path": cfg.get(
            "perceptual_loss_model_path",
            "checkpoints/7358_100ep_perceptual_net_best.pth",
        ),
        "model_variant": cfg.get("model_variant", "simple"),
        "model": cfg.get("model", {}),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
    }

    DereverberationModel, DereverberationLightningModule, model_variant = (
        _get_dereverb_classes(config["model_variant"])
    )

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
        persistent_workers=config["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )

    model_kwargs = dict(config["model"])
    if model_variant == "simple":
        model_kwargs["gradient_checkpointing"] = config["gradient_checkpointing"]

    dereverb_model = DereverberationModel(**model_kwargs)

    model = DereverberationLightningModule(
        model=dereverb_model,
        lr=config["lr"],
        loss=config["loss"],
        perceptual_loss_model_path=config["perceptual_loss_model_path"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=sys.argv[1] + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        # monitor="val_loss",
        # mode="min",
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", patience=5, mode="min"
    # )

    logger = TensorBoardLogger("logs", name=f"derevnet-{sys.argv[1]}")

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, DeviceStatsMonitor()],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
        precision=config["precision"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        detect_anomaly=config["detect_anomaly"],
        profiler="simple",
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), config["model_out"])
    print(f"model saved to {config['model_out']}")


if __name__ == "__main__":
    train()
