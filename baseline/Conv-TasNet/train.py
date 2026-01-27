from datasets import make_dataloader
from model import TasNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds


def train():
    set_seeds(42)

    params = {
        "epochs": 100,  # Epochs for training on synthetic data
        "batch_size": 32,
        "lr": 0.001,
        "model_out": "output/tasnet.pt",  # Checkpoint after synthetic training
    }

    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)

    train_loader = make_dataloader(is_train=True, directories=["../../datasets/LibriMix/data/LibriSpeech/train-clean-100"])
    val_loader = make_dataloader(is_train=False, directories=["../../datasets/LibriMix/data/LibriSpeech/dev-clean"], )

    model = TasNet()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="max",
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", patience=5, mode="min"
    # )

    logger = TensorBoardLogger("logs", name="tasnet")

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",  # Automatically use GPU if available
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), params["model_out"])
    print(f"model saved to {params['model_out']}")




if __name__ == "__main__":
    train()