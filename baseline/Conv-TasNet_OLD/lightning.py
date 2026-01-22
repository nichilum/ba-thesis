# -*- encoding: utf-8 -*-
"""
@Filename    :lightning.py
@Time        :2020/07/10 20:27:23
@Author      :Kai Li
@Version     :1.0
"""

from pytorch_lightning import LightningModule
import torch
from Loss import Loss
from torch import optim
from torch.utils.data import DataLoader
from Datasets import Datasets

# from pytorch_lightning.core.lightning import LightningModule
from model import ConvTasNet


class Lightning(LightningModule):
    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        norm="gLN",
        num_spks=2,
        activate="relu",
        causal=False,
        # optimizer
        lr=1e-3,
        # scheduler
        scheduler_mode="min",
        scheduler_factor=0.5,
        patience=2,
        # Dataset
        train_dirs=["../../datasets/LibriMix/data/LibriSpeech/train-clean-100"],
        val_dirs=["../../datasets/LibriMix/data/LibriSpeech/dev-clean"],
        sr=8000,
        # DataLoader
        batch_size=16,
        num_workers=2,
    ):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        # ----------training&validation&testing Param---------
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        # -----------------------model-----------------------
        self.convtasnet = ConvTasNet(N, L, B, H, P, X, R, norm, num_spks, activate)
        self.criterion = Loss()

    def forward(self, x):
        return self.convtasnet(x)

    # ---------------------
    # TRAINING STEP
    # ---------------------

    def training_step(self, batch, batch_idx):
        mix = batch["mix"]
        refs = batch["ref"]
        ests = self.forward(mix)
        ls_fn = Loss()
        loss = ls_fn.compute_loss(ests, refs)
        # Log in the current Lightning style to avoid silent logging issues.
        # If loss becomes non-finite, surface it clearly.
        if not torch.isfinite(loss):
            self.log("train_loss_non_finite", 1.0, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        mix = batch["mix"]  # [B, T]
        ref = batch["ref"]  # [B, T]

        est = self(mix)  # [B, T]

        loss = self.criterion.compute_loss(est, ref)

        if not torch.isfinite(loss):
            # Keep logging val_loss (will show up as nan) but also emit a flag for debugging.
            self.log(
                "val_loss_non_finite",
                1.0,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.scheduler_mode,
            factor=self.scheduler_factor,
            patience=self.patience,
            min_lr=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = Datasets(directories=self.train_dirs, sr=self.sample_rate)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        dataset = Datasets(directories=self.val_dirs, sr=self.sample_rate)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # Validation must not drop the last batch; dropping can make val empty for small datasets
            # and can lead to missing/NaN monitored metrics that trigger early stopping.
            drop_last=False,
            persistent_workers=True
        )
