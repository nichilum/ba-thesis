# -*- encoding: utf-8 -*-
"""
@Filename    :lightning.py
@Time        :2020/07/10 20:27:23
@Author      :Kai Li
@Version     :1.0
"""

import os
from pytorch_lightning import LightningModule
import torch
from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
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
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        mix = batch["mix"]
        refs = batch["ref"]
        ests = self.forward(mix)
        ls_fn = Loss()
        loss = ls_fn.compute_loss(ests, refs)
        return {"val_loss": loss}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

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
        )

    def val_dataloader(self):
        dataset = Datasets(directories=self.val_dirs, sr=self.sample_rate)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )
