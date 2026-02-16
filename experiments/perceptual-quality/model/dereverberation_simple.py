from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_tcn import TCN, TemporalConv1d
from model.perceptual_qualitynet import PerceptualLoss
from utils.metrics import si_snr


class DereverberationModel(nn.Module):
    """Encoder-TCN-Decoder dereverberation model (waveform-to-waveform)."""

    def __init__(
        self,
        encoder_channels: int = 256,  # 64
        tcn_channels: Sequence[int] = (256,) * 8,  # (64, 64, 64, 64, 64)
        kernel_size: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        use_norm: str = "weight_norm",
        activation: str = "relu",
        lookahead: int = 0,
        use_skip_connections: bool = True,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.encoder = TemporalConv1d(1, int(encoder_channels), kernel_size=1)

        self.tcn = TCN(
            num_inputs=int(encoder_channels),
            num_channels=list(map(int, tcn_channels)),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            causal=bool(causal),
            use_norm=use_norm,
            activation=activation,
            use_skip_connections=bool(use_skip_connections),
            input_shape="NCL",
            lookahead=int(lookahead),
            output_projection=int(encoder_channels),
            dilations=dilations or [2**i for i in range(len(tcn_channels))],
        )

        self.decoder = TemporalConv1d(int(encoder_channels), 1, kernel_size=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Args:
        audio: (batch, samples) or (batch, 1, samples)
        Returns:
        (batch, samples)
        """
        if audio.ndim == 2:
            x = audio.unsqueeze(1)
        elif audio.ndim == 3:
            x = audio
        else:
            raise ValueError("audio must have shape (N, L) or (N, 1, L)")

        x = self.encoder(x)
        x = self.tcn(x)
        x = self.decoder(x)

        x = x.squeeze(1)
        return x


class DereverberationLightningModule(pl.LightningModule):
    """Minimal Lightning wrapper.

    Expects batches either as:
    - dict with keys: 'reverb_audio' and ('clean_audio' or 'target_audio')
    - tuple/list: (reverb_audio, clean_audio)
    """

    def __init__(
        self,
        model: Optional[DereverberationModel] = None,
        lr: float = 1e-3,
        loss: str = "l1",
        perceptual_loss_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model or DereverberationModel()
        self.lr = float(lr)
        self.loss_name = (loss or "l1").lower()

        if self.loss_name == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_name == "sisnr":
            self.criterion = si_snr
        elif self.loss_name == "perceptual":
            loss = PerceptualLoss(perceptual_loss_model_path, device=self.device)
            self.criterion = loss
        else:
            raise ValueError("Unknown loss type: {}".format(loss))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model(audio)

    def _unpack_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            x = batch["reverb_audio"]
            y = batch["original_audio"]
            if y is None:
                raise KeyError("Batch dict must include 'original_audio'")
            return x, y
        raise TypeError("Unsupported batch format")

    def training_step(self, batch, batch_idx: int):
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if hasattr(self, "log"):
            self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if hasattr(self, "log"):
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
