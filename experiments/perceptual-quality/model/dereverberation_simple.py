from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
import pytorch_lightning as pl
from pytorch_tcn import TCN, TemporalConv1d
from model.perceptual_qualitynet import PerceptualLoss
from utils.metrics import si_snr


class DereverberationModel(nn.Module):
    """Encoder-TCN-Decoder dereverberation model (waveform-to-waveform).

    Defaults are chosen to keep Conv-TasNet-style temporal structure
    (X=8 blocks/repeat, R=3 repeats, P=3) while matching a ~4-5M parameter
    budget with this implementation.

    TODO: check this statement
    gradient_checkpointing: if True, recompute TCN activations during the
    backward pass instead of storing them. This trades ~30% more compute for
    a very large reduction in activation memory — critical when the sequence
    length is 176 k samples (44.1 kHz × 4 s) and the TCN has 24 blocks.
    """

    def __init__(
        self,
        encoder_channels: int = 256,
        tcn_channels: Sequence[int] = (128,) * 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
        use_norm: str = "layer_norm",
        activation: str = "relu",
        lookahead: int = 0,
        use_skip_connections: bool = True,
        num_blocks_per_repeat: int = 8,
        num_repeats: int = 4,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # if tcn_channels is None:
        #     tcn_channels = (265,) * (int(num_blocks_per_repeat) * int(num_repeats))

        # if dilations is None:
        #     base_dilations = [2**i for i in range(int(num_blocks_per_repeat))]
        #     dilations = base_dilations * int(num_repeats)

        if len(dilations) != len(tcn_channels):
            raise ValueError(
                "len(dilations) must match len(tcn_channels): "
                f"{len(dilations)} != {len(tcn_channels)}"
            )

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
            dilations=list(map(int, dilations)),
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

        # Gradient checkpointing: discard intermediate TCN activations and
        # recompute them on the backward pass. Saves ~(num_blocks - 1) /
        # num_blocks of TCN activation memory at the cost of one extra forward
        # pass through the TCN. With 24 blocks this typically cuts activation
        # memory by ~95 % for the TCN segment.
        if self.gradient_checkpointing and self.training:
            x = torch_checkpoint.checkpoint(self.tcn, x, use_reentrant=False)
        else:
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
            return batch[0], batch[1], None
        if isinstance(batch, dict):
            x = batch["reverb_audio"]
            y = batch["original_audio"]
            mask = batch.get("mask", None)  # (batch, samples), 1=valid 0=silence/pad
            if y is None:
                raise KeyError("Batch dict must include 'original_audio'")
            return x, y, mask
        raise TypeError("Unsupported batch format")

    def _masked_loss(self, y_hat, y, mask):
        if mask is None:
            return self.criterion(y_hat, y)

        if self.loss_name == "sisnr":
            # SI-SNR is a per-sequence metric, mask by zeroing out silent frames
            # then compute per-sample and mean
            y_hat_m = y_hat * mask
            y_m = y * mask
            return self.criterion(y_hat_m, y_m)

        elif self.loss_name == "perceptual":
            y_hat_m = y_hat * mask
            y_m = y * mask
            return self.criterion(y_hat_m, y_m)

        else:
            # L1 / MSE: compute elementwise then average over valid samples only
            # so the loss scale doesn't shrink for short utterances
            elementwise = (
                torch.abs(y_hat - y) if self.loss_name == "l1" else (y_hat - y) ** 2
            )
            masked = elementwise * mask
            loss = masked.sum() / mask.sum().clamp(min=1)
            return loss

    def training_step(self, batch, batch_idx: int):
        x, y, mask = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self._masked_loss(y_hat, y, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y, mask = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self._masked_loss(y_hat, y, mask)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    # def on_before_optimizer_step(self, optimizer, *args: Any, **kwargs: Any) -> None:
    #     grad_norm_dict = self._compute_grad_norms()
    #     if grad_norm_dict:
    #         self.log_grad_norm(grad_norm_dict)

    # def log_grad_norm(self, grad_norm_dict: Dict[str, torch.Tensor]) -> None:
    #     self.log_dict(
    #         grad_norm_dict,
    #         on_step=True,
    #         on_epoch=False,
    #         prog_bar=False,
    #         logger=True,
    #     )

    # def _compute_grad_norms(self) -> Dict[str, torch.Tensor]:
    #     grads = [p.grad for p in self.parameters() if p.grad is not None]
    #     if not grads:
    #         return {}

    #     device = grads[0].device
    #     l2_sq_sum = torch.zeros((), device=device)
    #     inf_max = torch.zeros((), device=device)

    #     for g in grads:
    #         g_detached = g.detach()
    #         l2_sq_sum = l2_sq_sum + torch.sum(g_detached * g_detached)
    #         inf_max = torch.maximum(inf_max, torch.max(torch.abs(g_detached)))

    #     l2_total = torch.sqrt(l2_sq_sum)
    #     return {
    #         "grad_norm/l2_total": l2_total,
    #         "grad_norm/inf_total": inf_max,
    #     }
