from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pytorch_tcn import TCN
from model.perceptual_qualitynet import PerceptualLoss


class _ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pad_2d_to_multiple(
    x: torch.Tensor, multiple_f: int, multiple_t: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad (N,C,F,T) on F/T dims to be divisible by given multiples.

    Returns padded tensor and pad tuple usable for unpadding.
    Pad format is (pad_left_t, pad_right_t, pad_left_f, pad_right_f).
    """
    if x.ndim != 4:
        raise ValueError("Expected 4D tensor (N,C,F,T)")
    f = x.shape[-2]
    t = x.shape[-1]

    pad_f = (multiple_f - (f % multiple_f)) % multiple_f
    pad_t = (multiple_t - (t % multiple_t)) % multiple_t

    # Only pad on the right to keep alignment simple.
    pad = (0, pad_t, 0, pad_f)
    if pad_f == 0 and pad_t == 0:
        return x, pad
    return F.pad(x, pad), pad


def _unpad_2d(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    pad_left_t, pad_right_t, pad_left_f, pad_right_f = pad
    if pad_left_t != 0 or pad_left_f != 0:
        raise ValueError("This helper assumes only right-padding was used")
    if pad_right_t:
        x = x[..., : -pad_right_t]
    if pad_right_f:
        x = x[..., : -pad_right_f, :]
    return x


class DereverberationModel(nn.Module):
    """Spectrogram U-Net (3 Conv blocks + pooling) + TCN + Multi-Head Attention + decoder.

    The network predicts a sigmoid time-frequency mask via a final 1x1 Conv2D, then applies
    it to the input STFT and reconstructs audio with iSTFT.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        center: bool = True,
        conv_channels: Sequence[int] = (32, 64, 128),
        conv_kernel_size: int = 3,
        pool_kernel: tuple[int, int] = (2, 2),
        tcn_channels: Sequence[int] = (128, 128, 128, 128, 128),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_causal: bool = True,
        tcn_use_norm: str = "weight_norm",
        tcn_activation: str = "relu",
        tcn_lookahead: int = 0,
        tcn_use_skip_connections: bool = False,
        tcn_dilations: Optional[Sequence[int]] = None,
        mha_heads: int = 4,
        mha_key_dim: int = 32,
    ):
        super().__init__()

        if len(conv_channels) != 3:
            raise ValueError("conv_channels must have exactly 3 values")

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.center = bool(center)

        c1, c2, c3 = map(int, conv_channels)

        # Encoder: 3 Conv blocks, each followed by MaxPool.
        self.enc1 = _ConvBlock2d(1, c1, kernel_size=int(conv_kernel_size))
        self.pool1 = nn.MaxPool2d(pool_kernel)
        self.enc2 = _ConvBlock2d(c1, c2, kernel_size=int(conv_kernel_size))
        self.pool2 = nn.MaxPool2d(pool_kernel)
        self.enc3 = _ConvBlock2d(c2, c3, kernel_size=int(conv_kernel_size))
        self.pool3 = nn.MaxPool2d(pool_kernel)

        # Bottleneck: TCN then Multi-Head Attention.
        # TCN is applied along time for each frequency bin (shared weights).
        self.tcn = TCN(
            num_inputs=c3,
            num_channels=list(map(int, tcn_channels)),
            kernel_size=int(tcn_kernel_size),
            dropout=float(tcn_dropout),
            causal=bool(tcn_causal),
            use_norm=tcn_use_norm,
            activation=tcn_activation,
            use_skip_connections=bool(tcn_use_skip_connections),
            input_shape="NCL",
            lookahead=int(tcn_lookahead),
            output_projection=c3,
            dilations=tcn_dilations
            or [2**i for i in range(len(tcn_channels))],
        )

        embed_dim = int(mha_heads) * int(mha_key_dim)
        if embed_dim != c3:
            raise ValueError(
                f"For key_dim={mha_key_dim} and heads={mha_heads}, "
                f"embed_dim is {embed_dim}, but encoder last channel is {c3}. "
                "Set conv_channels[-1] to heads*key_dim (e.g., 128)."
            )

        self.mha = nn.MultiheadAttention(
            embed_dim=c3,
            num_heads=int(mha_heads),
            batch_first=True,
        )

        # Decoder: upsampling + skip connections.
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = _ConvBlock2d(c3 + c3, c2, kernel_size=int(conv_kernel_size))
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = _ConvBlock2d(c2 + c2, c1, kernel_size=int(conv_kernel_size))
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _ConvBlock2d(c1 + c1, c1, kernel_size=int(conv_kernel_size))

        # Output: 1x1 Conv2D + sigmoid mask.
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)
        self.out_act = nn.Sigmoid()

        # Cached window tensor (registered as buffer for correct device moves).
        window = torch.hann_window(self.win_length)
        self.register_buffer("_stft_window", window, persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Args:
        audio: (batch, samples) or (batch, 1, samples)

        Returns:
        Dereverberated waveform: (batch, samples)
        """
        if audio.ndim == 3:
            if audio.shape[1] != 1:
                raise ValueError("If 3D, audio must have shape (N, 1, L)")
            audio_1d = audio[:, 0]
        elif audio.ndim == 2:
            audio_1d = audio
        else:
            raise ValueError("audio must have shape (N, L) or (N, 1, L)")

        # STFT
        stft = torch.stft(
            audio_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._stft_window.to(audio_1d.device),
            center=self.center,
            return_complex=True,
        )

        mag = torch.abs(stft)
        log_mag = torch.log(mag + 1e-8)
        x = log_mag.unsqueeze(1)  # (N,1,F,T)

        # Pad to make pooling/upsampling exact.
        x, pad = _pad_2d_to_multiple(x, multiple_f=8, multiple_t=8)

        # Encoder with skip connections.
        s1 = self.enc1(x)
        x = self.pool1(s1)
        s2 = self.enc2(x)
        x = self.pool2(s2)
        s3 = self.enc3(x)
        x = self.pool3(s3)

        # TCN along time per frequency.
        # x: (N,C,F,T) -> (N*F,C,T)
        n, c, f_bins, t_frames = x.shape
        x_tcn = x.permute(0, 2, 1, 3).contiguous().view(n * f_bins, c, t_frames)
        x_tcn = self.tcn(x_tcn)
        x = x_tcn.view(n, f_bins, c, t_frames).permute(0, 2, 1, 3).contiguous()

        # Multi-Head Attention across time-frequency bins.
        # Tokens: (N, F*T, C)
        x_tokens = x.permute(0, 2, 3, 1).contiguous().view(n, f_bins * t_frames, c)
        x_tokens, _ = self.mha(x_tokens, x_tokens, x_tokens, need_weights=False)
        x = x_tokens.view(n, f_bins, t_frames, c).permute(0, 3, 1, 2).contiguous()

        # Decoder: upsample and fuse skips.
        x = self.up3(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        mask = self.out_act(self.out_conv(x))
        mask = _unpad_2d(mask, pad)
        mask = mask.squeeze(1)  # (N,F,T)

        # Apply mask to complex STFT and invert.
        enhanced_stft = stft * mask
        enhanced = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._stft_window.to(audio_1d.device),
            center=self.center,
            length=audio_1d.shape[-1],
        )
        return enhanced


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
            self.criterion = self.sisnr_loss
        elif self.loss_name == "perceptual":
            loss = PerceptualLoss(perceptual_loss_model_path, device=self.device)
            self.criterion = loss
        else:
            raise ValueError("Unknown loss type: {}".format(loss))

    def sisnr_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss."""
        y_hat = y_hat - y_hat.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)

        s_target = (
            (y_hat * y).sum(dim=-1, keepdim=True)
            / (y.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
            * y
        )
        e_noise = y_hat - s_target

        si_snr = 10 * torch.log10(
            (s_target.norm(dim=-1) ** 2 + 1e-8) / (e_noise.norm(dim=-1) ** 2 + 1e-8)
        )
        return -si_snr.mean()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model(audio)

    def _unpack_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            x = batch["reverb_audio"]
            y = batch.get(
                "clean_audio",
                batch.get("target_audio", batch.get("original_audio")),
            )
            if y is None:
                raise KeyError(
                    "Batch dict must include 'clean_audio', 'target_audio', or 'original_audio'"
                )
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
