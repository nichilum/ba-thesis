import torch
import torch.nn as nn


class PerceptualQualityNet(nn.Module):
    """
    currently predicts odg score, reverb size and wetness, and combined quality score
    """

    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Shared encoder for spectral features
        self.encoder = nn.Sequential(
            # Input: (batch, 1, freq_bins, time)
            nn.Conv2d(1, 32, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed size output
        )

        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3)
        )

        # odg prediction head
        self.odg_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # size prediction head
        self.size_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Wetness prediction head
        self.wetness_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Combined quality head (learns optimal weighting)
        self.quality_head = nn.Sequential(
            nn.Linear(256 + 3, 64),  # 256 features + 3 predictions
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def compute_spectrogram(self, audio):
        """
        Convert audio waveform to log-magnitude spectrogram.

        Args:
            audio: (batch, samples) waveform

        Returns:
            log_mag: (batch, 1, freq_bins, time_frames) spectrogram
        """
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft).to(audio.device),
        )
        mag = torch.abs(stft)
        log_mag = torch.log(mag + 1e-8)
        return log_mag.unsqueeze(1)  # Add channel dimension

    def forward(self, audio, return_all=False):
        """
        Forward pass through the network.

        Args:
            audio: (batch, samples) waveform
            return_all: if True, return all predictions; else just quality

        Returns:
            quality: (batch, 1) predicted quality score [0, 1]
            or dict with all predictions if return_all=True
        """
        # Compute spectrogram
        spec = self.compute_spectrogram(audio)

        # Extract features
        features = self.encoder(spec)
        shared = self.shared_fc(features)

        # Individual predictions
        odg_pred = self.odg_head(shared)
        size_pred = self.size_head(shared)
        wetness_pred = self.wetness_head(shared)

        # Combined quality prediction
        combined_input = torch.cat([shared, odg_pred, size_pred, wetness_pred], dim=1)
        quality = self.quality_head(combined_input)

        if return_all:
            return {
                "quality": quality,
                "odg": odg_pred,
                "size": size_pred,
                "wetness": wetness_pred,
            }
        return quality


class DereverberationLoss(nn.Module):
    """
    Perceptual loss function for dereverberation using trained quality network.
    """

    def __init__(self, quality_net_path, device="cuda"):
        super().__init__()
        self.quality_net = PerceptualQualityNet()
        self.quality_net.load_state_dict(
            torch.load(quality_net_path, map_location=device)
        )
        self.quality_net.eval()
        self.quality_net.to(device)

        # Freeze quality network
        for param in self.quality_net.parameters():
            param.requires_grad = False

    def forward(self, output_audio, target_audio=None, alpha=0.1):
        """
        Compute perceptual loss for dereverberation.

        Args:
            output_audio: (batch, samples) dereverberated audio
            target_audio: (batch, samples) clean reference (optional)
            alpha: weight for MSE loss if target provided

        Returns:
            loss: scalar loss value
        """
        # Compute perceptual quality (no gradients needed)
        with torch.no_grad():
            quality_score = self.quality_net(output_audio)

        # Loss is inverse of quality (maximize quality = minimize loss)
        perceptual_loss = 1.0 - quality_score.mean()

        # Optional: combine with MSE for training stability
        if target_audio is not None:
            mse_loss = nn.functional.mse_loss(output_audio, target_audio)
            return perceptual_loss + alpha * mse_loss

        return perceptual_loss
