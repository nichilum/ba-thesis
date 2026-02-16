import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)

        return x


class MelSpectrogram(nn.Module):
    """Efficient mel-spectrogram computation using torchaudio-style approach.

    Simon King
    Professor of Speech Processing at the University of Edinburgh

    Good answer Danielle. But we should note that this paper is specifically about frequency scales for representing pitch (the perceptual correlate of fundamental frequency)
    rather than the more general spectral envelope information (e.g., formant frequencies) that is important for speech recognition.

    Regarding the choice of frequency scale for Automatic Speech Recognition (ASR), the key property we want is a non-linear scale that compresses the higher frequencies
    more than the lower ones. In other words, the resulting features (e.g., filterbank energies) use more co-efficients to describe the most important (i.e., most informative)
    frequency range for speech up to around 3 kHz, and fewer co-efficients for the higher frequencies that are less important (i.e, contain less information).

    All perceptual scales (Mel, Bark, etc) have this property. They will all work much the same for this application and the choice is made either through personal preference,
    or empirically by experimentation. The Mel scale is by far the most popular for ASR.
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=22050,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create mel filterbank
        mel_basis = self._create_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_basis", mel_basis)

        # Create window
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def _create_mel_filterbank(self, sr, n_fft, n_mels, fmin, fmax):
        """Create mel filterbank matrix."""

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        min_mel = hz_to_mel(torch.tensor(fmin, dtype=torch.float32))
        max_mel = hz_to_mel(torch.tensor(fmax, dtype=torch.float32))

        mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Create filterbank
        bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def forward(self, audio):
        """
        Convert audio to mel-spectrogram.

        Args:
            audio: (batch, samples) waveform

        Returns:
            mel_spec: (batch, 1, time_frames, n_mels) mel-spectrogram
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

        # Magnitude spectrogram
        mag = torch.abs(stft)  # (batch, freq_bins, time_frames)

        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_basis, mag)  # (batch, n_mels, time_frames)

        # Log compression
        log_mel = torch.log(mel_spec + 1e-10)

        # Transpose to (batch, time_frames, n_mels) and add channel dim
        log_mel = log_mel.transpose(1, 2).unsqueeze(1)

        return log_mel


class PerceptualQualityNet(nn.Module):
    """
    Optimized perceptual quality network inspired by Cnn14 architecture.
    Predicts ODG score, reverb size and wetness, and combined quality score.
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=22050,
    ):
        super().__init__()

        # Mel-spectrogram extractor (frozen parameters for efficiency)
        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # Freeze mel extractor
        for param in self.mel_extractor.parameters():
            param.requires_grad = False

        # Batch norm on input
        self.bn0 = nn.BatchNorm2d(n_mels)

        # Convolutional blocks (progressively increasing channels)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)

        # Shared feature layer
        self.fc_shared = nn.Linear(1024, 512, bias=True)

        # Task-specific heads
        self.odg_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.size_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.wetness_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Combined quality head
        self.quality_head = nn.Sequential(
            nn.Linear(512 + 3, 128),  # 512 features + 3 predictions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.init_weight()

    def init_weight(self):
        """Initialize weights for linear layers."""
        nn.init.xavier_uniform_(self.fc_shared.weight)
        nn.init.constant_(self.fc_shared.bias, 0)
        nn.init.constant_(self.bn0.weight, 1)
        nn.init.constant_(self.bn0.bias, 0)

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
        # Extract mel-spectrogram
        x = self.mel_extractor(audio)  # (batch, 1, time, n_mels)

        # Transpose for batch norm over frequency
        x = x.transpose(1, 3)  # (batch, n_mels, time, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch, 1, time, n_mels)

        # Convolutional feature extraction
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling (avg over frequency dimension)
        x = torch.mean(x, dim=3)

        # Max + mean pooling over time dimension
        x_max, _ = torch.max(x, dim=2)
        x_mean = torch.mean(x, dim=2)
        x = x_max + x_mean  # (batch, 1024)

        # Shared feature extraction
        x = F.dropout(x, p=0.5, training=self.training)
        shared = F.relu_(self.fc_shared(x))
        shared = F.dropout(shared, p=0.3, training=self.training)

        # Individual predictions
        odg_pred = self.odg_head(shared)
        size_pred = self.size_head(shared)
        wetness_pred = self.wetness_head(shared)

        # Combined quality prediction
        combined_input = torch.cat([shared, odg_pred, size_pred, wetness_pred], dim=1)
        quality_pred = self.quality_head(combined_input)

        if return_all:
            return {
                "quality": quality_pred,
                "odg": odg_pred,
                "size": size_pred,
                "wetness": wetness_pred,
            }
        return quality_pred


class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_net_path, device="cuda", sample_rate=44100):
        super().__init__()
        self.perceptual_net = PerceptualQualityNet(sample_rate=sample_rate)
        self.perceptual_net.load_state_dict(
            torch.load(perceptual_net_path, map_location=device, weights_only=True)
        )
        self.perceptual_net.eval()
        self.perceptual_net.to(device)

        # Freeze quality network
        for param in self.perceptual_net.parameters():
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
        quality = self.perceptual_net(output_audio)

        perceptual_loss = 1.0 - quality.mean()

        if target_audio is not None:
            mse_loss = F.mse_loss(output_audio, target_audio)
            return perceptual_loss + alpha * mse_loss

        return perceptual_loss
