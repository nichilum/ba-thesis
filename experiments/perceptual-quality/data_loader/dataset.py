import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import torchaudio


class DereverberationDataset(Dataset):
    def __init__(
        self,
        data,
        segment_length=44100 * 4,
        sample_rate=44100,
    ):

        self.data = data
        self.segment_length = segment_length
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reverb_audio = self.load_audio(self.data[idx]["reverberant_path"])
        size = self.data[idx]["size"]
        wetness = self.data[idx]["wetness"]
        odg = self.data[idx]["odg"]
        di = self.data[idx]["di"]

        # plt.plot(reverb_audio.cpu().numpy())
        # plt.show()

        odg_normalized = np.clip((odg + 4.0) / 4.0, 0, 1)

        quality_score = odg_normalized * (1 - wetness * 0.4) * (1 - size * 0.3)
        quality_score = np.clip(quality_score, 0, 1)

        return {
            "reverb_audio": reverb_audio,
            "quality_score": torch.tensor(quality_score, dtype=torch.float32),
            "odg": torch.tensor(odg_normalized, dtype=torch.float32),
            "size": torch.tensor(size, dtype=torch.float32),
            "wetness": torch.tensor(wetness, dtype=torch.float32),
        }

    def load_audio(self, path):
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()

        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)

        if sr != self.sample_rate:
            waveform = waveform.unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            waveform = waveform.squeeze(0)

        if waveform.shape[0] > self.segment_length:
            start = random.randint(0, waveform.shape[0] - self.segment_length)
            waveform = waveform[start : start + self.segment_length]
        else:
            pad_length = self.segment_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform
