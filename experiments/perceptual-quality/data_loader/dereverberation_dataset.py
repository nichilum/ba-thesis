import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
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
        original_audio = self.load_audio(self.data[idx]["original_path"])

        if reverb_audio.shape[0] > self.segment_length:
            start = random.randint(0, reverb_audio.shape[0] - self.segment_length)
            reverb_audio = reverb_audio[start : start + self.segment_length]
            original_audio = original_audio[start : start + self.segment_length]
        else:
            pad_length = self.segment_length - reverb_audio.shape[0]
            reverb_audio = torch.nn.functional.pad(reverb_audio, (0, pad_length))
            original_audio = torch.nn.functional.pad(original_audio, (0, pad_length))

        # TODO: live reverberate using RIRs

        return {
            "reverb_audio": reverb_audio,
            "original_audio": original_audio,
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

        return waveform
