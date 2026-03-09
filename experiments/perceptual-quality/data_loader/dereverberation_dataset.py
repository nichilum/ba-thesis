import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


class DereverberationDataset(Dataset):
    def __init__(
        self,
        data,
        segment_length=44100 * 4,
        sample_rate=44100,
        silence_thresh=-40,  # dBFS
        min_silence_len=100,
    ):
        self.data = data
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reverb_audio = self.load_audio(self.data[idx]["reverberant_path"])
        original_audio = self.load_audio(self.data[idx]["original_path"])

        mask = self.trim_to_longest_nonsilent(reverb_audio)

        if reverb_audio.shape[0] > self.segment_length:
            start = random.randint(0, reverb_audio.shape[0] - self.segment_length)
            reverb_audio = reverb_audio[start : start + self.segment_length]
            original_audio = original_audio[start : start + self.segment_length]
            mask = mask[start : start + self.segment_length]
        else:
            pad_length = self.segment_length - reverb_audio.shape[0]
            reverb_audio = torch.nn.functional.pad(reverb_audio, (0, pad_length))
            original_audio = torch.nn.functional.pad(original_audio, (0, pad_length))
            mask = torch.nn.functional.pad(mask, (0, pad_length))

        return {
            "reverb_audio": reverb_audio,
            "original_audio": original_audio,
            "mask": mask,
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

    def trim_to_longest_nonsilent(self, reverb_waveform: torch.Tensor):
        pcm = (reverb_waveform.numpy() * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            pcm.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1,
        )

        nonsilent_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
        )

        if not nonsilent_ranges:
            mask = torch.ones(reverb_waveform.shape[0], dtype=torch.float32)
            return mask

        mask = torch.zeros(reverb_waveform.shape[0], dtype=torch.float32)
        for start_ms, end_ms in nonsilent_ranges:
            start_sample = int(start_ms / 1000 * self.sample_rate)
            end_sample = min(
                int(end_ms / 1000 * self.sample_rate), reverb_waveform.shape[0]
            )
            mask[start_sample:end_sample] = 1.0

        return mask
