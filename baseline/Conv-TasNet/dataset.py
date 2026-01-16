import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchaudio
import glob
import os
import torchaudio.transforms as T
import sys
import numpy as np

sys.path.append("../../preprocessing/parameter-reverb")
from parameter_reverb import live_reverberate


def read_wav(fname):
    src, sr = torchaudio.load(fname, channels_first=True)
    return src.squeeze(), sr


def make_dataloader(
    is_train=True, data_kwargs=None, num_workers=4, chunk_size=32000, batch_size=16
):
    dataset = TestDataset(**data_kwargs, chunk_size=chunk_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=True,
    )


class TestDataset(Dataset):
    def __init__(self, sr=8000, chunk_size=32000, directories=None):
        super(TestDataset, self).__init__()
        self.sr = sr
        self.chunk_size = chunk_size

        self.samples = []
        for dir in directories:
            self.samples += glob.glob(os.path.join(dir, "*.wav"))
        self.samples = self.samples[:10]  # TODO: remove for prod

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dry_sample, or_sr = read_wav(self.samples[index])
        resampler = T.Resample(or_sr, self.sr)
        dry_resample = np.array(resampler(dry_sample)[0])
        wet_sample = live_reverberate(dry_resample, self.sr)

        return dry_resample, wet_sample
