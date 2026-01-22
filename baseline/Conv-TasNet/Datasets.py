# -*- encoding: utf-8 -*-
"""
@Filename    :Datasets.py
@Time        :2020/07/10 23:22:54
@Author      :Kai Li
@Version     :1.0
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from utils import handle_scp
import numpy as np
import soundfile as sf
import glob
import torchaudio.transforms as T
import sys
import os

sys.path.append("../../preprocessing/parameter-reverb")
from parameter_reverb import live_reverberate


def read_wav(fname):
    audio, sr = sf.read(fname, always_2d=True, dtype="float32")  # (frames, channels)
    # Convert to mono to match previous behavior.
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    wav = torch.from_numpy(audio)
    return wav, sr


def make_dataloader(
    is_train=True,
    data_kwargs=None,
    num_workers=0,
    chunk_size=32000,
    batch_size=16,  # from num_workers 4
):
    dataset = Datasets(**data_kwargs, chunk_size=chunk_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=True,
    )


class Datasets(Dataset):
    def __init__(self, sr=8000, chunk_size=32000, directories=None):
        super(Datasets, self).__init__()
        self.sr = sr
        self.chunk_size = chunk_size

        self.samples = []
        for dir in directories:
            self.samples += glob.glob(os.path.join(dir, "**/*.flac"), recursive=True)
        # self.samples = self.samples[:10]  # TODO: remove for prod
        # print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dry_sample, or_sr = read_wav(self.samples[index])

        resampler = T.Resample(or_sr, self.sr)
        dry = resampler(dry_sample)

        # --- ensure fixed length ---
        if dry.shape[0] < self.chunk_size:
            pad = self.chunk_size - dry.shape[0]
            dry = F.pad(dry, (0, pad))
        else:
            start = random.randint(0, dry.shape[0] - self.chunk_size)
            dry = dry[start : start + self.chunk_size]

        dry_np = dry.numpy()
        wet_np = live_reverberate(dry_np, self.sr)

        return {
            "mix": torch.from_numpy(wet_np).float(),
            "ref": torch.from_numpy(dry_np).float(),
        }
