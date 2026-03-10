from data_loader.dereverberation_dataset import DereverberationDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
from utils.load_data import load_data
from pathlib import Path
from importlib import import_module

from utils.seed import seed
import yaml
import sys


def main():
    data = load_data(Path(sys.argv[1]))

    train_dataset = DereverberationDataset(
        data.train_files,
        segment_length=44100 * 4,
        sample_rate=44100,
    )

    i = 0
    while True:
        print(i)
        train_dataset.__getitem__(i)
        i += 1


if __name__ == "__main__":
    main()
