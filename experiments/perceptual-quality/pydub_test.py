from data_loader.dereverberation_dataset import DereverberationDataset
from utils.load_data import load_data
from pathlib import Path

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
