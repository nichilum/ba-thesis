from functools import lru_cache
import json
from pathlib import Path
import pickle


def load_data(path: Path):
    match path.suffix:
        case ".jsonl":
            return load_jsonl(path)
        case ".pkl":
            return load_pickle(path)


class DataReturnObject(object):
    def __init__(self, train_files, val_files, test_files):
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files


@lru_cache(maxsize=None)
def load_pickle(path):
    with open(path, "rb") as f:
        splits = pickle.load(f)
        train_files, val_files, test_files = (
            splits["train"],
            splits["val"],
            splits["test"],
        )
    return DataReturnObject(
        train_files=train_files, val_files=val_files, test_files=test_files
    )


@lru_cache(maxsize=None)
def load_jsonl(path):
    train_files = []
    val_files = []
    test_files = []

    with open(path) as f:
        for line in f:
            line = json.loads(line)
            if line["split"] == "train":
                train_files.append(line)
            if line["split"] == "val":
                val_files.append(line)
            if line["split"] == "test":
                test_files.append(line)

    return DataReturnObject(
        train_files=train_files, val_files=val_files, test_files=test_files
    )
