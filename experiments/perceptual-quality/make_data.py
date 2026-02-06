import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import math
import logging
from pedalboard import Reverb
from pedalboard.io import AudioFile
import random
import subprocess
import shutil
from tqdm import tqdm

SEED = 42
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
assert math.isclose(sum(list(SPLIT.values())), 1), (
    "Train, val and test split are not valid"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

effect = Reverb(room_size=0, wet_level=0, dry_level=0.5)


def reverberate(file, output_dir):
    """
    dryness should be set so that when size, wet == 0 the original signal equals output signal

    size is an arbitrary/dimensionless number to assess the reverb length,
    when using rt60s 1.0 should be the longest rt60 value

    size and wetness must be in interval [0, 1]
    """
    size = random.uniform(0, 0.8)
    wetness = random.uniform(0, 0.8)
    effect.room_size = size
    effect.wet_level = wetness

    out_path = output_dir / file.name

    with AudioFile(file.as_posix()) as f:
        with AudioFile(out_path.as_posix(), "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)

                effected = effect(chunk, f.samplerate, reset=False)

                o.write(effected)

    peaq = calc_peaq(ref=file.as_posix(), test=out_path.as_posix())

    return {
        "original_path": file,
        "reverberant_path": out_path,
        "size": size,
        "wetness": wetness,
        "odg": peaq["odg"],
        "di": peaq["di"],
        # TODO: add pesq for weighing between general audio and speech?
    }


def calc_peaq(ref, test):
    out = subprocess.check_output(
        ["/usr/bin/python", "utils/peaq.py", "--ref", ref, "--test", test],
        text=True,
    )

    values = dict(line.split("=") for line in out.strip().splitlines())
    return {"odg": float(values["ODG"]), "di": float(values["DI"])}


def find_audio_files(data_paths):
    audio_files = []
    for dir in data_paths:
        audio_files += [
            p for p in Path(dir).rglob("*") if p.suffix.lower() in {".flac", ".wav"}
        ]
    logger.info(f"Total audio files found: {len(audio_files)}")
    return audio_files[:100]


def reverberate_audio_files(audio_files):
    output_dir = Path("data").resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.clean:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise RuntimeError(f"Data directory '{output_dir}' is not empty")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file in tqdm(audio_files):
        results.append(reverberate(file, output_dir))

    return results


def main():
    audio_files = find_audio_files(data_paths)
    reverberated_audio_files = reverberate_audio_files(audio_files)

    train, test_val = train_test_split(
        reverberated_audio_files, test_size=1 - SPLIT["train"], random_state=SEED
    )
    val, test = train_test_split(
        test_val,
        test_size=1 - (SPLIT["val"] / (SPLIT["val"] + SPLIT["test"])),
        random_state=SEED,
    )

    logger.info(
        f"Train set: {len(train)} files ({len(train) / len(reverberated_audio_files) * 100:.1f}%)"
    )
    logger.info(
        f"Eval set: {len(val)} files ({len(val) / len(reverberated_audio_files) * 100:.1f}%)"
    )
    logger.info(
        f"Test set: {len(test)} files ({len(test) / len(reverberated_audio_files) * 100:.1f}%)"
    )

    with open("data/data.pkl", "wb") as f:
        pickle.dump({"train": train, "val": val, "test": test}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Make Data",
        description="This program creates a data split based on given data paths",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Delete data directory",
    )
    parser.add_argument(
        "-dp",
        "--data-paths",
        nargs="+",
        default=[],
        help="Array of folders containing .wav or .flac files",
    )
    args = parser.parse_args()
    data_paths = list(args.data_paths)
    if not data_paths:
        parser.print_help()
        exit()

    main()
