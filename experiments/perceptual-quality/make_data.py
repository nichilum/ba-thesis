import argparse
import json
import math
import subprocess
from pathlib import Path
from tqdm import tqdm
import hashlib
import soundfile as sf
import numpy as np
from pedalboard import Reverb

SEED = 42
SIZE_RANGE = [0, 0.8]
WET_RANGE = [0, 0.8]

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
assert math.isclose(sum(SPLIT.values()), 1.0)


def assign_split(path: Path):
    name = path.name
    h = hashlib.md5(name.encode()).digest()
    r = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    if r < SPLIT["train"]:
        return "train"
    elif r < SPLIT["train"] + SPLIT["val"]:
        return "val"
    else:
        return "test"


def hash_to_random(filename: str, seed_offset: int = 0) -> float:
    """Generate reproducible random value [0, 1) from filename hash"""
    name = Path(filename).name
    h = hashlib.md5(f"{name}{seed_offset}".encode()).digest()
    return int.from_bytes(h[:4], "little") / 0xFFFFFFFF


def shard_path(base: Path, filename: str):
    h = hashlib.md5(filename.encode()).hexdigest()
    sub = h[:2]
    out = base / sub
    out.mkdir(parents=True, exist_ok=True)
    return (out / filename).with_suffix(".wav")


def find_audio_files(paths):
    for p in paths:
        files = (f for f in Path(p).rglob("*") if f.suffix.lower() in {".wav", ".flac"})
        yield from sorted(files, key=lambda f: str(f).lower())


def reverberate_file(
    in_path: str,
    out_path: str,
    room_size: float,
    wetness: float,
    block_size: int = 44100,
):
    effect = Reverb(
        room_size=room_size,
        wet_level=wetness,
        dry_level=0.5,
    )

    with (
        sf.SoundFile(in_path, "r") as fin,
        sf.SoundFile(
            out_path,
            "w",
            samplerate=fin.samplerate,
            channels=1,
            subtype="FLOAT",
        ) as fout,
    ):
        while True:
            block = fin.read(block_size, dtype="float32", always_2d=False)
            if len(block) == 0:
                break

            if block.ndim > 1:
                block = np.mean(block, axis=1)

            processed = effect(block, fin.samplerate, reset=True)
            fout.write(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data-paths", nargs="+", required=True)
    parser.add_argument("-o", "--out", default="data")
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument(
        "--use-peaq", action="store_true", help="Enable PEAQ quality metrics"
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.jsonl"

    processed = set()
    if args.resume and meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                processed.add(json.loads(line)["original_path"])

    peaq = None
    if args.use_peaq:
        peaq = subprocess.Popen(
            ["/usr/bin/python", "utils/peaq.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    with open(meta_path, "a") as meta:
        for file in tqdm(find_audio_files(args.data_paths)):
            if str(file) in processed:
                continue

            size = hash_to_random(file.name, seed_offset=0)
            size = np.interp(size, [0, 1], SIZE_RANGE)

            wet = hash_to_random(file.name, seed_offset=1)
            wet = np.interp(wet, [0, 1], WET_RANGE)

            out_path = shard_path(audio_dir, file.name)

            reverberate_file(
                in_path=str(file),
                out_path=str(out_path),
                room_size=size,
                wetness=wet,
            )

            if args.use_peaq:
                peaq.stdin.write(f"{file}\t{out_path}\n")
                peaq.stdin.flush()

                odg, di = peaq.stdout.readline().strip().split("\t")
                odg = float(odg)
                di = float(di)

                if np.isnan(odg) or np.isnan(di):
                    continue
            else:
                odg = 0.0
                di = 0.0

            record = {
                "original_path": str(file),
                "reverberant_path": str(out_path),
                "size": np.interp(size, SIZE_RANGE, [0, 1]),
                "wetness": np.interp(wet, WET_RANGE, [0, 1]),
                "odg": odg,
                "di": di,
                "split": assign_split(file),
            }

            json.dump(record, meta)
            meta.write("\n")
            meta.flush()

    if args.use_peaq:
        peaq.stdin.write("QUIT\n")
        peaq.stdin.flush()
        peaq.wait()


if __name__ == "__main__":
    main()
