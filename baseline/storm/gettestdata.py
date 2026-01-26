import soundfile as sf
import torchaudio.transforms as T
import glob
import os
from pathlib import Path

dir = "E:/Github/ba-thesis/datasets/LibriMix/data/LibriSpeech/train-clean-100"
samples = glob.glob(os.path.join(dir, "**/*.flac"), recursive=True)
samples = samples[:100]
for sample in samples:
    dry_audio, or_sr = sf.read(
        sample,
        always_2d=True,
        dtype="float32",
    )  # (frames, channels)

    resampler = T.Resample(or_sr, 16000)
    dry = resampler(dry_audio)

    path = Path(sample)

    sf.write(f"dry-test-data/{path.stem}.wav", dry, 16000)
