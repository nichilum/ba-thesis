from typing import Any
from pedalboard import load_plugin
from pedalboard.io import AudioFile
import random
import constants
import os
import numpy as np

from dotenv import load_dotenv

load_dotenv()

effect = load_plugin(os.getenv("VST_PATH"))


def mono_to_stereo(chunk: np.ndarray) -> np.ndarray:
    if chunk.ndim == 1:
        return np.stack([chunk, chunk], axis=0)
    if chunk.shape[0] == 1:
        return np.repeat(chunk, 2, axis=0)
    return chunk


def stereo_to_mono(chunk: np.ndarray) -> np.ndarray:
    if chunk.ndim == 2 and chunk.shape[0] == 2:
        return np.mean(chunk, axis=0)
    return chunk


def live_reverberate(chunk, samplerate):
    set_effect_attrib(random.choice, effect, constants.TAL_REVERB_DICT)

    is_mono = chunk.ndim == 1 or chunk.shape[0] == 1
    chunk = mono_to_stereo(chunk)
    effected = effect(chunk, samplerate, reset=False)
    if is_mono:
        effected = stereo_to_mono(effected)
    return effected


def offline_reverberate(audio_file_path, output_file_path):
    set_effect_attrib(random.choice, effect, constants.TAL_REVERB_DICT)

    with AudioFile(audio_file_path) as f:
        with AudioFile(output_file_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                is_mono = chunk.ndim == 1 or chunk.shape[0] == 1
                chunk = mono_to_stereo(chunk)
                effected = effect(chunk, f.samplerate, reset=False)
                if is_mono:
                    effected = stereo_to_mono(effected)
                o.write(effected)


def set_effect_attrib(choice, effect, parameter_dict: dict[str, Any]):
    for key in parameter_dict.keys():
        setattr(effect, key, choice(parameter_dict[key]))


if __name__ == "__main__":
    import soundfile as sf
    import torch
    import torchaudio.transforms as T
    import torch.nn.functional as F
    import math
    import glob
    import tqdm

    random.seed(42)

    chunk_size = 320000
    sr = 8000

    dir = "E:/Github/ba-thesis/datasets/LibriMix/data/LibriSpeech/train-clean-100"
    samples = glob.glob(os.path.join(dir, "**/*.flac"), recursive=True)
    samples = samples[:4000]

    for i, sample in tqdm.tqdm(enumerate(samples)):
        dry_audio, or_sr = sf.read(
            sample,
            always_2d=True,
            dtype="float32",
        )  # (frames, channels)
        if dry_audio.shape[1] > 1:
            dry_audio = dry_audio.mean(axis=1)
        else:
            dry_audio = dry_audio[:, 0]
        dry_sample = torch.from_numpy(dry_audio)

        resampler = T.Resample(or_sr, sr)
        dry = resampler(dry_sample)

        # --- ensure fixed length ---
        if dry.shape[0] < chunk_size:
            pad = chunk_size - dry.shape[0]
            dry = F.pad(dry, (0, pad))
        else:
            start = random.randint(0, dry.shape[0] - chunk_size)
            dry = dry[start : start + chunk_size]

        dry_np = dry.numpy()
        wet_np = live_reverberate(dry_np, sr)

        # print(dry_np)
        # print(wet_np)

        print("dry max", dry_np.max())
        print("dry min", dry_np.min())
        print("wet max", wet_np.max())
        print("wet min", wet_np.min())

        sf.write(f"test-data/tal-{i}-dry.wav", dry_np, sr)
        sf.write(f"test-data/tal-{i}-wet.wav", wet_np, sr)
        for val in wet_np:
            if math.isnan(val):
                print(sample)
                break
