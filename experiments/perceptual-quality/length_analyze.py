from utils.load_data import load_data
from utils.metrics import mse_mae_corr, si_snr
import torch
from pathlib import Path
import soundfile as sf
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from model.perceptual_qualitynet import PerceptualQualityNet
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import random

if __name__ == "__main__":
    sample_rate = 44100

    def load_audio(path):
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()

        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)

        if sr != sample_rate:
            waveform = waveform.unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            waveform = waveform.squeeze(0)

        return waveform

    data = load_data(Path(sys.argv[1]))

    full_duration = 0
    non_silent_duration = 0

    shuffled_train = data.train_files
    random.shuffle(shuffled_train)
    num_sample_points = 10

    for i in range(num_sample_points):
        index = random.randint(0, len(shuffled_train) - 1)
        sample_point_full_duration = 0
        sample_point_non_silent_duration = 0
        for ir in range(index - 5, index + 6):
            ref_audio = load_audio(shuffled_train[ir]["original_path"])
            pcm = (ref_audio.numpy() * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                pcm.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
            nonsilent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=100,
                silence_thresh=-40,
            )
            for nr in nonsilent_ranges:
                sample_point_non_silent_duration += nr[1] - nr[0]
            sample_point_full_duration += audio_segment.duration_seconds * 1000
        full_duration += sample_point_full_duration / 10
        non_silent_duration += sample_point_non_silent_duration / 10

    print(
        f"Duration of all train samples combined: {full_duration * len(shuffled_train) / num_sample_points / 1000 / 60} m"
    )
    print(
        f"Duration of all non silent utterances in training data: {non_silent_duration * len(shuffled_train) / num_sample_points / 1000 / 60} m"
    )
    print(
        f"Ratio of non silent duration to full duration: {non_silent_duration / full_duration}"
    )
