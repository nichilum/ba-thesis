from scipy.io import wavfile
from os import walk, path, makedirs
import random
import numpy as np
from process import simulate_room

VARIANTS_PER_SAMPLE = 1
OUTPUT_DIR = "./out"
makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)



sample_paths = [
    path.join(dirpath, f)
    for (dirpath, dirnames, filenames) in walk(
        "/mnt/h/thesis-data/samples/FSD50K.dev_audio"
    )
    for f in filenames
]


for sample_path in sample_paths:
    for i in range(VARIANTS_PER_SAMPLE):
        sample_in = sample_path

        samplerate_sample, sample = wavfile.read(sample_in)
        print(f"Processing {sample_in}, samplerate: {samplerate_sample}")

        sample_out, room = simulate_room(sample, samplerate_sample)

        out_filename = path.join(
            OUTPUT_DIR,
            f"{path.splitext(path.basename(sample_in))[0]}.wav",
        )

        room.mic_array.to_wav(
            out_filename,
            norm=True,
            bitdepth=np.int16,
        )