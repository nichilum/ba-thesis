from scipy.io import wavfile
from os import walk, path, makedirs
import random
from process import apply_convolution_reverb


IRS_PER_SAMPLE = 5
OUTPUT_DIR = "./out"
MASTER_SAMPLE_RATE = 44100

makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)


sample_paths = [
    path.join(dirpath, f)
    for (dirpath, dirnames, filenames) in walk(
        "/mnt/h/thesis-data/samples/FSD50K.dev_audio"
    )
    for f in filenames
]

ir_paths = [
    path.join(dirpath, f)
    for (dirpath, dirnames, filenames) in walk("/mnt/h/thesis-data/impulse_responses")
    for f in filenames
]


for sample_path in sample_paths:
    for ir_path in random.sample(ir_paths, IRS_PER_SAMPLE):
        sample_in = sample_path
        reverb_in = ir_path

        samplerate_sample, sample = wavfile.read(sample_in)
        samplerate_reverb, reverb = wavfile.read(reverb_in)

        reverb_to_render = apply_convolution_reverb(
            sample, samplerate_sample, reverb, samplerate_reverb, MASTER_SAMPLE_RATE
        )

        out_filename = path.join(
            OUTPUT_DIR,
            f"{path.splitext(path.basename(sample_in))[0]}_with_{path.splitext(path.basename(reverb_in))[0]}.wav",
        )
        wavfile.write(out_filename, MASTER_SAMPLE_RATE, reverb_to_render)
