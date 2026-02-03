import subprocess
import os
import csv
import soundfile
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
)
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import torch

RUN_PEAQ = True
REFS_DIRECTORY = "./refs"
TESTS_DIRECTORY = "./tests"
EXPORT_DIRECTORY = "./export"


if __name__ == "__main__":
    refs_files = set(os.listdir(REFS_DIRECTORY))
    tests_files = set(os.listdir(TESTS_DIRECTORY))
    intersection = refs_files.intersection(tests_files)
    wav_files = {f for f in intersection if f.lower().endswith(".wav")}

    mses = ["mse"]
    si_snrs = ["si_snr"]
    pesq_wbs = ["pesq_wb"]
    pesq_nbs = ["pesq_nb"]

    pesq_nb = PerceptualEvaluationSpeechQuality(8000, "nb")
    pesq_wb = PerceptualEvaluationSpeechQuality(16000, "wb")

    si_snr = ScaleInvariantSignalNoiseRatio()

    for file in wav_files:
        ref_filepath = os.path.join(REFS_DIRECTORY, file)
        test_filepath = os.path.join(TESTS_DIRECTORY, file)

        ref_np, sr = soundfile.read(ref_filepath)
        test_np, _ = soundfile.read(test_filepath)

        # ensure mono
        if ref_np.ndim > 1:
            ref_np = ref_np.mean(axis=1)
        if test_np.ndim > 1:
            test_np = test_np.mean(axis=1)
        assert len(ref_np) == len(test_np), "Files are not the same length"

        ref = torch.from_numpy(ref_np).float()
        test = torch.from_numpy(test_np).float()

        mses.append(mean_squared_error(ref_np, test_np))
        si_snrs.append(si_snr(test, ref).item())
        pesq_wbs.append(pesq_wb(test, ref).item())
        pesq_nbs.append(pesq_nb(test, ref).item())

    # write to csv
    filenames = list(wav_files)
    filenames.insert(0, "")
    export_file_name = f"export{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(
        os.path.join(EXPORT_DIRECTORY, export_file_name),
        "w",
    ) as export:
        wr = csv.writer(export, quoting=csv.QUOTE_ALL)
        wr.writerows([filenames, mses, si_snrs, pesq_wbs, pesq_nbs])

    if RUN_PEAQ:
        subprocess.run(["/usr/sbin/python", "peaq.py", "--file_name", export_file_name])
