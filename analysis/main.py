import subprocess
import os
import csv
import soundfile
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
)
import time
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

RUN_PEAQ = True
REFS_DIRECTORY = "../datasets/audio-set/output"
TESTS_DIRECTORY = "./tests"
EXPORT_DIRECTORY = "./export"

def get_id(filename):
    return filename.split(" ", 1)[0]


def trim_to_same_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    min_len = min(a.numel(), b.numel())
    return a[:min_len], b[:min_len]


def _rms(x: torch.Tensor) -> float:
    x = x.detach()
    return float(torch.sqrt(torch.mean(x * x) + 1e-12).cpu())


def safe_pesq(
    metric: PerceptualEvaluationSpeechQuality,
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int,
    min_seconds: float = 0.25,
    min_rms: float = 1e-4,
) -> float:
    preds, target = trim_to_same_length(preds, target)
    if preds.numel() == 0:
        return float("nan")

    if (preds.numel() / float(sample_rate)) < min_seconds:
        return float("nan")

    if _rms(preds) < min_rms or _rms(target) < min_rms:
        return float("nan")

    preds = preds.clamp(-1.0, 1.0)
    target = target.clamp(-1.0, 1.0)

    try:
        metric.reset()
        return float(metric(preds, target).item())
    except Exception as e:  # PESQ can throw NoUtterancesError for silent/no-speech clips
        if e.__class__.__name__ == "NoUtterancesError":
            return float("nan")
        raise


if __name__ == "__main__":
    refs_files = [f for f in os.listdir(REFS_DIRECTORY) if f.lower().endswith(".wav")]
    tests_files = [f for f in os.listdir(TESTS_DIRECTORY) if f.lower().endswith(".wav")]

    refs_by_id = {get_id(f): f for f in refs_files}
    tests_by_id = {get_id(f): f for f in tests_files}

    common_ids = sorted(set(refs_by_id.keys()) & set(tests_by_id.keys()))

    mses = ["mse"]
    si_snrs = ["si_snr"]
    pesq_wbs = ["pesq_wb"]
    pesq_nbs = ["pesq_nb"]
    
    pesq_nb = PerceptualEvaluationSpeechQuality(8000, "nb")
    pesq_wb = PerceptualEvaluationSpeechQuality(16000, "wb")

    si_snr = ScaleInvariantSignalNoiseRatio()

    for file_id in tqdm(common_ids):
        ref_filepath = os.path.join(REFS_DIRECTORY, refs_by_id[file_id])
        test_filepath = os.path.join(TESTS_DIRECTORY, tests_by_id[file_id])

        ref_np, ref_sr = soundfile.read(ref_filepath)
        test_np, test_sr = soundfile.read(test_filepath)

        # ensure mono
        if ref_np.ndim > 1:
            ref_np = ref_np.mean(axis=1)
        if test_np.ndim > 1:
            test_np = test_np.mean(axis=1)
    
        # print(len(ref_np), len(test_np))
        # print(ref_filepath, test_filepath)
        # assert len(ref_np) == len(test_np), "Files are not the same length"

        ref = torch.from_numpy(ref_np).float() #44100
        test = torch.from_numpy(test_np).float() #16000

        max_sr = max(ref_sr, test_sr)
        max_test_resampler = T.Resample(test_sr, max_sr, dtype=ref.dtype)
        max_ref_resampler = T.Resample(ref_sr, max_sr, dtype=test.dtype)

        test_upsampled = max_test_resampler(test)
        ref_upsampled = max_ref_resampler(ref)

        test_upsampled, ref_upsampled = trim_to_same_length(test_upsampled, ref_upsampled)

        pesq_wbs_test_resampler = T.Resample(test_sr, 16000, dtype=ref.dtype)
        pesq_wbs_ref_resampler = T.Resample(ref_sr, 16000, dtype=test.dtype)
        pesq_nbs_test_resampler = T.Resample(test_sr, 8000, dtype=ref.dtype)
        pesq_nbs_ref_resampler = T.Resample(ref_sr, 8000, dtype=test.dtype)

        pesq_wbs_test = pesq_wbs_test_resampler(test)
        pesq_wbs_ref = pesq_wbs_ref_resampler(ref)
        pesq_nbs_test = pesq_nbs_test_resampler(test)
        pesq_nbs_ref = pesq_nbs_ref_resampler(ref)

        pesq_wbs_test, pesq_wbs_ref = trim_to_same_length(pesq_wbs_test, pesq_wbs_ref)
        pesq_nbs_test, pesq_nbs_ref = trim_to_same_length(pesq_nbs_test, pesq_nbs_ref)

        mses.append(F.mse_loss(test_upsampled, ref_upsampled, reduction="mean").item())
        si_snr.reset()
        si_snrs.append(float(si_snr(test_upsampled, ref_upsampled).item()))
        pesq_wbs.append(safe_pesq(pesq_wb, pesq_wbs_test, pesq_wbs_ref, sample_rate=16000))
        pesq_nbs.append(safe_pesq(pesq_nb, pesq_nbs_test, pesq_nbs_ref, sample_rate=8000))

    # write to csv
    filenames = [""] + common_ids
    export_file_name = f"export{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(
        os.path.join(EXPORT_DIRECTORY, export_file_name),
        "w",
    ) as export:
        wr = csv.writer(export, quoting=csv.QUOTE_ALL)
        wr.writerows([filenames, mses, si_snrs, pesq_wbs, pesq_nbs])

    if RUN_PEAQ:
        subprocess.run(["/usr/sbin/python", "peaq.py", "--file_name", export_file_name])
