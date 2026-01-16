import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import dataset
import os
from dotenv import load_dotenv
import torch

load_dotenv()
os.add_dll_directory(os.getenv("FFMPEG_DLL_PATH"))

DATASET_PATHS = {"audio-set": "../../datasets/audio-set/output"}
testdataset = dataset.TestDataset(directories=DATASET_PATHS.values())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CONVTASNET_BASE_LIBRI2MIX.get_model().to(device)
model.eval()

si_snr = ScaleInvariantSignalNoiseRatio().to(device)

for i, data in enumerate(testdataset):
    dry, wet = data

    dry = torch.from_numpy(dry).float().to(device)
    wet = torch.from_numpy(wet).float().to(device)

    wet = wet.unsqueeze(0).unsqueeze(0)
    dry = dry.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        estimated_sources = model(wet)

    estimated_source_1 = estimated_sources[:, 0:1, :]
    estimated_source_2 = estimated_sources[:, 1:2, :]
    score = si_snr(estimated_source_1, dry)
    print(score.item())
    torchaudio.save(f"output/{i}-dry.wav", dry.squeeze(0), 8000)
    torchaudio.save(f"output/{i}-wet.wav", wet.squeeze(0), 8000)
    torchaudio.save(f"output/{i}-processed1.wav", estimated_source_1.squeeze(0), 8000)
    torchaudio.save(f"output/{i}-processed2.wav", estimated_source_2.squeeze(0), 8000)
