from model import TasNet
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from datasets import Datasets
import torch
import soundfile as sf
import time
from tqdm import tqdm

model = TasNet.load_from_checkpoint("checkpoints/MSE_epoch=125-val_loss=0.0009068398.ckpt")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = Datasets(directories=["../../datasets/LibriMix/data/LibriSpeech/test-clean"], chunk_size=32000)
# test_dataset = Datasets(directories=["../../analysis/refs"], chunk_size=80000)

si_snr = ScaleInvariantSignalNoiseRatio().to(device)

start = time.perf_counter()

scores = []
for i, data in tqdm(enumerate(test_dataset)):
    dry, wet, path = data["dry"], data["wet"], data["path"]

    # print(dry.shape, wet.shape)

    dry = dry.to(device)
    wet = wet.to(device)

    wet = wet.unsqueeze(0).unsqueeze(0)
    dry = dry.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        estimated_sources = model(wet)

    score = si_snr(estimated_sources, dry)
    scores.append(score.item())
    
    dry_np = dry.squeeze(0).squeeze(0).detach().cpu().numpy()  # (T,)
    wet_np = wet.squeeze(0).squeeze(0).detach().cpu().numpy()  # (T,)
    out_np = estimated_sources.squeeze(0).squeeze(0).detach().cpu().numpy()  # (T,)

    sr = 8000
    sf.write(f"output/{i}-dry.wav", dry_np, sr)
    sf.write(f"output/{i}-wet.wav", wet_np, sr)
    sf.write(f"output/{i}-out.wav", out_np, sr)

    # estimated_source_1 = estimated_sources[:, 0:1, :]
    # estimated_source_2 = estimated_sources[:, 1:2, :]
    # torchaudio.save(f"output/{i}-dry.wav", dry.squeeze(0), 8000)
    # torchaudio.save(f"output/{i}-wet.wav", wet.squeeze(0), 8000)
    # torchaudio.save(f"output/{i}-processed1.wav", estimated_source_1.squeeze(0), 8000)
    # torchaudio.save(f"output/{i}-processed2.wav", estimated_source_2.squeeze(0), 8000)

#write scores to file
with open("output/scores.txt", "w") as f:
    for score in scores:
        f.write(f"{score}\n")

end = time.perf_counter()
print(f"Processed {len(test_dataset)} files in {end - start} seconds.")