import torch
from torch.utils.data import DataLoader

from model.dereverberation_simple import DereverberationModel
from data_loader.dereverberation_dataset import DereverberationDataset
from utils.metrics import mse_msa_corr
from utils.load_data import load_data
import argparse
from tqdm import tqdm
from utils.seed import seed
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from utils.metrics import si_snr
import soundfile as sf
import seaborn as sns


def test_dereverb_net():
    config = {
        "data_split_file": Path("./data/metadata.jsonl"),
        "batch_size": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "segment_length": 44100 * 4,
        "sample_rate": 44100,
    }

    data = load_data(config["data_split_file"])

    test_dataset = DereverberationDataset(
        data.test_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model = DereverberationModel()

    checkpoint_obj = torch.load(args.checkpoint, map_location=config["device"])
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    else:
        state_dict = checkpoint_obj

    # PyTorch Lightning checkpoints typically prefix parameters with "model.".
    if isinstance(state_dict, dict) and all(
        str(k).startswith("model.") for k in state_dict.keys()
    ):
        state_dict = {str(k)[len("model.") :]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(config["device"])
    model.eval()

    best_mse = float("inf")

    with open(f"plots/{Path(args.checkpoint).stem}.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["MSE", "MAE", "SISNR", "Correlation"])
        with torch.no_grad():
            for batch in tqdm(test_loader):
                reverb_audio = batch["reverb_audio"].to(config["device"])

                targets = batch["original_audio"].to(config["device"]).unsqueeze(1)

                preds = model(reverb_audio)

                metrics = mse_msa_corr(preds, targets)
                si_snr_value = si_snr(preds, targets)
                if metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]
                    for i in preds:
                        sf.write(
                            f"output/{Path(args.checkpoint).stem}_{i}.wav",
                            i.cpu().numpy(),
                            config["sample_rate"],
                        )

                # print(f"MSE: {metrics['mse']:.4f}")
                # print(f"MAE: {metrics['mae']:.4f}")
                # print(f"SI-SNR: {si_snr_value:.4f}")
                # print(f"Correlation: {metrics['correlation']:.4f} \n")

                csv_writer.writerow(
                    [
                        f"{metrics['mse']}",
                        f"{metrics['mae']}",
                        f"{si_snr_value}",
                        f"{metrics['correlation']}",
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Test",
        description="Test checkpoints",
    )
    parser.add_argument(
        "checkpoint",
        default="checkpoints/perceptual_net_best.pth",
    )
    args = parser.parse_args()

    seed(42)
    test_dereverb_net()
