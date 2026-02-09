import torch
from torch.utils.data import DataLoader

from model.perceptual_qualitynet import PerceptualQualityNet
from data_loader.perceptual_quality_dataset import PerceptualDereverberationDataset
from utils.metrics import mse_msa_corr
from utils.load_data import load_data
import argparse
from tqdm import tqdm
from seed import seed


def test_perceptual_net():
    config = {
        "data_split_file": "./data/metadata.jsonl",
        "batch_size": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "segment_length": 44100 * 4,
        "sample_rate": 44100,
    }

    data = load_data(config["data_split_file"])

    test_dataset = PerceptualDereverberationDataset(
        data.test_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model = PerceptualQualityNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location=config["device"]))
    model = model.to(config["device"])
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            reverb_audio = batch["reverb_audio"].to(config["device"])

            all_targets.append(
                {
                    "quality": batch["quality"].to(config["device"]).unsqueeze(1),
                    "odg": batch["odg"].to(config["device"]).unsqueeze(1),
                    "size": batch["size"].to(config["device"]).unsqueeze(1),
                    "wetness": batch["wetness"].to(config["device"]).unsqueeze(1),
                }
            )

            preds = model(reverb_audio, return_all=True)
            all_predictions.append(preds)

    for key in ["quality", "odg", "size", "wetness"]:
        preds = torch.cat(list(map(lambda dict: dict[key], all_predictions)), dim=0)
        targets = torch.cat(list(map(lambda dict: dict[key], all_targets)), dim=0)
        metrics = mse_msa_corr(preds, targets)
        print(f"{key} MSE: {metrics['mse']:.4f}")
        print(f"{key} MAE: {metrics['mae']:.4f}")
        print(f"{key} Correlation: {metrics['correlation']:.4f} \n")


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
    test_perceptual_net()
