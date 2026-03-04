import torch
from torch.utils.data import DataLoader

from model.dereverberation_simple import DereverberationModel
from data_loader.dereverberation_dataset import DereverberationDataset
from utils.metrics import mse_mae_corr
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
import yaml


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as first_error:
        for prefix in ("model.", "module."):
            if any(str(key).startswith(prefix) for key in state_dict.keys()):
                remapped = {
                    str(key)[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if str(key).startswith(prefix)
                }
                if not remapped:
                    continue
                try:
                    model.load_state_dict(remapped)
                    return
                except RuntimeError:
                    pass

        raise RuntimeError(
            "Could not load checkpoint into DereverberationModel. Tried original keys and"
            " remapping common prefixes: model., module."
        ) from first_error


def _load_config(config_key: str) -> dict:
    with open("config.yaml", "r") as f:
        all_configs = yaml.safe_load(f)

    if config_key not in all_configs:
        available = ", ".join(sorted(all_configs.keys()))
        raise KeyError(
            f"Config key '{config_key}' not found in config.yaml. Available keys: {available}"
        )

    cfg = all_configs[config_key]
    use_cuda = torch.cuda.is_available()
    model_cfg = dict(cfg.get("model", {}))
    model_cfg = {k: v for k, v in model_cfg.items() if v is not None}

    if "gradient_checkpointing" in cfg:
        model_cfg.setdefault("gradient_checkpointing", cfg["gradient_checkpointing"])

    return {
        "data_split_file": Path(cfg.get("data_split_file", "./data/metadata.jsonl")),
        "batch_size": cfg.get("batch_size", 8),
        "device": "cuda" if use_cuda else "cpu",
        "segment_length": cfg.get("segment_length", 44100 * 4),
        "sample_rate": cfg.get("sample_rate", 44100),
        "model": model_cfg,
    }


def test_dereverb_net():
    config = _load_config(args.config_key)

    data = load_data(config["data_split_file"])

    test_dataset = DereverberationDataset(
        data.test_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model_kwargs = dict(config["model"])
    model_kwargs.setdefault("sr", config["sample_rate"])
    model = DereverberationModel(**model_kwargs)

    checkpoint_obj = torch.load(args.checkpoint, map_location=config["device"])
    state_dict = _extract_state_dict(checkpoint_obj)
    _load_model_state_dict(model, state_dict)
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

                metrics = mse_mae_corr(preds, targets)
                si_snr_value = si_snr(preds, targets)
                if metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]
                    for i, audio in enumerate(preds):
                        sf.write(
                            f"output/{Path(args.checkpoint).stem}_{i}_pred.wav",
                            audio.cpu().numpy(),
                            config["sample_rate"],
                        )
                    for i, audio in enumerate(targets):
                        sf.write(
                            f"output/{Path(args.checkpoint).stem}_{i}_target.wav",
                            audio.squeeze().cpu().numpy(),
                            config["sample_rate"],
                        )
                    for i, audio in enumerate(reverb_audio):
                        sf.write(
                            f"output/{Path(args.checkpoint).stem}_{i}_input.wav",
                            audio.cpu().numpy(),
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
        "config_key",
        help="Key in config.yaml used to configure model and data settings",
    )
    parser.add_argument(
        "checkpoint",
        default="checkpoints/perceptual_net_best.pth",
    )
    args = parser.parse_args()

    seed(42)
    test_dereverb_net()
