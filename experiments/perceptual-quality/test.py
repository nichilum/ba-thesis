import torch
from torch.utils.data import DataLoader
import os
import glob

from model.perceptual_qualitynet import PerceptualQualityNet, DereverberationLoss
from data_loader.dereverb_dataset import DereverberationDataset
from utils.metrics import compute_metrics


def test_quality_net():
    """Test the trained quality network."""
    config = {
        "data_dir": "/path/to/test/audio/files",
        "model_path": "checkpoints/quality_net_best.pth",
        "batch_size": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "segment_length": 48000,
        "sample_rate": 16000,
    }

    # Load test files
    test_files = glob.glob(os.path.join(config["data_dir"], "*.wav"))
    print(f"Testing on {len(test_files)} files")

    # Create dataset
    test_dataset = DereverberationDataset(
        test_files,
        segment_length=config["segment_length"],
        sample_rate=config["sample_rate"],
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Load model
    model = PerceptualQualityNet()
    model.load_state_dict(
        torch.load(config["model_path"], map_location=config["device"])
    )
    model = model.to(config["device"])
    model.eval()

    # Evaluate
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            reverb_audio = batch["reverb_audio"].to(config["device"])
            quality_target = batch["quality_score"].to(config["device"]).unsqueeze(1)

            preds = model(reverb_audio, return_all=True)

            all_predictions.append(preds["quality"])
            all_targets.append(quality_target)

    # Compute metrics
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(predictions, targets)

    print("\nTest Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")


def test_as_loss_function():
    """Demonstrate using the quality net as a loss function."""
    config = {
        "model_path": "checkpoints/quality_net_best.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Create loss function
    dereverb_loss = DereverberationLoss(config["model_path"], device=config["device"])

    # Example usage in training loop
    # Assuming you have:
    # - reverb_input: reverberated audio
    # - dereverb_output: output from your dereverberation model
    # - clean_target: clean reference audio

    # dummy_output = torch.randn(4, 48000).to(config['device'])
    # dummy_target = torch.randn(4, 48000).to(config['device'])

    # loss = dereverb_loss(dummy_output, dummy_target, alpha=0.1)
    # print(f"Loss: {loss.item():.4f}")

    print("Loss function initialized successfully")
    print("Use in training loop as:")
    print("  loss = dereverb_loss(dereverb_output, clean_target)")


if __name__ == "__main__":
    print("Testing Quality Network...")
    test_quality_net()

    print("\n" + "=" * 50)
    print("Testing as Loss Function...")
    test_as_loss_function()
