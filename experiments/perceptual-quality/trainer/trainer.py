import torch
import torch.nn as nn
from tqdm import tqdm


class QualityNetTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        device="cuda",
        save_path="checkpoints/quality_net.pth",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

        self.best_val_loss = float("inf")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        losses = {"quality": 0, "odg": 0, "size": 0, "wetness": 0}

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            reverb_audio = batch["reverb_audio"].to(self.device)

            # Ground truth targets
            odg_target = batch["odg"].to(self.device).unsqueeze(1)
            size_target = batch["size"].to(self.device).unsqueeze(1)
            wetness_target = batch["wetness"].to(self.device).unsqueeze(1)
            quality_target = batch["quality_score"].to(self.device).unsqueeze(1)

            # Forward pass
            preds = self.model(reverb_audio, return_all=True)

            # Multi-task loss
            loss_odg = self.mse_loss(preds["odg"], odg_target)
            loss_size = self.mse_loss(preds["size"], size_target)
            loss_wetness = self.mse_loss(preds["wetness"], wetness_target)
            loss_quality = self.mse_loss(preds["quality"], quality_target)

            # Weighted combination
            loss = (
                2.0 * loss_quality  # Main task
                + 1.0 * loss_odg  # Auxiliary tasks
                + 0.5 * loss_size
                + 0.5 * loss_wetness
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            losses["quality"] += loss_quality.item()
            losses["odg"] += loss_odg.item()
            losses["size"] += loss_size.item()
            losses["wetness"] += loss_wetness.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average losses
        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in losses.items()}, total_loss / n_batches

    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                reverb_audio = batch["reverb_audio"].to(self.device)
                quality_target = batch["quality_score"].to(self.device).unsqueeze(1)

                preds = self.model(reverb_audio, return_all=True)
                loss = self.mse_loss(preds["quality"], quality_target)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_losses, avg_train_loss = self.train_epoch()
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(
                f"  Quality: {train_losses['quality']:.4f}, "
                f"ODG: {train_losses['odg']:.4f}, "
                f"Size: {train_losses['size']:.4f}, "
                f"Wetness: {train_losses['wetness']:.4f}"
            )

            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Saved best model to {self.save_path}")
            else:
                torch.save(self.model.state_dict(), self.save_path)
