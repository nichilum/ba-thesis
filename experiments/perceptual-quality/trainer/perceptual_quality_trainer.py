import torch
import torch.nn as nn
from tqdm import tqdm


class PerceptualNetTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        patience=5,
        delta=0,
        es=True,
        device="cuda",
        save_path="checkpoints/perceptual_net.pth",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.best_val_loss = float("inf")

        self.patience = patience
        self.delta = delta

        self.es = es

        self.plots = {
            "test_loss_full": [],
            "test_loss_quality": [],
            "test_loss_size": [],
            "test_loss_wetness": [],
            "test_loss_odg": [],
            "val_loss": [],
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        losses = {"quality": 0, "odg": 0, "size": 0, "wetness": 0}

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            reverb_audio = batch["reverb_audio"].to(self.device)

            odg_target = batch["odg"].to(self.device).unsqueeze(1)
            size_target = batch["size"].to(self.device).unsqueeze(1)
            wetness_target = batch["wetness"].to(self.device).unsqueeze(1)
            quality_target = batch["quality"].to(self.device).unsqueeze(1)

            preds = self.model(reverb_audio, return_all=True)

            loss_odg = self.loss(preds["odg"], odg_target)
            loss_size = self.loss(preds["size"], size_target)
            loss_wetness = self.loss(preds["wetness"], wetness_target)
            loss_quality = self.loss(preds["quality"], quality_target)

            loss = self._loss_mat(loss_quality, loss_odg, loss_size, loss_wetness)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            losses["quality"] += loss_quality.item()
            losses["odg"] += loss_odg.item()
            losses["size"] += loss_size.item()
            losses["wetness"] += loss_wetness.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in losses.items()}, total_loss / n_batches

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                reverb_audio = batch["reverb_audio"].to(self.device)

                quality_target = batch["quality"].to(self.device).unsqueeze(1)
                odg_target = batch["odg"].to(self.device).unsqueeze(1)
                size_target = batch["size"].to(self.device).unsqueeze(1)
                wetness_target = batch["wetness"].to(self.device).unsqueeze(1)

                preds = self.model(reverb_audio, return_all=True)

                loss_quality = self.loss(preds["quality"], quality_target)
                loss_odg = self.loss(preds["odg"], odg_target)
                loss_size = self.loss(preds["size"], size_target)
                loss_wetness = self.loss(preds["wetness"], wetness_target)

                loss = self._loss_mat(loss_quality, loss_odg, loss_size, loss_wetness)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):
        early_stopping = EarlyStopping(
            patience=self.patience, delta=self.delta, verbose=True
        )

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_losses, avg_train_loss = self.train_epoch()
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(
                f"Quality: {train_losses['quality']:.4f}, "
                f"ODG: {train_losses['odg']:.4f}, "
                f"Size: {train_losses['size']:.4f}, "
                f"Wetness: {train_losses['wetness']:.4f}"
            )

            self.plots["test_loss_full"].append(avg_train_loss)
            self.plots["test_loss_quality"].append(train_losses["quality"])
            self.plots["test_loss_size"].append(train_losses["size"])
            self.plots["test_loss_odg"].append(train_losses["odg"])
            self.plots["test_loss_wetness"].append(train_losses["wetness"])

            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
                self.plots["val_loss"].append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Saved best model to {self.save_path}")
            else:
                torch.save(self.model.state_dict(), self.save_path)

            early_stopping.check_early_stop(val_loss)

            if early_stopping.stop_training and self.es:
                print(f"Early stopping at epoch {epoch}")
                break

    @staticmethod
    def _loss_mat(loss_quality, loss_odg, loss_size, loss_wetness):
        return (
            2.0 * loss_quality + 1.0 * loss_odg + 0.75 * loss_size + 0.75 * loss_wetness
        )


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
