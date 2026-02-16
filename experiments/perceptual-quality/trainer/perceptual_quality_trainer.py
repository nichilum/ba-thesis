import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class PerceptualNetTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr,
        patience,
        delta,
        es,
        device,
        save_path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.best_val_losses = {
            "full": float("inf"),
            "quality": float("inf"),
            "odg": float("inf"),
            "size": float("inf"),
            "wetness": float("inf"),
        }

        self.patience = patience
        self.delta = delta

        self.es = es

        self.plots = {
            "train_loss_full": [],
            "train_loss_quality": [],
            "train_loss_size": [],
            "train_loss_wetness": [],
            "train_loss_odg": [],
            "val_loss_full": [],
            "val_loss_quality": [],
            "val_loss_size": [],
            "val_loss_wetness": [],
            "val_loss_odg": [],
        }

        self.writer = SummaryWriter()

    def train_epoch(self):
        self.model.train()
        losses = {"full": 0, "quality": 0, "odg": 0, "size": 0, "wetness": 0}

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

            losses["full"] += loss.item()
            losses["quality"] += loss_quality.item()
            losses["odg"] += loss_odg.item()
            losses["size"] += loss_size.item()
            losses["wetness"] += loss_wetness.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in losses.items()}

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        losses = {"full": 0, "quality": 0, "odg": 0, "size": 0, "wetness": 0}

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

                losses["full"] += loss.item()
                losses["quality"] += loss_quality.item()
                losses["odg"] += loss_odg.item()
                losses["size"] += loss_size.item()
                losses["wetness"] += loss_wetness.item()

        n_batches = len(self.val_loader)
        return {k: v / n_batches for k, v in losses.items()}

    def train(self, epochs):
        early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_losses = self.train_epoch()
            print(
                f"Full: {train_losses['full']:.4f}, "
                f"Quality: {train_losses['quality']:.4f}, "
                f"ODG: {train_losses['odg']:.4f}, "
                f"Size: {train_losses['size']:.4f}, "
                f"Wetness: {train_losses['wetness']:.4f}"
            )

            self.plots["train_loss_full"].append(train_losses["full"])
            self.plots["train_loss_quality"].append(train_losses["quality"])
            self.plots["train_loss_size"].append(train_losses["size"])
            self.plots["train_loss_odg"].append(train_losses["odg"])
            self.plots["train_loss_wetness"].append(train_losses["wetness"])

            self.writer.add_scalars("Train Losses", train_losses, epoch)

            if self.val_loader is not None:
                val_losses = self.validate()
                print(
                    f"Full: {val_losses['full']:.4f}, "
                    f"Quality: {val_losses['quality']:.4f}, "
                    f"ODG: {val_losses['odg']:.4f}, "
                    f"Size: {val_losses['size']:.4f}, "
                    f"Wetness: {val_losses['wetness']:.4f}"
                )

                self.plots["val_loss_full"].append(val_losses["full"])
                self.plots["val_loss_quality"].append(val_losses["quality"])
                self.plots["val_loss_size"].append(val_losses["size"])
                self.plots["val_loss_odg"].append(val_losses["odg"])
                self.plots["val_loss_wetness"].append(val_losses["wetness"])

                self.writer.add_scalars("Val Losses", val_losses, epoch)

                for k, v in val_losses.items():
                    if v < self.best_val_losses[k]:
                        self.best_val_losses[k] = v
                        out_path = self.save_path(k, epoch + 1)
                        torch.save(self.model.state_dict(), out_path)
                        print(f"Saved best model to {out_path}")
            else:
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Saved latest model to {self.save_path}")

            early_stopping.check_early_stop(val_losses["full"])

            if early_stopping.stop_training and self.es:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()

    @staticmethod
    def _loss_mat(loss_quality, loss_odg, loss_size, loss_wetness):
        return (
            2.0 * loss_quality + 1.0 * loss_odg + 0.75 * loss_size + 0.75 * loss_wetness
        )


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
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
