import torch


def mse_mae_corr(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # Pearson correlation
    pred_mean = predictions.mean()
    target_mean = targets.mean()

    numerator = ((predictions - pred_mean) * (targets - target_mean)).sum()
    denominator = torch.sqrt(
        ((predictions - pred_mean) ** 2).sum() * ((targets - target_mean) ** 2).sum()
    )

    correlation = (numerator / (denominator + 1e-8)).item()

    return {"mse": mse, "mae": mae, "correlation": correlation}


def si_snr(y_hat: torch.Tensor, y: torch.Tensor):
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss."""
    y_hat = y_hat - y_hat.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    s_target = (
        (y_hat * y).sum(dim=-1, keepdim=True)
        / (y.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
        * y
    )
    e_noise = y_hat - s_target

    si_snr = 10 * torch.log10(
        (s_target.norm(dim=-1) ** 2 + 1e-8) / (e_noise.norm(dim=-1) ** 2 + 1e-8)
    )
    return -si_snr.mean()
