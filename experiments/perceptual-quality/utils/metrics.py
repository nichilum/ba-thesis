import torch


def mse_msa_corr(predictions, targets):
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
