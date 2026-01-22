# -*- encoding: utf-8 -*-
"""
@Filename    :Loss.py
@Time        :2020/07/09 22:11:13
@Author      :Kai Li
@Version     :1.0
"""

import torch
from itertools import permutations


class Loss(object):
    def sisnr(self, x, s, eps=1e-8):
        # x, s: (..., T)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        s = s - torch.mean(s, dim=-1, keepdim=True)

        # Project x onto s (target component)
        s_energy = torch.sum(s * s, dim=-1, keepdim=True)  # (..., 1)
        s_energy = torch.clamp(s_energy, min=eps)

        dot = torch.sum(x * s, dim=-1, keepdim=True)      # (..., 1)
        t = (dot / s_energy) * s                          # (..., T)

        # Compute SI-SNR = 10 * log10( ||t||^2 / ||x-t||^2 )
        t_pow = torch.sum(t * t, dim=-1)                  # (...)
        e_pow = torch.sum((x - t) * (x - t), dim=-1)       # (...)

        t_pow = torch.clamp(t_pow, min=eps)
        e_pow = torch.clamp(e_pow, min=eps)

        ratio = t_pow / e_pow
        ratio = torch.clamp(ratio, min=eps)

        return 10.0 * torch.log10(ratio)

    def compute_loss(self, est, ref):
        if isinstance(est, list):
            est = est[0]
        if isinstance(ref, list):
            ref = ref[0]

        def _summ(name: str, t: torch.Tensor) -> str:
            finite = torch.isfinite(t)
            n_bad = int((~finite).sum().item())
            shape = tuple(t.shape)
            with torch.no_grad():
                t_det = t.detach()

                if n_bad == t_det.numel():
                    return f"{name}: shape={shape}, bad={n_bad} (ALL), no finite values"

                fin_vals = t_det[finite]
                return (
                    f"{name}: shape={shape}, bad={n_bad}, "
                    f"min={fin_vals.min().item():.4g}, max={fin_vals.max().item():.4g}, "
                    f"mean={fin_vals.mean().item():.4g}"
                )

        est_ok = torch.isfinite(est).all()
        ref_ok = torch.isfinite(ref).all()
        if not est_ok or not ref_ok:
            raise ValueError(
                "Non-finite values detected before loss computation.\n"
                + _summ("est", est) + "\n"
                + _summ("ref", ref)
            )

        loss = -torch.mean(self.sisnr(est, ref))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss produced by SI-SNR.")
        return loss



if __name__ == "__main__":
    ests = torch.randn(4, 320)
    egs = torch.randn(4, 320)
    loss = Loss()
    print(loss.compute_loss(ests, egs))