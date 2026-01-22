# -*- encoding: utf-8 -*-
"""
@Filename    :Loss.py
@Time        :2020/07/09 22:11:13
@Author      :Kai Li
@Version     :1.0
"""

import torch
from itertools import permutations
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class Loss(object):
    def compute_loss(self, est, ref, device="cuda"):
        # ref = ref.unsqueeze(1)  # (B, 1, T)

        si_snr = ScaleInvariantSignalNoiseRatio().to(device)
        si_snr_val = si_snr(est, ref)

        #throw if nan
        if torch.isnan(si_snr_val):
            print("LOSS est:", est)
            print("LOSS ref:", ref)
            raise RuntimeError("SI-SNR loss is NaN.")

        return si_snr_val



if __name__ == "__main__":
    ests = torch.randn(4, 320)
    egs = torch.randn(4, 320)
    loss = Loss()
    print(loss.compute_loss(ests, egs))