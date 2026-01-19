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
        x = x - torch.mean(x, dim=-1, keepdim=True)
        s = s - torch.mean(s, dim=-1, keepdim=True)

        t = (
            torch.sum(x * s, dim=-1, keepdim=True)
            * s
            / (torch.sum(s**2, dim=-1, keepdim=True) + eps)
        )

        return 20 * torch.log10(
            torch.norm(t, dim=-1) / (torch.norm(x - t, dim=-1) + eps)
        )

    def compute_loss(self, est, ref):
        if isinstance(est, list):
            est = est[0]
        if isinstance(ref, list):
            ref = ref[0]

        return -torch.mean(self.sisnr(est, ref))


if __name__ == "__main__":
    ests = torch.randn(4, 320)
    egs = torch.randn(4, 320)
    loss = Loss()
    print(loss.compute_loss(ests, egs))
