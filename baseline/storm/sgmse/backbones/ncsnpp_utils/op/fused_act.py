import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

# NOTE: This module originates from StyleGAN2-style fused CUDA ops.
# In CPU-only environments (or when CUDA toolkit isn't installed), importing
# torch.utils.cpp_extension.load() will try to compile CUDA code and may crash
# with: "CUDA_HOME environment variable is not set".
#
# For this repo we want inference to work on CPU too, so we compile/load the
# extension only when CUDA is actually available and otherwise fall back to a
# pure PyTorch implementation.

fused = None


def _try_load_fused():
    global fused
    if fused is not None:
        return fused

    # Only attempt to compile the CUDA extension if CUDA is available.
    # This avoids hard-requiring a CUDA toolkit for CPU-only runs.
    if not torch.cuda.is_available():
        fused = False
        return fused

    try:
        from torch.utils.cpp_extension import load

        module_path = os.path.dirname(__file__)
        fused = load(
            "fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
    except Exception:
        # If compilation fails (missing nvcc, wrong toolchain, etc.), keep going
        # with the fallback implementation.
        fused = False

    return fused


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        fused_mod = _try_load_fused()
        if fused_mod is False:
            raise RuntimeError("Fused CUDA extension not available")

        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused_mod.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        fused_mod = _try_load_fused()
        if fused_mod is False:
            raise RuntimeError("Fused CUDA extension not available")

        gradgrad_out = fused_mod.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        fused_mod = _try_load_fused()
        if fused_mod is False:
            raise RuntimeError("Fused CUDA extension not available")

        empty = input.new_empty(0)
        out = fused_mod.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    # Always use the safe implementation on CPU.
    if input.device.type == "cpu":
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    # On CUDA, try the fused op, but fall back if compilation/loading failed.
    fused_mod = _try_load_fused()
    if fused_mod is False:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim),
                negative_slope=negative_slope,
            )
            * scale
        )

    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
