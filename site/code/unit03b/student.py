"""YOLOX Unit03-B student starter. Implement the TODOs without copying reference code.

Install PyTorch before running the tests; see START_HERE.md.
Use torch tensors in NCHW convention.
"""
import torch
from torch import nn


def focus_space_to_depth(x: torch.Tensor) -> torch.Tensor:
    """Input [N,C,H,W] where H and W are even.

    Return the channels from 2x2 offset positions in the official order:
    top-left, bottom-left, top-right, bottom-right.
    Output: [N,4*C,H/2,W/2]. No trainable weights.
    """
    raise NotImplementedError("Implement Focus channel rearrangement")


def conv_weight_count(cin: int, cout: int, kernel: int, depthwise: bool) -> int:
    """Return only the weight count of a standard or depthwise-separable convolution.

    Standard: spatial KxK across all input channels per output channel.
    Depthwise: spatial KxK per input channel, then 1x1 mixing.
    Do NOT count BatchNorm, bias, or activation in this function.
    """
    raise NotImplementedError("Compute convolution weights")


class DepthwiseSeparableConv(nn.Module):
    """Conv+BatchNorm+SiLU in the official Nano order.

    Requirements:
      depthwise = nn.Conv2d(cin,cin,kernel,stride,padding=(kernel-1)//2,
                            groups=cin,bias=False) followed by BN and SiLU.
      pointwise = nn.Conv2d(cin,cout,1,1,0,bias=False) followed by BN and SiLU.
      input/output: NCHW.
    """
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        raise NotImplementedError("Construct depthwise and pointwise modules")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Apply depthwise and pointwise in order")


def csp_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Both tensors are NCHW. Concatenate along channel axis (not H or W)."""
    raise NotImplementedError("Concatenate CSP branches")
