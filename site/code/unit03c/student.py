"""YOLOX Visual Academy / Unit 03-C: STUDENT STARTER (fill each TODO).

Tensor format: PyTorch NCHW. This exercise is about the exact connectivity
and output order, not yet implementing the complete YOLOX trainer.
Use your Unit03-B Conv/Depthwise implementation to construct conv_factory.
"""
import torch
from torch import nn


def upsample_nearest2x(x: torch.Tensor) -> torch.Tensor:
    """Input [N,C,H,W], return [N,C,2H,2W], nearest, no learnable parameters."""
    raise NotImplementedError('Implement nearest-neighbor 2x upsample')


def concat_channels(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Input same N/H/W; return [N, Ca+Cb, H, W], with all a channels first."""
    raise NotImplementedError('Concatenate along the channel axis, NOT add')


def pafpn_top_down(x2, x1, x0, lateral, c3_p4, reduce, c3_p3):
    """Return (p3, fpn1, fpn0), preserving the official YOLOPAFPN ordering.

    x2=dark3 [N, 64,52,52], x1=dark4 [N,128,26,26],
    x0=dark5 [N,256,13,13]. Use the supplied modules for 1x1 and CSP.
    """
    raise NotImplementedError('Wire lateral, upsample, concat, C3_p4, reduce, upsample, concat, C3_p3')


def pafpn_bottom_up(p3, fpn1, fpn0, bu2, c3_n3, bu1, c3_n4):
    """Return (p3, p4, p5) in stride order 8,16,32.

    bu2: 64->64 stride2, c3_n3: 128->128,
    bu1: 128->128 stride2, c3_n4: 256->256.
    """
    raise NotImplementedError('Wire bottom-up stride2, concat, C3_n3, stride2, concat, C3_n4')


class DecoupledRawHead(nn.Module):
    """Head wiring exercise: one stem and two independent branches per level.

    conv_factory(cin, cout) builds a Conv+BN+SiLU block with kernel3, stride1;
    for Nano use your Unit03-B Depthwise Separable Conv implementation.
    The stem is a 1x1 Conv+BN+SiLU. Predictor Conv2d layers use kernel 1.
    Return RAW [reg4, obj1, clsN] logits (NO sigmoid, NO box decode).
    """
    def __init__(self, num_classes=80, in_channels=(64, 128, 256), hidden=64, conv_factory=None):
        super().__init__()
        if conv_factory is None:
            raise ValueError('Supply conv_factory from your Unit 03-B implementation')
        # TODO: ModuleList of stems, cls_convs, reg_convs, cls_preds, reg_preds, obj_preds.
        # Two conv_factory blocks for cls, two separate blocks for reg, per level.
        raise NotImplementedError('Build separate head branches and 1x1 predictors')

    def forward(self, features):
        # TODO: For each level: stem, cls branch, reg branch, reg/obj/cls predictors.
        # Return 3 raw outputs in P3, P4, P5 order.
        raise NotImplementedError('Output order: [reg4, obj1, clsN] (raw logits)')
