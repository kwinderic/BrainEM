"""Register SegNeuron's MNet into the repo model registry with a dual-head
loss wrapper, mirroring the SAM2AFF CustomTrainer convention.

MNet outputs (affinity[3ch], foreground_mask[1ch]), both sigmoid-activated.
The framework dataloader is configured with MODEL.TARGET_OPT = ['2', '0'] so
`sample.out_target_l` = [affinity_target(3ch), fg_mask_target(1ch)] and
`sample.out_weight_l` = [aff_weight, mask_weight]. We compute a weighted MSE per
head (SegNeuron uses weighted MSE for both) and sum them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from connectomics.model.build import MODEL_MAP

from .model.mnet import MNet


def register_model(name):
    def _reg(cls):
        MODEL_MAP[name] = cls
        return cls
    return _reg


def _weighted_mse(pred, target, weight=None):
    """SegNeuron-style weighted MSE (normalized by voxel count * batch)."""
    if weight is None:
        return torch.mean((pred - target) ** 2)
    s = float(torch.prod(torch.tensor(pred.shape[2:]))) * pred.shape[0]
    return torch.sum(weight * (pred - target) ** 2) / s


@register_model("segneuron")
class SegNeuronModel(nn.Module):
    """MNet with a built-in dual-head loss (affinity + foreground mask).

    Compatible with the SAM2AFF-style CustomTrainer path:
        pred, loss, losses_vis = model(volume, target, weight, criterion=...)
    In test mode (criterion is None) it returns the fused affinity so the
    existing inference / postprocess path can consume it directly.
    """

    def __init__(self,
                 in_channel: int = 1,
                 filters=(32, 64, 96, 128, 256),
                 fmu: str = "sub",
                 fuse_mask_in_infer: bool = True,
                 **kwargs):
        super().__init__()
        self.mnet = MNet(in_channel, kn=tuple(filters), FMU=fmu)
        self.fuse_mask_in_infer = fuse_mask_in_infer

    def forward(self, x, target=None, weight=None, criterion=None):
        aff, mask = self.mnet(x)   # [B,3,...], [B,1,...], both in [0,1]

        if criterion is None and target is None:
            # Inference: fuse heads (suppress affinity where mask=background).
            if self.fuse_mask_in_infer:
                return torch.minimum(aff, mask)
            return aff

        # Training: dual-head weighted MSE.
        # target/weight are lists aligned with TARGET_OPT = ['2', '0'].
        aff_t, mask_t = target[0], target[1]
        aff_w = weight[0][0] if isinstance(weight[0], (list, tuple)) else weight[0]
        mask_w = weight[1][0] if isinstance(weight[1], (list, tuple)) else weight[1]
        aff_t = aff_t.to(aff.device)
        mask_t = mask_t.to(mask.device)
        aff_w = aff_w.to(aff.device) if aff_w is not None else None
        mask_w = mask_w.to(mask.device) if mask_w is not None else None

        loss_aff = _weighted_mse(aff, aff_t, aff_w)
        loss_mask = _weighted_mse(mask, mask_t, mask_w)
        loss = loss_aff + loss_mask
        losses_vis = {"loss_aff": loss_aff.detach(), "loss_mask": loss_mask.detach()}
        # Return the two heads concatenated rather than as a tuple: the
        # framework's training-time visualizer expects a single Tensor and
        # splits it back per TARGET_OPT via SplitActivation (split rule [3, 1]).
        # OUTPUT_ACT is 'none' for both heads, so no activation is re-applied
        # on top of MNet's sigmoids.
        return torch.cat([aff, mask], dim=1), loss, losses_vis
