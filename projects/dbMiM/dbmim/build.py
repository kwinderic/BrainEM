"""Register dbMiM's anisotropic UNETR affinity net into the repo model registry.

Mirrors the SegNeuron convention: the wrapper computes its own loss inside
forward and returns ``(pred, loss, losses_vis)`` in train mode, so the standard
CustomTrainer / _train_misc path works unchanged.

The loss is the reference's ``loss_type: mse`` + MAWS branch
(dbMiM/train_finetune.py:356-361 and :98-109):

    prob = sigmoid(logits)
    main = (prob - target) ** 2
    loss = sum(main * W) / sum(W),  W = channel_weight * spatial_weight

where ``spatial_weight`` is the membrane-aware map derived from the raw image
(NOT from labels -- see membrane_spatial_weight in train_finetune.py:481-504),
which is why this cannot go through the framework's MODEL.WEIGHT_OPT.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from connectomics.model.build import MODEL_MAP

from .models import UNETREMAffinityNet, membrane_edge_map_3d


def register_model(name):
    def _reg(cls):
        MODEL_MAP[name] = cls
        return cls
    return _reg


def membrane_spatial_weight(image: torch.Tensor,
                            n_channels: int,
                            membrane_weight: float,
                            axis_weights,
                            clip: float,
                            normalize: bool = True):
    """MAWS weight map: 1 + membrane_weight * edge(image), mean-normalized.

    Port of train_finetune.py:481-504. Returns None when disabled so the loss
    falls back to a plain channel-weighted mean.
    """
    if image is None or membrane_weight <= 0.0:
        return None
    edge = membrane_edge_map_3d(image.detach().float(),
                               axis_weights=axis_weights, clip=float(clip))
    if edge.shape[1] != 1:
        edge = edge.mean(dim=1, keepdim=True)
    weight = 1.0 + float(membrane_weight) * edge
    weight = weight.expand(-1, int(n_channels), -1, -1, -1).float()
    if normalize:
        weight = weight / weight.mean().clamp_min(1e-6)
    return weight


@register_model("dbmim_unetr_aniso_em")
class DBMiMAffinityModel(nn.Module):
    """UNETREMAffinityNet + built-in MSE/MAWS affinity loss.

    Train mode returns ``(logits, loss, losses_vis)``; test mode returns raw
    logits so the framework's INFERENCE.OUTPUT_ACT ('sigmoid') activates them.
    """

    def __init__(self,
                 in_channel: int = 1,
                 out_channel: int = 3,
                 volume_size=(32, 160, 160),
                 patch_size=(4, 16, 16),
                 embed_dim: int = 192,
                 depth: int = 6,
                 num_heads: int = 6,
                 feature_size: int = 32,
                 dropout: float = 0.05,
                 em_refine_depth: int = 2,
                 channel_bias_init=(-0.2, 0.0, 0.0),
                 membrane_weight: float = 0.75,
                 membrane_axis_weights=(0.25, 1.0, 1.0),
                 membrane_clip: float = 4.0,
                 membrane_normalize: bool = True,
                 channel_weights=(1.35, 1.0, 1.0),
                 **kwargs):
        super().__init__()
        self.net = UNETREMAffinityNet(
            in_channels=in_channel,
            out_channels=out_channel,
            volume_size=tuple(volume_size),
            patch_size=tuple(patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            feature_size=feature_size,
            dropout=dropout,
            em_refine_depth=em_refine_depth,
            channel_bias_init=list(channel_bias_init) if channel_bias_init else None,
        )
        self.membrane_weight = float(membrane_weight)
        self.membrane_axis_weights = list(membrane_axis_weights)
        self.membrane_clip = float(membrane_clip)
        self.membrane_normalize = bool(membrane_normalize)
        self.register_buffer(
            "channel_weights",
            torch.tensor([float(w) for w in channel_weights]).view(1, -1, 1, 1, 1),
            persistent=False)

    def forward(self, x, target=None, weight=None, criterion=None):
        logits = self.net(x)

        if target is None:
            return logits   # inference: OUTPUT_ACT applies the sigmoid

        # TARGET_OPT ['2'] -> a single affinity target.
        aff_t = target[0] if isinstance(target, (list, tuple)) else target
        aff_t = aff_t.to(device=logits.device, dtype=logits.dtype)

        main = (torch.sigmoid(logits) - aff_t).pow(2)

        w = self.channel_weights.to(device=logits.device, dtype=main.dtype)
        w = w.expand_as(main)
        spatial = membrane_spatial_weight(
            x, main.shape[1], self.membrane_weight, self.membrane_axis_weights,
            self.membrane_clip, self.membrane_normalize)
        if spatial is not None:
            w = w * spatial.to(device=logits.device, dtype=main.dtype).expand_as(main)

        loss = (main * w).sum() / w.sum().clamp_min(1.0)
        losses_vis = {"loss_aff_mse": loss.detach()}
        return logits, loss, losses_vis
