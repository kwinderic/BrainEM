"""
cad.py -- CAD (Co-detection of Affinities and Densities)

Wraps the 2D two-stream U-Net (CoDetectionCNN) and the 3D U-Net
(UNet_PNI_embedding) into a single nn.Module for the pytorch_connectomics
framework.

Training mode:
  1. 3D forward -> 12-channel multi-scale affinity loss (embedding_loss_norm5)
  2. Loop over z slice pairs:
     - 2D forward -> 2D slice loss
     - Cross-attention loss: 2D pred vs 3D pred
     - Interaction loss: 2D emb dot 3D emb (z direction only)
  3. Two-stage gating: for iter < start_ft only loss_3d + loss_2d_slice is used

Inference mode:
  2D model only, processed slice pair by slice pair, producing a
  [B, 3, D, H, W] affinity map

Source: CAD/scripts_2_5d_3d/main_CAD.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import *
from ..config import CfgMixin
from .model_2d import CoDetectionCNN
from .model_3d import UNet_PNI_embedding


# =============================================================================
# Loss functions
# =============================================================================

class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term


def _single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    """Affinity and loss for one (direction, offset) pair (same as PEA)."""
    B, C, D, H, W = embedding.shape
    ax = order % 3
    if ax == 0:
        affs = torch.sum(embedding[:, :, shift:, :, :] *
                         embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, shift:, :, :],
                               weightmap[:, order:order+1, shift:, :, :])
    elif ax == 1:
        affs = torch.sum(embedding[:, :, :, shift:, :] *
                         embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :, shift:, :],
                               weightmap[:, order:order+1, :, shift:, :])
    else:
        affs = torch.sum(embedding[:, :, :, :, shift:] *
                         embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :, :, shift:],
                               weightmap[:, order:order+1, :, :, shift:])
    return loss, affs


def embedding_loss_norm5(embedding, target, weightmap, criterion,
                         affs0_weight=1, shift=1, fill=True):
    """12-channel multi-scale affinity loss (same as PEA's SCM)."""
    embedding = F.normalize(embedding, p=2, dim=1)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]
    affs = torch.zeros_like(target)
    loss = 0
    for i, sh in enumerate(shifts):
        loss_i, affs_i = _single_offset_loss(embedding, i, sh, target, weightmap, criterion)
        loss += loss_i * affs0_weight if i < 3 else loss_i
        ax = i % 3
        if ax == 0:
            affs[:, i:i+1, sh:, :, :] = affs_i.clone().detach()
        elif ax == 1:
            affs[:, i:i+1, :, sh:, :] = affs_i.clone().detach()
        else:
            affs[:, i:i+1, :, :, sh:] = affs_i.clone().detach()
    return loss, affs


def embedding_loss_norm_trunc(embedding1, embedding2, target, weightmap,
                               criterion, affs0_weight=1, shift=1):
    """2D embedding -> 3-channel affinity (trunc mode: L2 normalize, dot product, clamp to [0,1])."""
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    H, W = embedding2.shape[2], embedding2.shape[3]

    affs0 = torch.sum(embedding1 * embedding2, dim=1, keepdim=True)
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])

    affs1 = torch.sum(embedding2[:, :, shift:, :] * embedding2[:, :, :H-shift, :],
                       dim=1, keepdim=True)
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
    affs1 = F.pad(affs1, (0, 0, 1, 0), mode='reflect')

    affs2 = torch.sum(embedding2[:, :, :, shift:] * embedding2[:, :, :, :W-shift],
                       dim=1, keepdim=True)
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
    affs2 = F.pad(affs2, (1, 0, 0, 0), mode='reflect')

    loss = affs0_weight * loss0 + loss1 + loss2
    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return loss, affs


# =============================================================================
# CAD main model
# =============================================================================

@register_model("cad")
class CAD(CfgMixin, nn.Module):
    """Co-detection of Affinities and Densities

    Args (passed from cfg via CfgMixin.parse_config):
        filters:             per-level channel counts of the 3D U-Net
        emd:                 embedding dimension
        filter_channel_2d:   base filter count of the 2D U-Net
        start_ft:            iteration at which joint fine-tuning starts
        ft_lr_ratio:         learning-rate scale for the 3D model
        affs0_weight_3d:     weight of the 3D short-range affinity
        affs0_weight_2d:     weight of the 2D short-range affinity
        w_3d/w_2d_slice/w_cross/w_interact: weights of the individual loss terms
    """

    @staticmethod
    def parse_config(cfg, kwargs):
        kwargs['emd'] = cfg.MODEL.CAD_EMBEDDING_DIM
        kwargs['filter_channel_2d'] = cfg.MODEL.CAD_FILTER_CHANNEL_2D
        kwargs['start_ft'] = cfg.MODEL.CAD_START_FT
        kwargs['ft_lr_ratio'] = cfg.MODEL.CAD_FT_LR_RATIO
        kwargs['affs0_weight_3d'] = cfg.MODEL.CAD_AFFS0_WEIGHT_3D
        kwargs['affs0_weight_2d'] = cfg.MODEL.CAD_AFFS0_WEIGHT_2D
        kwargs['w_3d'] = cfg.MODEL.CAD_LOSS_WEIGHT_3D
        kwargs['w_2d_slice'] = cfg.MODEL.CAD_LOSS_WEIGHT_2D_SLICE
        kwargs['w_cross'] = cfg.MODEL.CAD_LOSS_WEIGHT_CROSS
        kwargs['w_interact'] = cfg.MODEL.CAD_LOSS_WEIGHT_INTERACT
        return kwargs

    def __init__(self,
                 in_channel=1,
                 out_channel=3,
                 filters=(28, 36, 48, 64, 80),
                 emd=16,
                 filter_channel_2d=16,
                 start_ft=10000,
                 ft_lr_ratio=1.0,
                 affs0_weight_3d=10.0,
                 affs0_weight_2d=1.0,
                 w_3d=1.0,
                 w_2d_slice=1.0,
                 w_cross=1.0,
                 w_interact=1.0,
                 **kwargs):
        super().__init__()
        self.model_2d = CoDetectionCNN(
            n_channels=1, n_classes=emd, filter_channel=filter_channel_2d)
        self.model_3d = UNet_PNI_embedding(
            in_planes=in_channel, filters=filters, emd=emd)

        self.start_ft = start_ft
        self.ft_lr_ratio = ft_lr_ratio
        self.affs0_weight_3d = affs0_weight_3d
        self.affs0_weight_2d = affs0_weight_2d
        self.w_3d = w_3d
        self.w_2d_slice = w_2d_slice
        self.w_cross = w_cross
        self.w_interact = w_interact

        self.register_buffer('_iteration', torch.tensor(0, dtype=torch.long))

    @property
    def iteration(self):
        return self._iteration.item()

    @iteration.setter
    def iteration(self, val):
        self._iteration.fill_(val)

    def get_param_groups(self):
        """Separate 2D / 3D parameter groups, consumed by _build_optimizer."""
        return [
            {'params': list(self.model_2d.parameters()), 'lr_ratio': 1.0},
            {'params': list(self.model_3d.parameters()), 'lr_ratio': self.ft_lr_ratio},
        ]

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    @staticmethod
    def _embedding_to_affinity_3ch(embeddings):
        """Convert a [B, E, D, H, W] embedding into [B, 3, D, H, W] short-range affinities."""
        embeddings = F.normalize(embeddings, p=2, dim=1)
        B, C, D, H, W = embeddings.shape
        affs = torch.zeros(B, 3, D, H, W, device=embeddings.device)
        affs[:, 0:1, 1:, :, :] = torch.sum(
            embeddings[:, :, 1:, :, :] * embeddings[:, :, :D-1, :, :],
            dim=1, keepdim=True)
        affs[:, 1:2, :, 1:, :] = torch.sum(
            embeddings[:, :, :, 1:, :] * embeddings[:, :, :, :H-1, :],
            dim=1, keepdim=True)
        affs[:, 2:3, :, :, 1:] = torch.sum(
            embeddings[:, :, :, :, 1:] * embeddings[:, :, :, :, :W-1],
            dim=1, keepdim=True)
        # boundary padding + ReLU
        affs[:, 0:1, :1, :, :] = affs[:, 0:1, 1:2, :, :]
        affs[:, 1:2, :, :1, :] = affs[:, 1:2, :, 1:2, :]
        affs[:, 2:3, :, :, :1] = affs[:, 2:3, :, :, 1:2]
        affs = F.relu(affs)
        return affs

    def _forward_inference(self, volume):
        """Inference: run the 2D model over slice pairs and assemble a 3D affinity map.

        Args:
            volume: [B, 1, D, H, W]
        Returns:
            affinities: [B, 3, D, H, W]
        """
        B, _, D, H, W = volume.shape
        if D < 2:
            return torch.zeros(B, 3, D, H, W, device=volume.device)

        all_embeddings = []
        for z in range(D - 1):
            pair = volume[:, 0, z:z+2, :, :]   # [B, 2, H, W]
            emb1, emb2 = self.model_2d(pair)
            if z == 0:
                all_embeddings.append(emb1)
            all_embeddings.append(emb2)

        embeddings = torch.stack(all_embeddings, dim=2)  # [B, E, D, H, W]
        return self._embedding_to_affinity_3ch(embeddings)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def _forward_train(self, volume, target, weight):
        """Full CAD training forward pass

        Args:
            volume: [B, 1, D, H, W]
            target: List[4 x Tensor[B,3,D,H,W]]  multi-scale affinity labels
            weight: List[4 x List[1 x Tensor[B,3,D,H,W]]]  weights
        Returns:
            (pred_3d, total_loss, losses_vis)
        """
        device = volume.device
        B, _, D, H, W = volume.shape
        _criterion = WeightedMSE()

        # ── 3D loss ──
        # concatenate the multi-scale target/weight
        aff_target = torch.cat([t.to(device) for t in target], dim=1)  # [B,12,D,H,W]
        aff_weight = torch.cat([w[0].to(device) for w in weight], dim=1)

        embedding_3d = self.model_3d(volume)
        loss_3d, pred_3d_12ch = embedding_loss_norm5(
            embedding_3d, aff_target, aff_weight, _criterion,
            affs0_weight=self.affs0_weight_3d)

        # boundary padding of the 3D pred (first 3 channels)
        shift = 1
        pred_3d = pred_3d_12ch[:, :3].clone()
        pred_3d[:, 1, :, :shift, :] = pred_3d[:, 1, :, shift:shift*2, :]
        pred_3d[:, 2, :, :, :shift] = pred_3d[:, 2, :, :, shift:shift*2]
        pred_3d[:, 0, :shift, :, :] = pred_3d[:, 0, shift:shift*2, :, :]
        pred_3d = F.relu(pred_3d)

        # scale-0 target/weight (3 short-range channels)
        target_s0 = target[0].to(device)      # [B, 3, D, H, W]
        weight_s0 = weight[0][0].to(device)   # [B, 3, D, H, W]

        # ── 2D slice loop ──
        loss_2d_slice = torch.tensor(0.0, device=device)
        loss_cross = torch.tensor(0.0, device=device)
        loss_interaction = torch.tensor(0.0, device=device)

        phase2 = (self.iteration >= self.start_ft)

        # whether the 3D embedding is detached
        embedding_3d_for_interact = embedding_3d if phase2 else embedding_3d.detach()

        for z in range(D - 1):
            pair_input = volume[:, 0, z:z+2, :, :]  # [B, 2, H, W]
            emb1_2d, emb2_2d = self.model_2d(pair_input)

            # 2D slice loss
            target_z = target_s0[:, :3, z+1]    # [B, 3, H, W]
            weight_z = weight_s0[:, :3, z+1]    # [B, 3, H, W]
            size_z = weight_z.shape[-1]

            loss_tmp, pred_tmp = embedding_loss_norm_trunc(
                emb1_2d, emb2_2d, target_z, weight_z, _criterion,
                affs0_weight=self.affs0_weight_2d, shift=1)
            loss_2d_slice = loss_2d_slice + loss_tmp

            # Cross-attention loss
            if phase2:
                loss_cross_tmp = _criterion(
                    pred_tmp[:, :, :, shift:],
                    pred_3d[:, :3, z+1, :, shift:],
                    weight_z[:, :, :, shift:])
            else:
                loss_cross_tmp = _criterion(
                    pred_tmp[:, :, shift:, :],
                    pred_3d[:, :3, z+1, shift:, :].detach(),
                    weight_z[:, :, shift:, :])
            loss_cross = loss_cross + loss_cross_tmp

            # Interaction loss (z direction only)
            shifts_12ch = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]
            loss_3d_tmp = torch.tensor(0.0, device=device)
            for order, sh in enumerate(shifts_12ch):
                if order % 3 != 0:          # z direction only
                    continue
                if z + sh > D - 2:          # bounds check
                    continue
                emb2_2d_norm = F.normalize(emb2_2d, p=2, dim=1)
                if sh == 1:
                    emb_3d_slice = F.normalize(
                        embedding_3d_for_interact[:, :, z+1+sh], p=2, dim=1)
                    affs0 = torch.sum(emb2_2d_norm * emb_3d_slice,
                                      dim=1, keepdim=True)
                    affs0 = torch.clamp(affs0, 0.0, 1.0)
                    loss_3d_tmp = _criterion(
                        affs0,
                        aff_target[:, order:order+1, z+2, :, :],
                        aff_weight[:, order:order+1, z+2, :, :])
                else:
                    emb1_2d_norm = F.normalize(emb1_2d, p=2, dim=1)
                    emb_3d_z = F.normalize(
                        embedding_3d_for_interact[:, :, z+sh], p=2, dim=1)
                    emb_3d_z1 = F.normalize(
                        embedding_3d_for_interact[:, :, z+1+sh], p=2, dim=1)
                    affs0_1 = torch.clamp(
                        torch.sum(emb1_2d_norm * emb_3d_z, dim=1, keepdim=True),
                        0.0, 1.0)
                    affs0_2 = torch.clamp(
                        torch.sum(emb2_2d_norm * emb_3d_z1, dim=1, keepdim=True),
                        0.0, 1.0)
                    loss_t1 = _criterion(
                        affs0_1,
                        aff_target[:, order:order+1, z+1, :, :],
                        aff_weight[:, order:order+1, z+1, :, :])
                    loss_t2 = _criterion(
                        affs0_2,
                        aff_target[:, order:order+1, z+2, :, :],
                        aff_weight[:, order:order+1, z+2, :, :])
                    loss_3d_tmp = loss_t1 + loss_t2
            loss_interaction = loss_interaction + loss_3d_tmp

        # ── loss combination ──
        loss_3d_w = self.w_3d * loss_3d
        loss_2d_w = self.w_2d_slice * loss_2d_slice
        loss_cross_w = self.w_cross * loss_cross
        loss_interact_w = self.w_interact * loss_interaction

        if phase2:
            total_loss = loss_3d_w + loss_2d_w + loss_cross_w + loss_interact_w
        else:
            total_loss = loss_3d_w + loss_2d_w

        self.iteration = self.iteration + 1

        losses_vis = {
            'loss_3d': loss_3d.item(),
            'loss_2d_slice': loss_2d_slice.item(),
            'loss_cross': loss_cross.item(),
            'loss_interaction': loss_interaction.item(),
            'phase': 2 if phase2 else 1,
        }

        return pred_3d, total_loss, losses_vis

    # -----------------------------------------------------------------
    # forward entry point
    # -----------------------------------------------------------------
    def forward(self, inputs, target=None, weight=None, criterion=None):
        """
        Inference (criterion=None): return pred [B, 3, D, H, W]
        Training (criterion!=None): return (pred, loss, losses_vis)
        """
        if criterion is None:
            return self._forward_inference(inputs)
        return self._forward_train(inputs, target, weight)
