import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributed as dist
# from typing import override
from typing_extensions import override

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from connectomics.model.utils import model_init
from connectomics.model.arch import UNet3D
from connectomics.model.block import conv3d_norm_act
from connectomics.model.block import *
from connectomics.model.utils.misc import get_norm_3d, get_norm_1d

from interface import E2EMixin, AutoLossMixin


# Use affinity prediction as queries
# replace SparseInst head with connected components labeling 


from skimage.measure import label

import mahotas
from scipy import ndimage
from .ds_conv import DCN_Conv

import matplotlib
import matplotlib.pyplot as plt

import cv2
from skimage.measure import label as skilabel

from .affinity_knet import AffinityKNet


def get_same_conv(Ci, Co, kernel_size=(3, 3, 3), dilation=(1, 1, 1), **kw):
    padding = (
        dilation[0] * (kernel_size[0]-1)//2, 
        dilation[1] * (kernel_size[1]-1)//2,
        dilation[2] * (kernel_size[2]-1)//2
    )
    return conv3d_norm_act(Ci, Co, kernel_size=kernel_size, 
                    dilation=dilation, padding=padding, **kw)


class RefineHead(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 8
        share_kws = dict(norm_mode="none", act_mode="relu")
        self.convs1 = nn.ModuleList([
            get_same_conv(4, channels, kernel_size=(1, 1, 1), 
                            dilation=(1, 1, 1), **share_kws),
            get_same_conv(4, channels, kernel_size=(3, 3, 3), 
                            dilation=(1, 1, 1), **share_kws),
            get_same_conv(4, channels, kernel_size=(3, 3, 3), 
                            dilation=(4, 4, 2), **share_kws),
            get_same_conv(4, channels, kernel_size=(3, 3, 3), 
                            dilation=(8, 8, 4), **share_kws)
        ])
        self.convs2 = get_same_conv(channels, 1, kernel_size=(3, 3, 3), 
                dilation=(1, 1, 1), norm_mode="none", act_mode="none")


    def forward(self, logits, affinities):
        # logits: B, N, D, H, W
        # affinities: B, 3, D, H, W
        B, N, D, H, W = logits.shape

        # inds = logits.sum([2,3,4]).argsort(descending=True)
        # inds = inds[:, :100]

        x = logits.view(B*N, 1, D, H, W)
        y = affinities.repeat(N, 1, 1, 1, 1)    # B*N, 3, D, H, W

        x = torch.cat([x, y], dim=1)        # B*N, 4, D, H, W
        z = 0
        for conv in self.convs1:
            z = z + conv(x)                 # B*N, C, D, H, W
        z = self.convs2(z)
        z = z.reshape(B, N, D, H, W)

        return logits + z



class AffinityKNetRefine(AffinityKNet):
    """
    Refine the predictions with affinities
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.refine_head = RefineHead()

        self.eval()
        self.refine_head.train()

        for p in self.parameters():
            p.requires_grad = False
        for p in self.refine_head.parameters():
            p.requires_grad = True


    def forward(self, x, label=None):

        with torch.no_grad():
            # backbone
            feats = self._forward_backbone(x)
            x = self.fusion_block(feats)

            # feature extraction
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            # affinity prediction
            if self.with_affinity:
                pred_affinity = self.affinity_branch(x)
                # init_masks = get_connected_components(pred_affinity)        # N, D, H, W
            else:
                pred_affinity = None

            # AGFP
            if self.with_agfp:
                x = self.agfp(x)

            # instance segmentation
            mask_features = self.mask_branch(x)     # B, C, D, H, W

            if self.affinity_for_mask:
                pred_affinity_norm = pred_affinity.detach().sigmoid()
                pred_affinity_norm = pred_affinity_norm - 0.5
                mask_features = torch.cat([mask_features, pred_affinity_norm], dim=1)
                mask_features = self.affinity_proj(mask_features)

            pred_masks, pred_kernels, recorded_init_masks = self.inst_decoder(x, mask_features, pred_affinity)

        # TODO: call refine head here
        pred_masks[-1] = self.refine_head(pred_masks[-1], pred_affinity)
        pass

        if self.inference_without_bg:
            pred_masks[-1][0,0].zero_()
            print('zero bg mask for inference')

        output = {
            "pred_masks": pred_masks[-1],
            "pred_kernel": pred_kernels[-1],
            "pixel_feature": mask_features,
            "aux_outputs": [],
            "recorded_init_masks": recorded_init_masks      # None if not set
        }

        if self.with_affinity:
            output['pred_affinity'] = pred_affinity

        output['aux_groups'] = []
        # pred_masks[-1], shape B, N, D, H, W; required init_masks.shape B, D, H, W

        return output
