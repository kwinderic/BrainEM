from __future__ import print_function, division
from typing import Optional, List
from collections import OrderedDict

import os
import logging
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from connectomics.model.block import *
from connectomics.model.utils import model_init


from .utils import *
from .config import CfgMixin
# from sam2.sam2.sam2_video_predictor import SAM2VideoPredictor
# from sam2.sam2.build_sam import build_sam2_video_predictor


# from .utils import return_loss, register_model
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from torch.nn import init
import torch.distributed as dist
from skimage.measure import label as skilabel
import cv2
import mahotas
from scipy import ndimage

from .sam2.training.model.sam2 import *
from .sam2.training.utils.data_utils import BatchedVideoDatapoint


import logging
import numpy as np


sam2_config_dict= {
        "tiny": {
            "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_tiny.pt"
        },
        "small": {
            "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_small.pt"
        },
        "base+": {
            "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        },
        "large": {
            "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_large.pt"
        }
    }


    
class MaskBranch(nn.Module):
    '''BCHDW -> BCHDW'''
    def __init__(self, 
            num_convs=4, 
            kernel_dim=64,
            in_filter=64, 
            out_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            shared_kwargs={}
        ):
        super().__init__()

        in_filters = [in_filter] + [out_filter] * (num_convs - 1)
        self.mask_convs = nn.Sequential(*[
            conv3d_norm_act(f, out_filter, kernel_size=kernel_size,
                    padding=padding, **shared_kwargs) \
            for f in in_filters
        ])
        self.projection = nn.Conv3d(out_filter, kernel_dim, 
                kernel_size=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv3d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)
    
class SAM2AFFTrain(SAM2Train):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder,**kwargs)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    # def forward(self, input, prompts):
    #     # Precompute image features on all frames
    #     self.num_frames = input.size(2)
    #     input = input.permute(0,2,1,3,4).flatten(0,1)
    #     if input.shape[1] == 1:  # 检查通道数
    #         input = input.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
    #     backbone_out = self.forward_image(input)
    #     # backbone_out = self.forward_image(input.flat_img_batch)
    #     backbone_out = self.prepare_prompt_inputs(backbone_out, prompts)
    #     previous_stages_out = self.forward_tracking(backbone_out, prompts)
    #     return previous_stages_out

    def forward_encode(self,input):
        
        
        # Precompute image features on all frames
        self.num_frames = input.size(2)
        input = input.permute(0,2,1,3,4)
        # if input.shape[1] == 1:  # 检查通道数
        #     input = input.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
        input_flat = input.flatten(0,1)
        backbone_out = self.forward_image(input_flat)
        
        return backbone_out
        
class FPN(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out

        return out
        
@register_model("sam2aff")
class SAM2AFF(nn.Module,CfgMixin):
    """Leveraging Query-Proposal to generate prompt for SAM2

    Args:
        block_type (str): the block type at each U-Net stage. Default: ``'residual'``
        in_channel (int): number of input channels. Default: 1
        out_channel (int): number of output channels. Default: 3
        filters (List[int]): number of filters at each U-Net stage. Default: [28, 36, 48, 64, 80]
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        isotropy (List[bool]): specify each U-Net stage is isotropic or anisotropic. All elements will
            be `True` if :attr:`is_isotropic` is `True`. Default: [False, False, False, True, True]
        pad_mode (str): one of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'replicate'``
        act_mode (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, ``'efficient_swish'`` or ``'none'``. Default: ``'relu'``
        norm_mode (str): one of ``'bn'``, ``'sync_bn'`` ``'in'`` or ``'gn'``. Default: ``'bn'``
        init_mode (str): one of ``'xavier'``, ``'kaiming'``, ``'selu'`` or ``'orthogonal'``. Default: ``'orthogonal'``
        pooling (bool): downsample by max-pooling if `True` else using stride. Default: `False`
        blurpool (bool): apply blurpool as in Zhang 2019 (https://arxiv.org/abs/1904.11486). Default: `False`
    """
    sam2_config_dict= {
        "tiny": {
            "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_tiny.pt"
        },
        "small": {
            "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_small.pt"
        },
        "base+": {
            "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        },
        "large": {
            "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "../sam2/checkpoints/sam2.1_hiera_large.pt"
        }
    }
    block_dict = {
        'residual': BasicBlock3d,
        'residual_pa': BasicBlock3dPA,
        'residual_se': BasicBlock3dSE,
        'residual_se_pa': BasicBlock3dPASE,
        # -------------------
        'residual_half': BasicBlock3dHalf,
    }
    # 
        # sam_layers = (
        #             []
        #             #   + list(net.image_encoder.parameters())
        #             #   + list(net.sam_prompt_encoder.parameters())
        #             + list(self.sam2_predictor.sam_mask_decoder.parameters())
        #             )
        # mem_layers = (
        #             []
        #             + list(self.sam2_predictor.obj_ptr_proj.parameters())
        #             + list(self.sam2_predictor.memory_encoder.parameters())
        #             + list(self.sam2_predictor.memory_attention.parameters())
        #             + list(self.sam2_predictor.mask_downsample.parameters())
        #             )
    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 return_feats: Optional[list] = None,
                 sam2_config: str = 'tiny',
                 fusion_channel: int=64,
                 conv_in_channel:int=64,
                 is_freeze_encoder: bool = False,
                 image_size:int=256,
                 conv_after_fusion:bool = True,
                 fpn_setting:int=1,
                 **kwargs):
        super().__init__()
        self.is_freeze_encoder = is_freeze_encoder
        self.conv_after_fusion = conv_after_fusion
        return_init_mask=True
        # self.use_init_feature = use_init_feature
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        sam_in_channel = 3
        
        fusion_channel = 32
        
        self.backbone_fusion = FPN(
            d_model = fusion_channel,
            backbone_channel_list = [256, 256, 256],
            kernel_size = 3,
            padding=1,
            fpn_interp_model = "bilinear",
            fuse_type='avg'
        )

        # self.backbone_fusion = FPN(
        #     d_model = fusion_channel,
        #     backbone_channel_list = [256, 64, 32],
        #     kernel_size = 3,
        #     padding=1,
        #     fpn_interp_model = "bilinear",
        #     fuse_type='avg'
        # )
        
        self.conv_image_feature = nn.Sequential(
            conv3d_norm_act(fusion_channel, fusion_channel, (1,3,3), padding=(0,1,1),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        self.conv_pos_enc = nn.Sequential(
            conv3d_norm_act(256, 32, (1,1,1), padding=(0,0,0),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        # initialization
        model_init(self, mode=init_mode)

        """Build SAM2""" #TODO:path search
        net = "sam2aff.model.SAM2AFFTrain"
        self.sam2 = build_sam2aff(self.sam2_config_dict[sam2_config]["cfg"],
                                                         self.sam2_config_dict[sam2_config]["checkpoint"],net=net,
                                                         image_size = image_size) 
        self.affinity_branch = MaskBranch(
            num_convs=1, kernel_dim=3, 
            in_filter=fusion_channel, out_filter=fusion_channel, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1),
            shared_kwargs=self.shared_kwargs
        )
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()

    def _freeze_sam2_encoder(self):
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
            # if True:
                print(f"{name}: requires_grad={param.requires_grad}")

    def _input_resize(self,inputs,size=1024):
        # Step 1: resize to 1024x1024 for each slice along D
        B, C, D, H, W = inputs.shape
        inputs = inputs.view(B * D, 1, H, W)  # merge B and D to apply 2D interpolation
        inputs = F.interpolate(inputs, size=(size, size), mode='bilinear', align_corners=False)
        inputs = inputs.view(B, 1, D, size, size)  # reshape back
        return inputs

    def _input_transform(self, inputs):
        """
        inputs: Tensor of shape (B, 1, D, H, W), values in [0, 1]
        Returns: normalized RGB tensor of shape (B, 3, D, 1024, 1024)
        """
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)

        # Step 2: repeat grayscale to 3 channels
        inputs = inputs.repeat(1, 3, 1, 1, 1)  # shape: (B, 3, D, 1024, 1024)

        # Step 3: normalize
        img_mean = torch.tensor(img_mean, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        img_std = torch.tensor(img_std, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        
        return (inputs - img_mean) / img_std

    @return_loss
    def forward(self, x):
        B,C,D,H,W = x.shape
        x = self._input_transform(x)
        
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        feature_fpn = self.sam2.image_encoder(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn).permute(1,0,2,3).unsqueeze(0)
        feature_fusion = F.interpolate(feature_fusion, size=(D,H,W), mode='trilinear', align_corners=False)

        # if self.conv_after_fusion:
        #     sam_image_features_fusion = self.conv_image_feature(sam_image_features_fusion)
        affinity = self.affinity_branch(feature_fusion)

        return affinity
        
