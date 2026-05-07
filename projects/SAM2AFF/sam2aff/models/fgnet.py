from __future__ import print_function, division
from typing import Optional, List
from collections import OrderedDict

import logging
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from connectomics.model.block import *
from connectomics.model.utils import model_init

from ..model import *

class DepthwiseBlock3d(nn.Module):
    expansion = 4

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: Union[int, tuple] = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 projection: bool = False,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 isotropic: bool = False):
                #  isotropic: bool = True):
        super().__init__()
        # if isotropic:
        #     # kernel_size, padding = 3, dilation
        #     kernel_size, padding = 5, 2*dilation
        #     # kernel_size, padding = 7, 3*dilation
        # else:
        #     # kernel_size, padding = (1, 3, 3), (0, dilation, dilation)
        #     kernel_size, padding = (1, 5, 5), (0, 2*dilation, 2*dilation)
        #     # kernel_size, padding = (1, 7, 7), (0, 3*dilation, 3*dilation)

        kernel_size, padding = (3, 7, 7), (1*dilation, 3*dilation, 3*dilation)
        assert in_planes == planes
        
        # [DWConv(C->C) -> Norm] - > [PointConv(C->4C) -> Act] -> [PointConv(4C->C) -> Norm] -> 
        self.conv = nn.Sequential(
            # DWConv(C->C) -> Norm
            conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=planes, padding=padding, pad_mode=pad_mode, 
                            norm_mode=norm_mode, act_mode='none'),

            # PointConv(C->4C) -> Act
            conv3d_norm_act(planes, self.expansion * planes, kernel_size=(1,1,1), 
                            padding=(0,0,0), bias=True, 
                            norm_mode='none', act_mode=act_mode),

            # PointConv(4C->C) -> Norm
            conv3d_norm_act(self.expansion * planes, planes, kernel_size=(1,1,1),
                            padding=(0,0,0),
                            norm_mode=norm_mode, act_mode='none')
        )

    def forward(self, x):
        y = self.conv(x)
        return y + x
    

class HelperMixin:
    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x + y

    def _get_kernal_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (5, 5, 5), (2, 2, 2)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)


    def _get_stride(self, is_isotropic, previous, i):
        if self.pooling or previous == i:
            return 1

        return self._get_downsample(is_isotropic)

    def _get_downsample(self, is_isotropic):
        if not is_isotropic:
            return (1, 2, 2)
        return 2

    def _make_pooling_layer(self, is_isotropic, previous, i):
        if self.pooling and previous != i:
            kernel_size = stride = self._get_downsample(is_isotropic)
            return nn.MaxPool3d(kernel_size, stride)

        return nn.Identity()


class EncoderFPN(nn.Module, HelperMixin):
    def __init__(self, block, filters, isotropy, pooling, shared_kwargs):
        super(EncoderFPN, self).__init__()

        self.depth = len(filters)
        self.pooling = pooling
        self.shared_kwargs = shared_kwargs

        # encoding path
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i-1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                block(filters[i], filters[i], **self.shared_kwargs)
            )
            self.layers.append(layer)

            downsample = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                        stride=stride, padding=padding, **self.shared_kwargs)
            )
            self.downsamples.append(downsample)

    def forward(self, x):
        feats = [None] * self.depth

        feats[0] = self.layers[0](self.downsamples[0](x))
        for i in range(1, self.depth):
            feats[i] = self.layers[i](
                self.downsamples[i](feats[i-1])
            )
        # for i in range(len(feats)):
        #     print(f"encoder feature shapes:{feats[i].shape}")
        return feats

class DecoderFPN(nn.Module, HelperMixin):
    """Decoder of U-Net with Local Attention (LA) mechanism."""

    def __init__(self, block, filters, isotropy, pooling, shared_kwargs):
        super(DecoderFPN, self).__init__()

        self.depth = len(filters)
        self.pooling = pooling
        self.shared_kwargs = shared_kwargs

        # disable all activation functions in projection layers
        proj_conv_kwargs = deepcopy(shared_kwargs)
        proj_conv_kwargs["act_mode"] = "none"

        # decoding path
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()

        norm_mode = 'ln'
        for i in range(self.depth-1):
            self.norms.append(get_norm_3d(norm_mode,filters[i]))
            self.convs.append(nn.Conv3d(filters[i+1],filters[i],1))

    def forward(self, x,y=None):
        feats = [None] * self.depth

        feats[-1] = x[-1]
        
        for i in range(self.depth-2,-1,-1):
            feats[i] = self.norms[i](
                self._upsample_add(self.convs[i](feats[i+1]),x[i])
            )
        # for i in range(len(feats)):
        #     print(f"decoder feature shapes:{feats[i].shape}")
        return feats[0]     # return the highest res. feature map


    def _upsample(self, x, y_refer):
        """Upsample feature map x to the size of y_refer.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y_refer.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x


@register_model("fgnet_3d")
class FGNet(nn.Module, HelperMixin):
    """3D residual FPN-style architecture with Depthwise Conv.
    based on dwunet_3d_v2: 
        - Change stem layer to 2 conv3x3 instead of 1 conv5x5.
        - Use GroupNorm (by default).
    """
    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 return_feats: Optional[list] = None,
                 **kwargs):
        super(FGNet, self).__init__()

        block = DepthwiseBlock3d
        logging.warning(f'Force using DepthwiseBlock3d for DWUNet3D. {block_type} is ignored.')

        self.depth = len(filters)
        self.do_return_feats = (return_feats is not None)
        self.return_feats = return_feats
        print(f"Return feature maps from 3D FPN-Net? {self.do_return_feats}")

        assert not self.do_return_feats

        if is_isotropic:
            isotropy = [True] * self.depth
        # assert len(filters) == len(isotropy)

        # block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        # self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
        #                                padding=padding_io, **self.shared_kwargs)
        self.conv_in = [
            conv3d_norm_act(in_channel, filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs),
            conv3d_norm_act(filters[0], filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs)
        ]
        self.conv_in = nn.Sequential(*self.conv_in)
        
        # stages
        self.num_stages = 3
        self.loss_weight = "avg"

        self.heads = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(nn.Sequential(
                EncoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs),
                DecoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs)
            ))
            self.heads.append(
                conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
            )

        # initialization
        model_init(self, mode=init_mode)

    def forward(self, inputs, target=None, weight=None, criterion=None):
        x = self.conv_in(inputs)

        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)

        preds = []
        for i, feats in enumerate(stage_feats):
            # pred = self.heads[i](feats[0])
            pred = self.heads[i](feats)
            preds.append(pred)

        # calculate loss
        if criterion:
            list_loss, full_losses_vis = [], dict()

            for t in range(self.num_stages):
                loss, losses_vis = criterion(preds[t], target, weight)
                list_loss.append(loss)
                full_losses_vis.update({k+f"_iter{t}": v for k, v in losses_vis.items()})

            full_losses_vis.update(losses_vis)

            if self.loss_weight == "avg":
                loss = sum(list_loss) / len(list_loss)
            elif self.loss_weight == "sum":
                loss = sum(list_loss)
            elif self.loss_weight == "last":
                loss = list_loss[-1] + 0 * sum(list_loss[:-1])      #  avoid unused parameter warning
            else:
                raise NotImplementedError

            return preds[-1], loss, full_losses_vis

        return preds[-1]