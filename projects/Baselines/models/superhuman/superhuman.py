"""
models.py -- model code for the `python main.py -c=seg_3d` pipeline

config: seg_3d.yaml -> model_type: 'superhuman' -> UNet_PNI

Dependencies:
  UNet_PNI
    └── resBlock_pni
          └── conv3dBlock ─── getConv3d, getBN, getRelu, init_conv
    └── conv3dBlock
    └── upsampleBlock ──────── init_conv

Usage:
    from model.models import UNet_PNI
    model = UNet_PNI(in_planes=1, out_planes=3, filters=[28,36,48,64,80])
"""

import torch
import torch.nn as nn
from ..model import *

# =============================================================================
# Utilities
# =============================================================================

def init_conv(m, init_mode):
    """Weight initialization for convolution layers."""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight)
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif init_mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)
        elif init_mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def getRelu(mode='elu'):
    """Activation factory: 'relu' | 'elu' | 'leaky<slope>'"""
    if mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode[:5] == 'leaky':
        return nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))
    raise ValueError('Unknown relu mode: ' + mode)


def getBN(out_planes, bn_mode='async', bn_momentum=0.1):
    """3D BatchNorm factory: 'async' (standard BN) | 'sync' (standard BN used here as a substitute)"""
    return nn.BatchNorm3d(out_planes, momentum=bn_momentum)


def getConv3d(in_planes, out_planes, kernel_size, stride, padding,
              bias, pad_mode='zero', init_mode='', dilation_size=(1, 1, 1)):
    """Build a 3D convolution layer with zero / replicate padding."""
    if pad_mode == 'zero':
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            dilation=dilation_size, padding=padding,
                            stride=stride, bias=bias)]
    elif pad_mode == 'replicate':
        pad = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad3d(pad),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, dilation=dilation_size, bias=bias)]
    else:
        raise ValueError('Unknown pad_mode: ' + pad_mode)
    if init_mode:
        init_conv(layers[-1], init_mode)
    return layers


def conv3dBlock(in_planes, out_planes,
                kernel_size=[(3, 3, 3)], stride=[1], padding=[0],
                bias=[True], pad_mode=['zero'], bn_mode=[''], relu_mode=[''],
                init_mode='kaiming_normal', bn_momentum=0.1, dilation_size=None):
    """VGG-style 3D conv block: stackable [Conv -> BN -> ReLU]"""
    layers = []
    if dilation_size is None:
        dilation_size = [(1, 1, 1)] * len(in_planes)
    for i in range(len(in_planes)):
        if in_planes[i] > 0:
            layers += getConv3d(in_planes[i], out_planes[i], kernel_size[i],
                                stride[i], padding[i], bias[i],
                                pad_mode[i], init_mode, dilation_size[i])
        if bn_mode[i] != '':
            layers.append(getBN(out_planes[i], bn_mode[i], bn_momentum))
        if relu_mode[i] != '':
            layers.append(getRelu(relu_mode[i]))
    return nn.Sequential(*layers)


def upsampleBlock(in_planes, out_planes, up=(1, 2, 2), mode='bilinear',
                  kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                  bias=True, init_mode=''):
    """3D upsampling block
    mode:
      'bilinear'   trilinear interpolation + 1x1x1 conv
      'nearest'    nearest interpolation + 1x1x1 conv
      'transpose'  transposed convolution (dense)
      'transposeS' depthwise-separable transposed convolution (sparse, recommended)
    """
    if mode == 'bilinear':
        layers = [nn.Upsample(scale_factor=up, mode='trilinear', align_corners=True),
                  nn.Conv3d(in_planes, out_planes, kernel_size, stride=stride,
                            padding=padding, bias=bias)]
    elif mode == 'nearest':
        layers = [nn.Upsample(scale_factor=up, mode='nearest'),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias)]
    elif mode == 'transpose':
        layers = [nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size,
                                     stride=up, bias=bias)]
    elif mode == 'transposeS':
        layers = [nn.ConvTranspose3d(in_planes, in_planes, kernel_size=up,
                                     stride=up, bias=bias, groups=in_planes),
                  nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)]
    else:
        raise ValueError('Unknown upsample mode: ' + mode)
    out = nn.Sequential(*layers)
    for m in out._modules.values():
        init_conv(m, init_mode)
    return out


# =============================================================================
# Residual block
# =============================================================================

class resBlock_pni(nn.Module):
    """PNI residual block (for anisotropic EM images)

    Structure:
      block1: 1x3x3 conv+BN+ReLU  (channel projection)
      block2: 3x3x3 conv+BN+ReLU -> 3x3x3 conv+BN  (two residual layers)
      block3: BN
      block4: ReLU
    ref: https://github.com/torms3/Superhuman
    """
    def __init__(self, in_planes, out_planes,
                 pad_mode='zero', bn_mode='async', relu_mode='elu',
                 init_mode='kaiming_normal', bn_momentum=0.1):
        super().__init__()
        self.block1 = conv3dBlock(
            [in_planes], [out_planes], [(1, 3, 3)], [1], [(0, 1, 1)],
            [False], [pad_mode], [bn_mode], [relu_mode], init_mode, bn_momentum)
        self.block2 = conv3dBlock(
            [out_planes] * 2, [out_planes] * 2, [(3, 3, 3)] * 2, [1] * 2,
            [(1, 1, 1)] * 2, [False] * 2, [pad_mode] * 2,
            [bn_mode, ''], [relu_mode, ''], init_mode, bn_momentum)
        self.block3 = getBN(out_planes, bn_mode, bn_momentum)
        self.block4 = getRelu(relu_mode) if relu_mode else None

    def forward(self, x):
        residual = self.block1(x)
        out = self.block3(residual + self.block2(residual))
        if self.block4 is not None:
            out = self.block4(out)
        return out


# =============================================================================
# UNet_PNI (superhuman)
# Paper: Superhuman Accuracy on the SNEMI3D Connectomics Challenge
# https://arxiv.org/abs/1706.00120
# =============================================================================

@register_model("superhuman")
class UNet_PNI(nn.Module):
    """5-level UNet with PNI residual blocks for 3D affinity prediction

    Input:  (B, 1, D, H, W)   recommended size: (B, 1, 18, 160, 160)
    Output: (B, 3, D, H, W)   z/y/x affinity maps in [0,1]

    Key args (mapped from seg_3d.yaml -> MODEL):
        filters       : per-level channel counts, default [28,36,48,64,80]
        upsample_mode : 'bilinear' | 'transposeS' | ...
        merge_mode    : 'add' (additive skip) | 'cat' (concatenated skip)
        pad_mode      : 'zero' | 'replicate'
        bn_mode       : 'async' (standard BN)
        relu_mode     : 'elu' | 'relu' | 'leaky<slope>'
        init_mode     : 'kaiming_normal' | ...
        if_sigmoid    : whether to apply sigmoid to the output
    """
    def __init__(self,
                 in_planes=1,
                 out_planes=3,
                 filters=(28, 36, 48, 64, 80),
                 upsample_mode='transposeS',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 **kwargs):
        super().__init__()
        f = [list(filters)[0]] + list(filters)   # f[0..5]
        self.merge_mode = merge_mode
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # Input embedding: 1x5x5 (anisotropic, no convolution along z)
        self.embed_in = conv3dBlock(
            [in_planes], [f[0]], [(1, 5, 5)], [1], [(0, 2, 2)],
            [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)

        # encoder
        self.conv0 = resBlock_pni(f[0], f[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv1 = resBlock_pni(f[1], f[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = resBlock_pni(f[2], f[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv3 = resBlock_pni(f[3], f[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # bottleneck
        self.center = resBlock_pni(f[4], f[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # decoder (4 symmetric levels)
        self.up0, self.cat0, self.conv4 = self._dec_block(f[5], f[4], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up1, self.cat1, self.conv5 = self._dec_block(f[4], f[3], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up2, self.cat2, self.conv6 = self._dec_block(f[3], f[2], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up3, self.cat3, self.conv7 = self._dec_block(f[2], f[1], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # Output embedding: 1x5x5
        self.embed_out = conv3dBlock(
            [f[0]], [f[0]], [(1, 5, 5)], [1], [(0, 2, 2)],
            [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)

        # Output head: 1x1x1 -> out_planes
        self.out_put = conv3dBlock([f[0]], [out_planes], [(1, 1, 1)], init_mode=init_mode)

    @staticmethod
    def _dec_block(f_in, f_skip, upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum):
        up   = upsampleBlock(f_in, f_skip, (1, 2, 2), upsample_mode, init_mode=init_mode)
        ch   = f_skip if merge_mode == 'add' else f_skip * 2
        cat  = conv3dBlock([0], [ch], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        conv = resBlock_pni(ch, f_skip, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        return up, cat, conv

    def _merge(self, up, skip, cat_layer):
        if self.merge_mode == 'add':
            return cat_layer(up + skip)
        return cat_layer(torch.cat([up, skip], dim=1))


    @return_loss
    def forward(self, x):
        e   = self.embed_in(x)
        c0  = self.conv0(e)
        c1  = self.conv1(self.pool0(c0))
        c2  = self.conv2(self.pool1(c1))
        c3  = self.conv3(self.pool2(c2))
        ctr = self.center(self.pool3(c3))

        d0 = self.conv4(self._merge(self.up0(ctr), c3, self.cat0))
        d1 = self.conv5(self._merge(self.up1(d0),  c2, self.cat1))
        d2 = self.conv6(self._merge(self.up2(d1),  c1, self.cat2))
        d3 = self.conv7(self._merge(self.up3(d2),  c0, self.cat3))

        embed_out = self.embed_out(d3)
        out = self.out_put(embed_out)

        if self.show_feature:
            return embed_out
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 18, 160, 160)
    model = UNet_PNI(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')
    out = model(x)
    print(f'in: {list(x.shape)}  ->  out: {list(out.shape)}')
    # expected: in: [1, 1, 18, 160, 160]  ->  out: [1, 3, 18, 160, 160]
