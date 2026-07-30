"""
pea.py -- full code for the `python main.py -c=ac3ac4` pipeline (model + losses + EMA input)

config: ac3ac4.yaml -> model_type: 'pea' -> UNet_PNI_embedding_deep

Dependencies:
  UNet_PNI_embedding_deep
    └── resBlock_pni
          └── conv3dBlock ─── getConv3d, getBN, getRelu, init_conv
    └── conv3dBlock
    └── upsampleBlock ──────── init_conv

  Losses:
    embedding_loss_norm5     <- SCM: 12-channel multi-scale cosine affinity
    embedding_loss_norm1     <- EPM: 3-channel short-range cosine affinity (deep supervision)
    ema_embedding_loss_norm5 <- CCM: 12-channel cross affinity
    convert_consistency_flip <- CCM: coordinate alignment of ema_embedding
    WeightedMSE              <- weighted MSE criterion

  EMA input generation:
    make_ema_input           <- three stages: intensity jitter -> cutout -> flip

Usage:
    from model.pea import UNet_PNI_embedding_deep
    model = UNet_PNI_embedding_deep(
        in_planes=1, out_planes=12, filters=[28,36,48,64,80],
        upsample_mode='bilinear', merge_mode='add', emd=16)
    embedding = model(x)  # inference: x: [B,1,18,160,160] -> [B,emd,18,160,160]
    pred, loss, vis = model(x, target, weight, criterion)  # training: framework call

Total training loss:
    loss = embedding_loss_norm5(embedding, ...)          # SCM
         + ema_embedding_loss_norm5(embedding, ema_emb)  # CCM
         + embedding_loss_norm1(emd1, down4[:,:3], ...)  # EPM ×4
         + embedding_loss_norm1(emd2, down3[:,:3], ...)
         + embedding_loss_norm1(emd3, down2[:,:3], ...)
         + embedding_loss_norm1(emd4, down1[:,:3], ...)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import *
from .utils import *



# =============================================================================
# Loss functions
# =============================================================================



class WeightedMSE(nn.Module):
    """Weighted MSE, the base criterion for all losses

    L = sum[ weight * (pred - target)^2 ] / (B * spatial_size)
    Call: criterion(pred, target, weightmap)
    """
    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        else:
            return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight=None):
        return self.weighted_mse_loss(pred, target, weight)


# -----------------------------------------------------------------------------
# Internal helper: affinity loss for a single direction and offset
# -----------------------------------------------------------------------------

def _single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    """Affinity and loss for one (direction, offset) pair

    Direction is given by order % 3: 0 -> z axis, 1 -> y axis, 2 -> x axis
    """
    B, C, D, H, W = embedding.shape
    ax = order % 3

    if ax == 0:
        affs = torch.sum(embedding[:, :, shift:,   :,      :     ] *
                         embedding[:, :, :D-shift,  :,      :     ], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, shift:,   :,      :     ],
                               weightmap[:, order:order+1, shift:,   :,      :     ])
    elif ax == 1:
        affs = torch.sum(embedding[:, :, :,      shift:,   :     ] *
                         embedding[:, :, :,      :H-shift,  :     ], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :,      shift:,   :     ],
                               weightmap[:, order:order+1, :,      shift:,   :     ])
    else:
        affs = torch.sum(embedding[:, :, :,      :,      shift: ] *
                         embedding[:, :, :,      :,      :W-shift], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :,      :,      shift: ],
                               weightmap[:, order:order+1, :,      :,      shift: ])

    return loss, affs


def _ema_single_offset_loss(embedding, ema_embedding, order, shift, target, weightmap, criterion):
    """Cross-stream variant: dot-product affinity between embedding and ema_embedding."""
    B, C, D, H, W = embedding.shape
    ax = order % 3

    if ax == 0:
        affs = torch.sum(embedding[:, :, shift:,   :,      :     ] *
                         ema_embedding[:, :, :D-shift,  :,      :     ], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, shift:,   :,      :     ],
                               weightmap[:, order:order+1, shift:,   :,      :     ])
    elif ax == 1:
        affs = torch.sum(embedding[:, :, :,      shift:,   :     ] *
                         ema_embedding[:, :, :,      :H-shift,  :     ], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :,      shift:,   :     ],
                               weightmap[:, order:order+1, :,      shift:,   :     ])
    else:
        affs = torch.sum(embedding[:, :, :,      :,      shift: ] *
                         ema_embedding[:, :, :,      :,      :W-shift], dim=1, keepdim=True)
        loss = criterion(affs, target[:, order:order+1, :,      :,      shift: ],
                               weightmap[:, order:order+1, :,      :,      shift: ])

    return loss, affs


# -----------------------------------------------------------------------------
# SCM: 12-channel multi-scale cosine affinity loss (final embedding)
# -----------------------------------------------------------------------------

def embedding_loss_norm5(embedding, target, weightmap, criterion,
                         affs0_weight=1, shift=1, fill=True):
    """12-channel multi-scale affinity loss

    shifts = [1,1,1, 2, 3,3,3, 9,9, 4, 27,27]
    channels 0-2  (shift=1): z/y/x short range -> weight * affs0_weight
    channels 3-11 (shift>1): long-range offsets -> weight 1

    Returns: (scalar loss, affs tensor with the same shape as target)
    """
    embedding = F.normalize(embedding, p=2, dim=1)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    affs = torch.zeros_like(target)
    loss = 0
    for i, sh in enumerate(shifts):
        loss_i, affs_i = _single_offset_loss(embedding, i, sh, target, weightmap, criterion)
        loss += loss_i * affs0_weight if i < 3 else loss_i
        ax = i % 3
        if ax == 0:
            affs[:, i:i+1, sh:,  :,   :  ] = affs_i.clone().detach()
        elif ax == 1:
            affs[:, i:i+1, :,   sh:,  :  ] = affs_i.clone().detach()
        else:
            affs[:, i:i+1, :,   :,   sh: ] = affs_i.clone().detach()

    return loss, affs


# -----------------------------------------------------------------------------
# EPM: 3-channel short-range cosine affinity loss (supervision of emd1-emd4)
# -----------------------------------------------------------------------------

def embedding_loss_norm1(embedding, target, weightmap, criterion,
                         affs0_weight=1, shift=1, fill=True):
    """3-channel affinity loss (z/y/x, shift=1)

    Applied to intermediate decoder outputs (emd1-emd4) against downsampled
    affinity labels.
    Returns: (scalar loss, affs tensor with the same shape as target)
    """
    embedding = F.normalize(embedding, p=2, dim=1)
    B, C, D, H, W = embedding.shape

    affs0 = torch.sum(embedding[:, :, shift:, :, :] * embedding[:, :, :D-shift, :, :], dim=1, keepdim=True)
    affs1 = torch.sum(embedding[:, :, :, shift:, :] * embedding[:, :, :, :H-shift, :], dim=1, keepdim=True)
    affs2 = torch.sum(embedding[:, :, :, :, shift:] * embedding[:, :, :, :, :W-shift], dim=1, keepdim=True)

    loss0 = criterion(affs0, target[:, 0:1, shift:, :, :], weightmap[:, 0:1, shift:, :, :])
    loss1 = criterion(affs1, target[:, 1:2, :, shift:, :], weightmap[:, 1:2, :, shift:, :])
    loss2 = criterion(affs2, target[:, 2:3, :, :, shift:], weightmap[:, 2:3, :, :, shift:])

    loss = affs0_weight * loss0 + loss1 + loss2

    affs = torch.zeros_like(target)
    affs[:, 0:1, shift:, :, :] = affs0.clone().detach()
    affs[:, 1:2, :, shift:, :] = affs1.clone().detach()
    affs[:, 2:3, :, :, shift:] = affs2.clone().detach()

    return loss, affs


# -----------------------------------------------------------------------------
# CCM: coordinate alignment (un-flip ema_embedding back to the original frame)
# -----------------------------------------------------------------------------

def _augment_reverse_torch(data, rule):
    """Inverse flip/transpose of a single sample's embedding tensor [C, D, H, W]

    Exactly inverts simple_augment:
      simple_augment order:  z -> x -> y -> xy transpose
      this function:         xy transpose -> y -> x -> z

    rule: length-4 array, rule[i] in {0, 1}
      rule[0] -> z flip   rule[1] -> x flip
      rule[2] -> y flip   rule[3] -> xy transpose
    """
    assert len(data.shape) == 4   # [C, D, H, W]
    if rule[3]: data = data.permute(0, 1, 3, 2)   # inverse xy-transpose
    if rule[2]: data = torch.flip(data, [2])        # inverse y-flip
    if rule[1]: data = torch.flip(data, [3])        # inverse x-flip
    if rule[0]: data = torch.flip(data, [1])        # inverse z-flip
    return data


def convert_consistency_flip(ema_embedding, rules):
    """Un-flip each sample's ema_embedding back into the original coordinate frame

    ema_imgs are randomly flipped before being fed to the model, so the spatial
    coordinates of the resulting ema_embedding are flipped too. This function
    reverses each rule so that both embeddings share the same coordinates.

    Input:
      ema_embedding  [B, C, D, H, W]  GPU tensor
      rules          [B, 4]           GPU tensor from the DataLoader
    Output:
      aligned        [B, C, D, H, W]  coordinate-aligned ema_embedding
    """
    B = ema_embedding.shape[0]
    ema_embedding = ema_embedding.detach().clone()
    rules_np = rules.data.cpu().numpy().astype(np.uint8)
    out = []
    for k in range(B):
        out.append(_augment_reverse_torch(ema_embedding[k], rules_np[k]))
    return torch.stack(out, dim=0)


# -----------------------------------------------------------------------------
# CCM: 12-channel cross affinity loss (EMA consistency)
# -----------------------------------------------------------------------------

def ema_embedding_loss_norm5(embedding, ema_embedding, target, weightmap, criterion,
                              affs0_weight=1, shift=1, fill=True):
    """EMA consistency loss: 12-channel cross affinity between the two embeddings

    Dot-product affinities between the original embedding and the EMA-augmented
    embedding are compared against the same ground-truth labels, forcing the two
    branches to agree.

    Returns: (scalar loss, affs tensor with the same shape as target)
    """
    embedding     = F.normalize(embedding,     p=2, dim=1)
    ema_embedding = F.normalize(ema_embedding, p=2, dim=1)
    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]

    affs = torch.zeros_like(target)
    loss = 0
    for i, sh in enumerate(shifts):
        loss_i, affs_i = _ema_single_offset_loss(
            embedding, ema_embedding, i, sh, target, weightmap, criterion)
        loss += loss_i * affs0_weight if i < 3 else loss_i
        ax = i % 3
        if ax == 0:
            affs[:, i:i+1, sh:,  :,   :  ] = affs_i.clone().detach()
        elif ax == 1:
            affs[:, i:i+1, :,   sh:,  :  ] = affs_i.clone().detach()
        else:
            affs[:, i:i+1, :,   :,   sh: ] = affs_i.clone().detach()

    return loss, affs


# =============================================================================
# EMA input generation (data layer, runs on CPU)
# =============================================================================

class IntensityAugment:
    """Random contrast and brightness jitter

    Makes the EMA branch input photometrically different from the original
    input while keeping the semantics of each pixel (which neuron it belongs
    to) unchanged.
    """
    def __call__(self, imgs, contrast_factor=0.1, brightness_factor=0.1):
        imgs = imgs * (1 + contrast_factor)
        imgs = imgs + brightness_factor
        imgs = np.clip(imgs, 0, 1)
        return imgs


def gen_mask(imgs,
             min_mask_counts=0,
             max_mask_counts=60,
             min_mask_size=(5, 10, 10),
             max_mask_size=(10, 20, 20)):
    """Random cutout occlusion, returns a binary mask (0 = masked, 1 = kept)

    Input:  imgs  [D, H, W]  numpy float32
    Output: mask  [D, H, W]  numpy float32, 0 or 1
    """
    crop_size = list(imgs.shape)          # [D, H, W]
    mask = np.ones_like(imgs, dtype=np.float32)
    mask_counts = random.randint(min_mask_counts, max_mask_counts)
    sz  = random.randint(min_mask_size[0], max_mask_size[0])
    sxy = random.randint(min_mask_size[1], max_mask_size[1])
    for _ in range(mask_counts):
        mz = random.randint(0, crop_size[0] - sz)
        my = random.randint(0, crop_size[1] - sxy)
        mx = random.randint(0, crop_size[2] - sxy)
        mask[mz:mz+sz, my:my+sxy, mx:mx+sxy] = 0
    return mask


def simple_augment(data, rule):
    """Flip/transpose numpy 3D data [D, H, W] according to rule

    rule: length-4 binary array
      rule[0]=1 -> z flip   rule[1]=1 -> x flip
      rule[2]=1 -> y flip   rule[3]=1 -> swap x and y (transpose)

    rule must be stored with the batch and passed into the training loop so
    convert_consistency_flip can invert it and realign the coordinate frames.
    """
    assert data.ndim == 3
    if rule[0]: data = data[::-1, :, :]
    if rule[1]: data = data[:, :, ::-1]
    if rule[2]: data = data[:, ::-1, :]
    if rule[3]: data = np.transpose(data, (0, 2, 1))
    return data


class Filp_EMA:
    """Random flip wrapper: samples a rule and flips ema_imgs accordingly

    Call:   ema_imgs, rule = Filp_EMA()(ema_imgs)
    Input:  data  [D, H, W]  numpy float32
    Output: (augmented_data [D,H,W],  rule [4,] uint8)
    """
    def __call__(self, data):
        rule = np.random.randint(2, size=4)
        data = simple_augment(data, rule)
        return data, rule


def make_ema_input(imgs,
                   if_ema_intensity=True,
                   if_ema_mask=True,
                   if_ema_flip=True,
                   min_mask_counts=0,
                   max_mask_counts=60,
                   min_mask_size=(5, 10, 10),
                   max_mask_size=(10, 20, 20)):
    """Generate the EMA branch input in three stages

    original imgs
      -> IntensityAugment   brightness/contrast jitter
      -> gen_mask           random cutout occlusion
      -> Filp_EMA           random flip + recorded rule
      -> ema_imgs, rule

    Input:  imgs  [D, H, W]  numpy float32 in [0,1]
    Output:
      ema_imgs  [D, H, W]  numpy float32
      rule      [4,]       numpy uint8, all zeros means no flip was applied
    """
    _aug_intensity = IntensityAugment()
    _aug_flip      = Filp_EMA()

    ema_imgs = imgs.copy()

    if if_ema_intensity:
        ema_imgs = _aug_intensity(ema_imgs)

    if if_ema_mask:
        mask = gen_mask(ema_imgs,
                        min_mask_counts=min_mask_counts,
                        max_mask_counts=max_mask_counts,
                        min_mask_size=min_mask_size,
                        max_mask_size=max_mask_size)
        ema_imgs = ema_imgs * mask

    if if_ema_flip:
        ema_imgs, rule = _aug_flip(ema_imgs)
    else:
        rule = np.zeros(4, dtype=np.uint8)

    return ema_imgs, rule


    
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
# UNet_PNI_embedding_deep (PEA)
# Paper: Pixel Embedded Affinity
# Based on the Superhuman UNet, with added deep embedding supervision (5 outputs)
# =============================================================================

@register_model("pea")
class UNet_PNI_embedding_deep(nn.Module):
    """5-level UNet with PNI residual blocks and deep embedding supervision

    Input:  (B, 1, D, H, W)   recommended size: (B, 1, 18, 160, 160)
    Output: (emd1, emd2, emd3, emd4, embedding)
      emd1  <- projection of the center level -> emd dims
      emd2  <- projection of conv4 (1st decoder level)
      emd3  <- projection of conv5 (2nd decoder level)
      emd4  <- projection of conv6 (3rd decoder level)
      embedding <- projection of the final output

    Key args (mapped from ac3ac4.yaml -> MODEL):
        filters       : per-level channel counts, default [28,36,48,64,80]
        upsample_mode : 'bilinear' | 'transposeS' | ...
        merge_mode    : 'add' (additive skip) | 'cat' (concatenated skip)
        pad_mode      : 'zero' | 'replicate'
        bn_mode       : 'async' (standard BN)
        relu_mode     : 'elu' | 'relu' | 'leaky<slope>'
        init_mode     : 'kaiming_normal' | ...
        emd           : embedding projection dimension, default 16
    """
    def __init__(self,
                 in_planes=1,
                 out_planes=12,
                 filters=(28, 36, 48, 64, 80),
                 upsample_mode='bilinear',
                 decode_ratio=1,
                 merge_mode='add',
                 pad_mode='zero',
                 bn_mode='async',
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 emd=16,
                 show_feature=False,
                 # aliases from build_model kwargs
                 in_channel=None,
                 out_channel=None,
                 **kwargs):
        super().__init__()
        # build_model passes in_channel/out_channel; map to in_planes/out_planes
        if in_channel is not None:
            in_planes = in_channel
        if out_channel is not None:
            out_planes = out_channel
        f = [list(filters)[0]] + list(filters)   # f[0..5]
        self.merge_mode = merge_mode
        self.emd = emd

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

        # Output projection heads: each level -> emd dims
        self.out_put  = conv3dBlock([f[0]], [emd], [(1, 1, 1)], init_mode=init_mode)  # final embedding
        self.out_put1 = conv3dBlock([f[5]], [emd], [(1, 1, 1)], init_mode=init_mode)  # center
        self.out_put2 = conv3dBlock([f[4]], [emd], [(1, 1, 1)], init_mode=init_mode)  # conv4
        self.out_put3 = conv3dBlock([f[3]], [emd], [(1, 1, 1)], init_mode=init_mode)  # conv5
        self.out_put4 = conv3dBlock([f[2]], [emd], [(1, 1, 1)], init_mode=init_mode)  # conv6

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

    def _forward_features(self, x):
        """UNet forward pass, returns (emd1, emd2, emd3, emd4, embedding)."""
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

        embedding = self.out_put(embed_out)   # final embedding
        emd1 = self.out_put1(ctr)             # center level
        emd2 = self.out_put2(d0)              # 1st decoder level
        emd3 = self.out_put3(d1)              # 2nd decoder level
        emd4 = self.out_put4(d2)              # 3rd decoder level

        return emd1, emd2, emd3, emd4, embedding

    def _make_ema_input_torch(self, volume):
        """GPU-side EMA input generation: intensity jitter + cutout + random flip

        Args:
            volume: [B, 1, D, H, W] GPU tensor
        Returns:
            ema_volume: [B, 1, D, H, W]
            rules: [B, 4] tensor
        """
        B = volume.shape[0]
        ema = volume.clone()

        # 1. intensity jitter
        ema = ema * (1 + 0.1 * (torch.rand(B, 1, 1, 1, 1, device=volume.device) * 2 - 1))
        ema = ema + 0.1 * (torch.rand(B, 1, 1, 1, 1, device=volume.device) * 2 - 1)
        ema = ema.clamp(0, 1)

        # 2. Cutout (per sample)
        for b in range(B):
            mask = torch.ones_like(ema[b])
            n_masks = torch.randint(0, 61, (1,)).item()
            for _ in range(n_masks):
                sz = torch.randint(5, 11, (1,)).item()
                sxy = torch.randint(10, 21, (1,)).item()
                D, H, W = ema.shape[2:]
                mz = torch.randint(0, max(D - sz, 1), (1,)).item()
                my = torch.randint(0, max(H - sxy, 1), (1,)).item()
                mx = torch.randint(0, max(W - sxy, 1), (1,)).item()
                mask[:, mz:mz+sz, my:my+sxy, mx:mx+sxy] = 0
            ema[b] = ema[b] * mask

        # 3. random flip + record rules
        rules = torch.randint(0, 2, (B, 4), device=volume.device)
        for b in range(B):
            r = rules[b]
            if r[0]: ema[b] = torch.flip(ema[b], [1])  # z
            if r[1]: ema[b] = torch.flip(ema[b], [3])  # x
            if r[2]: ema[b] = torch.flip(ema[b], [2])  # y
            if r[3]: ema[b] = ema[b].permute(0, 1, 3, 2).clone()  # xy transpose

        return ema, rules

    @staticmethod
    def _embedding_to_affinity(embedding):
        """Convert an embedding into a 3-channel short-range affinity map (inference)

        Cosine similarity with shift=1 along the z/y/x axes of the L2-normalized
        embedding.
        Returns: Tensor [B, 3, D, H, W]
        """
        embedding = F.normalize(embedding, p=2, dim=1)
        B, C, D, H, W = embedding.shape
        affs = torch.zeros(B, 3, D, H, W, device=embedding.device)
        affs[:, 0:1, 1:, :, :] = torch.sum(
            embedding[:, :, 1:, :, :] * embedding[:, :, :D-1, :, :],
            dim=1, keepdim=True)
        affs[:, 1:2, :, 1:, :] = torch.sum(
            embedding[:, :, :, 1:, :] * embedding[:, :, :, :H-1, :],
            dim=1, keepdim=True)
        affs[:, 2:3, :, :, 1:] = torch.sum(
            embedding[:, :, :, :, 1:] * embedding[:, :, :, :, :W-1],
            dim=1, keepdim=True)
        # Boundary padding: fill the zero border left by the shift with neighbouring
        # values to avoid small holes at inference time
        affs[:, 0:1, :1, :, :] = affs[:, 0:1, 1:2, :, :]   # z border
        affs[:, 1:2, :, :1, :] = affs[:, 1:2, :, 1:2, :]   # y border
        affs[:, 2:3, :, :, :1] = affs[:, 2:3, :, :, 1:2]   # x border
        affs = F.relu(affs)                                   # keep non-negative
        return affs

    def forward(self, inputs, target=None, weight=None, criterion=None):
        """Forward pass and loss computation (framework calling convention)

        Test mode (criterion=None):
            return pred                     # Tensor [B, 12, D, H, W] multi-scale affinity

        Training mode (criterion!=None):
            return pred, loss, losses_vis   # matches the return_loss interface

        Provided by the framework:
            target: List[4 x Tensor[B,3,D,H,W]]  multi-scale affinity labels
            weight: List[4 x List[1 x Tensor[B,3,D,H,W]]]  weight maps
            criterion: framework Criterion (ignored, WeightedMSE is used internally)
        """
        # ── main branch forward ──
        emd1, emd2, emd3, emd4, embedding = self._forward_features(inputs)

        if criterion is None:
            return self._embedding_to_affinity(embedding)

        _criterion = WeightedMSE()

        # ── unpack target/weight ──
        # target: List[4 x Tensor[B,3,D,H,W]] → Tensor[B,12,D,H,W]
        aff_target = torch.cat([t.to(inputs.device) for t in target], dim=1)
        # weight: List[4 x List[1 x Tensor[B,3,D,H,W]]] → Tensor[B,12,D,H,W]
        aff_weight = torch.cat([w[0].to(inputs.device) for w in weight], dim=1)

        # ── SCM: 12-channel multi-scale affinity loss (final embedding) ──
        loss_emb, affs_emb = embedding_loss_norm5(embedding, aff_target, aff_weight, _criterion)

        # ── CCM: EMA consistency loss ──
        ema_volume, rules = self._make_ema_input_torch(inputs)
        _, _, _, _, ema_embedding = self._forward_features(ema_volume)
        ema_embedding = convert_consistency_flip(ema_embedding, rules)
        loss_cross, _ = ema_embedding_loss_norm5(
            embedding, ema_embedding, aff_target, aff_weight, _criterion)

        # ── EPM: multi-scale deep supervision ──
        # down1-4 are produced by downsampling the shift=1 target and weight
        short_target = target[0].to(inputs.device)   # [B,3,D,H,W]
        short_weight = weight[0][0].to(inputs.device) # [B,3,D,H,W]
        short_tw = torch.cat([short_target, short_weight], dim=1)  # [B,6,D,H,W]

        down1 = F.avg_pool3d(short_tw, (1, 2, 2))    # 2x
        down2 = F.avg_pool3d(short_tw, (1, 4, 4))    # 4x
        down3 = F.avg_pool3d(short_tw, (1, 8, 8))    # 8x
        down4 = F.avg_pool3d(short_tw, (1, 16, 16))  # 16x

        loss_emd1, _ = embedding_loss_norm1(emd1, down4[:, :3], down4[:, 3:], _criterion)
        loss_emd2, _ = embedding_loss_norm1(emd2, down3[:, :3], down3[:, 3:], _criterion)
        loss_emd3, _ = embedding_loss_norm1(emd3, down2[:, :3], down2[:, 3:], _criterion)
        loss_emd4, _ = embedding_loss_norm1(emd4, down1[:, :3], down1[:, 3:], _criterion)

        # ── total loss ──
        loss = loss_emb + loss_cross + loss_emd1 + loss_emd2 + loss_emd3 + loss_emd4

        losses_vis = {
            'loss_embedding': loss_emb.item(),
            'loss_cross':     loss_cross.item(),
            'loss_emd1':      loss_emd1.item(),
            'loss_emd2':      loss_emd2.item(),
            'loss_emd3':      loss_emd3.item(),
            'loss_emd4':      loss_emd4.item(),
        }

        pred = self._embedding_to_affinity(embedding)
        return pred, loss, losses_vis





# =============================================================================
# Smoke test
# =============================================================================

if __name__ == '__main__':
    # ── model test (inference mode) ──
    x = torch.randn(1, 1, 18, 160, 160)
    model = UNet_PNI_embedding_deep(
        in_planes=1, out_planes=12,
        filters=[28, 36, 48, 64, 80],
        upsample_mode='bilinear', merge_mode='add', emd=16)
    pred = model(x)
    print(f'pred (inference): {list(pred.shape)}')

    # ── training mode test (simulated framework call) ──
    # mimic the target and weight produced by the DataLoader
    target = [torch.randn(1, 3, 18, 160, 160) for _ in range(4)]
    weight = [[torch.ones(1, 3, 18, 160, 160)] for _ in range(4)]
    criterion = WeightedMSE()  # placeholder, forward uses its own criterion

    pred, loss, losses_vis = model(x, target=target, weight=weight, criterion=criterion)
    print(f'pred (train): {list(pred.shape)}')
    print(f'loss: {loss.item():.4f}')
    print(f'losses_vis: {losses_vis}')
    loss.backward()
    print('backward OK')

    # ── EMA input test ──
    imgs = np.random.rand(18, 160, 160).astype(np.float32)
    ema_imgs, rule = make_ema_input(imgs)
    print(f'ema_imgs: {ema_imgs.shape}, rule: {rule}')
