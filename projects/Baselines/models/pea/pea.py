"""
pea.py — python main.py -c=ac3ac4 链路所需全部代码（模型 + 损失 + EMA 输入）

config: ac3ac4.yaml → model_type: 'pea' → UNet_PNI_embedding_deep

依赖关系:
  UNet_PNI_embedding_deep
    └── resBlock_pni
          └── conv3dBlock ─── getConv3d, getBN, getRelu, init_conv
    └── conv3dBlock
    └── upsampleBlock ──────── init_conv

  损失函数:
    embedding_loss_norm5     ← SCM: 12 通道多尺度余弦亲和力
    embedding_loss_norm1     ← EPM: 3 通道短程余弦亲和力（中间层监督）
    ema_embedding_loss_norm5 ← CCM: 12 通道交叉亲和力
    convert_consistency_flip ← CCM: ema_embedding 坐标对齐
    WeightedMSE              ← 带权重 MSE criterion

  EMA 输入生成:
    make_ema_input           ← 三步串联: 亮度扰动 → cutout → 翻转

使用:
    from model.pea import UNet_PNI_embedding_deep
    model = UNet_PNI_embedding_deep(
        in_planes=1, out_planes=12, filters=[28,36,48,64,80],
        upsample_mode='bilinear', merge_mode='add', emd=16)
    embedding = model(x)  # 推理: x: [B,1,18,160,160] → [B,emd,18,160,160]
    pred, loss, vis = model(x, target, weight, criterion)  # 训练: 框架标准调用

训练总损失:
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
# 损失函数
# =============================================================================



class WeightedMSE(nn.Module):
    """带权重的 MSE，所有 loss 的底层 criterion

    L = Σ[ weight * (pred - target)² ] / (B * spatial_size)
    调用: criterion(pred, target, weightmap)
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
# 内部辅助：单方向单偏移量的亲和力损失
# -----------------------------------------------------------------------------

def _single_offset_loss(embedding, order, shift, target, weightmap, criterion):
    """计算一个 (方向, 偏移) 对的亲和力和损失

    方向由 order % 3 决定: 0 → z 轴, 1 → y 轴, 2 → x 轴
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
    """交叉流版本：embedding 和 ema_embedding 之间的点积亲和力"""
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
# SCM: 12 通道多尺度余弦亲和力损失（最终 embedding）
# -----------------------------------------------------------------------------

def embedding_loss_norm5(embedding, target, weightmap, criterion,
                         affs0_weight=1, shift=1, fill=True):
    """12 通道多尺度亲和力损失

    shifts = [1,1,1, 2, 3,3,3, 9,9, 4, 27,27]
    channels 0-2  (shift=1): z/y/x 短程  → weight * affs0_weight
    channels 3-11 (shift>1): 长程偏移    → weight 1

    返回: (scalar loss, affs tensor 同 target 形状)
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
# EPM: 3 通道短程余弦亲和力损失（中间层 emd1–emd4 监督）
# -----------------------------------------------------------------------------

def embedding_loss_norm1(embedding, target, weightmap, criterion,
                         affs0_weight=1, shift=1, fill=True):
    """3 通道亲和力损失 (z/y/x, shift=1)

    用于中间解码层输出 (emd1–emd4)，对应下采样的亲和力标签。
    返回: (scalar loss, affs tensor 同 target 形状)
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
# CCM: 坐标对齐（将 ema_embedding 逆翻转到原始坐标系）
# -----------------------------------------------------------------------------

def _augment_reverse_torch(data, rule):
    """对单个样本的 embedding tensor [C, D, H, W] 做逆翻转/转置

    与 simple_augment 的操作完全相反：
      simple_augment 顺序:  z → x → y → xy转置
      本函数逆序:           xy转置 → y → x → z

    rule: 长度 4 的数组，rule[i] ∈ {0, 1}
      rule[0] → z轴翻转   rule[1] → x轴翻转
      rule[2] → y轴翻转   rule[3] → xy转置
    """
    assert len(data.shape) == 4   # [C, D, H, W]
    if rule[3]: data = data.permute(0, 1, 3, 2)   # 逆 xy-transpose
    if rule[2]: data = torch.flip(data, [2])        # 逆 y-flip
    if rule[1]: data = torch.flip(data, [3])        # 逆 x-flip
    if rule[0]: data = torch.flip(data, [1])        # 逆 z-flip
    return data


def convert_consistency_flip(ema_embedding, rules):
    """将 batch 内每个样本的 ema_embedding 逆翻转，对齐到原始坐标系

    ema_imgs 经过随机翻转后送入 model，输出的 ema_embedding 空间坐标
    也随之翻转。本函数按 rule 逆序还原，使两路 embedding 坐标一致。

    输入:
      ema_embedding  [B, C, D, H, W]  GPU tensor
      rules          [B, 4]           GPU tensor，来自 DataLoader
    输出:
      aligned        [B, C, D, H, W]  坐标已对齐的 ema_embedding
    """
    B = ema_embedding.shape[0]
    ema_embedding = ema_embedding.detach().clone()
    rules_np = rules.data.cpu().numpy().astype(np.uint8)
    out = []
    for k in range(B):
        out.append(_augment_reverse_torch(ema_embedding[k], rules_np[k]))
    return torch.stack(out, dim=0)


# -----------------------------------------------------------------------------
# CCM: 12 通道交叉亲和力损失（EMA 一致性）
# -----------------------------------------------------------------------------

def ema_embedding_loss_norm5(embedding, ema_embedding, target, weightmap, criterion,
                              affs0_weight=1, shift=1, fill=True):
    """EMA 一致性损失：两路 embedding 的交叉 12 通道亲和力

    通过计算原始 embedding 和 EMA 增强 embedding 的点积亲和力，
    与相同的 ground-truth 标签比较，强制两路输出一致。

    返回: (scalar loss, affs tensor 同 target 形状)
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
# EMA 输入生成（数据层，CPU 端运行）
# =============================================================================

class IntensityAugment:
    """对比度 + 亮度随机扰动

    让 EMA 路输入与原始输入在光度上不同，
    但同一位置像素的语义（属于哪个神经元）不变。
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
    """随机 cutout 遮挡，生成二值 mask（0=遮挡，1=保留）

    输入: imgs  [D, H, W]  numpy float32
    输出: mask  [D, H, W]  numpy float32，0 或 1
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
    """对 numpy 3D 数据 [D, H, W] 按 rule 做翻转/转置

    rule: 长度 4 的二值数组
      rule[0]=1 → z 轴翻转   rule[1]=1 → x 轴翻转
      rule[2]=1 → y 轴翻转   rule[3]=1 → xy 轴互换（转置）

    rule 必须随 batch 一起保存并传入训练循环，
    供 convert_consistency_flip 做逆变换对齐坐标系。
    """
    assert data.ndim == 3
    if rule[0]: data = data[::-1, :, :]
    if rule[1]: data = data[:, :, ::-1]
    if rule[2]: data = data[:, ::-1, :]
    if rule[3]: data = np.transpose(data, (0, 2, 1))
    return data


class Filp_EMA:
    """封装随机翻转：生成 rule 并对 ema_imgs 做翻转

    调用: ema_imgs, rule = Filp_EMA()(ema_imgs)
    输入: data  [D, H, W]  numpy float32
    输出: (augmented_data [D,H,W],  rule [4,] uint8)
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
    """三步串联生成 EMA 路输入

    原始 imgs
      → IntensityAugment   亮度/对比度扰动
      → gen_mask           随机 cutout 遮挡
      → Filp_EMA           随机翻转 + 记录 rule
      → ema_imgs, rule

    输入: imgs  [D, H, W]  numpy float32，值域 [0,1]
    输出:
      ema_imgs  [D, H, W]  numpy float32
      rule      [4,]       numpy uint8，全 0 表示未做翻转
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
# 工具函数
# =============================================================================




def init_conv(m, init_mode):
    """卷积层权重初始化"""
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
    """激活函数工厂: 'relu' | 'elu' | 'leaky<slope>'"""
    if mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode[:5] == 'leaky':
        return nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))
    raise ValueError('Unknown relu mode: ' + mode)


def getBN(out_planes, bn_mode='async', bn_momentum=0.1):
    """3D BatchNorm 工厂: 'async'（标准 BN）| 'sync'（此处用标准 BN 替代）"""
    return nn.BatchNorm3d(out_planes, momentum=bn_momentum)


def getConv3d(in_planes, out_planes, kernel_size, stride, padding,
              bias, pad_mode='zero', init_mode='', dilation_size=(1, 1, 1)):
    """3D 卷积层构建，支持 zero / replicate padding"""
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
    """VGG 风格 3D 卷积块：可堆叠 [Conv → BN → ReLU]"""
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
    """3D 上采样块
    mode:
      'bilinear'   — trilinear 插值 + 1x1x1 conv
      'nearest'    — nearest 插值 + 1x1x1 conv
      'transpose'  — 转置卷积（密集）
      'transposeS' — 深度可分离转置卷积（稀疏，推荐）
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
# 残差块
# =============================================================================

class resBlock_pni(nn.Module):
    """PNI 残差块（各向异性 EM 图像专用）

    结构:
      block1: 1×3×3 conv+BN+ReLU  （维度对齐）
      block2: 3×3×3 conv+BN+ReLU → 3×3×3 conv+BN  （残差两层）
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
# 论文: Pixel Embedded Affinity
# 基于 Superhuman UNet，增加深层 embedding 监督（5 路输出）
# =============================================================================

@register_model("pea")
class UNet_PNI_embedding_deep(nn.Module):
    """5 层 UNet + PNI 残差块 + 深层 embedding 监督

    输入: (B, 1, D, H, W)   推荐尺寸: (B, 1, 18, 160, 160)
    输出: (emd1, emd2, emd3, emd4, embedding)
      emd1  ← center 层投影 → emd 维
      emd2  ← conv4（第 1 解码层）投影
      emd3  ← conv5（第 2 解码层）投影
      emd4  ← conv6（第 3 解码层）投影
      embedding ← 最终输出投影

    关键参数 (对应 ac3ac4.yaml → MODEL):
        filters       : 各层通道数，默认 [28,36,48,64,80]
        upsample_mode : 'bilinear' | 'transposeS' 等
        merge_mode    : 'add'（相加跳接）| 'cat'（拼接跳接）
        pad_mode      : 'zero' | 'replicate'
        bn_mode       : 'async'（标准 BN）
        relu_mode     : 'elu' | 'relu' | 'leaky<slope>'
        init_mode     : 'kaiming_normal' 等
        emd           : embedding 投影维度，默认 16
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

        # 输入嵌入：1×5×5（各向异性处理，z 方向不卷）
        self.embed_in = conv3dBlock(
            [in_planes], [f[0]], [(1, 5, 5)], [1], [(0, 2, 2)],
            [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)

        # 编码器
        self.conv0 = resBlock_pni(f[0], f[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv1 = resBlock_pni(f[1], f[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = resBlock_pni(f[2], f[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv3 = resBlock_pni(f[3], f[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # 瓶颈
        self.center = resBlock_pni(f[4], f[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # 解码器（4 层对称）
        self.up0, self.cat0, self.conv4 = self._dec_block(f[5], f[4], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up1, self.cat1, self.conv5 = self._dec_block(f[4], f[3], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up2, self.cat2, self.conv6 = self._dec_block(f[3], f[2], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up3, self.cat3, self.conv7 = self._dec_block(f[2], f[1], upsample_mode, merge_mode, pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # 输出嵌入：1×5×5
        self.embed_out = conv3dBlock(
            [f[0]], [f[0]], [(1, 5, 5)], [1], [(0, 2, 2)],
            [True], [pad_mode], [''], [relu_mode], init_mode, bn_momentum)

        # 输出投影头：各层 → emd 维
        self.out_put  = conv3dBlock([f[0]], [emd], [(1, 1, 1)], init_mode=init_mode)  # 最终 embedding
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
        """UNet 前向传播，返回 (emd1, emd2, emd3, emd4, embedding)"""
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

        embedding = self.out_put(embed_out)   # 最终 embedding
        emd1 = self.out_put1(ctr)             # center 层
        emd2 = self.out_put2(d0)              # 第 1 解码层
        emd3 = self.out_put3(d1)              # 第 2 解码层
        emd4 = self.out_put4(d2)              # 第 3 解码层

        return emd1, emd2, emd3, emd4, embedding

    def _make_ema_input_torch(self, volume):
        """GPU 端 EMA 输入生成：亮度扰动 + cutout + 随机翻转

        Args:
            volume: [B, 1, D, H, W] GPU tensor
        Returns:
            ema_volume: [B, 1, D, H, W]
            rules: [B, 4] tensor
        """
        B = volume.shape[0]
        ema = volume.clone()

        # 1. 亮度扰动
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

        # 3. 随机翻转 + 记录 rules
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
        """将 embedding 转换为 3 通道短程亲和力图（推理用）

        对 L2 归一化后的 embedding，沿 z/y/x 轴以 shift=1 计算余弦相似度。
        返回: Tensor [B, 3, D, H, W]
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
        # 边界填充：将 shift 导致的零值边界用邻近值填充，避免推理时出现小孔洞
        affs[:, 0:1, :1, :, :] = affs[:, 0:1, 1:2, :, :]   # z 边界
        affs[:, 1:2, :, :1, :] = affs[:, 1:2, :, 1:2, :]   # y 边界
        affs[:, 2:3, :, :, :1] = affs[:, 2:3, :, :, 1:2]   # x 边界
        affs = F.relu(affs)                                   # 确保非负
        return affs

    def forward(self, inputs, target=None, weight=None, criterion=None):
        """前向传播 + 损失计算（适配框架标准调用接口）

        测试模式 (criterion=None):
            return pred                     # Tensor [B, 12, D, H, W] 多尺度亲和力

        训练模式 (criterion!=None):
            return pred, loss, losses_vis   # 与 return_loss 接口一致

        框架传入:
            target: List[4 x Tensor[B,3,D,H,W]]  多尺度亲和力标签
            weight: List[4 x List[1 x Tensor[B,3,D,H,W]]]  权重图
            criterion: 框架 Criterion（忽略，内部使用 WeightedMSE）
        """
        # ── 主路前向 ──
        emd1, emd2, emd3, emd4, embedding = self._forward_features(inputs)

        if criterion is None:
            return self._embedding_to_affinity(embedding)

        _criterion = WeightedMSE()

        # ── 解包 target/weight ──
        # target: List[4 x Tensor[B,3,D,H,W]] → Tensor[B,12,D,H,W]
        aff_target = torch.cat([t.to(inputs.device) for t in target], dim=1)
        # weight: List[4 x List[1 x Tensor[B,3,D,H,W]]] → Tensor[B,12,D,H,W]
        aff_weight = torch.cat([w[0].to(inputs.device) for w in weight], dim=1)

        # ── SCM: 12 通道多尺度亲和力损失（最终 embedding）──
        loss_emb, affs_emb = embedding_loss_norm5(embedding, aff_target, aff_weight, _criterion)

        # ── CCM: EMA 一致性损失 ──
        ema_volume, rules = self._make_ema_input_torch(inputs)
        _, _, _, _, ema_embedding = self._forward_features(ema_volume)
        ema_embedding = convert_consistency_flip(ema_embedding, rules)
        loss_cross, _ = ema_embedding_loss_norm5(
            embedding, ema_embedding, aff_target, aff_weight, _criterion)

        # ── EPM: 多尺度中间层监督 ──
        # 从 shift=1 的 target 和 weight 下采样生成 down1-4
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

        # ── 总损失 ──
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
# 快速测试
# =============================================================================

if __name__ == '__main__':
    # ── 模型测试（推理模式）──
    x = torch.randn(1, 1, 18, 160, 160)
    model = UNet_PNI_embedding_deep(
        in_planes=1, out_planes=12,
        filters=[28, 36, 48, 64, 80],
        upsample_mode='bilinear', merge_mode='add', emd=16)
    pred = model(x)
    print(f'pred (inference): {list(pred.shape)}')

    # ── 训练模式测试（模拟框架调用）──
    # 模拟 DataLoader 产出的 target 和 weight
    target = [torch.randn(1, 3, 18, 160, 160) for _ in range(4)]
    weight = [[torch.ones(1, 3, 18, 160, 160)] for _ in range(4)]
    criterion = WeightedMSE()  # 占位，forward 内部会用自己的

    pred, loss, losses_vis = model(x, target=target, weight=weight, criterion=criterion)
    print(f'pred (train): {list(pred.shape)}')
    print(f'loss: {loss.item():.4f}')
    print(f'losses_vis: {losses_vis}')
    loss.backward()
    print('backward OK')

    # ── EMA 输入测试 ──
    imgs = np.random.rand(18, 160, 160).astype(np.float32)
    ema_imgs, rule = make_ema_input(imgs)
    print(f'ema_imgs: {ema_imgs.shape}, rule: {rule}')
