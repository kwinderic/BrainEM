"""
models.py — python main.py -c=seg_3d 链路所需模型代码

config: seg_3d.yaml → model_type: 'superhuman' → UNet_PNI

依赖关系:
  UNet_PNI
    └── resBlock_pni
          └── conv3dBlock ─── getConv3d, getBN, getRelu, init_conv
    └── conv3dBlock
    └── upsampleBlock ──────── init_conv

使用:
    from model.models import UNet_PNI
    model = UNet_PNI(in_planes=1, out_planes=3, filters=[28,36,48,64,80])
"""

import torch
import torch.nn as nn
from ..model import *

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
# UNet_PNI (superhuman)
# 论文: Superhuman Accuracy on the SNEMI3D Connectomics Challenge
# https://arxiv.org/abs/1706.00120
# =============================================================================

@register_model("superhuman")
class UNet_PNI(nn.Module):
    """5 层 UNet + PNI 残差块，用于 3D affinity 预测

    输入: (B, 1, D, H, W)   推荐尺寸: (B, 1, 18, 160, 160)
    输出: (B, 3, D, H, W)   z/y/x 三方向亲和力图，值域 [0,1]

    关键参数 (对应 seg_3d.yaml → MODEL):
        filters       : 各层通道数，默认 [28,36,48,64,80]
        upsample_mode : 'bilinear' | 'transposeS' 等
        merge_mode    : 'add'（相加跳接）| 'cat'（拼接跳接）
        pad_mode      : 'zero' | 'replicate'
        bn_mode       : 'async'（标准 BN）
        relu_mode     : 'elu' | 'relu' | 'leaky<slope>'
        init_mode     : 'kaiming_normal' 等
        if_sigmoid    : 输出是否过 sigmoid
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

        # 输出头：1×1×1 → out_planes
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
    # 预期: in: [1, 1, 18, 160, 160]  ->  out: [1, 3, 18, 160, 160]
