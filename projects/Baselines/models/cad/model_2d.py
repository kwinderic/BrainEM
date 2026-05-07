"""
CoDetectionCNN — 双流 2D U-Net

输入: [B, 2, H, W]  两张连续切片作为 2 通道
输出: (embedding1, embedding2)  各 [B, emd, H, W]

架构:
  每张切片各自通过 inc 编码 → 第一层 down 各自编码
  → cat 拼接 → 共享深层编码 (down[1..3])
  → 共享上层解码 (up1, up2)
  → 分离解码最后两层 (up3_t/up3_tn, up4_t/up4_tn)
  → 各自输出头 (outc_t, outc_tn)

来源: CAD/scripts_2_5d_3d/CoDetectionCNN.py + network_parts.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 2D 基础模块 (来自 network_parts.py)
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# CoDetectionCNN
# =============================================================================

class CoDetectionCNN(nn.Module):
    """双流 2D U-Net

    两张切片 (slice_z, slice_z+1) 作为 2 通道输入。
    低层分别编码，中层 cat 合并共享编码，高层解码再分开，
    产出两个 embedding 图（一张对应每个切片）。

    参数:
        n_channels:    每张切片的通道数 (默认 1)
        n_classes:     输出 embedding 维度 (默认 16)
        filter_channel: 基础滤波器数 (默认 16)
    """
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16):
        super().__init__()
        fc = filter_channel

        # 各切片独立的 inc
        self.inc = Inconv(n_channels, fc)

        # down[0]: 各切片独立下采样; down[1..3]: cat 后共享
        self.down = nn.ModuleList([Down(fc, fc * 2)])
        self.down.append(Down(fc * 4, fc * 4))     # 输入 = cat(t_enc1, tn_enc1)
        self.down.append(Down(fc * 4, fc * 8))
        self.down.append(Down(fc * 8, fc * 8))

        # 共享解码上层
        self.up1 = Up(fc * 16, fc * 4)
        self.up2 = Up(fc * 8, fc * 2)

        # 分离解码下层 (per slice)
        self.up3_t  = Up(fc * 4, fc)
        self.up3_tn = Up(fc * 4, fc)
        self.up4_t  = Up(fc * 2, 32)
        self.up4_tn = Up(fc * 2, 32)

        # 输出头
        self.outc_t  = Outconv(32, n_classes)
        self.outc_tn = Outconv(32, n_classes)

    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W]
        Returns:
            pred_t:  [B, emd, H, W]  — slice z 的 embedding
            pred_tn: [B, emd, H, W]  — slice z+1 的 embedding
        """
        x_inp1 = x[:, 0:1, :, :]   # slice z
        x_inp2 = x[:, 1::, :, :]   # slice z+1

        t_enc, tn_enc = [0] * 2, [0] * 2
        enc = [0] * 4

        # 各自编码第一层
        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)

        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])

        # cat 合并 → 共享深层编码
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        # 共享解码上层
        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        # 分离解码
        t_dec  = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec  = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t  = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn
