"""
CoDetectionCNN -- two-stream 2D U-Net

Input:  [B, 2, H, W]  two consecutive slices as 2 channels
Output: (embedding1, embedding2)  each [B, emd, H, W]

Architecture:
  each slice encoded separately by inc, then by the first down level
  -> concatenated -> shared deep encoder (down[1..3])
  -> shared upper decoder (up1, up2)
  -> separate decoders for the last two levels (up3_t/up3_tn, up4_t/up4_tn)
  -> separate output heads (outc_t, outc_tn)

Source: CAD/scripts_2_5d_3d/CoDetectionCNN.py + network_parts.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 2D building blocks (from network_parts.py)
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
    """Two-stream 2D U-Net

    Two slices (slice_z, slice_z+1) are fed as 2 input channels.
    Shallow levels are encoded separately, mid levels are concatenated and
    encoded jointly, and the deep decoder splits again to produce one
    embedding map per slice.

    Args:
        n_channels:    channels per slice (default 1)
        n_classes:     output embedding dimension (default 16)
        filter_channel: base filter count (default 16)
    """
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16):
        super().__init__()
        fc = filter_channel

        # shared by both slices
        self.inc = Inconv(n_channels, fc)

        # down[0]: applied per slice; down[1..3]: shared after concatenation
        self.down = nn.ModuleList([Down(fc, fc * 2)])
        self.down.append(Down(fc * 4, fc * 4))     # input = cat(t_enc1, tn_enc1)
        self.down.append(Down(fc * 4, fc * 8))
        self.down.append(Down(fc * 8, fc * 8))

        # shared upper decoder
        self.up1 = Up(fc * 16, fc * 4)
        self.up2 = Up(fc * 8, fc * 2)

        # separate lower decoders (per slice)
        self.up3_t  = Up(fc * 4, fc)
        self.up3_tn = Up(fc * 4, fc)
        self.up4_t  = Up(fc * 2, 32)
        self.up4_tn = Up(fc * 2, 32)

        self.outc_t  = Outconv(32, n_classes)
        self.outc_tn = Outconv(32, n_classes)

    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W]
        Returns:
            pred_t:  [B, emd, H, W]  embedding of slice z
            pred_tn: [B, emd, H, W]  embedding of slice z+1
        """
        x_inp1 = x[:, 0:1, :, :]   # slice z
        x_inp2 = x[:, 1::, :, :]   # slice z+1

        t_enc, tn_enc = [0] * 2, [0] * 2
        enc = [0] * 4

        # per-slice encoding
        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)

        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])

        # concatenate -> shared deep encoder
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        # shared upper decoder
        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        # separate decoders
        t_dec  = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec  = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t  = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn
