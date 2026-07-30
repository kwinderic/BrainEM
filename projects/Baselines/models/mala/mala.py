"""
mala.py -- MALA UNet3D model

config: MALA-BASE.yaml -> ARCHITECTURE: 'mala' -> UNet3D_MALA

Dependencies:
  UNet3D_MALA
    └── nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d

Usage:
    from models.mala.mala import UNet3D_MALA
    model = UNet3D_MALA(output_nc=3, if_sigmoid=True, init_mode='kaiming')
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ..model import *


# =============================================================================
# UNet3D_MALA
# Paper: Large Scale Image Segmentation with Structured Loss based Deep Learning
#        for Connectome Reconstruction (MALA)
# =============================================================================

@register_model("mala")
class UNet3D_MALA(nn.Module):
    """3-level UNet (MALA style) for 3D affinity prediction

    Input:  (B, 1, D, H, W)   recommended size: (B, 1, 84, 268, 268)
    Output: (B, out_planes, D', H', W')   z/y/x affinity maps

    Key args:
        in_planes     : input channels, default 1
        out_planes    : output channels, default 3 (z/y/x affinity)
        if_sigmoid    : whether to apply sigmoid to the output
        init_mode     : 'kaiming' | 'xavier' | 'orthogonal'
    """
    def __init__(self,
                 in_planes=1,
                 out_planes=3,
                 if_sigmoid=False,
                 init_mode='kaiming',
                 show_feature=False,
                 **kwargs):
        super().__init__()
        self.if_sigmoid = if_sigmoid
        self.init_mode = init_mode
        self.show_feature = show_feature

        # encoder
        self.conv1 = nn.Conv3d(in_planes, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.conv5 = nn.Conv3d(60, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv6 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        # bottleneck
        self.conv7 = nn.Conv3d(300, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv8 = nn.Conv3d(1500, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # decoder
        self.dconv1 = nn.ConvTranspose3d(1500, 1500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1500, bias=False)
        self.conv9 = nn.Conv3d(1500, 300, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv10 = nn.Conv3d(600, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv11 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.dconv2 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300, bias=False)
        self.conv12 = nn.Conv3d(300, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60, bias=False)
        self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # output head
        self.conv18 = nn.Conv3d(12, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if self.init_mode == 'kaiming':
                    init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
                elif self.init_mode == 'xavier':
                    init.xavier_normal_(m.weight)
                elif self.init_mode == 'orthogonal':
                    init.orthogonal_(m.weight)
                else:
                    raise AttributeError('No this init mode!')

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[3] - upsampled.size()[3]) // 2
            cc = (bypass.size()[2] - upsampled.size()[2]) // 2
            assert(c > 0)
            assert(cc > 0)
            bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
        return torch.cat((upsampled, bypass), 1)

    def _forward_impl(self, x):
        """Network forward pass, returns the prediction."""
        # encoder
        conv1 = F.leaky_relu(self.conv1(x), 0.005)
        conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
        pool1 = self.pool1(conv2)
        conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
        conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
        pool2 = self.pool2(conv4)
        conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
        conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
        pool3 = self.pool3(conv6)
        conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
        conv8 = F.leaky_relu(self.conv8(conv7), 0.005)

        # decoder
        dconv1 = self.dconv1(conv8)
        conv9 = self.conv9(dconv1)
        mc1 = self.crop_and_concat(conv9, conv6, crop=True)
        conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
        conv11 = F.leaky_relu(self.conv11(conv10), 0.005)

        dconv2 = self.dconv2(conv11)
        conv12 = self.conv12(dconv2)
        mc2 = self.crop_and_concat(conv12, conv4, crop=True)
        conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
        conv14 = F.leaky_relu(self.conv14(conv13), 0.005)

        dconv3 = self.dconv3(conv14)
        conv15 = self.conv15(dconv3)
        mc3 = self.crop_and_concat(conv15, conv2, crop=True)
        conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
        conv17 = F.leaky_relu(self.conv17(conv16), 0.005)

        output = self.conv18(conv17)
        if self.if_sigmoid:
            output = torch.sigmoid(output)
        if self.show_feature:
            return conv8, conv11, conv14, conv17, output
        return output

    @staticmethod
    def _center_crop(tensor, target_size):
        """Center-crop the spatial dims of tensor to target_size (D, H, W)."""
        _, _, d, h, w = tensor.shape
        td, th, tw = target_size
        sd = (d - td) // 2
        sh = (h - th) // 2
        sw = (w - tw) // 2
        return tensor[:, :, sd:sd+td, sh:sh+th, sw:sw+tw]

    def forward(self, inputs, target=None, weight=None, criterion=None):
        pred = self._forward_impl(inputs)
        if criterion is None:      # inference mode
            return pred
        # Training mode: target/weight are at INPUT_SIZE while pred is at OUTPUT_SIZE,
        # so target/weight must be center-cropped to align with pred.
        out_size = pred.shape[2:]   # (D', H', W')
        if isinstance(target, (list, tuple)):
            target = [self._center_crop(t, out_size) for t in target]
        else:
            target = self._center_crop(target, out_size)
        loss, losses_vis = criterion(pred, target, weight)
        return pred, loss, losses_vis


if __name__ == '__main__':
    import numpy as np
    model = UNet3D_MALA(if_sigmoid=True, init_mode='kaiming').to('cuda:0')
    x = torch.tensor(np.random.random((1, 1, 84, 268, 268)).astype(np.float32)).to('cuda:0')
    out = model(x)
    print(f'in: {list(x.shape)}  ->  out: {list(out.shape)}')
