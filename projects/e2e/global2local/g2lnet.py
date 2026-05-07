import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributed as dist

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from connectomics.model.utils import model_init
from connectomics.model.arch import UNet3D
from connectomics.model.block import conv3d_norm_act
from connectomics.model.block import *
from connectomics.model.utils.misc import get_norm_3d, get_norm_1d

from ..model.sparse_unet_v2_old import SparseUNet3DV2



class G2LNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.global_net = SparseUNet3DV2()
        self.local_net  = SparseUNet3DV2()


    def forward(self, x):

        # global path: x -> global output, global queries, global features

        # local path: x, global queries -> local output, local queries, local features

        # return global/local outputs, global/local features, global/local queries

        if self.training:
            return {"global": output_g, "local": output_l}

        return 
