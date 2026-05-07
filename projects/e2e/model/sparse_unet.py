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

from interface import E2EMixin

class FusionBlock(nn.Module):
    def __init__(self, 
            in_filters=[28, 36, 48, 64, 80],
            out_level=0, 
            out_filter=64, 
            mid_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            shared_kwargs={}
        ):
        super().__init__()

        self.out_level = out_level
        self.shared_kwargs = shared_kwargs

        self.pre_convs = nn.ModuleList([
            conv3d_norm_act(f, mid_filter, kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs) \
            for f in in_filters
        ])
        self.post_convs = nn.Sequential(
            conv3d_norm_act(mid_filter, out_filter, kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs)
        )

    def forward(self, feats):
        target_size = feats[self.out_level].shape[2:]

        out = 0
        for i, x in enumerate(feats):
            x = self.pre_convs[i](x)
            x = F.interpolate(x, size=target_size, 
                    mode='trilinear', align_corners=True)
            out += x
        return self.post_convs(out)


class SelfAttentionBlock(nn.Module):
    # https://github1s.com/karpathy/nanoGPT/blob/HEAD/model.py#L31-L76

    def __init__(self, n_embd, n_head=8, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                            dropout_p=self.dropout if self.training else 0, 
                            is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class InstanceBranch(nn.Module):

    def __init__(self, 
            num_convs=4, 
            num_masks=100,
            kernel_dim=64,
            in_filter=64, 
            out_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            use_matchness=False,
            shared_kwargs={}
        ):
        super().__init__()

        self.shared_kwargs = shared_kwargs

        in_filters = [in_filter] + [out_filter] * (num_convs - 1)
        self.inst_convs = nn.Sequential(*[
            conv3d_norm_act(f, out_filter, kernel_size=kernel_size,
                    padding=padding, **self.shared_kwargs) \
            for f in in_filters
        ])

        # iam prediction, a simple conv
        self.iam_convs = conv3d_norm_act(out_filter, num_masks, 
                kernel_size=kernel_size, padding=padding, 
                **self.shared_kwargs)

        # outputs
        self.mask_kernel = nn.Linear(out_filter, kernel_dim)
        # self.objectness = nn.Linear(out_filter, 1)
        if use_matchness:
            self.matchness = nn.Sequential(
                SelfAttentionBlock(out_filter),
                nn.Linear(out_filter, 1)
            )

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):        # TODO
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            # init.constant_(module.bias, bias_value)
        # init.normal_(self.iam_conv.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_convs(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))   # BxNx(HW) * Bx(HW)xC = BxNxC
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None]
        # predict classification & segmentation kernel & objectness
        pred_kernel = self.mask_kernel(inst_features)

        if hasattr(self, 'objectness'):
            pred_scores = self.objectness(inst_features)
        else:
            pred_scores = None
        if hasattr(self, 'matchness'):
            pred_match = self.matchness(inst_features)
        else:
            pred_match = None
        return pred_kernel, pred_scores, pred_match, iam



class MaskBranch(nn.Module):

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



class SparseUNet3D(UNet3D, E2EMixin):
    """
    Image -> UNet3D -> Fusion -> inst_convs -> [iam] -> [kernel], [score]
                           \                               \ 
                            \--> mask_convs -------------> (*) --> [masks]
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
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 return_feats: Optional[list] = None,
                 use_matchness: bool = False,
                 **kwargs):
        super(UNet3D, self).__init__()

        self.depth = len(filters)
        self.do_return_feats = (return_feats is not None)
        self.return_feats = return_feats
        print(f"Return feature maps from 3D U-Net? {self.do_return_feats}")

        if is_isotropic:
            isotropy = [True] * self.depth
        assert len(filters) == len(isotropy)

        block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
                                padding=padding_io, **self.shared_kwargs)
        # self.conv_out = conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                                        # padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i-1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs),
                block(filters[i], filters[i], **self.shared_kwargs))
            self.down_layers.append(layer)

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j-1], kernel_size,
                                padding=padding, **self.shared_kwargs),
                block(filters[j-1], filters[j-1], **self.shared_kwargs)])
            self.up_layers.append(layer)

        # task branches
        fusion_channel = 64
        fusion_level = 1

        self.fusion_block = FusionBlock(
            filters, fusion_level, 
            fusion_channel, fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )

        # initialization
        model_init(self, mode=init_mode)

        self.shared_kwargs['norm_mode'] = 'none'
        self.inst_branch = InstanceBranch(
            num_convs=4, num_masks=100, kernel_dim=64, 
            in_filter=fusion_channel+3, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs,
            use_matchness=use_matchness
        )
        self.mask_branch = MaskBranch(
            num_convs=4, kernel_dim=64, 
            in_filter=fusion_channel+3, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )

        if dist.is_initialized():
            self.norm_logits = nn.SyncBatchNorm(1, eps=1e-3, momentum=0.01)
            self.norm_kernel = nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01)
        else:
            self.norm_logits = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
            self.norm_kernel = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)


    def _forward_backbone(self, x):
        x = self.conv_in(x)

        down_x = [None] * (self.depth-1)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
            down_x[i] = x

        out = []
        x = self.down_layers[-1](x)
        out.append(x)

        for j in range(self.depth-1):
            i = self.depth-2-j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)
            out.append(x)

        return out[::-1]    # P0, P1, P2, P3, P4
        # x = self.conv_out(x)
        # return x


    @torch.no_grad()
    def compute_coordinates(self, x):
        d, h, w = x.size(2), x.size(3), x.size(4)
        z_loc = torch.linspace(-1, 1, d, device=x.device)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        
        z_loc, y_loc, x_loc = torch.meshgrid(z_loc, y_loc, x_loc)
        z_loc = z_loc.expand([x.shape[0], 1, -1, -1, -1])
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1, -1])

        locations = torch.cat([z_loc, x_loc, y_loc], 1)
        return locations.to(x)


    def forward(self, x):
        # backbone
        feats = self._forward_backbone(x)
        x = self.fusion_block(feats)

        # head
        coord_features = self.compute_coordinates(x)
        x = torch.cat([coord_features, x], dim=1)
        pred_kernel, pred_scores, pred_match, iam = self.inst_branch(x)
        mask_features = self.mask_branch(x)     # B, C, D, H, W

        mask_features = F.normalize(mask_features, p=2, dim=1)
        pred_kernel = self.norm_kernel(pred_kernel.transpose(1,2)).transpose(1,2)

        # pred_kernel: 1, 100, 64;  pred_scores: 1, 100, 1;  iam: 1, 100, 17, 129, 129
        # mask_features: 1, 64, 17, 129, 129

        N = pred_kernel.shape[1]
        # mask_features: BxCxDxHxW
        B, C, D, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, 
                mask_features.view(B, C, D * H * W)
            ).unsqueeze(dim=1)              # B, 1, N, DHW

        pred_masks = self.norm_logits(pred_masks).view(B, N, D, H, W)

        # pred_masks = BN

        # pred_masks = F.interpolate(
        #     pred_masks, scale_factor=self.scale_factor,
        #     mode='bilinear', align_corners=False)

        output = {
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
            "pred_match": pred_match,
            "pixel_feature": mask_features
        }

        return output
