import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributed as dist
# from typing import override
from typing_extensions import override

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from connectomics.model.utils import model_init
from connectomics.model.arch import UNet3D
from connectomics.model.block import conv3d_norm_act
from connectomics.model.block import *
from connectomics.model.utils.misc import get_norm_3d, get_norm_1d

from interface import E2EMixin, AutoLossMixin

# Use MaskFormer head

from ..affinity_knet import (
    get_connected_components,
    get_zwise_connected_components,
    get_opened_connected_components,
    get_seeds,
    watershed,
    get_watershed_fragments,
    FusionBlock,
    SelfAttention,
    MLP,
    SelfAttentionBlock,
    ConnectedComponetsHead,
    Predictor,
    KernelUpdator,
    MaskBranch
)
from .transformer_predictor import TransformerPredictor


class InstDecoder(nn.Module):
    def __init__(self, 
            fusion_channel, 
            num_iter=2, 
            init_mask_method='connected_components', 
            num_masks=100, 
            num_learned_masks=100,
            return_init_mask=False,
            share_weights=False,
            bg_conv_as_bias=False,
            bg_conv_share=False,
            bg_conv_norm='bn'
        ):
        super().__init__()
        kernel_dim = 64

        # self.rpn = ConnectedComponetsHead(
        #     num_convs=4, num_masks=num_masks, kernel_dim=64, 
        #     in_filter=fusion_channel+3, out_filter=fusion_channel, 
        #     kernel_size=(1, 3, 3), padding=(0, 1, 1),
        #     init_mask_method=init_mask_method,
        #     num_learned_masks=num_learned_masks,
        #     return_init_mask=return_init_mask
        # )
        # self.num_iter = num_iter
        self.num_iter = 1
        
        self.share_weights = share_weights
        self.bg_conv_as_bias = bg_conv_as_bias
        self.bg_conv_share = bg_conv_share
        self.bg_conv_norm = bg_conv_norm
        
        assert not self.share_weights, "Not implemented yet"
        assert not bg_conv_as_bias

        self.bg_convs = nn.ModuleList(
            [conv3d_norm_act(fusion_channel, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_mode=bg_conv_norm) \
                # for _ in range(num_iter + 1)])          # norm_mode='bn'
                for _ in range(num_iter)])          # norm_mode='bn'

        # MaskFormer head
        dim = 64
        self.seg_head = TransformerPredictor(
            use_learnable_queries=True,
            in_channels = fusion_channel + 3,
            mask_classification = False,
            num_classes = 0, 
            hidden_dim = dim, 
            num_queries = 100, 
            nheads = 8,
            dropout = 0, 
            dim_feedforward = 2048,
            enc_layers = 0,
            dec_layers = 2, 
            # dec_layers = 1, 
            pre_norm = True,
            deep_supervision = True,
            mask_dim = dim, 
            enforce_input_project = False 
        )


    def forward(self, x, mask_features, pred_affinity, init_masks=None):
        mask_features = F.normalize(mask_features, p=2, dim=1)

        pred_masks_list = []

        out = self.seg_head(x, mask_features)

        for i in range(self.num_iter):
            bg_mask = self.bg_convs[i](mask_features)
            
            if i < self.num_iter - 1:
                pred_masks = out['aux_outputs'][i]['pred_masks']        # B, N, D, H, W
            else:
                pred_masks = out['pred_masks']

            pred_masks_list.append(torch.cat([bg_mask, pred_masks], dim=1))

        return pred_masks_list, [None], None    # pred_kernel_list=[]



class AffinityKNetMaskFormerOnlyLearn(UNet3D, E2EMixin):
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
                 with_affinity: bool = False,
                 num_iter: int = 2,
                 affinity_convs: int = 4,
                 affinity_for_mask: bool = False,
                 init_mask_method: str = 'connected_components',
                 num_masks: int = 100,
                 num_learned_masks: int = 100,
                 return_init_mask: bool = False,
                 aux_inst_decoder: bool = False,
                 with_agfp: bool = False,
                 inference_without_bg: bool = False,
                 inst_decoder_share_weights: bool = False,
                 inst_decoder_bg_conv_as_bias: bool = False,
                 inst_decoder_bg_conv_share: bool = False,
                 inst_decoder_bg_conv_norm: str = 'bn',
                 feed_gt_mask: bool = False,
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
        self.inst_decoder = InstDecoder(fusion_channel, num_iter, init_mask_method, num_masks, num_learned_masks, return_init_mask, 
                                        share_weights=inst_decoder_share_weights,
                                        bg_conv_as_bias=inst_decoder_bg_conv_as_bias,
                                        bg_conv_share=inst_decoder_bg_conv_share,
                                        bg_conv_norm=inst_decoder_bg_conv_norm)
        self.mask_branch = MaskBranch(
            num_convs=4, kernel_dim=64, 
            in_filter=fusion_channel+3, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )

        self.with_affinity = with_affinity
        if self.with_affinity:
            # self.affinity_branch = MaskBranch(
            #     num_convs=4, kernel_dim=3, 
            #     in_filter=fusion_channel+3, out_filter=fusion_channel, 
            #     kernel_size=(1, 3, 3), padding=(0, 1, 1),
            #     shared_kwargs=self.shared_kwargs
            # )
            self.affinity_branch = MaskBranch(
                num_convs=affinity_convs, kernel_dim=3, 
                in_filter=fusion_channel+3, out_filter=fusion_channel, 
                kernel_size=(3, 3, 3), padding=(1, 1, 1),
                shared_kwargs=self.shared_kwargs
            )

        self.affinity_for_mask = affinity_for_mask
        if self.affinity_for_mask:
            self.affinity_proj = conv3d_norm_act(fusion_channel+3, fusion_channel, 
                    kernel_size=(3, 3, 3), padding=(1, 1, 1), **self.shared_kwargs)

        self.aux_inst_decoder = aux_inst_decoder

        self.with_agfp = with_agfp
        if self.with_agfp:
            self.agfp = AGFP(fusion_channel+3, out_channel=fusion_channel+3, num_convs=2, kernel_size=3, num_iter=10, 
                            **self.shared_kwargs)
        self.inference_without_bg = inference_without_bg

        self.feed_gt_mask = feed_gt_mask


    @staticmethod
    def parse_config(cfg, kwargs):
        kwargs['num_iter'] = cfg.MODEL.NUM_ITER
        kwargs['with_affinity'] = cfg.MODEL.WITH_AFFINITY
        kwargs['affinity_convs'] = cfg.MODEL.AFFINITY_CONVS
        kwargs['affinity_for_mask'] = cfg.MODEL.AFFINITY_FOR_MASK
        kwargs['init_mask_method'] = cfg.MODEL.INIT_MASK_METHOD
        kwargs['num_masks'] = cfg.MODEL.NUM_MASKS
        kwargs['num_learned_masks']= cfg.MODEL.NUM_LEARNED_MASKS
        kwargs['return_init_mask']= cfg.MODEL.RETURN_INIT_MASK
        kwargs['aux_inst_decoder']= cfg.MODEL.AUX_INST_DECODER
        kwargs['with_agfp']= cfg.MODEL.WITH_AGFP
        kwargs['inference_without_bg']= cfg.MODEL.INFERENCE_WITHOUT_BG
        kwargs['inst_decoder_share_weights']= cfg.MODEL.INST_DECODER_SHARE_WEIGHTS
        kwargs['inst_decoder_bg_conv_share']= cfg.MODEL.INST_DECODER_BG_CONV_SHARE
        kwargs['inst_decoder_bg_conv_as_bias'] = cfg.MODEL.INST_DECODER_BG_CONV_AS_BIAS
        kwargs['inst_decoder_bg_conv_norm']= cfg.MODEL.INST_DECODER_BG_CONV_NORM
        kwargs['feed_gt_mask'] = cfg.MODEL.FEED_GT_MASK
        return kwargs


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


    def forward(self, x, label=None):

        # backbone
        feats = self._forward_backbone(x)
        x = self.fusion_block(feats)

        # feature extraction
        coord_features = self.compute_coordinates(x)
        x = torch.cat([coord_features, x], dim=1)

        # affinity prediction
        if self.with_affinity:
            pred_affinity = self.affinity_branch(x)
            # init_masks = get_connected_components(pred_affinity)        # N, D, H, W
        else:
            pred_affinity = None

        # AGFP
        if self.with_agfp:
            x = self.agfp(x)

        # instance segmentation
        mask_features = self.mask_branch(x)     # B, C, D, H, W

        if self.affinity_for_mask:
            pred_affinity_norm = pred_affinity.detach().sigmoid()
            pred_affinity_norm = pred_affinity_norm - 0.5
            mask_features = torch.cat([mask_features, pred_affinity_norm], dim=1)
            mask_features = self.affinity_proj(mask_features)

        pred_masks, pred_kernels, recorded_init_masks = self.inst_decoder(x, mask_features, pred_affinity)

        # if not self.training and self.inference_without_bg:
        if self.inference_without_bg:
            pred_masks[-1][0,0].zero_()
            print('zero bg mask for inference')

        output = {
            "pred_masks": pred_masks[-1],
            "pred_kernel": pred_kernels[-1],
            "pixel_feature": mask_features,
            "aux_outputs": [
                {'pred_masks': m, 'pred_kernel': k} \
                    for m, k in zip(
                        pred_masks[:-1], 
                        pred_kernels[:-1]
                    )
            ],
            "recorded_init_masks": recorded_init_masks      # None if not set
        }

        if self.with_affinity:
            output['pred_affinity'] = pred_affinity

        output['aux_groups'] = []
        # pred_masks[-1], shape B, N, D, H, W; required init_masks.shape B, D, H, W
        if self.aux_inst_decoder:
            pred_masks_aux, pred_kernels_aux, _ = \
                self.inst_decoder(x, mask_features, pred_affinity, init_masks=pred_masks[-1].argmax(1).detach())
            aux_group_output = {
                "pred_masks": pred_masks_aux[-1],
                "pred_kernel": pred_kernels_aux[-1],
                "aux_outputs": [
                    {'pred_masks': m, 'pred_kernel': k} \
                        for m, k in zip(
                            pred_masks_aux[:-1], 
                            pred_kernels_aux[:-1]
                        )
                ],
            }
            output['aux_groups'].append(aux_group_output)

        if self.training and self.feed_gt_mask:
            from scipy.ndimage import zoom
            H0, W0 = label.shape[-2:]
            H1, W1 = pred_masks[-1].shape[-2:]

            device = label.device
            resized_label = zoom(label.cpu().numpy(), (1, 1, H1/H0, W1/W0), order=0)
            resized_label = label.new_tensor(resized_label)

            pred_masks_aux, pred_kernels_aux, _ = \
                self.inst_decoder(x, mask_features, pred_affinity, init_masks=resized_label)
            aux_group_output = {
                "pred_masks": pred_masks_aux[-1],
                "pred_kernel": pred_kernels_aux[-1],
                "aux_outputs": [
                    {'pred_masks': m, 'pred_kernel': k} \
                        for m, k in zip(
                            pred_masks_aux[:-1], 
                            pred_kernels_aux[:-1]
                        )
                ],
            }
            output['aux_groups'].append(aux_group_output)

        return output
