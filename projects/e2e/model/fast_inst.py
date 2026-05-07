import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from connectomics.model.utils import model_init
from connectomics.model.arch import UNet3D
from connectomics.model.block import conv3d_norm_act
from connectomics.model.block import *
from connectomics.model.utils.misc import get_norm_3d, get_norm_1d

from .utils import QueryProposal, \
    CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP
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


class FastInstDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim: int = 64,
            num_queries: int = 100,
            num_aux_queries: int = 8,
            nheads: int = 8,
            dim_feedforward: int = 64,
            # dec_layers: int = 3,
            # dec_layers: int = 2,
            dec_layers: int = 1,
            pre_norm: bool = False,
            mask_dim: int = 64
    ) -> None:
        super().__init__()

        num_classes = 1

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None

        meta_pos_size = int(round(math.sqrt(self.num_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size, meta_pos_size))
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)

        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes=num_classes)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()
        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

        self.decoder_query_norm_layers = nn.ModuleList()
        # self.class_embed_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
            # self.class_embed_layers.append(MLP(hidden_dim, hidden_dim, num_classes + 1, 3))
            self.mask_embed_layers.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            self.mask_features_layers.append(nn.Linear(hidden_dim, mask_dim))


    def forward(self, x):
        # features: shape N, C, D, H, W

        # TODO: use multi-scale features
        # bs = x[0].shape[0]
        # proposal_size = x[1].shape[-2:]
        # pixel_feature_size = x[2].shape[-2:]

        bs = x.shape[0]
        proposal_size = pixel_feature_size = x.shape[-3:]

        # generate positional embedding
        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="trilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="trilinear", align_corners=False)

        # flatten the image features
        pixel_features = x.flatten(2).permute(2, 0, 1)      # N, C, DHW -> DHW, N, C
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        # generate query features
        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x, proposal_pos_embeds
        )
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        # prediction
        outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1, return_attn_mask=True
        )
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]

        # multi-layer predictions
        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i
            )
            outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
                query_features, pixel_features, pixel_feature_size, i, return_attn_mask=True,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        # TODO: guided predictions

        output = {
            'proposal_cls_logits': proposal_cls_logits,
            'query_locations': query_locations,
            "pred_masks": predictions_mask[-1],
            "pred_scores": predictions_class[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, query_locations
            )
        }
        return output


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, output_query_locations):
        return [
            {
                "query_locations": output_query_locations,
                "pred_logits": a,
                "pred_masks": b,
            }
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds, pos=query_pos_embeds
        )
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, pixel_features, memory_mask=attn_mask, query_pos=query_pos_embeds, pos=pixel_pos_embeds
        )
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds
        )
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features


    def forward_prediction_heads(self, query_features, pixel_features, pixel_feature_size, idx_layer,
                                 return_attn_mask=False, return_gt_attn_mask=False,
                                 targets=None, query_locations=None):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](query_features[:self.num_queries])
        decoder_query_features = decoder_query_features.transpose(0, 1)
        # if self.training or idx_layer + 1 == self.num_layers:
        #     outputs_class = self.class_embed_layers[idx_layer + 1](decoder_query_features)
        # else:
        #     outputs_class = None
        outputs_class = None
        outputs_mask_embed = self.mask_embed_layers[idx_layer + 1](decoder_query_features)
        outputs_mask_features = self.mask_features_layers[idx_layer + 1](pixel_features.transpose(0, 1))

        outputs_mask = torch.einsum("bqc,blc->bql", outputs_mask_embed, outputs_mask_features)
        outputs_mask = outputs_mask.reshape(-1, self.num_queries, *pixel_feature_size)

        if return_attn_mask:
            # outputs_mask.shape: b, q, d, h, w
            attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, 0, 0, self.num_aux_queries), "constant", 1)
            attn_mask = (attn_mask < 0.).flatten(2)  # b, q, dhw
            invalid_query = attn_mask.all(-1, keepdim=True)  # b, q, 1
            attn_mask = (~ invalid_query) & attn_mask  # b, q, dhw
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        matching_indices = None
        gt_attn_mask = None

        return outputs_class, outputs_mask, attn_mask, matching_indices, gt_attn_mask




class FastUNet3D(UNet3D, E2EMixin):
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
        self.inst_decoder = FastInstDecoder(fusion_channel)


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


        output = self.inst_decoder(x)


        # # head
        # coord_features = self.compute_coordinates(x)
        # x = torch.cat([coord_features, x], dim=1)
        # pred_kernel, pred_scores, pred_match, iam = self.inst_branch(x)
        # mask_features = self.mask_branch(x)     # B, C, D, H, W

        # mask_features = F.normalize(mask_features, p=2, dim=1)
        # pred_kernel = self.norm_kernel(pred_kernel.transpose(1,2)).transpose(1,2)

        # # pred_kernel: 1, 100, 64;  pred_scores: 1, 100, 1;  iam: 1, 100, 17, 129, 129
        # # mask_features: 1, 64, 17, 129, 129

        # N = pred_kernel.shape[1]
        # # mask_features: BxCxDxHxW
        # B, C, D, H, W = mask_features.shape
        # pred_masks = torch.bmm(pred_kernel, 
        #         mask_features.view(B, C, D * H * W)
        #     ).view(B, N, D, H, W)

        # pred_masks = self.norm_logits(pred_masks.unsqueeze(dim=1)).squeeze(dim=1)

        # # pred_masks = BN

        # # pred_masks = F.interpolate(
        # #     pred_masks, scale_factor=self.scale_factor,
        # #     mode='bilinear', align_corners=False)

        # output = {
        #     "pred_masks": pred_masks,
        #     "pred_scores": pred_scores,
        #     "pred_match": pred_match
        # }

        return output
