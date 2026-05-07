from ..model import *
from .fgnet import DepthwiseBlock3d,HelperMixin,EncoderFPN,DecoderFPN,FGNet

class EncoderFPNWITHSAM(nn.Module, HelperMixin):
    def __init__(self, block, filters, isotropy, pooling, shared_kwargs,fusion_channel):
        super(EncoderFPNWITHSAM, self).__init__()

        self.depth = len(filters)
        self.pooling = pooling
        self.shared_kwargs = shared_kwargs

        # encoding path
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        self.rho_convs = nn.ModuleList()
        self.beta_convs = nn.ModuleList()
        proj_conv_kwargs = deepcopy(shared_kwargs)
        proj_conv_kwargs["act_mode"] = "none"
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i-1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                block(filters[i], filters[i], **self.shared_kwargs)
            )
            self.layers.append(layer)

            downsample = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                        stride=stride, padding=padding, **self.shared_kwargs)
            )
            self.downsamples.append(downsample)

            conv = conv3d_norm_act(fusion_channel,filters[i], (1, 3, 3),
                            padding=(0, 1, 1), **proj_conv_kwargs)
            self.rho_convs.append(conv)
            conv = conv3d_norm_act(fusion_channel,filters[i], (1, 3, 3),
                            padding=(0, 1, 1), **proj_conv_kwargs)
            self.beta_convs.append(conv)
            

    def forward(self, x,sam_feats):
        feats = [None] * self.depth

        feats[0] = self.layers[0](self.downsamples[0](x))
        for i in range(1, self.depth):
            feats[i] = self.downsamples[i](feats[i-1])
            if i>=1 and i<=3:
                rho_feat = self.rho_convs[i](sam_feats[i-1])
                rho = self._upsample(rho_feat.sigmoid(), feats[i])

                beta_feat = self.beta_convs[i](sam_feats[i-1])
                beta = self._upsample(beta_feat, feats[i])
                feats[i] = rho*feats[i] + beta
            feats[i] = self.layers[i](
                feats[i]
            )
        # for i in range(len(feats)):
        #     print(f"encoder feature shapes:{feats[i].shape}")
        return feats
    def _upsample(self, x, y_refer):
        """Upsample feature map x to the size of y_refer.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y_refer.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x

class DecoderFPN(nn.Module, HelperMixin):
    """Decoder of U-Net with Local Attention (LA) mechanism."""

    def __init__(self, block, filters, isotropy, pooling, shared_kwargs):
        super(DecoderFPN, self).__init__()

        self.depth = len(filters)
        self.pooling = pooling
        self.shared_kwargs = shared_kwargs

        # disable all activation functions in projection layers
        proj_conv_kwargs = deepcopy(shared_kwargs)
        proj_conv_kwargs["act_mode"] = "none"

        # decoding path
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()

        norm_mode = 'ln'
        for i in range(self.depth-1):
            self.norms.append(get_norm_3d(norm_mode,filters[i]))
            self.convs.append(nn.Conv3d(filters[i+1],filters[i],1))

    def forward(self, x,y=None):
        feats = [None] * self.depth

        feats[-1] = x[-1]
        
        for i in range(self.depth-2,-1,-1):
            feats[i] = self.norms[i](
                self._upsample_add(self.convs[i](feats[i+1]),x[i])
            )
        # for i in range(len(feats)):
        #     print(f"decoder feature shapes:{feats[i].shape}")
        return feats[0]     # return the highest res. feature map


    def _upsample(self, x, y_refer):
        """Upsample feature map x to the size of y_refer.

        When pooling layer is used, the input size is assumed to be even,
        therefore :attr:`align_corners` is set to `False` to avoid feature
        mis-match. When downsampling by stride, the input size is assumed
        to be 2n+1, and :attr:`align_corners` is set to `True`.
        """
        align_corners = False if self.pooling else True
        x = F.interpolate(x, size=y_refer.shape[2:], mode='trilinear',
                          align_corners=align_corners)
        return x


        
@register_model("fgnetv2")
class FGNETV2(nn.Module,CfgMixin,HelperMixin):
    """FGNetV2新版本, 做了一些细节优化"""

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
                 sam2_config: str = 'tiny',
                 fusion_channel: int=64,
                 conv_in_channel:int=64,
                 is_freeze_encoder: bool = False,
                 image_size:int=256,
                 conv_after_fusion:bool = True,
                 fpn_setting:int=1,
                 fg_stage_num:int=3,
                 **kwargs):
        super().__init__()
        self.is_freeze_encoder = is_freeze_encoder

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        fusion_channel = 64

        block = DepthwiseBlock3d
        logging.warning(f'Force using DepthwiseBlock3d for DWUNet3D. {block_type} is ignored.')

        self.backbone_fpn = FPN(
            d_model = fusion_channel,
            backbone_channel_list = [256, 256, 256],
            kernel_size = 3,
            padding=1,
            fpn_interp_model = "bilinear",
            fuse_type='avg'
        )
        """Corase Decoder for SAM2"""
        
        self.coarse_decoder = DecoderFPN(block, filters=[fusion_channel]*3, isotropy=[True,True,True], pooling = pooling, shared_kwargs = self.shared_kwargs)
        self.coarse_head = conv3d_norm_act(fusion_channel, out_channel, (5,5,5), bias=True,
                        padding=(2,2,2), pad_mode=pad_mode, act_mode='none', norm_mode='none')

        """Build FGNet"""

        if is_isotropic:
            isotropy = [True] * self.depth

        self.pooling, self.blurpool = pooling, blurpool

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = [
            conv3d_norm_act(in_channel, filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs),
            conv3d_norm_act(filters[0], filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs)
        ]
        self.conv_in = nn.Sequential(*self.conv_in)
        
        # stages
        self.num_stages = fg_stage_num
        self.loss_weight = "avg"

        self.heads = nn.ModuleList()
        self.encoder_stages = nn.ModuleList()  # 编码器阶段
        self.decoder_stages = nn.ModuleList()  # 解码器阶段
        
        for i in range(self.num_stages):
            # 单独初始化编码器和解码器
            encoder = EncoderFPNWITHSAM(block, filters, isotropy, self.pooling, self.shared_kwargs, fusion_channel)
            decoder = DecoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs)
            
            self.encoder_stages.append(encoder)
            self.decoder_stages.append(decoder)
            self.heads.append(
                conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
            )

        

        model_init(self, mode=init_mode)

        """Build SAM2""" 
        net = "sam2aff.model.SAM2AFFTrain"
        self.sam2 = build_sam2aff(sam2_config_dict[sam2_config]["cfg"],
                                                         sam2_config_dict[sam2_config]["checkpoint"],net=net,
                                                         image_size = image_size) 
        self._freeze_sam2_other()
        
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()

    def _freeze_sam2_other(self):
            for param in self.sam2.parameters():
                param.requires_grad = False
            for param in self.sam2.image_encoder.parameters():
                param.requires_grad = True
            for name, param in self.named_parameters():
                if param.requires_grad is False:
                # if True:
                    print(f"{name}: requires_grad={param.requires_grad}")

    def _freeze_sam2_encoder(self):
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
            # if True:
                print(f"{name}: requires_grad={param.requires_grad}")

    def _input_resize(self,inputs,size=1024):
        # Step 1: resize to 1024x1024 for each slice along D
        B, C, D, H, W = inputs.shape
        inputs = inputs.view(B * D, 1, H, W)  # merge B and D to apply 2D interpolation
        inputs = F.interpolate(inputs, size=(size, size), mode='bilinear', align_corners=False)
        inputs = inputs.view(B, 1, D, size, size)  # reshape back
        return inputs

    def _input_transform(self, inputs):
        """
        inputs: Tensor of shape (B, 1, D, H, W), values in [0, 1]
        Returns: normalized RGB tensor of shape (B, 3, D, 1024, 1024)
        """
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)

        # Step 2: repeat grayscale to 3 channels
        inputs = inputs.repeat(1, 3, 1, 1, 1)  # shape: (B, 3, D, 1024, 1024)

        # Step 3: normalize
        img_mean = torch.tensor(img_mean, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        img_std = torch.tensor(img_std, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        
        return (inputs - img_mean) / img_std
    
    
    def forward(self, inputs, target=None, weight=None, criterion=None):

        
        B,C,D,H,W = inputs.shape

        """corase"""
        x = self._input_transform(inputs)
        
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        sam_feature = self.sam2.image_encoder(x)['backbone_fpn']
        sam_feature = self.backbone_fpn(sam_feature)
        sam_feature = [f.permute(1,0,2,3).unsqueeze(0) for f in sam_feature]

        x = self.coarse_decoder(sam_feature)
        x = self.coarse_head(x)
        
        
        corase_affinity = F.interpolate(x, size=(D,H,W), mode='trilinear', align_corners=False)
        
        """refine"""
        x = self.conv_in(inputs)

        stage_feats = []
        for i in range(self.num_stages):
            x = self.encoder_stages[i](x, sam_feature)  
            x = self.decoder_stages[i](x)  
            stage_feats.append(x)

        preds = []
        preds.append(corase_affinity)
        for i, feats in enumerate(stage_feats):
            # pred = self.heads[i](feats[0])
            pred = self.heads[i](feats)
            preds.append(pred)

        """calculate loss"""

        if criterion:
            list_loss, full_losses_vis = [], dict()

            for t in range(self.num_stages+1):
                loss, losses_vis = criterion(preds[t], target, weight)
                list_loss.append(loss)
                full_losses_vis.update({k+f"_iter{t}": v for k, v in losses_vis.items()})

            full_losses_vis.update(losses_vis)

            if self.loss_weight == "avg":
                loss = sum(list_loss) / len(list_loss)
            elif self.loss_weight == "sum":
                loss = sum(list_loss)
            elif self.loss_weight == "last":
                loss = list_loss[-1] + 0 * sum(list_loss[:-1])      #  avoid unused parameter warning
            else:
                raise NotImplementedError

            return preds[-1], loss, full_losses_vis

        return preds[-1]












@register_model("fgnet_local_refine")
class FGNETLOCALREFINE(SAM2AFF,HelperMixin):
    """Leveraging Query-Proposal to generate prompt for SAM2"""

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
                 sam2_config: str = 'tiny',
                 fusion_channel: int=64,
                 conv_in_channel:int=64,
                 is_freeze_encoder: bool = False,
                 image_size:int=256,
                 conv_after_fusion:bool = True,
                 fpn_setting:int=1,
                 fg_stage_num:int=3,
                 **kwargs):
        super().__init__()
        self.is_freeze_encoder = is_freeze_encoder
        self.conv_after_fusion = conv_after_fusion
        return_init_mask=True
        # self.use_init_feature = use_init_feature
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        sam_in_channel = 3
        
        fusion_channel = 64
        
        self.backbone_fusion = FPN(
            d_model = fusion_channel,
            backbone_channel_list = [256, 256, 256],
            kernel_size = 3,
            padding=1,
            fpn_interp_model = "bilinear",
            fuse_type='avg'
        )

        # self.backbone_fusion = FPN(
        #     d_model = fusion_channel,
        #     backbone_channel_list = [256, 64, 32],
        #     kernel_size = 3,
        #     padding=1,
        #     fpn_interp_model = "bilinear",
        #     fuse_type='avg'
        # )
        
        self.conv_image_feature = nn.Sequential(
            conv3d_norm_act(fusion_channel, fusion_channel, (1,3,3), padding=(0,1,1),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        self.conv_pos_enc = nn.Sequential(
            conv3d_norm_act(256, 32, (1,1,1), padding=(0,0,0),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        # initialization
        self.affinity_branch = MaskBranch(
            num_convs=4, kernel_dim=3, 
            in_filter=fusion_channel, out_filter=fusion_channel, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1),
            shared_kwargs=self.shared_kwargs
        )
        

        """ ———FGNet BEGIN———"""

        block = DepthwiseBlock3d
        logging.warning(f'Force using DepthwiseBlock3d for DWUNet3D. {block_type} is ignored.')

        # self.depth = len(filters)
        # self.do_return_feats = (return_feats is not None)
        # self.return_feats = return_feats
        # print(f"Return feature maps from 3D FPN-Net? {self.do_return_feats}")

        # assert not self.do_return_feats

        if is_isotropic:
            isotropy = [True] * self.depth
        # assert len(filters) == len(isotropy)

        # block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = [
            conv3d_norm_act(in_channel, filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs),
            conv3d_norm_act(filters[0], filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs)
        ]
        self.conv_in = nn.Sequential(*self.conv_in)
        
        # stages
        self.num_stages = fg_stage_num
        self.loss_weight = "avg"

        self.heads = nn.ModuleList()
        self.encoder_stages = nn.ModuleList()  # 编码器阶段
        self.decoder_stages = nn.ModuleList()  # 解码器阶段
        
        for i in range(self.num_stages):
            # 单独初始化编码器和解码器
            encoder = EncoderFPNWITHSAM(block, filters, isotropy, self.pooling, self.shared_kwargs, fusion_channel)
            decoder = DecoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs)
            
            self.encoder_stages.append(encoder)
            self.decoder_stages.append(decoder)
            self.heads.append(
                conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
            )
        
        self.decoder_stages_local = nn.ModuleList()  # 解码器阶段
        self.heads_local = nn.ModuleList()
        
        for i in range(self.num_stages):
            
            decoder = DecoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs)
            
            self.decoder_stages_local.append(decoder)
            self.heads_local.append(
                conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
            )
        
        self.attention_pool = nn.Conv3d(64,3,kernel_size=1)

        """——————FGNet END——————"""
        model_init(self, mode=init_mode)

        """Build SAM2""" #TODO:path search
        net = "sam2aff.model.SAM2AFFTrain"
        self.sam2 = build_sam2aff(self.sam2_config_dict[sam2_config]["cfg"],
                                                         self.sam2_config_dict[sam2_config]["checkpoint"],net=net,
                                                         image_size = image_size) 
        self._freeze_sam2_other()
        
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()

    def _freeze_sam2_other(self):
            for param in self.sam2.parameters():
                param.requires_grad = False
            for param in self.sam2.image_encoder.parameters():
                param.requires_grad = True
            for name, param in self.named_parameters():
                if param.requires_grad is False:
                # if True:
                    print(f"{name}: requires_grad={param.requires_grad}")

    def _freeze_sam2_encoder(self):
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
            # if True:
                print(f"{name}: requires_grad={param.requires_grad}")

    def _input_resize(self,inputs,size=1024):
        # Step 1: resize to 1024x1024 for each slice along D
        B, C, D, H, W = inputs.shape
        inputs = inputs.view(B * D, 1, H, W)  # merge B and D to apply 2D interpolation
        inputs = F.interpolate(inputs, size=(size, size), mode='bilinear', align_corners=False)
        inputs = inputs.view(B, 1, D, size, size)  # reshape back
        return inputs

    def _input_transform(self, inputs):
        """
        inputs: Tensor of shape (B, 1, D, H, W), values in [0, 1]
        Returns: normalized RGB tensor of shape (B, 3, D, 1024, 1024)
        """
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)

        # Step 2: repeat grayscale to 3 channels
        inputs = inputs.repeat(1, 3, 1, 1, 1)  # shape: (B, 3, D, 1024, 1024)

        # Step 3: normalize
        img_mean = torch.tensor(img_mean, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        img_std = torch.tensor(img_std, dtype=torch.float32, device=inputs.device).view(1, 3, 1, 1, 1)
        
        return (inputs - img_mean) / img_std
    
    
    def forward(self, inputs, target=None, weight=None, criterion=None):

        
        B,C,D,H,W = inputs.shape
        x = self._input_transform(inputs)
        
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        feature_fpn = self.sam2.image_encoder(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn)
        feature_fusion = [f.permute(1,0,2,3).unsqueeze(0) for f in feature_fusion]
        affinity = self.affinity_branch(feature_fusion[0])
        
        affinity = F.interpolate(affinity, size=(D,H,W), mode='trilinear', align_corners=False)
        
        x = self.conv_in(inputs)

        stage_feats = []
        stage_feats_local = []
        
        for i in range(self.num_stages):
            encoder_output,attention_map = self.encoder_stages[0](x, feature_fusion)
            x = self.decoder_stages[i](encoder_output)  # 解码器接收编码器输出
            stage_feats.append(x)
            x = self.decoder_stages_local[i](encoder_output)  # 解码器接收编码器输出
            stage_feats_local.append(x)

        preds = []
        preds.append(affinity)
        preds_local = []
        for i, feats in enumerate(stage_feats):
            # pred = self.heads[i](feats[0])
            pred = self.heads[i](feats)
            preds.append(pred)
        for i, feats in enumerate(stage_feats_local):
            # pred = self.heads[i](feats[0])
            pred_local = self.heads_local[i](feats)
            preds_local.append(pred_local)    

        
        # calculate loss
        if criterion:
            list_loss, full_losses_vis = [], dict()

            for t in range(self.num_stages+1):
                loss, losses_vis = criterion(preds[t], target, weight)
                list_loss.append(loss)
                full_losses_vis.update({k+f"_iter{t}": v for k, v in losses_vis.items()})

            attention = self.attention_pool(attention_map[0])
            attention = F.interpolate(attention,size = inputs.shape[2:], mode='trilinear', align_corners=False)

            mask = (attention_map > threshold).float()

            for t in range(self.num_stages):
                loss, losses_vis = criterion(preds_local[t], target, weight)
                # Only sum over masked regions
                masked_bce = (bce_loss * mask).sum()
                mask_sum = mask.sum()
                bce_loss_masked = masked_bce / (mask_sum + 1e-8)

                # Optional: Local Dice Loss (if you want it)
                pred_sigmoid = torch.sigmoid(pred)
                pred_masked = pred_sigmoid * mask
                target_masked = target * mask

                intersection = (pred_masked * target_masked).sum()
                union = pred_masked.sum() + target_masked.sum()
                dice_coeff = (2. * intersection + 1e-8) / (union + 1e-8)
                dice_loss_masked = 1.0 - dice_coeff

                # Combine (you can adjust weights or use only BCE)
                total_loss = bce_loss_masked + dice_loss_masked

                list_loss.append(loss)
                full_losses_vis.update({k+f"local_iter{t}": v for k, v in losses_vis.items()})

            full_losses_vis.update(losses_vis)

            if self.loss_weight == "avg":
                loss = sum(list_loss) / len(list_loss)
            elif self.loss_weight == "sum":
                loss = sum(list_loss)
            elif self.loss_weight == "last":
                loss = list_loss[-1] + 0 * sum(list_loss[:-1])      #  avoid unused parameter warning
            else:
                raise NotImplementedError

            return preds[-1], loss, full_losses_vis

        return preds[-1]

