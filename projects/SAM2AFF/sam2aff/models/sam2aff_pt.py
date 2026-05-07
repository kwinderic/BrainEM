from ..model import *
from .fgnet import DepthwiseBlock3d,HelperMixin,EncoderFPN,DecoderFPN,FGNet

class FGNet(nn.Module, HelperMixin):
    """3D residual FPN-style architecture with Depthwise Conv.
    based on dwunet_3d_v2: 
        - Change stem layer to 2 conv3x3 instead of 1 conv5x5.
        - Use GroupNorm (by default).
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
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 return_feats: Optional[list] = None,
                 **kwargs):
        super(FGNet, self).__init__()

        block = DepthwiseBlock3d
        logging.warning(f'Force using DepthwiseBlock3d for DWUNet3D. {block_type} is ignored.')

        self.depth = len(filters)
        self.do_return_feats = (return_feats is not None)
        self.return_feats = return_feats
        print(f"Return feature maps from 3D FPN-Net? {self.do_return_feats}")

        assert not self.do_return_feats

        if is_isotropic:
            isotropy = [True] * self.depth
        # assert len(filters) == len(isotropy)

        # block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        # self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
        #                                padding=padding_io, **self.shared_kwargs)
        self.conv_in = [
            conv3d_norm_act(in_channel, filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs),
            conv3d_norm_act(filters[0], filters[0], (1,3,3), padding=(0,1,1), **self.shared_kwargs)
        ]
        self.conv_in = nn.Sequential(*self.conv_in)
        
        # stages
        self.num_stages = 3
        self.loss_weight = "avg"

        self.heads = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(nn.Sequential(
                EncoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs),
                DecoderFPN(block, filters, isotropy, self.pooling, self.shared_kwargs)
            ))
            self.heads.append(
                conv3d_norm_act(filters[0], out_channel, kernel_size_io, bias=True,
                        padding=padding_io, pad_mode=pad_mode, act_mode='none', norm_mode='none')
            )

        # initialization
        model_init(self, mode=init_mode)

    def forward(self, inputs, target=None, weight=None, criterion=None):
        x = self.conv_in(inputs)

        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)

        preds = []
        for i, feats in enumerate(stage_feats):
            # pred = self.heads[i](feats[0])
            pred = self.heads[i](feats)
            preds.append(pred)

        # calculate loss
        if criterion:
            list_loss, full_losses_vis = [], dict()

            for t in range(self.num_stages):
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

@register_model("sam2aff_pt")
class SAM2AFFPT(SAM2AFF):
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
    
    @return_loss
    def forward(self, x):

        
        B,C,D,H,W = x.shape
        x = self._input_transform(x)
        
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        feature_fpn = self.sam2.image_encoder(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn).permute(1,0,2,3).unsqueeze(0)
        affinity = self.affinity_branch(feature_fusion)
        
        affinity = F.interpolate(affinity, size=(D,H,W), mode='trilinear', align_corners=False)

        return affinity

@register_model("sam2aff_pt_orinorm")
class SAM2AFFPTORINORM(SAM2AFF):
    """Leveraging Query-Proposal to generate prompt for SAM2"""
    @return_loss
    def forward(self, x):

        
        B,C,D,H,W = x.shape
        # x = self._input_transform(x)
        
        x = x.repeat(1, 3, 1, 1, 1)
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        feature_fpn = self.sam2.image_encoder(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn).permute(1,0,2,3).unsqueeze(0)
        affinity = self.affinity_branch(feature_fusion)
        
        affinity = F.interpolate(affinity, size=(D,H,W), mode='trilinear', align_corners=False)

        return affinity
        
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

@register_model("sam2aff_pt_fg")
class SAM2AFFPTFG(SAM2AFF,HelperMixin):
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
        for i in range(self.num_stages):
            # 单独调用编码器和解码器
            encoder_output = self.encoder_stages[i](x, feature_fusion)  # 传递两个参数
            x = self.decoder_stages[i](encoder_output)  # 解码器接收编码器输出
            stage_feats.append(x)

        preds = []
        preds.append(affinity)
        for i, feats in enumerate(stage_feats):
            # pred = self.heads[i](feats[0])
            pred = self.heads[i](feats)
            preds.append(pred)

        # calculate loss
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

