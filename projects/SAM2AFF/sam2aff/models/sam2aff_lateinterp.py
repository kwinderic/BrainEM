from ..model import *

@register_model("sam2aff_lateinterp")
class SAM2AFFLATEINTERP(SAM2AFF):
    """use feature after mask conv"""
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
        # self.use_init_feature = use_init_feature
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        sam_in_channel = 3
        
        fusion_channel = 32
        
        # self.backbone_fusion = FPN(
        #     d_model = fusion_channel,
        #     backbone_channel_list = [256, 256, 256],
        #     kernel_size = 3,
        #     padding=1,
        #     fpn_interp_model = "bilinear",
        #     fuse_type='avg'
        # )

        self.backbone_fusion = FPN(
            d_model = fusion_channel,
            backbone_channel_list = [256, 64, 32],
            kernel_size = 3,
            padding=1,
            fpn_interp_model = "bilinear",
            fuse_type='avg'
        )
        
        self.conv_image_feature = nn.Sequential(
            conv3d_norm_act(fusion_channel, fusion_channel, (1,3,3), padding=(0,1,1),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        self.conv_pos_enc = nn.Sequential(
            conv3d_norm_act(256, 32, (1,1,1), padding=(0,0,0),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        # initialization
        model_init(self, mode=init_mode)

        """Build SAM2""" #TODO:path search
        net = "sam2aff.model.SAM2AFFTrain"
        self.sam2 = build_sam2aff(self.sam2_config_dict[sam2_config]["cfg"],
                                                         self.sam2_config_dict[sam2_config]["checkpoint"],net=net,
                                                         image_size = image_size) 
        self.affinity_branch = MaskBranch(
            num_convs=1, kernel_dim=3, 
            in_filter=fusion_channel, out_filter=fusion_channel, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1),
            shared_kwargs=self.shared_kwargs
        )
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()

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
        feature_fpn = self.sam2.forward_image(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn).permute(1,0,2,3).unsqueeze(0)
        affinity = self.affinity_branch(feature_fusion)
        
        affinity = F.interpolate(affinity, size=(D,H,W), mode='trilinear', align_corners=False)

        # if self.conv_after_fusion:
        #     sam_image_features_fusion = self.conv_image_feature(sam_image_features_fusion)
        

        return affinity


class EfficientDetailEnhancer(nn.Module):
    """高效细节增强模块，修复通道数问题"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 2  # 每个分支输出out_channels//2通道
        
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b3], dim=1)  # 拼接后通道数为out_channels
        out = self.fusion(out)
        return out


class MemoryEfficientAttention(nn.Module):
    """内存高效的注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        reduce_channels = max(8, in_channels // 8)  # 确保最小通道数
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, reduce_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(reduce_channels, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        spatial_weights = torch.sigmoid(avg_out.expand_as(x))
        return x * spatial_weights


class ResidualDetailBlock(nn.Module):
    """残差细节块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enhancer = EfficientDetailEnhancer(in_channels, out_channels)
        self.attention = MemoryEfficientAttention(out_channels)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        out = self.enhancer(x)
        out = self.attention(out)
        out += residual
        return F.relu(out)


class BoundaryRefinementModule(nn.Module):
    """边界细化模块"""
    def __init__(self, in_channels, num_directions=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, num_directions, kernel_size=1)
        
    def forward(self, x):
        edge_features = F.relu(self.bn1(self.conv1(x)))
        edge_map = self.conv2(edge_features)
        return edge_map


class HighRes3DBoundaryNet(nn.Module):
    """高分辨率3D边界预测网络"""
    def __init__(self, in_channels=1, num_directions=3):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.block1 = nn.Sequential(
            ResidualDetailBlock(16, 16),
        )
        
        self.expand = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = nn.Sequential(
            ResidualDetailBlock(32, 32),
            ResidualDetailBlock(32, 32),
        )
        
        self.prediction = BoundaryRefinementModule(32, num_directions)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.expand(x)
        x = self.block2(x)
        boundary = self.prediction(x)
        return boundary


@register_model("sam2aff_lateinterp_sigma")
class SAM2AFFLATEINTERPSIGMA(SAM2AFF):
    """use feature after mask conv"""
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
                 sigma:float=0.5,
                 resize_input:bool=False,
                 **kwargs):
        super().__init__()
        self.is_freeze_encoder = is_freeze_encoder
        self.conv_after_fusion = conv_after_fusion
        self.sigma =sigma
        self.resize_input =resize_input
        if self.resize_input:
            image_size =1024
        # self.use_init_feature = use_init_feature
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}
        sam_in_channel = 3
        
        fusion_channel = 32
        
        # self.backbone_fusion = FPN(
        #     d_model = fusion_channel,
        #     backbone_channel_list = [256, 256, 256],
        #     kernel_size = 3,
        #     padding=1,
        #     fpn_interp_model = "bilinear",
        #     fuse_type='avg'
        # )

        self.backbone_fusion = FPN(
            d_model = fusion_channel,
            backbone_channel_list = [256, 64, 32],
            kernel_size = 3,
            padding=1,
            fpn_interp_model = "bilinear",
            fuse_type='avg'
        )
        
        self.conv_image_feature = nn.Sequential(
            conv3d_norm_act(fusion_channel, fusion_channel, (1,3,3), padding=(0,1,1),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        self.conv_pos_enc = nn.Sequential(
            conv3d_norm_act(256, 32, (1,1,1), padding=(0,0,0),pad_mode= 'replicate',act_mode= 'elu',norm_mode= 'gn')
        )
        # initialization
        model_init(self, mode=init_mode)

        """Build SAM2""" #TODO:path search
        net = "sam2aff.model.SAM2AFFTrain"
        self.sam2 = build_sam2aff(self.sam2_config_dict[sam2_config]["cfg"],
                                                         self.sam2_config_dict[sam2_config]["checkpoint"],net=net,
                                                         image_size = image_size) 
        self.affinity_branch = MaskBranch(
            num_convs=1, kernel_dim=3, 
            in_filter=fusion_channel, out_filter=fusion_channel, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1),
            shared_kwargs=self.shared_kwargs
        )
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()
        
        self.local_refine_net = HighRes3DBoundaryNet()

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
    def forward(self, inputs):
        B,C,D,H,W = inputs.shape
        if self.resize_input:
            x = self._input_resize(inputs)
            x = self._input_transform(x)
        else:
            x = self._input_transform(inputs)
        
        x = x.permute(0,2,1,3,4)
        x = x.flatten(0,1)
        feature_fpn = self.sam2.forward_image(x)['backbone_fpn']

        
        feature_fusion = self.backbone_fusion(feature_fpn).permute(1,0,2,3).unsqueeze(0)
        affinity_sam = self.affinity_branch(feature_fusion)
        
        affinity_sam = F.interpolate(affinity_sam, size=(D,H,W), mode='trilinear', align_corners=False)
        
        affinity_local = self.local_refine_net(inputs)

        # if self.conv_after_fusion:
        #     sam_image_features_fusion = self.conv_image_feature(sam_image_features_fusion)
        
        affinity = self.sigma * affinity_sam +(1-self.sigma)*affinity_local

        return affinity
