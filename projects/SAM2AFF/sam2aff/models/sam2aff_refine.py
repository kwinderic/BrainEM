from ..model import *

class SmallObjectEnhancer(nn.Module):
    """
    小目标边界增强模块 - 输入原图和预计算的affinity，输出优化后的affinity
    """
    def __init__(self, img_channels=3, affinity_channels=3, hidden_channels=64):
        super().__init__()
        
        # 1. 原图特征提取（保留小目标细节）
        self.img_encoder = nn.Sequential(
            # 使用2D卷积，在HW平面上处理每个D切片
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 2. Affinity特征处理（仅在H/W维度上采样至原图分辨率）
        self.affinity_processor = nn.Sequential(
            # 仅在H/W维度上进行4倍上采样
            nn.Conv2d(affinity_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 3. 小目标注意力机制
        self.small_obj_attn = nn.Sequential(
            nn.Conv2d(64+32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # 输出注意力权重图
        )
        
        # 4. 特征融合与增强
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(64+32+1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, affinity_channels, kernel_size=1),
            nn.Sigmoid()  # 输出优化后的affinity
        )
        
    def forward(self, original_img, input_affinity):
        """
        输入:
            original_img: 原始图像 [B, C, D, H, W]
            input_affinity: 预计算的affinity [B, affinity_channels, D, H/4, W/4]
        输出:
            enhanced_affinity: 增强后的affinity [B, affinity_channels, D, H, W]
            attention_map: 注意力图 [B, 1, D, H, W]
        """
        batch_size, _, depth, height, width = original_img.shape
        
        # 重塑输入以处理每个D切片
        original_img = original_img.permute(0, 2, 1, 3, 4).flatten(0,1)
        input_affinity = input_affinity.permute(0, 2, 1, 3, 4).flatten(0,1)
        
        # 提取原图特征
        img_features = self.img_encoder(original_img)  # [B*D, 64, H, W]
        
        scale_factor = original_img.shape[-1] / input_affinity.shape[-1]
        input_affinity = F.interpolate(
            input_affinity, 
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )  # [B*D, affinity_channels, H, W]
        
        # 处理affinity特征
        affinity_features = self.affinity_processor(input_affinity)  # [B*D, 32, H, W]
        
        # 拼接特征并计算小目标注意力
        concat_features = torch.cat([img_features, affinity_features], dim=1)  # [B*D, 96, H, W]
        attention_map = self.small_obj_attn(concat_features)  # [B*D, 1, H, W]
        
        # 融合特征并生成增强后的affinity
        fusion_input = torch.cat([concat_features, attention_map], dim=1)  # [B*D, 97, H, W]
        refined_affinity = self.feature_fusion(fusion_input)  # [B*D, affinity_channels, H, W]
        
        # 重塑回原始维度
        attention_map = attention_map.reshape(batch_size, depth, 1, height, width).permute(0, 2, 1, 3, 4)
        refined_affinity = refined_affinity.reshape(batch_size, depth, -1, height, width).permute(0, 2, 1, 3, 4)
        input_affinity = input_affinity.reshape(batch_size, depth, -1, height, width).permute(0, 2, 1, 3, 4)
        
        # 应用注意力机制：在注意力高的区域强化affinity
        enhanced_affinity = input_affinity * (1 - attention_map) + refined_affinity * attention_map
        
        return enhanced_affinity, attention_map

class ZAxisAffinityCapture(nn.Module):
    """
    Z轴方向Affinity捕捉模块
    """
    def __init__(self, img_channels=3, hidden_channels=64):
        """
        参数:
            img_channels: 输入图像的通道数
            hidden_channels: 隐藏层通道数
        """
        super().__init__()
        
        # 特征提取网络 - 从原图中提取用于计算亲和性的特征
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(img_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels//2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)  
        )
        
        # 亲和性计算头 - 计算不同Z轴距离的亲和性
        self.affinity_head = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1)
        )
    def forward(self,x):
        return self.affinity_head(self.feature_extractor(x))


@register_model("sam2aff_refine")
class SAM2AFFREFINE(SAM2AFF):
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
            num_convs=1, kernel_dim=2, 
            in_filter=fusion_channel, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )
        if self.is_freeze_encoder:
            self._freeze_sam2_encoder()
        
        self.refine_net = SmallObjectEnhancer(img_channels=1,affinity_channels=2,hidden_channels=64)
        self.affinity_branch_z = ZAxisAffinityCapture(1,64)

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
        
        affinity_yx, attention_map = self.refine_net(inputs,affinity_sam)

        affinity_z = self.affinity_branch_z(inputs)

        affinity = torch.cat([affinity_z,affinity_yx],dim=1)

        return affinity,attention_map
    

    