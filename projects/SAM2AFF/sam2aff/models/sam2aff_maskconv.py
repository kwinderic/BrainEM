from ..model import *

@register_model("sam2aff_maskconv")
class SAM2AFFMASKCONV(SAM2AFF):
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
                 init_mask_method='connected_components', 
                 num_masks:int =100, 
                 num_learned_masks:int=100,
                 return_init_mask=False,
                 image_size:int=256,
                 prompts_method:str = None,
                 feature_for_affinity:str = 'fpn_after_mask_conv',
                 conv_after_fusion:bool = True,
                 fpn_setting:int=1,
                 query_prompt_method:str=None,
                 sa_before_decode:int = 0,
                 sa_during_decode:int = 0,
                 decode_iter_num:int=1,
                 branch_channel:int=64,
                 use_pred_bg:bool=False,
                 **kwargs):
        super().__init__()
        self.is_freeze_encoder = is_freeze_encoder
        self.prompts_method = prompts_method
        self.feature_for_affinity = feature_for_affinity
        self.conv_after_fusion = conv_after_fusion
        self.query_prompt_method = query_prompt_method
        self.num_learned_masks = num_learned_masks
        self.sa_before_decode = sa_before_decode
        self.sa_during_decode = sa_during_decode
        self.decode_iter_num = decode_iter_num
        return_init_mask=True
        if self.prompts_method=='init_mask':
            return_init_mask=True
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
        feature_fusion = F.interpolate(feature_fusion, size=(D,H,W), mode='trilinear', align_corners=False)

        # if self.conv_after_fusion:
        #     sam_image_features_fusion = self.conv_image_feature(sam_image_features_fusion)
        affinity = self.affinity_branch(feature_fusion)

        return affinity


