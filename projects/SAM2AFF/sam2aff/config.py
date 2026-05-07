from yacs.config import CfgNode

from .utils import register_model

@register_model('cfg_mixin')
class CfgMixin:
    
    @staticmethod
    def parse_config(cfg, kwargs):
        kwargs['sam2_config'] = cfg.MODEL.SAM2_CONFIG
        kwargs['fusion_channel'] = cfg.MODEL.FUSION_CHANNEL
        kwargs['affinity_convs'] = cfg.MODEL.AFFINITY_CONVS
        kwargs['is_freeze_encoder']=cfg.MODEL.IS_FREEZE_ENCODER
        kwargs['image_size'] = (cfg.MODEL.INPUT_SIZE)[-1]
        kwargs['conv_after_fusion'] = cfg.MODEL.CONV_AFTER_FUSION
        kwargs['fpn_setting']=cfg.MODEL.FPN_SETTING
        kwargs['sigma']=cfg.MODEL.SIGMA
        kwargs['resize_input']=cfg.MODEL.RESIZE_INPUT
        kwargs['fg_stage_num']=cfg.MODEL.FG_STAGE_NUM
        return kwargs
    
def add_sam2aff_config(_C:CfgNode):
    r'''EMSAM2 specific configurations'''
    
    _C.MODEL.CONV_IN_CHANNEL = 64
    _C.MODEL.FUSION_CHANNEL = 64
    _C.MODEL.USE_MATCHNESS = False
    _C.MODEL.NUM_ITER = 2
    _C.MODEL.WITH_AFFINITY = False
    _C.MODEL.AFFINITY_CONVS = 4
    _C.MODEL.IS_FREEZE_ENCODER = False
    _C.MODEL.SAM2_CONFIG = 'tiny'
    _C.MODEL.CONV_AFTER_FUSION = True
    _C.MODEL.FPN_SETTING = 1
    _C.MODEL.SIGMA = 0.5
    _C.MODEL.RESIZE_INPUT = False
    _C.MODEL.FG_STAGE_NUM = 1

    _C.DATASET.MEAN = 0.5
    _C.DATASET.STD = 0.5

    _C.SOLVER.IS_TWO_STAGE = False
    _C.SOLVER.TWO_STAGE_ITER = 50000

    
 