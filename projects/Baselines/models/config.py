from yacs.config import CfgNode

from .utils import register_model

@register_model('cfg_mixin')
class CfgMixin:
    
    @staticmethod
    def parse_config(cfg, kwargs):

        return kwargs
    
def add_custom_config(_C:CfgNode):
    r'''LDINO specific configurations'''
    _C.MODEL.USE_MATCHNESS = False

    # Adam/AdamW extra params not exposed by the framework's build_optimizer.
    # Defaults match PyTorch defaults so other models are unaffected.
    _C.SOLVER.ADAM_EPS = 1e-8
    _C.SOLVER.AMSGRAD = False

    # CAD config
    _C.MODEL.CAD_EMBEDDING_DIM = 16
    _C.MODEL.CAD_FILTER_CHANNEL_2D = 16
    _C.MODEL.CAD_START_FT = 10000
    _C.MODEL.CAD_FT_LR_RATIO = 1.0
    _C.MODEL.CAD_AFFS0_WEIGHT_3D = 10.0
    _C.MODEL.CAD_AFFS0_WEIGHT_2D = 1.0
    _C.MODEL.CAD_LOSS_WEIGHT_3D = 1.0
    _C.MODEL.CAD_LOSS_WEIGHT_2D_SLICE = 1.0
    _C.MODEL.CAD_LOSS_WEIGHT_CROSS = 1.0
    _C.MODEL.CAD_LOSS_WEIGHT_INTERACT = 1.0

    
 