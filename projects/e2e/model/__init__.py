from connectomics.model.build import MODEL_MAP

from interface import E2EMixin
from .sparse_unet import SparseUNet3D
from .fast_inst import FastUNet3D
from .sparse_unet_v2 import SparseUNet3DV2
from .sparse_unet_v2_old import SparseUNet3DV2 as SparseUNet3DV2Old
from .sparse_unet_v3 import SparseUNet3DV3
from .affinity_knet import AffinityKNet
from .affinity_knet_refine import AffinityKNetRefine
from .affinity_knet_coarse_mask import AffinityKNetCoarseMask
# from .affinity_knet_maskformer import AffinityKNetMaskFormer, AffinityKNetMaskFormerOnlyLearn


MODEL_MAP['e2e_mixin'] = E2EMixin
MODEL_MAP['sparse_unet_3d'] = SparseUNet3D
MODEL_MAP['fast_unet_3d'] = FastUNet3D
MODEL_MAP['sparse_unet_3d_v2'] = SparseUNet3DV2
MODEL_MAP['sparse_unet_3d_v2_old'] = SparseUNet3DV2Old
MODEL_MAP['sparse_unet_3d_v3'] = SparseUNet3DV3
MODEL_MAP['affinity_knet'] = AffinityKNet
MODEL_MAP['affinity_knet_refine'] = AffinityKNetRefine
MODEL_MAP['affinity_knet_coarse_mask'] = AffinityKNetCoarseMask
# MODEL_MAP['affinity_knet_maskformer'] = AffinityKNetMaskFormer
# MODEL_MAP['affinity_knet_maskformer_onlylearn'] = AffinityKNetMaskFormerOnlyLearn