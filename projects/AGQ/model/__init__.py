from connectomics.model.build import MODEL_MAP

from interface import E2EMixin
from .affinity_knet import AffinityKNet

MODEL_MAP['e2e_mixin'] = E2EMixin
MODEL_MAP['affinity_knet'] = AffinityKNet
