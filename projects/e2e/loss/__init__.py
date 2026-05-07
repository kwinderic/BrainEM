from .sparse_inst import SparseInstCriterion3D, SparseInstMatcher3D
from .panoptic import PanopticCriterion3D, PanopticMatcher3D
from .fast_inst import FastInstCriterion3D, FastInstMatcher3D


# def build_criterion(style='instance'):
def build_criterion(style='fastinst'):
    assert style in {'instance', 'panoptic', 'fastinst'}

    if style == 'instance':
        matcher = SparseInstMatcher3D()
        criterion = SparseInstCriterion3D(matcher)
    elif style == 'panoptic':
        matcher = PanopticMatcher3D()
        criterion = PanopticCriterion3D(matcher)
    elif style == 'fastinst':
        matcher = FastInstMatcher3D()
        criterion = FastInstCriterion3D(matcher)

    return criterion
