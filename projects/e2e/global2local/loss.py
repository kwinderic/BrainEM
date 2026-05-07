import torch.nn as nn

from ..loss.panoptic import PanopticCriterion3D, PanopticMatcher3D


class G2LCriterion3D(nn.Module):
    """
    TODO:
        - mask loss on G/L outputs
        - contrastive loss on G/L features
        - [contrastive loss on G/L queries]
    """

    def __init__(self) -> None:
        self.matcher = PanopticMatcher3D()
        self.mask_criterion = PanopticCriterion3D()

    def forward(self, outputs, targets, weight):
        pass
