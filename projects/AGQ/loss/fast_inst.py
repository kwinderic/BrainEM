# Modified from https://github.com/bytedance/kmax-deeplab/blob/main/kmax_deeplab/modeling/criterion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from .sparse_inst import (
    compute_mask_iou, 
    is_dist_avail_and_initialized, 
    get_world_size,
    # dice_score
)


# https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
def divide_no_nan(x: torch.Tensor, y: torch.Tensor):
    return torch.nan_to_num(x / y, nan=0.0, posinf=0.0, neginf=0.0)


def dice_score(inputs, targets):
    # inputs = inputs.sigmoid()
    inputs = F.softmax(inputs, dim=0)
    inputs = inputs.flatten(1) # N x HW

    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        matched_cls_prob: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        reduction: str='sum'
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    inputs = inputs.softmax(1) # B N HW
    if pixel_gt_void_mask is not None:
        # https://github.com/google-research/deeplab2/blob/main/model/loss/base_loss.py#L111
        inputs = inputs.masked_fill(pixel_gt_void_mask.unsqueeze(1), 0) # remove void pixels.

    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)

    loss *= matched_cls_prob

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    return loss.sum()



def softmax_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        reduction: str = 'mean',
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.cross_entropy(inputs, targets, reduction="none") # B x HW
    if pixel_gt_void_mask is not None:
        loss = loss.masked_fill(pixel_gt_void_mask, 0) # remove void pixels.

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    loss = divide_no_nan(loss_sum_per_sample, num_non_zero)

    if reduction == 'none':
        return loss
    return loss.mean()




class FastInstCriterion3D(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py
    # default_weight_dict = dict(loss_mask=5.0, loss_dice=2.0, loss_objectness=1.0)
    # default_weight_dict = dict(loss_mask=1.0, loss_dice=2.0, loss_objectness=1.0)
    default_weight_dict = dict(loss_mask=0.3, loss_dice=3.0)

    def __init__(self, matcher, **update_weight_dict):
        super().__init__()

        self.num_classes = 1
        self.matcher = matcher
        self.weight_dict = deepcopy(self.default_weight_dict)
        self.weight_dict.update(update_weight_dict)


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def loss_masks_with_iou_objectness(self, outputs, target_masks, indices, num_instances, bg_index):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xDxHxW
        assert "pred_masks" in outputs
        # assert "pred_scores" in outputs

        # src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        num_masks = [m.size(0) for m in target_masks]

        # Pad and permute the target_mask to B x N x D x H x W
        pad_target_masks = torch.zeros_like(src_masks)
        pad_target_masks_o = torch.cat([
                    m[J] for m, (_, J) in zip(target_masks, indices)]).to(pad_target_masks)
        pad_target_masks[src_idx] = pad_target_masks_o

        # Note that for instance segmentation masks may overlap with each other, here we normalize
        # the mask to ensure they sum to one
        pad_target_masks = pad_target_masks / torch.clamp(pad_target_masks.sum(1, keepdim=True), min=1.0)

        src_masks = src_masks.flatten(2)                                # B, N, DHW
        pad_target_masks = pad_target_masks.flatten(2)                  # B, N, DHW
        pixel_gt_void_mask = (pad_target_masks.sum(1) < 1).flatten(1)   # B x DHW,  BG mask
        # with torch.no_grad():
        #     ious = compute_mask_iou(src_masks, pad_target_masks)

        # fill channel bg_mask to background
        pad_target_masks[:,bg_index] = 1 - pad_target_masks.sum(1)


        matched_cls_prob = torch.full(
            src_masks.shape[:2], 0, dtype=src_masks.dtype, device=src_masks.device
        ) # B x N
        matched_cls_prob[src_idx] = 1

        # tgt_iou_scores = ious
        # src_iou_scores = src_iou_scores[src_idx]
        # tgt_iou_scores = tgt_iou_scores.flatten(0)
        # src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            # "loss_dice": dice_loss(src_masks, pad_target_masks) / num_instances,
            "loss_dice": dice_loss(src_masks, pad_target_masks, matched_cls_prob, pixel_gt_void_mask, reduction='mean'),
            # "loss_mask": softmax_ce_loss(src_masks, pad_target_masks, pixel_gt_void_mask)
            "loss_mask": softmax_ce_loss(src_masks, pad_target_masks, None)
        }
        return losses


    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals

        proposal_size = output_proposals["proposal_cls_logits"].shape[-3:]
        proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()    # B, #class, DHW

        # default as bg class 
        target_classes = self.num_classes * torch.ones([
            proposal_cls_logits.size(0), proposal_size.numel()],
            device=proposal_cls_logits.device
        )
        target_classes = target_classes.to(torch.int64)

        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = 0     # 0 for fg

        loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        return {"loss_proposal": loss_proposal}


    def forward(self, outputs, targets, weight, bg_index=0):

        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}
        target_masks = self.prepare_target_masks(targets, outputs)

        # Compute proposal loss for FastInst's IA-guided queries
        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher.forward_proposal(output_proposals, target_masks)
            proposal_loss_dict = self.loss_proposals(output_proposals, target_masks, indices)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, target_masks)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(t.size(0) for t in target_masks)        # num_instances in this GPU
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()         # average num_instances per GPU

        # Compute all the requested losses
        losses = dict()
        losses.update(
            self.loss_masks_with_iou_objectness(
                    outputs, target_masks, indices, num_instances, bg_index=bg_index)
        )
        losses.update(
            proposal_loss_dict
        )
        self._apply_weight(losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, target_masks)
                l_dict = self.loss_masks_with_iou_objectness(
                        aux_outputs, target_masks, indices, num_instances, bg_index=bg_index)
                self._apply_weight(l_dict)

                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return sum(losses.values()), losses


    def _apply_weight(self, losses):
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]


    def prepare_target_masks(self, targets, outputs):
        """
        Prepare and resize target masks. (1) decompose all instance masks 
        for the tensor of instance ids. (2) resize.
        shape change: (B, D, H', W') -> list of (Ni, D, H, W)
        """
        pred_masks = outputs["pred_masks"][0]      # B, N, D, H, W

        out = []
        for tgt in targets[0]:
            ids = tgt.unique()
            fg_masks = []
            for i in ids:
                if i > 0:
                    fg_masks.append(tgt == i)
            fg_masks = torch.stack(fg_masks, dim=0)     # (Ni, D, H', W')
            fg_masks = F.interpolate(
                        fg_masks[:, None].to(pred_masks), 
                        size=pred_masks.shape[1:], 
                        mode="trilinear", 
                        align_corners=False
                ).squeeze(1)
            out.append(fg_masks)
        return out



class FastInstMatcher3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.mask_score = dice_score
        self.cost_class = 2.0
        self.cost_mask = 5.0
        self.cost_location = 1000.0

    def forward_proposal(self, outputs, tgt_masks):
        # proposal_cls_logits: B, 2 (#class), D, H, W

        bs = outputs["proposal_cls_logits"].shape[0]
        proposal_size = outputs["proposal_cls_logits"].shape[-3:]

        indices = []
        for b in range(bs):
            proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1)\
                .transpose(0, 1).softmax(-1)        # DHW, #class

            # if tgt_mask
            # TODO: resize tgt_masks

            # cost location for proposal not inside the instance region
            cost_location = - tgt_masks[b].flatten(1).transpose(0, 1)   # DHW, #obj

            # cost class = - prob_y. In our case, 0 for fg and 1 for bg
            cost_class = - proposal_cls_prob[:, :1].detach()            # DHW, 1

            # Calculate proposal cost matrix and assign each obj a location
            C = self.cost_class * cost_class + self.cost_location * cost_location
            C = C.reshape(proposal_size.numel(), -1).cpu()         # DHW, #obj
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) \
                for (i, j) in indices
        ]


    def forward(self, outputs, tgt_masks, bg_index=0):
        with torch.no_grad():
            B, N, D, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks'].float()

            indices = []
            for b in range(B):
                this_pred_masks = pred_masks[b].flatten(1)          # N, DHW
                this_tgt_masks = tgt_masks[b].flatten(1)            # K, DHW

                with autocast(enabled=False):
                    mask_score = self.mask_score(this_pred_masks, this_tgt_masks)
                    mask_score[bg_index] = 0        # avoid background prediction matching with any fg masks

                # Nx(Number of gts)
                C = mask_score.cpu()

                # hungarian matching
                row_ind, col_ind = linear_sum_assignment(C, maximize=True)
                indices.append((row_ind, col_ind))

        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        return indices
