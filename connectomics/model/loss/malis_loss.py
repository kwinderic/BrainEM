"""MALIS (Maximin Affinity Learning of Image Segmentation) Loss.

Ported from the C++ implementation in:
  - malis_cpp.cpp (Srini Turaga, MIT)
  - malis_loss.py (Funke et al.)

The MALIS loss computes structured weights for affinity prediction:
  loss = Σ weight_ij * (pred_aff_ij - gt_aff_ij)²

where weight_ij is high only for "critical" edges that cause merge/split
errors in the maximum spanning tree segmentation. This focuses learning
on the most impactful boundaries.
"""

from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
from numba import njit


def _nodelist_like(shape, nhood):
    """Build edge node lists for an affinity graph.

    Args:
        shape: (D, H, W) volume shape.
        nhood: (nEdgeType, 3) neighborhood offsets.

    Returns:
        (node1, node2): each of shape (nEdgeType, D, H, W), flat node indices.
        node2 uses nVert as sentinel for invalid edges.
    """
    nEdge = nhood.shape[0]
    nVert = int(np.prod(shape))
    nodes = np.arange(nVert, dtype=np.int64).reshape(shape)
    node1 = np.tile(nodes, (nEdge, 1, 1, 1))
    node2 = np.full(node1.shape, nVert, dtype=np.int64)  # sentinel = nVert

    D, H, W = shape
    for e in range(nEdge):
        dz, dy, dx = int(nhood[e, 0]), int(nhood[e, 1]), int(nhood[e, 2])
        node2[e,
              max(0, -dz):min(D, D - dz),
              max(0, -dy):min(H, H - dy),
              max(0, -dx):min(W, W - dx)] = \
            nodes[max(0, dz):min(D, D + dz),
                  max(0, dy):min(H, H + dy),
                  max(0, dx):min(W, W + dx)]
    return node1, node2


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _union(parent, rank, a, b):
    if rank[a] < rank[b]:
        a, b = b, a
    parent[b] = a
    if rank[a] == rank[b]:
        rank[a] += 1
    return a


@njit(cache=True)
def _malis_loss_weights_numba(seg, node1, node2, edge_weight, pos, nVert):
    """Numba-accelerated MALIS loss weight computation.

    Kruskal's maximum spanning tree with union-find and pair counting.
    """
    nEdge = len(node1)

    # Union-Find
    parent = np.arange(nVert, dtype=np.int64)
    rank = np.zeros(nVert, dtype=np.int64)

    # Overlap storage: use larger buffer (4x nVert) to handle relocations
    max_labels = nVert * 4
    overlap_labels = np.zeros(max_labels, dtype=np.int64)
    overlap_counts = np.zeros(max_labels, dtype=np.int64)
    overlap_offset = np.zeros(nVert, dtype=np.int64)
    overlap_size = np.zeros(nVert, dtype=np.int64)

    # Initialize
    idx = 0
    for i in range(nVert):
        overlap_offset[i] = idx
        if seg[i] != 0:
            overlap_labels[idx] = seg[i]
            overlap_counts[idx] = 1
            overlap_size[i] = 1
            idx += 1
        else:
            overlap_size[i] = 0

    free_ptr = idx

    # Filter valid edges and sort by weight descending
    valid_count = 0
    for i in range(nEdge):
        if node1[i] < nVert and node2[i] < nVert:
            valid_count += 1

    valid_edges = np.empty(valid_count, dtype=np.int64)
    valid_weights = np.empty(valid_count, dtype=np.float32)
    j = 0
    for i in range(nEdge):
        if node1[i] < nVert and node2[i] < nVert:
            valid_edges[j] = i
            valid_weights[j] = edge_weight[i]
            j += 1

    order = np.argsort(-valid_weights)
    sorted_edges = valid_edges[order]

    nPairPerEdge = np.zeros(nEdge, dtype=np.int64)

    for idx in range(len(sorted_edges)):
        e = sorted_edges[idx]
        u = node1[e]
        v = node2[e]
        set1 = _find(parent, u)
        set2 = _find(parent, v)

        if set1 != set2:
            off1 = overlap_offset[set1]
            sz1 = overlap_size[set1]
            off2 = overlap_offset[set2]
            sz2 = overlap_size[set2]

            # Count pairs
            n_pair = np.int64(0)
            for i1 in range(sz1):
                label1 = overlap_labels[off1 + i1]
                count1 = overlap_counts[off1 + i1]
                for i2 in range(sz2):
                    label2 = overlap_labels[off2 + i2]
                    count2 = overlap_counts[off2 + i2]
                    if pos == 1 and label1 == label2:
                        n_pair += count1 * count2
                    elif pos == 0 and label1 != label2:
                        n_pair += count1 * count2

            nPairPerEdge[e] = n_pair

            # Union: make the larger cluster the new root
            if sz1 >= sz2:
                new_root = _union(parent, rank, set1, set2)
                if new_root != set1:
                    # swap references
                    overlap_offset[new_root] = off1
                    overlap_size[new_root] = sz1
                other_off = off2
                other_sz = sz2
            else:
                new_root = _union(parent, rank, set2, set1)
                if new_root != set2:
                    overlap_offset[new_root] = off2
                    overlap_size[new_root] = sz2
                other_off = off1
                other_sz = sz1

            off_new = overlap_offset[new_root]
            sz_new = overlap_size[new_root]

            # Merge: add other's labels into new_root
            for i2 in range(other_sz):
                label2 = overlap_labels[other_off + i2]
                count2 = overlap_counts[other_off + i2]

                found = False
                for i1 in range(sz_new):
                    if overlap_labels[off_new + i1] == label2:
                        overlap_counts[off_new + i1] += count2
                        found = True
                        break

                if not found:
                    # Need to append - check if there's contiguous space
                    new_pos = off_new + sz_new
                    if new_pos >= max_labels:
                        break  # safety: buffer full
                    # Check if slot is free (relocate if needed)
                    if new_pos >= free_ptr:
                        overlap_labels[new_pos] = label2
                        overlap_counts[new_pos] = count2
                        sz_new += 1
                        free_ptr = new_pos + 1
                    else:
                        # Relocate new_root to free space
                        relocated_off = free_ptr
                        if relocated_off + sz_new + 1 >= max_labels:
                            break  # safety
                        for k in range(sz_new):
                            overlap_labels[relocated_off + k] = overlap_labels[off_new + k]
                            overlap_counts[relocated_off + k] = overlap_counts[off_new + k]
                        overlap_labels[relocated_off + sz_new] = label2
                        overlap_counts[relocated_off + sz_new] = count2
                        sz_new += 1
                        overlap_offset[new_root] = relocated_off
                        off_new = relocated_off
                        free_ptr = relocated_off + sz_new

            overlap_size[new_root] = sz_new

    return nPairPerEdge


@njit(cache=True)
def _seg_from_aff_numba(gt_aff, node1, node2, nVert):
    """Reconstruct segmentation from GT affinity via connected components."""
    C = gt_aff.shape[0]
    parent = np.arange(nVert, dtype=np.int64)
    rank = np.zeros(nVert, dtype=np.int64)

    for e in range(C):
        aff_flat = gt_aff[e].ravel()
        n1_flat = node1[e].ravel()
        n2_flat = node2[e].ravel()
        for i in range(len(aff_flat)):
            if aff_flat[i] > 0.5 and n1_flat[i] < nVert and n2_flat[i] < nVert:
                a = _find(parent, n1_flat[i])
                b = _find(parent, n2_flat[i])
                if a != b:
                    _union(parent, rank, a, b)

    seg = np.empty(nVert, dtype=np.int64)
    for i in range(nVert):
        seg[i] = _find(parent, i)
    return seg


def _compute_malis_weights(pred_aff, gt_aff, gt_seg, nhood, node1_flat, node2_flat, nVert):
    """Compute MALIS loss weights (positive + negative pass).

    Args:
        pred_aff: (C, D, H, W) predicted affinities.
        gt_aff: (C, D, H, W) ground-truth affinities.
        gt_seg: (nVert,) ground-truth segmentation labels.
        nhood: (C, 3) neighborhood offsets.
        node1_flat, node2_flat: (C*D*H*W,) edge endpoints.
        nVert: number of vertices.

    Returns:
        weights: (C, D, H, W) float32 MALIS weights.
    """
    weights = np.zeros_like(pred_aff, dtype=np.float32)

    for pass_type in [0, 1]:  # negative pass, positive pass
        # Constrain affinities for this pass
        pass_aff = pred_aff.copy()
        constraint = gt_aff == (1 - pass_type)
        pass_aff[constraint] = float(1 - pass_type)

        n_pair = _malis_loss_weights_numba(
            gt_seg,
            node1_flat,
            node2_flat,
            pass_aff.ravel().astype(np.float32),
            pass_type,
            nVert)

        pass_weights = n_pair.reshape(pred_aff.shape).astype(np.float32)

        # Edges with gt_aff == (1-pass_type) don't contribute
        pass_weights[gt_aff == (1 - pass_type)] = 0

        # Normalize
        total = pass_weights.sum()
        if total > 0:
            pass_weights /= total

        weights += pass_weights

    return weights


class MalisLoss(nn.Module):
    """MALIS loss for affinity-based neuron segmentation.

    Computes structured loss weights via maximum spanning tree, then
    applies weighted MSE: loss = Σ w_ij * (pred_ij - target_ij)²

    The weights focus learning on "critical" edges that would cause
    merge/split errors in the resulting segmentation.

    Compatible with framework loss interface: forward(pred, target, weight_mask).
    """

    def __init__(self):
        super().__init__()
        self.nhood = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
        self._node_cache = {}

    def _get_nodes(self, shape):
        """Cache node lists for a given volume shape."""
        if shape not in self._node_cache:
            node1, node2 = _nodelist_like(shape, self.nhood)
            nVert = int(np.prod(shape))
            self._node_cache[shape] = (
                node1.ravel().astype(np.int64),
                node2.ravel().astype(np.int64),
                node1,  # keep for _seg_from_aff
                node2,
                nVert)
        return self._node_cache[shape]

    def forward(self, pred, target, weight_mask=None):
        """
        Args:
            pred: (B, 3, D, H, W) predicted affinities (after sigmoid).
            target: (B, 3, D, H, W) GT affinities (binary 0/1).
            weight_mask: optional, additional weight mask.

        Returns:
            loss: scalar tensor.
        """
        B = pred.shape[0]
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        vol_shape = tuple(pred.shape[2:])  # (D, H, W)
        node1_flat, node2_flat, node1_4d, node2_4d, nVert = self._get_nodes(vol_shape)

        for b in range(B):
            pred_np = pred[b].detach().cpu().numpy().astype(np.float32)
            gt_aff_np = target[b].detach().cpu().numpy().astype(np.float32)

            # Reconstruct segmentation from GT affinity
            gt_seg = _seg_from_aff_numba(gt_aff_np, node1_4d, node2_4d, nVert)

            # Compute MALIS weights
            malis_w = _compute_malis_weights(
                pred_np, gt_aff_np, gt_seg, self.nhood,
                node1_flat, node2_flat, nVert)

            malis_w_tensor = torch.from_numpy(malis_w).to(
                device=pred.device, dtype=pred.dtype)

            # Weighted MSE
            diff_sq = (pred[b] - target[b]) ** 2
            loss_b = (malis_w_tensor * diff_sq).sum()

            total_loss = total_loss + loss_b

        return total_loss / max(B, 1)
