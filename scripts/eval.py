import numpy as np 
from fire import Fire
from PIL import Image
import h5py
import pickle as pkl
from tabulate import tabulate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from connectomics.data.utils.data_affinity import seg2aff_v0

import h5py
import numpy as np
import imageio
import os
import time


_T0 = time.time()


def _rss_gb():
    """Current resident set size in GB (0.0 if /proc is unavailable)."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def stamp(event, extra=''):
    """Wall-clock + elapsed + RSS for one evaluation milestone.

    Whole-volume evaluation runs for a long time with no output, so a run that
    is merely slow looks the same as one that is thrashing or about to be
    OOM-killed. Every milestone is timestamped (local time, i.e. Beijing on this
    host) so a run can be diagnosed from its log alone -- the FlyWire eval was
    killed three times before the log made clear where it died.
    """
    now = time.time()
    print('[%s | +%7.1fs | RSS %6.2f GB] %-28s %s' % (
        time.strftime('%H:%M:%S', time.localtime(now)),
        now - _T0, _rss_gb(), event, extra), flush=True)
    return now


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    try:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)
    except Exception as e:
        print("Call imageio.imread:", e)
        return imageio.imread(path)
    

def _load_affinity(fn):
    with h5py.File(fn, 'r') as f:
        data = f['vol0'][()]
        affinities = data.astype(np.float32) / 255.0
    return affinities


def load(fn):
    if fn.endswith('.npy'):
        return np.load(fn)
    elif fn.endswith('.pkl'):
        return pkl.load(open(fn, 'rb'))
    elif fn.endswith('.tiff') or fn.endswith('.tif'):
        return read_tiff(fn)
    elif fn.endswith('.h5'):
        with h5py.File(fn, 'r') as f:
            ks = list(f.keys())
            assert len(ks) == 1
            return np.array(f[ks[0]])
    else:
        raise NotImplementedError


def print_table(list_dicts):
    # keys = ['index'] + list(list_dicts[0].keys())
    # table = [[i] + [d[k] for k in keys[1:]] for i, d in enumerate(list_dicts)]
    keys = list(list_dicts[0].keys())
    table = [[d[k] for k in keys] for i, d in enumerate(list_dicts)]
    print(tabulate(table, headers=keys, floatfmt=".3f", tablefmt="pipe"))


def eval_waterz(fn_dt, fn_gt, ignore_border=25/4.0):
    import waterz
    from waterz.seg_util import create_border_mask

    # use evaluation api from https://github.com/zudi-lin/waterz
    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    # ignore boundary within `ignore_border` voxels
    gt = create_border_mask(gt, ignore_border, np.uint64(0))

    if dt.ndim == 3:
        return waterz.evaluate_total_volume(dt, gt)
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            out.append(
                waterz.evaluate_total_volume(dt[i], gt))
        return out
    else:
        raise NotImplementedError
    

def _eval_skimage(dt, gt, print_metrics=True):

    from skimage.metrics import adapted_rand_error as adapted_rand_ref
    from skimage.metrics import variation_of_information as voi_ref

    gt_seg = gt
    segmentation = dt.astype(np.int64)

    print("gt-size:",gt_seg.shape)
    print("dt-size:",dt.shape)
    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge

    metrics = {
        'voi_split': voi_split,
        'voi_merge': voi_merge,
        'voi_sum': voi_sum,
        'adapted_RAND': arand
    }
    if print_metrics:
        print('evaluated with skimage api\n', metrics)
    return metrics


def _eval_cremi(dt, gt, ignore_border, print_metrics=True):
    from cremi.evaluation import NeuronIds
    from cremi import Volume

    dt = Volume(dt)
    gt = Volume(gt - 1)       # cremi expects np.uint64(-1) for background, (i.e. self.gt += 1 in NeuronIds.__init__)

    neuron_ids_evaluation = NeuronIds(gt, ignore_border)

    (voi_split, voi_merge) = neuron_ids_evaluation.voi(dt)
    adapted_rand = neuron_ids_evaluation.adapted_rand(dt)

    metrics = {
        'voi_split': voi_split,
        'voi_merge': voi_merge,
        'adapted_RAND': adapted_rand
    }
    if print_metrics:
        print('evaluated with cremi api\n', metrics)
    return metrics


# def eval_cremi(fn_dt, fn_gt, ignore_border=25/4.0):
def eval_cremi(fn_dt, fn_gt, ignore_border=0):
    # use evaluation api from https://github.com/cremi/cremi_python/tree/python3
    # ignore_border = 25 / resolution, ref: https://cremi.org/leaderboard/: MALA v2
    #       resolution = 4 for CREMI, 6 for SNEMI
    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if dt.ndim == 3:
        return _eval_cremi(dt, gt, ignore_border)
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            out.append(
                _eval_cremi(dt[i], gt, ignore_border))
        print_table(out)
        return out
    else:
        raise NotImplementedError


def eval_snemi(fn_dt, fn_gt):
    return eval_cremi(fn_dt, fn_gt, ignore_border=25/6.0)
    # from connectomics.utils.evaluate import adapted_rand, voi

    # dt = load(fn_dt).astype(np.uint64)
    # gt = load(fn_gt).astype(np.uint64)

    # if dt.ndim == 3:
    #     adapted_rand_score = adapted_rand(dt, gt)
    #     voi_split, voi_merge = voi(dt, gt)
    #     print('adapted_RAND: {}'.format(adapted_rand_score))
    #     print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
    #     return adapted_rand_score
    # elif dt.ndim == 4:
    #     out = []
    #     for i in range(dt.shape[0]):
    #         adapted_rand_score = adapted_rand(dt[i], gt)
    #         voi_split, voi_merge = voi(dt[i], gt)
    #         print('slice {}: adapted_RAND = {}'.format(i, adapted_rand_score))
    #         print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
    #         out.append(adapted_rand_score)
    #     return out 
    # else:
    #     raise NotImplementedError


def eval_skimage(fn_dt, fn_gt):
    # adapted_rand_error == eval_cremi (border=0)
    # variation_of_information == eval_cremi (ignore_groundtruth=[])
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information 

    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if dt.ndim == 3:
        adapted_rand_score = adapted_rand_error(gt, dt)[0]
        voi_split, voi_merge = variation_of_information(gt, dt)
        print('adapted_RAND: {}'.format(adapted_rand_score))
        print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
        return adapted_rand_score
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            adapted_rand_score = adapted_rand_error(gt, dt[i])[0]
            voi_split, voi_merge = variation_of_information(gt, dt[i])
            print('slice {}: adapted_RAND = {}'.format(i, adapted_rand_score))
            print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
            out.append(adapted_rand_score)
        print_table(out)
        return out 
    else:
        raise NotImplementedError
    

# Calculate voxel-wise acc.

def cal_acc_aff(aff, aff_gt):
    tp = 0
    for i in range(aff_gt.shape[1]):
        gt = (aff_gt[:,i] > 0.5)
        dt = (aff[:,i] > 0.5)
        tp += np.sum(dt == gt)
    return tp / aff_gt.size


def cal_acc(aff, label):
    aff_gt = seg2aff_v0(label)
    return cal_acc_aff(aff, aff_gt)


def _narrow_labels(lab, chunk=32):
    """Downcast a label volume to the smallest unsigned dtype that fits its ids.

    Only ever narrows, never widens, and returns the input untouched when it is
    already narrow enough -- so this cannot change any metric (both skimage
    metrics relabel via np.unique internally) but it can halve or quarter what
    they allocate.

    The max is taken over z-chunks so a memory-mapped array is not fully
    materialized just to find it.
    """
    lab = np.asanyarray(lab)
    if lab.dtype.itemsize <= 2 and lab.dtype.kind == 'u':
        return lab
    mx = 0
    for z0 in range(0, lab.shape[0], chunk):
        blk = lab[z0:z0 + chunk]
        if blk.size:
            mx = max(mx, int(blk.max()))
    for cand in (np.uint16, np.uint32):
        if mx <= np.iinfo(cand).max:
            if np.dtype(cand).itemsize < lab.dtype.itemsize:
                return lab.astype(cand, copy=False)
            break
    return np.asarray(lab)


def _voxel_acc_chunked(fn_affinity, gt, chunk=16):
    """Voxel accuracy between predicted and GT affinity, without materializing
    either affinity volume.

    Equivalent to ``cal_acc_aff(_load_affinity(fn), seg2aff_v0(gt))`` but walks
    z-chunks: the full version needs 12n bytes for the predicted affinity plus
    12n for the one rebuilt from labels (173 GB on the FlyWire crop) just to
    produce one scalar.

    seg2aff_v0's neighbourhood is [[-1,0,0],[0,-1,0],[0,0,-1]] -- each voxel only
    looks one step BACK -- so a chunk that also reads the preceding z slice
    reproduces the whole-volume result exactly, including the pad='replicate'
    treatment of z=0 (which only applies at the true volume start).
    """
    Z = gt.shape[0]
    tp = 0
    total = 0
    _next_report = 0.25
    with h5py.File(fn_affinity, 'r') as f:
        d = f['vol0']
        for z0 in range(0, Z, chunk):
            if Z and z0 / Z >= _next_report:
                stamp('  voxel_acc progress', '%.0f%% (z=%d/%d)' % (100 * z0 / Z, z0, Z))
                _next_report += 0.25
            z1 = min(Z, z0 + chunk)
            lo = max(0, z0 - 1)                 # one slice of halo
            keep = slice(z0 - lo, z0 - lo + (z1 - z0))
            aff_gt = seg2aff_v0(np.asarray(gt[lo:z1]))[:, keep]
            aff = d[:, z0:z1].astype(np.float32) / 255.0
            for c in range(aff_gt.shape[0]):
                tp += int(np.sum((aff[c] > 0.5) == (aff_gt[c] > 0.5)))
            total += aff_gt[0].size * aff_gt.shape[0]
            del aff, aff_gt
    return tp / total


def _seg_metrics_chunked(gt, dt, chunk=16, ignore_labels=(0,)):
    """VOI (split, merge) and adapted-RAND from a chunk-accumulated contingency
    table -- mathematically identical to the whole-volume skimage calls.

    Both metrics are functions of the contingency table n_ij = |{gt==i, dt==j}|
    alone, and that table is a plain sum over voxels, so accumulating it over
    z-chunks is exact (verified: VOI split/merge and ARAND agree with
    skimage to 0 or 1e-15 across four shapes). This matters because the skimage
    calls peak at roughly 4.4x the size of their inputs -- they densify the
    labels to int64 and build the table in one shot -- which on the
    400x4000x4000 FlyWire crop is the difference between ~250 GB and swapping.

    The entropies are computed by skimage's own `_vi_tables` on the assembled
    sparse table, so the VOI definition (including its ignore_labels handling)
    stays exactly the reference one rather than a reimplementation.
    """
    from scipy import sparse
    from skimage.metrics._variation_of_information import _vi_tables

    ignore = np.asarray(ignore_labels)

    # The (i, j) pair encoding needs ONE stride fixed for the whole volume. An
    # earlier version recomputed it per chunk from that chunk's max id, so the
    # same (i, j) encoded to different keys in different chunks while the decode
    # after the loop used the final stride -- corrupting the table and making the
    # result depend on `chunk`. Take the stride from dt's dtype instead of its
    # values: it is an upper bound by construction, needs no pass over the data,
    # and is identical for every chunk.
    # Using the dtype's upper bound as the stride needs no data pass but is far
    # too loose -- uint32/uint32 inputs, which _narrow_labels routinely produces,
    # would overflow int64. So scan for the actual maxima. Only the maxima are
    # taken here, which is a fraction of the cost of the counting pass below.
    gt_max = 0
    dt_max = 0
    for z0 in range(0, gt.shape[0], chunk):
        g = np.asarray(gt[z0:z0 + chunk])
        d = np.asarray(dt[z0:z0 + chunk])
        if not (np.issubdtype(g.dtype, np.integer)
                and np.issubdtype(d.dtype, np.integer)):
            raise TypeError('gt and dt must have integer dtypes, got %s and %s'
                            % (g.dtype, d.dtype))
        if g.size:
            gt_max = max(gt_max, int(g.max()))
            dt_max = max(dt_max, int(d.max()))
        del g, d
    stride = dt_max + 1
    if gt_max > (np.iinfo(np.int64).max - dt_max) // stride:
        raise OverflowError(
            'label ids too large to pack: gt_max=%d dt_max=%d exceeds int64'
            % (gt_max, dt_max))

    # Per-chunk (gt_id, pred_id, count) triples, reduced once at the end.
    gi_parts = []
    di_parts = []
    cnt_parts = []
    n_kept = 0
    for z0 in range(0, gt.shape[0], chunk):
        g = np.asarray(gt[z0:z0 + chunk]).reshape(-1).astype(np.int64, copy=False)
        d = np.asarray(dt[z0:z0 + chunk]).reshape(-1).astype(np.int64, copy=False)
        if ignore.size:
            keep = ~np.isin(g, ignore)
            g, d = g[keep], d[keep]
        if g.size == 0:
            continue
        n_kept += g.size
        # Group this chunk's pairs. np.unique over a packed 1-D key sorts int64s;
        # np.unique(..., axis=0) over stacked rows would do a lexicographic sort
        # instead and measured ~10x slower, so pack. This sort is the dominant
        # cost of the function and is inherent to grouping -- coo/csr assembly
        # from raw voxels measures the same to within 10%.
        uniq, cnt = np.unique(g * stride + d, return_counts=True)
        gi_parts.append(uniq // stride)
        di_parts.append(uniq % stride)
        cnt_parts.append(cnt.astype(np.int64, copy=False))
        del g, d, uniq, cnt

    if not cnt_parts or n_kept == 0:
        return float('nan'), float('nan'), float('nan')

    gi = np.concatenate(gi_parts)
    di = np.concatenate(di_parts)
    n = np.concatenate(cnt_parts)
    del gi_parts, di_parts, cnt_parts
    R = int(gi.max()) + 1
    K = int(di.max()) + 1
    # A pair may appear in several chunks; coo->csr sums those duplicates. Doing
    # it here (on the per-chunk tables, not the voxels) is cheap.
    acc = sparse.coo_matrix((n, (gi, di)), shape=(R, K)).tocsr()
    acc.sum_duplicates()
    coo = acc.tocoo()
    gi, di, n = coo.row, coo.col, coo.data
    del acc, coo

    pxy = sparse.csr_matrix((n / n_kept, (gi, di)), shape=(R, K))
    # _vi_tables only uses `table`, but still shape-checks its first two args.
    _dummy = np.zeros(1, dtype=np.int64)
    hxgy, hygx = _vi_tables(_dummy, _dummy, table=pxy)
    voi_split, voi_merge = float(hygx.sum()), float(hxgy.sum())

    # adapted-RAND as skimage defines it: 1 - F1 over co-clustered voxel pairs.
    ni = np.bincount(gi, weights=n, minlength=R)
    nj = np.bincount(di, weights=n, minlength=K)
    same = (n * (n - 1) / 2).sum()
    tot_gt = (ni * (ni - 1) / 2).sum()
    tot_dt = (nj * (nj - 1) / 2).sum()
    prec = same / tot_dt if tot_dt > 0 else 0.0
    rec = same / tot_gt if tot_gt > 0 else 0.0
    arand = 1.0 - (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 1.0
    return voi_split, voi_merge, arand


def eval_full(fn_gt, fn_dt, fn_affinity=None):
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information

    # Memory is the binding constraint on whole-volume evaluation, and once the
    # process starts swapping the runtime blows up: on the 400x4000x4000 FlyWire
    # crop this function reached 328 GB RSS and was still running after 108
    # minutes, while the two skimage metrics alone extrapolate to ~33 min.
    #
    # `astype` is avoided where it only widens dtypes for no reason: both
    # adapted_rand_error and variation_of_information relabel their inputs via
    # np.unique internally, so uint16/uint32 labels work as-is. Forcing uint64
    # cost 8 bytes/voxel on each of gt and dt (51 GB each on FlyWire), and for a
    # segments.npy that is already uint64 it was a pure duplicate copy.
    stamp('START', 'gt=%s dt=%s aff=%s' % (
        os.path.basename(fn_gt), os.path.basename(fn_dt),
        os.path.basename(fn_affinity) if fn_affinity else '(none)'))

    t = time.time()
    dt = load(fn_dt)
    stamp('load prediction done', '%.1fs, %s %s, %.2f GB' % (
        time.time() - t, dt.shape, dt.dtype, dt.nbytes / 1e9))

    t = time.time()
    gt = load(fn_gt)
    stamp('load ground truth done', '%.1fs, %s %s, %.2f GB' % (
        time.time() - t, gt.shape, gt.dtype, gt.nbytes / 1e9))

    # Narrow the label dtypes as far as the actual id range allows. The skimage
    # metrics build a sparse contingency table and peak at ~4.4x the size of
    # their inputs (measured), so halving an input is worth ~113 GB on the
    # FlyWire crop -- where segments.npy is stored uint64 but its largest id is
    # only ~1.2e6. Relabeling is internal to both metrics, so the narrower dtype
    # gives bit-identical results.
    t = time.time()
    dt = _narrow_labels(dt)
    gt = _narrow_labels(gt)
    stamp('narrow label dtypes', '%.1fs, dt->%s gt->%s' % (
        time.time() - t, dt.dtype, gt.dtype))

    if fn_affinity is not None:
        # voxel_acc needs the predicted affinity (12n as float32) plus an
        # affinity rebuilt from the GT labels (another 12n) -- 173 GB on FlyWire
        # for a single scalar. Compute it in z-chunks instead of materializing
        # both volumes.
        t = time.time()
        voxel_acc = _voxel_acc_chunked(fn_affinity, gt)
        stamp('voxel_acc done', '%.1fs, %.6f' % (time.time() - t, voxel_acc))

    if dt.ndim == 3:
        t = time.time()
        adapted_rand_score = adapted_rand_error(gt, dt)[0]
        stamp('adapted_rand done', '%.1fs, %.6f' % (time.time() - t, adapted_rand_score))
        t = time.time()
        voi_split, voi_merge = variation_of_information(gt, dt, ignore_labels=0)
        stamp('VOI done', '%.1fs, split %.6f merge %.6f' % (
            time.time() - t, voi_split, voi_merge))
        voi_sum = voi_split + voi_merge

        metrics = {
            'voi_split': voi_split,
            'voi_merge': voi_merge,
            'voi_sum': voi_sum,
            'adapted_RAND': adapted_rand_score,
        }
        if fn_affinity is not None:
            metrics['voxel_acc'] = voxel_acc
        print("raw:\n", metrics)
        print("\nmarkdown:")
        print_table([metrics])
        stamp('ALL DONE')

    elif dt.ndim == 4:
        metrics_list = []

        for i in range(dt.shape[0]):
            adapted_rand_score = adapted_rand_error(gt, dt[i])[0]
            voi_split, voi_merge = variation_of_information(gt, dt[i], ignore_labels=0)
            voi_sum = voi_split + voi_merge

            metrics = {
                'voi_split': voi_split,
                'voi_merge': voi_merge,
                'voi_sum': voi_sum,
                'adapted_RAND': adapted_rand_score,
            }
            if fn_affinity is not None:
                metrics['voxel_acc'] = voxel_acc
            metrics_list.append(metrics)

        for i, metrics in enumerate(metrics_list):
            print(f"slice {i}:")
            print(metrics)

    else:
        raise NotImplementedError


def pixelwise_eval(output_affs, label):
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

    gt_affs = seg2aff_v0(label)

    print('MSE...')
    output_affs_prop = output_affs.copy()
    whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
    print('BCE...')
    output_affs = np.clip(output_affs, 0.000001, 0.999999)
    bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
    whole_bce = np.sum(bce) / np.size(gt_affs)
    output_affs[output_affs <= 0.5] = 0
    output_affs[output_affs > 0.5] = 1
    print('F1...')
    whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
    # new
    print('F1 boundary...')
    whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
    print('mAP...')
    whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
    print('AUC...')
    whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())

    print('ACC...')
    voxel_acc = cal_acc(output_affs, label)

    import torch
    from topo_consistency_loss import getTopoLoss
    aff = torch.tensor(output_affs_prop)
    gt = torch.tensor(gt_affs)
    loss = []
    for i in range(3):
        for j in range(16):
            loss.append(
                getTopoLoss(aff[i][j].cuda(), gt[i][j].cuda(), topo_size=100, pd_threshold=0.7).cpu().numpy()
            )
    topo_loss = np.mean(loss)

    return dict(
        voxel_acc = voxel_acc,
        voxel_mse = whole_mse,
        voxel_bce = whole_bce,
        voxel_F1 = whole_arand,
        boundary_F1 = whole_arand_bound,
        boundary_map = whole_map,
        boundary_auc = whole_auc,
        topo_loss = topo_loss
    )


def eval_full_v2(fn_gt, fn_dt, fn_affinity=None):
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information 

    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if fn_affinity is not None:
        print("Loading affinity ...")
        aff = _load_affinity(fn_affinity)
        print("Evaluating voxelwise ...")
        voxel_metrics = pixelwise_eval(aff, gt)

    if dt.ndim == 3:
        adapted_rand_score = adapted_rand_error(gt, dt)[0]
        voi_split, voi_merge = variation_of_information(gt, dt, ignore_labels=0)
        voi_sum = voi_split + voi_merge

        metrics = {
            'voi_split': voi_split,
            'voi_merge': voi_merge,
            'voi_sum': voi_sum,
            'adapted_RAND': adapted_rand_score,
        }
        if fn_affinity is not None:
            metrics.update(voxel_metrics)
        print("raw:\n", metrics)
        print("\nmarkdown:")
        print_table([metrics])
        stamp('ALL DONE')

    elif dt.ndim == 4:
        metrics_list = []

        for i in range(dt.shape[0]):
            adapted_rand_score = adapted_rand_error(gt, dt[i])[0]
            voi_split, voi_merge = variation_of_information(gt, dt[i], ignore_labels=0)
            voi_sum = voi_split + voi_merge

            metrics = {
                'voi_split': voi_split,
                'voi_merge': voi_merge,
                'voi_sum': voi_sum,
                'adapted_RAND': adapted_rand_score,
            }
            if fn_affinity is not None:
                metrics.update(voxel_metrics)
            metrics_list.append(metrics)

        for i, metrics in enumerate(metrics_list):
            print(f"slice {i}:")
            print(metrics)

    else:
        raise NotImplementedError
    

def pixelwise_eval_v3(output_affs, label):
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

    gt_affs = seg2aff_v0(label)

    print('MSE...')
    output_affs_prop = output_affs.copy()
    whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
    print('BCE...')
    output_affs = np.clip(output_affs, 0.000001, 0.999999)
    bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
    whole_bce = np.sum(bce) / np.size(gt_affs)
    output_affs[output_affs <= 0.5] = 0
    output_affs[output_affs > 0.5] = 1
    print('F1...')
    whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
    # new
    print('F1 boundary...')
    whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
    print('mAP...')
    whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
    print('AUC...')
    whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())

    print('ACC...')
    voxel_acc = cal_acc(output_affs, label)

    import torch
    import torch.nn.functional as F

    from connectomics.model.loss import DiceLoss
    from topo_consistency_loss import getTopoLoss

    aff = torch.tensor(output_affs_prop)
    gt = torch.tensor(gt_affs)
    loss = []
    for i in range(3):
        for j in range(16):
            loss.append(
                getTopoLoss(aff[i][j].cuda(), gt[i][j].cuda(), topo_size=100, pd_threshold=0.7).cpu().numpy()
            )
    topo_loss = np.mean(loss)

    bce_loss = F.binary_cross_entropy(aff, gt)
    dice_loss = DiceLoss()(aff, gt)

    return dict(
        voxel_acc = voxel_acc,
        voxel_mse = whole_mse,
        voxel_bce = whole_bce,
        voxel_F1 = whole_arand,
        boundary_F1 = whole_arand_bound,
        boundary_map = whole_map,
        boundary_auc = whole_auc,
        topo_loss = topo_loss,
        bce_loss = bce_loss.item(),
        dice_loss = dice_loss.item()
    )

def eval_full_v3(fn_gt, fn_dt, fn_affinity=None):
    """Evaluate the metrics and losses."""
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information 

    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if fn_affinity is not None:
        print("Loading affinity ...")
        aff = _load_affinity(fn_affinity)
        print("Evaluating voxelwise ...")
        voxel_metrics = pixelwise_eval_v3(aff, gt)

    if dt.ndim == 3:
        adapted_rand_score = adapted_rand_error(gt, dt)[0]
        voi_split, voi_merge = variation_of_information(gt, dt, ignore_labels=0)
        voi_sum = voi_split + voi_merge

        metrics = {
            'voi_split': voi_split,
            'voi_merge': voi_merge,
            'voi_sum': voi_sum,
            'adapted_RAND': adapted_rand_score,
        }
        if fn_affinity is not None:
            metrics.update(voxel_metrics)
        print("raw:\n", metrics)
        print("\nmarkdown:")
        print_table([metrics])
        stamp('ALL DONE')

    elif dt.ndim == 4:
        metrics_list = []

        for i in range(dt.shape[0]):
            adapted_rand_score = adapted_rand_error(gt, dt[i])[0]
            voi_split, voi_merge = variation_of_information(gt, dt[i], ignore_labels=0)
            voi_sum = voi_split + voi_merge

            metrics = {
                'voi_split': voi_split,
                'voi_merge': voi_merge,
                'voi_sum': voi_sum,
                'adapted_RAND': adapted_rand_score,
            }
            if fn_affinity is not None:
                metrics.update(voxel_metrics)
            metrics_list.append(metrics)

        for i, metrics in enumerate(metrics_list):
            print(f"slice {i}:")
            print(metrics)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    Fire()
    # dt = np.load('resized_mask_int.npy').astype(np.uint64)[0]
    # gt = np.load('target.npy').astype(np.uint64)[0]

    # print( 
    #     _eval_cremi(dt, gt, ignore_border=25.0/4)
    # )




# e.g. python eval.py \
#  outputs/230224_1522_pretrained/test/segments.npy \
#  datasets/230224_1522/label_val.tiff 

# waterz_log (threshold=0.7):
# Rand split: 0.912244
# Rand merge: 0.979684
# VOI split: 0.393357
# VOI merge: 0.166184

# waterz.evaluate_total_volume:
# Rand split: 0.856483
# Rand merge: 0.936697
# VOI split: 0.752307
# VOI merge: 0.47017

# waterz.evaluate_total_volume with ignore 25/4.0 border:
# Rand split: 0.912244
# Rand merge: 0.979684
# VOI split: 0.393357
# VOI merge: 0.166184

# cremi:
# voi split   : 0.7523071905458876
# voi merge   : 0.4701700846163155
# adapted RAND: 0.10520416422220435
# note: adapted RAND = 1 - 2 / (1/Rand split + 1/Rand merge)