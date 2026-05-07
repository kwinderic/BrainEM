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


def eval_full(fn_gt, fn_dt, fn_affinity=None):
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information 

    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if fn_affinity is not None:
        aff = _load_affinity(fn_affinity)
        voxel_acc = cal_acc(aff, gt)

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
            metrics['voxel_acc'] = voxel_acc
        print("raw:\n", metrics)
        print("\nmarkdown:")
        print_table([metrics])

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