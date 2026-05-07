import os
import os.path as osp
import pickle as pkl
import numpy as np 
from tqdm import tqdm
from collections import defaultdict
import logging

from scripts.eval import load, _eval_cremi, _eval_skimage



def zyx_from_fn(filename):
    # z, x, y = map(int, os.path.splitext(os.path.basename(filename))[0].split('_'))            # FIB25
    try:
        z, y, x = map(int, osp.splitext(osp.basename(filename))[0].split('_'))
        return z, y, x
    except:
        return -1, -1, -1


def blockwise_eval(label_path, preds_path, eval_info_path, 
                   block_shape=(17, 257, 257), 
                   ignore_border=0
                ):
    block_metrics  = defaultdict(list)

    d, h, w = block_shape
    fns = os.listdir(preds_path)
    label = load(label_path)

    for fn in tqdm(fns):
        z, y, x = zyx_from_fn(fn)
        if z < 0:
            logging.warning(f'Invalid filename: {fn}')
            continue

        block_pred = pkl.load(open(osp.join(preds_path, fn), 'rb'))['mask_int']
        block_label = label[z:z+d, y:y+h, x:x+w]

        # metric = _eval_cremi(block_pred.astype(np.int64), block_label, ignore_border, print_metrics=False)
        # for k, v in metric.items():
        #     block_metrics[k].append(v)

        metric_skimage = _eval_skimage(block_pred.astype(np.int64), block_label, print_metrics=False)
        for k, v in metric_skimage.items():
            block_metrics['ski_' + k].append(v)

    for k in block_metrics:
        print(f'{k}: {np.mean(block_metrics[k])}')
    
    os.makedirs(osp.dirname(eval_info_path), exist_ok=True)
    with open(eval_info_path, 'wb') as f:
        pkl.dump(block_metrics, f)
    print(f'Evaluation info saved to {eval_info_path}')
    return block_metrics


if __name__ == '__main__':
    blockwise_eval(
        'datasets/AC3-AC4/AC3_labels.h5',
        'exps/full_pipeline/AC3AC4/outputs/affinity_knet_cc_half_esc',
        'exps/full_pipeline/AC3AC4/outputs/affinity_knet_cc_half_esc/eval_info.pkl'
    )
