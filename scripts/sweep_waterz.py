"""Sweep waterz agglomeration thresholds over one affinity volume.

post_process.py runs a single threshold and overwrites segments.npy, so scanning
N thresholds through it costs N watersheds (identical every time -- watershed
only sees the affinity, not the threshold) and leaves the metrics scattered
across the log with no summary.

This script does the watershed once and then walks the thresholds in a single
waterz.agglomerate call. waterz merges incrementally and yields one
segmentation per threshold, so an ascending sweep is the same total work as the
single largest threshold. Metrics for every threshold land in one CSV.

Usage (from the pytorch_connectomics project root):
    python -u scripts/sweep_waterz.py \
        --res_h5    exps/.../CremiC/result.h5 \
        --label     datasets/cremi/sample_C_test_labels.h5 \
        --out_dir   exps/.../CremiC \
        --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

Writes:
    <out_dir>/sweep_waterz.csv          one row per threshold
    <out_dir>/segments_t<thresh>.npy    only with --save_segments
"""
import csv
import os
import os.path as osp
import sys
import time

import h5py
import numpy as np
from fire import Fire

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from post_process import relabel, stamp, watershed  # noqa: E402


def _parse_thresholds(thresholds):
    """Accept 0.1,0.2 / [0.1, 0.2] / 0.5 and return a sorted float list."""
    if isinstance(thresholds, str):
        vals = [float(t) for t in thresholds.replace(' ', '').split(',') if t]
    elif isinstance(thresholds, (int, float)):
        vals = [float(thresholds)]
    else:
        vals = [float(t) for t in thresholds]
    # waterz merges incrementally: the sweep must be ascending, or a later
    # threshold would ask for a coarser segmentation than one already produced.
    return sorted(set(vals))


def _load_affinity(res_h5):
    """Load affinity as float32 in [0,1].

    Two on-disk conventions in this project:
      * pytorch_connectomics / CAD: key 'vol0', uint8 = float*255
      * PEA:    key 'main', float32 already in [0,1]
    Pick whichever key is present and scale only when the dtype says to.
    """
    with h5py.File(res_h5, 'r') as f:
        keys = list(f.keys())
        key = next((k for k in ('vol0', 'main') if k in keys), None)
        if key is None:
            if len(keys) != 1:
                raise KeyError('no vol0/main in %s, and %d keys to choose from: %s'
                               % (res_h5, len(keys), keys))
            key = keys[0]
        dset = f[key]
        stamp('h5 open', f'key={key} shape={dset.shape} dtype={dset.dtype}')
        data = dset[()]
    affinities = data.astype(np.float32, copy=(data.dtype != np.float32))
    if data.dtype == np.uint8:
        affinities /= 255.0
    del data
    if affinities.ndim != 4 or affinities.shape[0] < 3:
        raise ValueError('expected affinity (C>=3, D, H, W), got %s'
                         % (affinities.shape,))
    # Some models emit extra channels beyond the 3 waterz needs.
    return np.ascontiguousarray(affinities[:3])


# Metrics are rounded to 3 decimals on the way into the csv. Differences past
# the third decimal are far below run-to-run variation, and full float repr makes
# the csv unreadable (0.736901234567... columns).
NDIGITS = 3


def _metrics(gt, seg):
    from skimage.metrics import adapted_rand_error, variation_of_information
    arand = adapted_rand_error(gt, seg)[0]
    voi_split, voi_merge = variation_of_information(gt, seg, ignore_labels=0)
    return {
        'voi_split': round(float(voi_split), NDIGITS),
        'voi_merge': round(float(voi_merge), NDIGITS),
        # Summed before rounding, so the column is the true sum rather than the
        # sum of two rounded numbers.
        'voi_sum': round(float(voi_split) + float(voi_merge), NDIGITS),
        'adapted_RAND': round(float(arand), NDIGITS),
    }


# Named waterz scoring functions. 'mean' is what post_process.py uses; 'quantile'
# is what CAD's inference_affs.py uses, so a sweep over CAD affinity should pass
# --scoring quantile to stay comparable with the numbers CAD prints itself.
SCORING_FUNCTIONS = {
    'mean': ('OneMinus<EdgeStatisticValue<RegionGraphType, '
             'MeanAffinityProvider<RegionGraphType, ScoreValue>>>'),
    'quantile': ('OneMinus<HistogramQuantileAffinity<RegionGraphType, '
                 '50, ScoreValue, 256>>'),
}


def main(res_h5,
         label=None,
         out_dir=None,
         thresholds='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
         seed_method='maxima_distance',
         scoring='mean',
         save_segments=False,
         csv_name='sweep_waterz.csv'):
    thresholds = _parse_thresholds(thresholds)
    if scoring not in SCORING_FUNCTIONS and '<' not in str(scoring):
        raise ValueError('unknown --scoring %r; use one of %s or a raw waterz '
                         'C++ type string' % (scoring, list(SCORING_FUNCTIONS)))
    out_dir = out_dir or osp.dirname(osp.abspath(res_h5))
    os.makedirs(out_dir, exist_ok=True)
    stamp('START', f'{res_h5} thresholds={thresholds}')

    affinities = _load_affinity(res_h5)

    gt = None
    if label:
        from eval import cal_acc, load, print_table
        gt = load(label).astype(np.uint64)
        voxel_acc = round(float(cal_acc(affinities, gt)), NDIGITS)
        stamp('voxel acc', '%.3f' % voxel_acc)
    else:
        from eval import print_table
        print('No --label given: sweeping without metrics (segments only).')
        voxel_acc = None

    t0 = time.time()
    fragments = watershed(affinities, seed_method)
    stamp('watershed done', '%.1fs, %d fragments' % (
        time.time() - t0, int(fragments.max())))
    fragments = fragments.astype(np.uint64, copy=True)

    sf = SCORING_FUNCTIONS.get(scoring, scoring)   # named, or a raw C++ type string
    stamp('scoring function', '%s -> %s' % (scoring, sf))

    def _agglomerate(force_rebuild):
        import waterz
        return waterz.agglomerate(affinities, thresholds,
                                  fragments=fragments,
                                  force_rebuild=force_rebuild,
                                  scoring_function=sf,
                                  discretize_queue=256)

    try:
        # force_rebuild=False reuses the cached cython module in ~/.cython/inline
        # (same reasoning as post_process.py); the except branch recompiles if
        # the cache is missing or stale.
        gen = _agglomerate(False)
        gen = iter(gen)
        first = next(gen)
    except ModuleNotFoundError as e:
        print(f'rebuild waterz for {e}')
        gen = iter(_agglomerate(True))
        first = next(gen)

    rows = []
    for i, thresh in enumerate(thresholds):
        seg = first if i == 0 else next(gen)
        stamp('agglomerate done', 'thresh=%.3f' % thresh)
        # waterz reuses its internal buffer across yields, so copy before
        # relabel rewrites it in place -- otherwise the next threshold would
        # continue merging from compacted ids.
        seg = np.array(seg, dtype=np.uint64, copy=True)
        seg = relabel(seg)
        n_seg = int(seg.max())

        row = {'threshold': thresh, 'num_segments': n_seg}
        if gt is not None:
            row.update(_metrics(gt, seg))
            row['voxel_acc'] = voxel_acc
        rows.append(row)
        print('thresh=%.3f  segments=%d  %s' % (
            thresh, n_seg,
            ' '.join('%s=%.3f' % (k, v) for k, v in row.items()
                     if k not in ('threshold', 'num_segments'))), flush=True)

        if save_segments:
            fn = osp.join(out_dir, 'segments_t%s.npy' % ('%g' % thresh))
            np.save(fn, seg)
            stamp('saved', fn)
        del seg

    csv_path = osp.join(out_dir, csv_name)
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    stamp('csv written', csv_path)

    print('\nmarkdown:')
    print_table(rows)
    if gt is not None:
        best = min(rows, key=lambda r: r['voi_sum'])
        print('\nbest voi_sum: %.3f at threshold %g '
              '(voi_split=%.3f voi_merge=%.3f arand=%.3f)' % (
                  best['voi_sum'], best['threshold'], best['voi_split'],
                  best['voi_merge'], best['adapted_RAND']))
    stamp('ALL DONE')


if __name__ == '__main__':
    Fire(main)
