import h5py
import mahotas
import numpy as np
from scipy import ndimage
import sys
# sys.path.append('/home/chenhang/data1/pytorch_connectomics/waterz')

import waterz
import numpy as np
from fire import Fire
from PIL import Image
import os 
import os.path as osp 
import time 
from concurrent.futures import ThreadPoolExecutor


def tick():
    return time.time()


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
    """Wall-clock + elapsed + RSS for one pipeline milestone.

    Whole-volume post-processing on the large crops runs for a long time with no
    output, which makes it impossible to tell a slow stage from a hung one. Every
    milestone is timestamped so a run can be diagnosed from its log alone.
    """
    now = time.time()
    print('[%s | +%7.1fs | RSS %6.2f GB] %-26s %s' % (
        time.strftime('%H:%M:%S', time.localtime(now)),
        now - _T0, _rss_gb(), event, extra), flush=True)
    return now


def watershed(affs, seed_method, use_mahotas_watershed=True):
    """2D per-slice watershed over the xy boundary map.

    Memory notes (this runs on whole volumes, so every full-size array counts;
    n = number of voxels):
      * affs_xy is built in place. ``1.0 - 0.5*(affs[1] + affs[2])`` allocates
        the sum and then a second array for the expression, i.e. 2n instead of n.
      * fragments is allocated at its final dtype. ``np.zeros_like(affs[0])``
        would build a float32 array (4n) and ``.astype(np.uint64)`` then copies
        it into a fresh 8n array -- 4n of that is pure waste, measured at
        +0.8 GB on a 200x1000x1000 volume.
      * fragments starts as uint32: ids are assigned per slice and the running
        total stays far below 2^32 (asserted below). waterz needs uint64, so the
        caller widens it right before the call and drops the narrow copy.
    """
    _t = time.time()
    affs_xy = affs[1] + affs[2]          # n
    affs_xy *= -0.5                      # in place
    affs_xy += 1.0                       # in place
    stamp('  ws: affs_xy built', '%.1fs' % (time.time() - _t))
    depth = affs_xy.shape[0]
    fragments = np.zeros(affs_xy.shape, dtype=np.uint32)

    # Slices are independent, and mahotas releases the GIL, so this parallelises
    # with plain threads (no pickling of slices). cwatershed dominates: measured
    # 80% of the per-slice cost, and the whole loop went 12.6 s -> 2.7 s on a
    # 25x1250x1250 volume with 16 threads.
    #
    # Ids must still come out exactly as the serial version numbered them, and
    # each slice's base id depends on how many seeds every earlier slice found.
    # So run seeding first (parallel, independent), take the prefix sum, and only
    # then offset and watershed -- rather than sharing a running counter.
    n_threads = min(int(os.environ.get("POSTPROC_THREADS", 16)), max(depth, 1))

    def _seed(z):
        return get_seeds(affs_xy[z], next_id=1, method=seed_method)

    stamp('  ws: seeding START', '%d slices, %d threads' % (depth, n_threads))
    _t = time.time()
    if n_threads > 1:
        with ThreadPoolExecutor(n_threads) as ex:
            seeded = list(ex.map(_seed, range(depth)))
    else:
        seeded = [_seed(z) for z in range(depth)]
    stamp('  ws: seeding done', '%.1fs' % (time.time() - _t))

    # Exclusive prefix sum of seed counts == the serial next_id at each slice.
    bases = np.cumsum([0] + [ns for _, ns in seeded[:-1]]) + 1
    total_ids = int(bases[-1] + seeded[-1][1]) if depth else 1
    assert total_ids < 2**32, (
        f"fragment ids reached {total_ids}, too many for uint32 -- widen the "
        f"fragments dtype in watershed()")

    def _flood(z):
        seeds, _ = seeded[z]
        # get_seeds built ids at base 1; shift to this slice's range. The zero
        # background must stay zero.
        shifted = seeds.copy()
        nz = shifted > 0
        shifted[nz] += int(bases[z]) - 1
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], shifted)
        else:
            fragments[z] = ndimage.watershed_ift(
                (255.0*affs_xy[z]).astype(np.uint8), shifted)
        seeded[z] = None      # release the seed array as soon as it is used

    stamp('  ws: flood START', '%d total seed ids' % total_ids)
    _t = time.time()
    if n_threads > 1:
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(_flood, range(depth)))
    else:
        for z in range(depth):
            _flood(z)
    stamp('  ws: flood done', '%.1fs' % (time.time() - _t))

    del affs_xy
    return fragments

def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds


def relabel(seg, chunk=16):
    """Compact segment ids to 1..N, in place.

    Rewrites `seg` rather than returning `mapping[seg]`: fancy-indexing a whole
    volume allocates a second full-size array (8n for uint64), and on top of the
    incoming segmentation that was the peak of the whole script.

    A whole-volume ``np.unique`` is also avoided -- it flattens and sorts a full
    copy, a transient 8n (measured +0.49 GB on a 49 Mvox uint64 volume). Marking
    occupied ids in a boolean lookup while walking z-chunks needs only
    (maxid + 1) bytes plus one chunk at a time, and keeps everything in the
    segmentation's own dtype (no int64 round-trip of the voxel data).
    """
    maxid = int(seg.max())
    if maxid == 0:
        return seg
    present = np.zeros(maxid + 1, dtype=bool)
    for z0 in range(0, seg.shape[0], chunk):
        present[seg[z0:z0 + chunk]] = True
    uid = np.flatnonzero(present)
    uid = uid[uid > 0]
    if uid.size == 0:
        return seg

    mapping = np.zeros(maxid + 1, dtype=seg.dtype)
    mapping[uid] = np.arange(1, uid.size + 1, dtype=seg.dtype)
    # Apply per z-chunk so the temporary is chunk/depth of the volume.
    for z0 in range(0, seg.shape[0], chunk):
        sl = slice(z0, z0 + chunk)
        seg[sl] = mapping[seg[sl]]
    return seg


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)


def main(res_h5, out_dir=".", thresh=0.5,gt_tif=None, aff_thresholds=[0.05, 0.995], 
        #  seg_thresholds=[0.1, 0.3, 0.6]):
         seg_thresholds=[0.4, 0.5, 0.6]):
        #  seg_thresholds=[0.2, 0.4, 0.6, 0.8, 0.9, 0.95]):
    # Memory: this is a whole-volume CPU pass, so live arrays dominate. With
    # n = voxel count the original peaked at ~48n (nothing was ever released);
    # freeing each array as soon as it is consumed brings the peak to ~28n, and
    # the peak now sits mid-run instead of at the end. For the 400x4000x4000
    # FlyWire crop that is 179 GB instead of 307 GB.
    stamp('START', f'{res_h5} thresh={thresh}')
    with h5py.File(res_h5, 'r') as f:
        dset = f['vol0']
        stamp('h5 open', f'shape={dset.shape} dtype={dset.dtype} '
                         f'chunks={dset.chunks} compression={dset.compression}')

        t = tick()
        data = dset[()]                            # 3n  uint8
        stamp('h5 read done', '%.1fs, %.2f GB raw' % (
            tick() - t, data.nbytes / 1e9))

        t = tick()
        affinities = data.astype(np.float32)       # 12n float32
        affinities /= 255.0                        # in place
        del data                                   # -3n: unused from here on
        stamp('uint8 -> float32 done', '%.1fs' % (tick() - t))

        print("!!thresh:",thresh)
        if gt_tif is not None:
            seg_gt = read_tiff(gt_tif)
            seg_gt = seg_gt.astype(np.uint32)
        else:
            seg_gt = None

        if not osp.exists(out_dir):
            os.makedirs(out_dir)


        t0 = tick()
        stamp('watershed START')
        fragments = watershed(affinities, 'maxima_distance')   # 4n uint32
        # np.save(osp.join(out_dir, 'fragments.npy'), fragments)
        # fragments = watershed(affinities, 'grid')
        stamp('watershed done', '%.1fs, %d fragments' % (
            tick() - t0, int(fragments.max())))

        # waterz requires uint64 fragments; widen only now and drop the narrow
        # copy, so the two dtypes coexist for one statement instead of the whole
        # watershed pass.
        t = tick()
        fragments = fragments.astype(np.uint64, copy=True)     # +8n, then -4n
        stamp('fragments -> uint64', '%.1fs' % (tick() - t))

        t1 = tick()
        print(f'WaterZ: {t1 - t0:.2f} seconds')

        # fragments = waterz.watershed(affinities, 'maxima_distance')
        
        t0 = tick()
        stamp('agglomerate START')
        sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
        try:
            # force_rebuild=False reuses the cached cython module in
            # ~/.cython/inline. Rebuilding costs 6.5 s per run and produces the
            # identical module (verified), so it is pure overhead here -- the
            # except branch below still recompiles if the cache is missing or
            # stale, which is what it was there for.
            segmentation = list(waterz.agglomerate(affinities, [thresh],
                                                fragments=fragments,
                                                force_rebuild=False,
                                                scoring_function=sf,
                                                discretize_queue=256))[0]
        except ModuleNotFoundError as e:
            print(f"rebuid waterz for {e}")
            segmentation = list(waterz.agglomerate(affinities, [thresh],
                                                fragments=fragments,
                                                force_rebuild=True,
                                                scoring_function=sf,
                                                discretize_queue=256))[0]
        stamp('agglomerate done', '%.1fs' % (tick() - t0))

        # Both inputs are consumed by now. Releasing them here is what takes the
        # relabel step off the memory peak: affinities alone is 12n.
        del fragments                              # -8n
        del affinities                             # -12n
        stamp('freed affinities+fragments')

        # waterz already returns uint64; the old `.astype(np.uint64)` copied the
        # whole volume for nothing (numpy copies even when the dtype matches).
        assert segmentation.dtype == np.uint64, segmentation.dtype
        t = tick()
        segmentation = relabel(segmentation)       # in place
        stamp('relabel done', '%.1fs' % (tick() - t))
        t1 = tick()

        print('the max id = %d' % np.max(segmentation))
        print(f'Agglomerate: {t1 - t0:.2f} seconds')

        # f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
        # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
        # f.close()

        t = tick()
        np.save(osp.join(out_dir, 'segments.npy'), segmentation)
        stamp('np.save done', '%.1fs -> %s' % (
            tick() - t, osp.join(out_dir, 'segments.npy')))
        stamp('ALL DONE')


if __name__ == '__main__':
    Fire(main)
    # e.g.
    # main('outputs/230224_1522/test/result.h5', 'outputs/230224_1522/test/', 'datasets/230224_1522/label_val.tiff') 
    # main('outputs/230224_1522/test/result.h5', 'outputs/230224_1522/test/', 'datasets/230224_1522/label_val.tiff') 


# waterz at thresholds [0.1, 0.3, 0.6]
# Compiling waterz in /home/chenhang/.cython/inline
# Preparing segmentation volume...
# counting regions and sizes...
# creating region graph for 403095 nodes
# creating statistics provider
# extracting region graph...
# Region graph number of edges: 3066744
# merging until threshold 0.1
# computing initial scores
# merging until 0.1
# min edge score 0.00195312
# threshold exceeded
# merged 363589 edges
# extracting segmentation
# evaluating current segmentation against ground-truth
#         Rand split: 0.798939
#         Rand merge: 0.98868
#         VOI split: 1.15475
#         VOI merge: 0.0853848
# Storing record...
# merging until threshold 0.3
# merging until 0.3
# min edge score 0.103516
# threshold exceeded
# merged 24233 edges
# extracting segmentation
# evaluating current segmentation against ground-truth
#         Rand split: 0.862964
#         Rand merge: 0.987727
#         VOI split: 0.759203
#         VOI merge: 0.0977759
# Storing record...
# merging until threshold 0.6
# merging until 0.6
# min edge score 0.302734
# threshold exceeded
# merged 12658 edges
# extracting segmentation
# evaluating current segmentation against ground-truth
#         Rand split: 0.893469
#         Rand merge: 0.976649
#         VOI split: 0.4828
#         VOI merge: 0.160527