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


def tick():
    return time.time()


def watershed(affs, seed_method, use_mahotas_watershed=True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds
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


def relabel(seg):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-background samples
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1 # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]


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
    with h5py.File(res_h5, 'r') as f:
        data = f['vol0'][()]

        affinities = data.astype(np.float32) / 255.0

        print("!!thresh:",thresh)
        if gt_tif is not None:
            seg_gt = read_tiff(gt_tif)
            seg_gt = seg_gt.astype(np.uint32)
        else:
            seg_gt = None

        if not osp.exists(out_dir):
            os.makedirs(out_dir)


        t0 = tick()
        print('Waterz segmentation...')
        fragments = watershed(affinities, 'maxima_distance')
        np.save(osp.join(out_dir, 'fragments.npy'), fragments)
        # fragments = watershed(affinities, 'grid')

        t1 = tick()
        print(f'WaterZ: {t1 - t0:.2f} seconds')

        # fragments = waterz.watershed(affinities, 'maxima_distance')
        
        t0 = tick()
        sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
        try:
            segmentation = list(waterz.agglomerate(affinities, [thresh],
                                                fragments=fragments,
                                                # force_rebuild=False,
                                                force_rebuild=True,
                                                scoring_function=sf,
                                                discretize_queue=256))[0]
        except ModuleNotFoundError as e:
            print(f"rebuid waterz for {e}")
            segmentation = list(waterz.agglomerate(affinities, [thresh],
                                                fragments=fragments,
                                                force_rebuild=True,
                                                scoring_function=sf,
                                                discretize_queue=256))[0]
        segmentation = relabel(segmentation).astype(np.uint64)
        t1 = tick()

        print('the max id = %d' % np.max(segmentation))
        print(f'Agglomerate: {t1 - t0:.2f} seconds')

        # f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
        # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
        # f.close()

        np.save(osp.join(out_dir, 'segments.npy'), segmentation)


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