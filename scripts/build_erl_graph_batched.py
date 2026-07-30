"""Build an ERL skeleton graph from a large GT volume, in object batches.

Why this exists: em_erl's seg_to_graph hands the whole volume to
kimimaro.skeletonize with parallel=num_thread, and each worker allocates its own
float32 buffer the size of the volume. On the 400x4000x4000 FlyWire crop that is
23.8 GiB per worker, so -t 16 dies with ArrayMemoryError even on a 500 GB node.

The fix is to batch over OBJECTS rather than space: kimimaro.skeletonize takes an
`object_ids` argument, so we call it repeatedly on the same volume with a slice of
the id list each time, and keep parallel=1 so only one large buffer is live. Peak
memory is then roughly (volume + one float32 buffer) regardless of object count.

Batching by object rather than by spatial tile also avoids the harder problem
that tiling introduces: a neurite crossing a tile boundary would be skeletonized
as two fragments, which would inflate the split count and depress ERL. Every
object here is skeletonized against the full volume exactly once.

Usage:
    python scripts/build_erl_graph_batched.py \\
        --seg datasets/EM/Flywire/flywire_crop_labels.h5 \\
        --out exps/.../flywire_full_gt_graph_vox.npz \\
        --resolution 1,1,1 --batch 2000

Resolution note: pass 1,1,1 to get ERL in voxels (what em_erl's j0126 workflow
does, and what the cross-source graphs here use). Values are then comparable
across models on this volume but not across volumes.
"""
import os
import os.path as osp
import sys
import time

import numpy as np
from fire import Fire

# em_erl is a separate repo (https://github.com/PytorchConnectomics/em_erl).
# `pip install -e em_erl` makes this import work; otherwise point EM_ERL at the
# clone.
EM_ERL = os.environ.get('EM_ERL')
if EM_ERL:
    sys.path.insert(0, EM_ERL)


def _rss_gb():
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


_T0 = time.time()


def stamp(msg, extra=''):
    print('[+%7.1fs | RSS %6.2f GB] %-26s %s'
          % (time.time() - _T0, _rss_gb(), msg, extra), flush=True)


def main(seg,
         out,
         resolution='1,1,1',
         batch=2000,
         dust_size=100,
         length_threshold=0,
         num_thread=1):
    """Skeletonize `seg` in batches of `batch` object ids and save an ERLGraph."""
    import kimimaro
    from em_erl.erl import skel_to_erlgraph
    from em_erl.io import read_vol

    # Fire parses "1,1,1" into a tuple, so accept both that and a raw string.
    if isinstance(resolution, str):
        res = [float(v) for v in resolution.replace(' ', '').split(',') if v]
    else:
        res = [float(v) for v in resolution]
    if len(res) != 3:
        raise ValueError('--resolution needs 3 values (z,y,x), got %r' % (resolution,))

    stamp('reading volume', seg)
    labels = read_vol(seg)
    stamp('volume loaded', '%s %s' % (labels.shape, labels.dtype))

    ids = np.unique(labels)
    ids = ids[ids > 0]
    stamp('object ids', '%d objects, batch=%d -> %d batches'
          % (len(ids), batch, int(np.ceil(len(ids) / batch))))

    # kimimaro's anisotropy is xyz while our --resolution is zyx.
    aniso = (res[2], res[1], res[0])

    all_skels = {}
    n_batches = int(np.ceil(len(ids) / batch))
    for bi in range(n_batches):
        chunk = list(ids[bi * batch:(bi + 1) * batch])
        t0 = time.time()
        skels = kimimaro.skeletonize(
            labels,
            teasar_params={
                'scale': 1.5,
                'const': 500,
                'pdrf_exponent': 4,
                'pdrf_scale': 100000,
                'soma_detection_threshold': 1100,
                'soma_acceptance_threshold': 3500,
                'soma_invalidation_scale': 1.0,
                'soma_invalidation_const': 300,
                'max_paths': 50,
            },
            object_ids=chunk,
            dust_threshold=dust_size,
            anisotropy=aniso,
            fix_branching=True,
            fix_borders=True,
            progress=False,
            # parallel=1 on purpose: each extra worker copies the whole volume's
            # float32 distance field, which is what breaks on big crops.
            parallel=num_thread,
            parallel_chunk_size=100,
        )
        all_skels.update(skels)
        stamp('batch %d/%d' % (bi + 1, n_batches),
              '%d ids -> %d skeletons (%.1fs, %d total)'
              % (len(chunk), len(skels), time.time() - t0, len(all_skels)))

    del labels
    stamp('skeletonized', '%d skeletons total' % len(all_skels))

    # Nodes come out of kimimaro already in physical units (anisotropy applied),
    # so skeleton_resolution stays None here.
    graph = skel_to_erlgraph(all_skels, skeleton_resolution=None,
                             length_threshold=length_threshold)
    graph.print_info()
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    graph.save_npz(out)
    stamp('saved', out)


if __name__ == '__main__':
    Fire(main)
