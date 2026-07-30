"""Self-supervised pretraining dataset for SegNeuron (TF-free, bug-fixed port).

Pretext task (from SegNeuron Pretrain/): take an unlabeled EM patch, split it
into 3D blocks, randomly mask 50-70% of blocks (replace with Gaussian noise),
and ask MNet to reconstruct (a) the clean image and (b) its per-slice HOG map.
Loss (in the trainer) = 0.2 * MSE(img) + MSE(hog).

Volumes are plain .tif/.h5 grayscale stacks (no labels needed).
"""
from __future__ import annotations

import random

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from skimage.feature import hog
from torch.utils.data import Dataset


class SimpleAugment:
    """Random flips (x/y/z) + xy transpose, matching the reference."""
    def __call__(self, vol):
        if random.random() < 0.5:
            vol = vol[::-1]
        if random.random() < 0.5:
            vol = vol[:, ::-1]
        if random.random() < 0.5:
            vol = vol[:, :, ::-1]
        if random.random() < 0.5:
            vol = vol.transpose(0, 2, 1)
        return np.ascontiguousarray(vol)


class SegNeuronPretrainDataset(Dataset):
    def __init__(self, volumes, crop=(20, 128, 128), length=100000):
        """
        volumes: list of numpy 3D arrays (uint8 EM stacks), loaded in memory.
        crop: (z, y, x) patch size. Block/mask math assumes divisibility.
        """
        self.volumes = volumes
        self.crop = crop
        self.aug = SimpleAugment()
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        cz, cy, cx = self.crop
        v = self.volumes[random.randrange(len(self.volumes))]   # fixed: randrange (ref had off-by-one)
        z = random.randint(0, v.shape[0] - cz)
        y = random.randint(0, v.shape[1] - cy)
        x = random.randint(0, v.shape[2] - cx)
        img = v[z:z + cz, y:y + cy, x:x + cx].copy()
        img = self.aug(img)
        return make_ssl_targets(img.astype(np.float32) / 255.0)


def make_ssl_targets(img: np.ndarray):
    """Build (masked_img, clean_img, hog) for one [0,1] float crop [Z,Y,X].

    Split out of the dataset so the crop itself can come from the framework's
    dataloader (which already handles multi-volume pools, h5/tif mixes and DDP
    sharding) while this pretext-task construction stays identical.

    Masked-out blocks are filled with GAUSSIAN NOISE matched to the crop's own
    mean/std, not zeros -- that is what the official pretraining does
    (SegNeuron/Pretrain/pretrain_provider.py:57-94).
    """
    cz, cy, cx = img.shape
    noise = np.random.normal(loc=img.mean(), scale=img.std() + 1e-6,
                             size=img.shape).astype(np.float32)

    # Per-slice HOG map.
    hog_stack = np.stack([
        hog(img[s], pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)[1]
        for s in range(cz)
    ]).astype(np.float32)

    # 3D block masking: random block size, 50-70% masked, noise-filled.
    z_bx = random.choice([b for b in (2, 4, 5) if cz % b == 0]) or 2
    xy_choices = [b for b in (4, 8, 16) if cy % b == 0 and cx % b == 0]
    xy_bx = random.choice(xy_choices)
    ratio = 0.5 + 0.2 * random.random()
    nf, nh, nw = cz // z_bx, cy // xy_bx, cx // xy_bx
    to_patch = Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)',
                         p1=xy_bx, p2=xy_bx, pf=z_bx)
    to_img = Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',
                       f=nf, h=nh, w=nw, p1=xy_bx, p2=xy_bx, pf=z_bx, c=1)
    mask = np.ones_like(img)
    patches = to_patch(torch.tensor(mask.reshape(1, 1, cz, cy, cx)))
    n_blk = nf * nh * nw
    drop = np.random.choice(n_blk, size=int(n_blk * ratio), replace=False)
    patches[:, drop, :] = 0
    mask = to_img(patches).numpy().squeeze()

    img_mask = mask * img + (1 - mask) * noise

    return (torch.from_numpy(img_mask[None].astype(np.float32)),
            torch.from_numpy(img[None].astype(np.float32)),
            torch.from_numpy(hog_stack[None]))


def load_volumes(specs):
    """specs: list of 'path' or 'path:dataset'. Returns list of numpy stacks."""
    import h5py
    import imageio
    vols = []
    for spec in specs:
        path, _, ds = spec.partition(":")
        if path.endswith((".tif", ".tiff")):
            vols.append(np.asarray(imageio.volread(path)))
        else:
            with h5py.File(path, "r") as f:
                # Use the explicit dataset if given, else the sole/first key
                # (the pool mixes 'volumes' and 'inputs' conventions).
                key = ds if ds else list(f.keys())[0]
                vols.append(f[key][...])
    return vols
