"""Unlabeled-volume dataset for dbMiM masked-image pretraining.

Random crops from a pool of unlabeled EM stacks (h5 / tif), normalized and
augmented exactly as in the reference dbMiM pipeline:

  * normalization is a per-array 1/99-percentile contrast stretch with a
    heuristic /255 (only applied when max > 2.0) -- dbMiM/dbmim/datasets.py:21-34.
    This is NOT the framework's mean/std normalize_image.
  * augmentation order and probabilities follow augment_image_and_label
    (dbMiM/dbmim/datasets.py:60-132): rot90 xy (p=1 over k=0..3 when enabled),
    flip x 0.5, flip y 0.5, flip z 0.2, intensity gain/bias 0.35,
    gamma 0.35, gaussian noise 0.25.

Volumes are loaded once into memory (the pretraining pool is a fixed set of
stacks), then cropped per sample, mirroring SegNeuron's SSL loader so the two
SSL projects behave the same way under DDP.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset


def normalize_volume(volume: np.ndarray,
                     pct_lo: float = 1.0,
                     pct_hi: float = 99.0) -> np.ndarray:
    """Percentile contrast stretch (reference normalize_volume)."""
    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[None, ...]
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    volume = volume.astype(np.float32)
    # Heuristic /255: skipped for volumes already in [0, 1] (or [0, 2]).
    if volume.max(initial=0) > 2.0:
        volume = volume / 255.0
    lo = float(np.percentile(volume, pct_lo))
    hi = float(np.percentile(volume, pct_hi))
    if hi > lo:
        volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    return volume


def augment_volume(volume: torch.Tensor,
                   rotate_xy: bool = True,
                   gamma: bool = True,
                   gamma_range=(0.7, 1.5),
                   gamma_probability: float = 0.35,
                   noise_std: float = 0.025,
                   intensity_probability: float = 0.35) -> torch.Tensor:
    """Reference augmentation order/probabilities (image-only path)."""
    if rotate_xy:
        k = int(torch.randint(0, 4, ()).item())
        if k:
            volume = torch.rot90(volume, k, dims=(-2, -1))
    if torch.rand(()) < 0.5:
        volume = volume.flip(-1)
    if torch.rand(()) < 0.5:
        volume = volume.flip(-2)
    if torch.rand(()) < 0.2:
        volume = volume.flip(-3)
    if torch.rand(()) < float(intensity_probability):
        gain = 0.85 + 0.30 * torch.rand((), device=volume.device)
        bias = 0.10 * (torch.rand((), device=volume.device) - 0.5)
        volume = (volume * gain + bias).clamp(0.0, 1.0)
    if gamma and gamma_probability > 0.0 and torch.rand(()) < gamma_probability:
        lo, hi = float(gamma_range[0]), float(gamma_range[1])
        if hi < lo:
            lo, hi = hi, lo
        exponent = lo + (hi - lo) * torch.rand((), device=volume.device)
        volume = volume.clamp(1e-4, 1.0).pow(exponent).clamp(0.0, 1.0)
    if torch.rand(()) < 0.25:
        volume = (volume + torch.randn_like(volume) * float(noise_std)).clamp(0.0, 1.0)
    return volume


def load_volumes(specs, pct_lo: float = 1.0, pct_hi: float = 99.0):
    """specs: list of 'path' or 'path:dataset'. Returns normalized float stacks.

    The h5 dataset key is optional: the pool mixes conventions
    (datasets/cremi uses 'volumes', datasets/EM/cremi uses 'inputs'), so fall
    back to the file's sole/first key.
    """
    import h5py
    import imageio
    vols = []
    for spec in specs:
        path, _, ds = spec.partition(":")
        if path.endswith((".tif", ".tiff")):
            arr = np.asarray(imageio.volread(path))
        else:
            with h5py.File(path, "r") as f:
                key = ds if ds else list(f.keys())[0]
                arr = f[key][...]
        vols.append(normalize_volume(arr, pct_lo, pct_hi))
    return vols


class DBMiMPretrainDataset(IterableDataset):
    """Endless stream of random unlabeled EM crops for masked-image pretraining.

    Iterable (not indexed) on purpose, following the convention of
    A simple SSL patch loader: "every call
    samples a fresh batch independently -- no epoch concept". SSL pretraining
    is defined by an iteration count, not by passes over a finite set, so
    there is no dataset length to size and no DistributedSampler to shard.
    (An indexed Dataset of length iters*batch would additionally be wrong under
    DDP, where the sampler hands each rank only 1/world_size of the indices and
    the loader then stops at iters/world_size steps.)

    Each worker/rank seeds its own RNG so they don't draw identical crops.
    """

    def __init__(self, volumes, crop=(32, 160, 160),
                 augment: bool = True, noise_std: float = 0.025,
                 seed: int = 0):
        self.volumes = volumes
        self.crop = tuple(int(c) for c in crop)
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

        cz, cy, cx = self.crop
        usable = [i for i, v in enumerate(volumes)
                  if v.shape[0] >= cz and v.shape[1] >= cy and v.shape[2] >= cx]
        if not usable:
            raise RuntimeError(
                f"no volume can fit a {self.crop} crop; shapes="
                f"{[v.shape for v in volumes]}")
        if len(usable) != len(volumes):
            dropped = [(i, volumes[i].shape) for i in range(len(volumes))
                       if i not in set(usable)]
            print(f"[dbmim-ssl] skipping {len(dropped)} volume(s) too small for "
                  f"crop {self.crop}: {dropped}")
        self.usable = usable

    def _sample(self, rng):
        cz, cy, cx = self.crop
        v = self.volumes[self.usable[rng.integers(len(self.usable))]]
        z = int(rng.integers(v.shape[0] - cz + 1))
        y = int(rng.integers(v.shape[1] - cy + 1))
        x = int(rng.integers(v.shape[2] - cx + 1))
        patch = v[z:z + cz, y:y + cy, x:x + cx]
        t = torch.from_numpy(np.ascontiguousarray(patch))[None]  # [1,Z,Y,X]
        if self.augment:
            t = augment_volume(t, noise_std=self.noise_std)
        return t

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        rng = np.random.default_rng(self.seed + 100003 * rank + wid)
        while True:
            yield self._sample(rng)
