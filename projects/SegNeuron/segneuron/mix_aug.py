"""SegNeuron's cross-volume mixing augmentations (frequency + spatial).

Port of the two style/content mixing steps in the official
`Train_and_Inference/supervised_provider.py`, which are enabled only for
supervised training (`freq_mix_prob: 0.25`, `spa_mix_prob: 0.25` in the
official supervised YAML; the pretrain YAML has neither).

Both draw a *second* patch from a different source volume, so they can't be
expressed as an ordinary per-sample augmentor in this framework (those see one
sample at a time). They are applied by `SegNeuronVolumeDataset`
(segneuron/mix_dataset.py) instead.

1. Frequency mixing (FDA): replace the low-frequency amplitude band of the
   sample's 2D FFT (per z-slice) with that of a patch from another volume.
   Transfers imaging "style" while keeping structure/phase, so the targets are
   unchanged. Applied to the raw [0,1] image before normalization.
2. Spatial mixing: composite a random xy window from another volume's patch
   over the sample. The official code blends the already-computed affinity /
   foreground targets (not the raw label ids), so this module exposes the mask
   plus a blend helper and the dataset applies it to image and targets.
"""
from __future__ import annotations

import numpy as np


def _low_freq_mutate_np(amp_src: np.ndarray, amp_trg: np.ndarray, L: float = 0.1):
    """Swap the centered low-frequency square (half-width b = floor(min(h,w)*L))
    of the source amplitude with the target's."""
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    h, w = a_src.shape
    b = int(np.floor(np.amin((h, w)) * L))
    c_h, c_w = int(np.floor(h / 2.0)), int(np.floor(w / 2.0))

    a_src[c_h - b:c_h + b + 1, c_w - b:c_w + b + 1] = \
        a_trg[c_h - b:c_h + b + 1, c_w - b:c_w + b + 1]
    return np.fft.ifftshift(a_src, axes=(-2, -1))


def _fda_2d(src_img: np.ndarray, trg_img: np.ndarray, L: float = 0.01):
    fft_src = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg_img, axes=(-2, -1))

    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    amp_src_ = _low_freq_mutate_np(amp_src, amp_trg, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    return np.real(np.fft.ifft2(fft_src_, axes=(-2, -1)))


def fda_source_to_target_3d(src_img: np.ndarray, trg_img: np.ndarray,
                            L: float = 0.01) -> np.ndarray:
    """Slice-wise FDA over a (z,y,x) volume, as in the official
    FDA_source_to_target_np_3D."""
    return np.stack([_fda_2d(src_img[z], trg_img[z], L) for z in range(src_img.shape[0])])


def freq_mix(image: np.ndarray, other_image: np.ndarray,
             L: float = 0.001) -> np.ndarray:
    """Style-mix `image` toward `other_image` and clip back to [0,1].

    Both inputs are (z,y,x) float in [0,1]. The official call site uses
    L=0.001 and clips negatives/overshoot after the inverse FFT.
    """
    out = fda_source_to_target_3d(image, other_image, L)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_spatial_mix_mask(shape_zyx, window_frac: float = 70.0 / 128.0,
                          random_state: np.random.RandomState = None) -> np.ndarray:
    """Mask over a (z,y,x) patch: 1 keeps the original sample, 0 selects the
    other patch inside a random xy window spanning all z.

    The official code hardcodes a 70x70 window at offset 0..58 for a 128x128
    crop; kept as a fraction here so it scales with MODEL.INPUT_SIZE.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    _, ny, nx = shape_zyx
    wy = max(1, min(ny, int(round(ny * window_frac))))
    wx = max(1, min(nx, int(round(nx * window_frac))))
    oy = random_state.randint(0, ny - wy + 1)
    ox = random_state.randint(0, nx - wx + 1)

    mask = np.ones(shape_zyx, dtype=np.float32)
    mask[:, oy:oy + wy, ox:ox + wx] = 0.0
    return mask


def blend_with_mask(a: np.ndarray, b: np.ndarray, mask_zyx: np.ndarray) -> np.ndarray:
    """Composite `a` (where mask==1) with `b` (where mask==0).

    Broadcasts the (z,y,x) mask over a leading channel axis, so it works for
    the image (1 or no channel), the 3-channel affinity target, and the
    1-channel foreground target alike -- matching the official code, which
    blends the already-computed affinity maps rather than the raw labels
    (blending label ids would invent nonexistent ids and create spurious
    affinities at the window seam).
    """
    m = mask_zyx
    if a.ndim == mask_zyx.ndim + 1:
        m = mask_zyx[np.newaxis, ...]
    elif a.ndim != mask_zyx.ndim:
        raise ValueError(f"cannot broadcast mask {mask_zyx.shape} onto {a.shape}")
    return (m * a + (1.0 - m) * b).astype(a.dtype)
