"""VolumeDataset subclass applying SegNeuron's cross-volume mixing augmentations.

The official supervised provider draws a second patch from a *different* source
volume for both frequency mixing (style transfer, targets unchanged) and spatial
mixing (window composite of image + targets). A standard framework augmentor only
ever sees one sample, so the mixing is done here, overriding `_process_targets`:

  * frequency mixing edits the raw [0,1] image before `_process_image`
    normalization, matching the official call site;
  * spatial mixing blends the already-computed affinity / foreground targets
    (the official code blends target maps, not raw label ids) and needs a full
    second pass through target generation, so it runs after `super()` built the
    first sample's targets.

Both are no-ops unless their probability fires, and both need more than one
source volume in the pool -- with a single-volume dataset there is no "other"
volume to mix with, so they self-disable (logged once).
"""
from __future__ import annotations

import numpy as np

from connectomics.data.dataset.dataset_volume import VolumeDataset

from .mix_aug import blend_with_mask, freq_mix, make_spatial_mix_mask


class SegNeuronVolumeDataset(VolumeDataset):
    def __init__(self, *args,
                 freq_mix_prob: float = 0.0,
                 spa_mix_prob: float = 0.0,
                 freq_mix_l: float = 0.001,
                 spa_mix_window_frac: float = 70.0 / 128.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_mix_prob = float(freq_mix_prob)
        self.spa_mix_prob = float(spa_mix_prob)
        self.freq_mix_l = float(freq_mix_l)
        self.spa_mix_window_frac = float(spa_mix_window_frac)

        self._mix_enabled = (self.mode == 'train' and len(self.volume) > 1)
        if (self.freq_mix_prob > 0 or self.spa_mix_prob > 0) and not self._mix_enabled:
            print(f"[SegNeuron] cross-volume mixing disabled: mode={self.mode}, "
                  f"{len(self.volume)} source volume(s) -- needs train mode and >1 volume")
        elif self._mix_enabled:
            print(f"[SegNeuron] cross-volume mixing on {len(self.volume)} volumes: "
                  f"freq_mix_prob={self.freq_mix_prob}, spa_mix_prob={self.spa_mix_prob}")

    def _sample_other_volume_idx(self, exclude_idx: int) -> int:
        """Pick a source volume index != exclude_idx (official picks a different
        dataset family; the closest equivalent here is a different volume)."""
        n = len(self.volume)
        j = np.random.randint(n - 1)
        return j if j < exclude_idx else j + 1

    def _other_patch(self, exclude_idx: int, vol_size):
        """Crop a patch (and its label) from a different source volume, applying
        the same augmentor so its statistics match the primary sample."""
        other_idx = self._sample_other_volume_idx(exclude_idx)
        # _get_pos_train picks a volume by sample count; force our chosen volume
        # by cropping directly at a random position inside it.
        pos = self._get_pos_train(vol_size)
        pos = [other_idx] + list(pos[1:])
        pos = np.array(pos, dtype=int)
        # Clamp the position to the chosen volume's bounds -- sample counts
        # differ per volume, so a position valid for one may overrun another.
        limits = np.array(self.volume[other_idx].shape) - np.array(vol_size)
        pos[1:] = np.clip(pos[1:], 0, np.maximum(limits, 0))

        _, other_volume, other_label, other_valid = self._crop_with_pos(pos, vol_size)

        if self.augmentor is not None and other_label is not None:
            data = {'image': other_volume, 'label': other_label,
                    'valid_mask': other_valid}
            augmented = self.augmentor(data, random_seed=np.random.randint(0, 1000))
            other_volume, other_label = augmented['image'], augmented['label']
            other_valid = augmented['valid_mask']

        return other_volume, other_label, other_valid

    def _process_targets(self, sample):
        if not self._mix_enabled or self.return_clean_input:
            return super()._process_targets(sample)

        pos, out_volume, out_label, out_valid = sample
        vol_size = self.sample_volume_size

        # ---- 1. frequency mixing: style only, targets untouched ----
        if np.random.rand() < self.freq_mix_prob:
            other_volume, _, _ = self._other_patch(int(pos[0]), vol_size)
            out_volume = freq_mix(out_volume, other_volume, L=self.freq_mix_l)

        # ---- 2. spatial mixing: composite image + targets over an xy window ----
        do_spa = np.random.rand() < self.spa_mix_prob
        other_sample = None
        if do_spa:
            other_volume, other_label, other_valid = self._other_patch(
                int(pos[0]), vol_size)
            if other_label is None:
                do_spa = False
            else:
                other_sample = (pos, other_volume, other_label, other_valid)

        primary = super()._process_targets((pos, out_volume, out_label, out_valid))
        if not do_spa:
            return primary

        # Unlabeled path returns (pos, volume) -- nothing to blend targets for.
        if len(primary) < 4:
            return primary

        other = super()._process_targets(other_sample)
        if len(other) < 4:
            return primary

        _, vol_a, tgt_a, wgt_a = primary
        _, vol_b, tgt_b, _ = other

        mask = make_spatial_mix_mask(tuple(vol_a.shape[-3:]),
                                     window_frac=self.spa_mix_window_frac)
        vol_mixed = blend_with_mask(vol_a, vol_b, mask)
        tgt_mixed = [blend_with_mask(a, b, mask) for a, b in zip(tgt_a, tgt_b)]
        # Weights are recomputed from the primary sample only; the official code
        # likewise carries the primary patch's area mask through the mix.
        return pos, vol_mixed, tgt_mixed, wgt_a
