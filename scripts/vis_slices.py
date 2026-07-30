"""Visualize raw image, GT, predicted affinity, and predicted segmentation for selected slices."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import h5py
import tifffile
from PIL import Image

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

def load(fn):
    if fn.endswith('.npy'):
        return np.load(fn)
    elif fn.endswith('.h5'):
        with h5py.File(fn, 'r') as f:
            ks = list(f.keys())
            assert len(ks) == 1, f"Expected 1 key, got {ks}"
            return np.array(f[ks[0]])
    elif fn.endswith(('.tif', '.tiff')):
        return read_tiff(fn)
    else:
        raise NotImplementedError(f"Unsupported format: {fn}")


def get_lookup(N=200001):
    np.random.seed(42)
    lookup = 0.2 + 0.8 * np.random.rand(N, 3)
    lookup[:, 1] = 0.8
    lookup[:, 2] = 0.8
    lookup = matplotlib.colors.hsv_to_rgb(lookup)
    lookup[0, :] = np.ones(3)
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Raw image h5")
    parser.add_argument('--label', type=str, required=True, help="GT label h5")
    parser.add_argument('--affinity', type=str, default=None,
                        help="Predicted affinity h5 (result.h5). Optional: methods "
                             "that emit instances directly (e.g. FFN) have none, "
                             "and the affinity panel is then omitted.")
    parser.add_argument('--segment', type=str, required=True, help="Predicted segmentation npy (segments.npy)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for images")
    parser.add_argument('--slices', type=str, default=None,
                        help="Comma-separated slice indices, e.g. '0,10,20'. Default: 5 evenly spaced.")
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    image = load(args.image)
    label = load(args.label)
    affinity = load(args.affinity) if args.affinity else None
    segment = load(args.segment)
    print(f"  image:    {image.shape} {image.dtype}")
    print(f"  label:    {label.shape} {label.dtype}")
    print(f"  affinity: "
          f"{affinity.shape if affinity is not None else '(none)'}"
          f"{' ' + str(affinity.dtype) if affinity is not None else ''}")
    print(f"  segment:  {segment.shape} {segment.dtype}")

    num_slices = image.shape[0]
    if args.slices is not None:
        indices = [int(s) for s in args.slices.split(',')]
    else:
        indices = np.linspace(0, num_slices - 1, min(5, num_slices), dtype=int).tolist()

    lut = get_lookup()

    npanel = 4 if affinity is not None else 3
    for z in indices:
        fig, axs = plt.subplots(1, npanel, figsize=(5 * npanel, 5))

        # raw image
        axs[0].imshow(image[z], cmap='gray')
        axs[0].set_title(f"Image (z={z})")
        axs[0].axis('off')

        # GT label
        axs[1].imshow(lut[label[z] % len(lut)])
        axs[1].set_title(f"GT (z={z})")
        axs[1].axis('off')

        k = 2
        if affinity is not None:
            # affinity -> RGB (3, D, H, W) => (H, W, 3)
            aff_slice = affinity[:, z].transpose(1, 2, 0)
            axs[k].imshow(aff_slice)
            axs[k].set_title(f"Pred Affinity (z={z})")
            axs[k].axis('off')
            k += 1

        # predicted segmentation
        seg_z = segment[z] if segment.ndim == 3 else segment[0, z]
        axs[k].imshow(lut[seg_z % len(lut)])
        axs[k].set_title(f"Pred Seg (z={z}, {len(np.unique(seg_z)) - 1} ids)")
        axs[k].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"vis_slice_{z:04d}.png")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"Done. {len(indices)} images saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
