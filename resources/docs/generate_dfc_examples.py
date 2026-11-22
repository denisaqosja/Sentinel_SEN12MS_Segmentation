"""Regenerate the report figure using local RGB TIFFs and paired DFC labels.

Run from the repository root with the segment environment:
    python resources/docs/generate_dfc_examples.py
"""
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/s2-segmentation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Datasets/DFC_Public_Dataset/sensors'
CLASSES = [(1, 'Forest'), (2, 'Shrubland'), (4, 'Grassland'), (5, 'Wetlands'),
           (6, 'Croplands'), (7, 'Urban / built-up'), (9, 'Barren'), (10, 'Water')]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def main():
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), facecolor='white')
    selected = [('train', 'BlackForest'), ('train', 'Mumbai'),
                ('val', 'Chabarovsk'), ('val', 'MexicoCity')]
    for col, (split, region) in enumerate(selected):
        paths = sorted((DATA / split / 's2').glob(f'*_s2_{region}_*.tif'))
        # Favor class diversity among 16 evenly spaced candidates per region.
        candidates = [paths[i] for i in np.linspace(0, len(paths)-1, 16, dtype=int)]
        path = max(candidates, key=lambda p: len(np.unique(tifffile.imread(
            DATA / split / "dfc" / p.name.replace("_s2_", "_dfc_")))))
        mask_path = DATA / split / 'dfc' / path.name.replace('_s2_', '_dfc_')
        image = tifffile.imread(path)
        mask = tifffile.imread(mask_path).squeeze()
        assert image.shape == (256, 256, 3), (path, image.shape)
        assert mask.shape == image.shape[:2], (mask_path, mask.shape)
        axes[0, col].imshow(np.clip(image.astype(np.float32) / 3000, 0, 1))
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for (raw_id, _), color in zip(CLASSES, COLORS):
            colored[mask == raw_id] = matplotlib.colors.to_rgb(color) * np.array([255]*3)
        axes[1, col].imshow(colored, interpolation='nearest')
        season, patch = path.stem.split('_')[1], path.stem.split('_')[-1]
        axes[0, col].set_title(f'{region} · {split}\n{season} / {patch}', fontsize=11)
        print(f'{split}: {path.name}; labels={np.unique(mask).tolist()}')
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel('Sentinel-2 RGB', fontsize=12)
            axes[1, col].set_ylabel('DFC ground truth', fontsize=12)
    handles = [Patch(facecolor=c, label=f'{name} ({raw})') for (raw, name), c in zip(CLASSES, COLORS)]
    handles.append(Patch(facecolor='#000000', label='Ignored / unmapped'))
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.015))
    fig.suptitle('DFC2020: paired optical samples and land-cover masks', fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.14, hspace=0.08, wspace=0.06)
    destination = Path(__file__).parent / 'figures/dfc_samples_masks.png'
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor='white')
    plt.close(fig)
    print(destination)


if __name__ == '__main__':
    main()
