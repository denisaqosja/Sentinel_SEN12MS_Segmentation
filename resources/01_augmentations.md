# Augmentations in the Sentinel-2 segmentation project

This report describes the current implementation in [`augmentations.py`](../src/data/augmentations.py), its configuration in [`dfc.yaml`](../configs/dfc.yaml), and reproducible examples generated from a real DFC2020 training patch. These are measured effects on data and targets, **not evidence of improved validation accuracy**. No model was retrained for this report.

## Data and experiment setup

- Main image: `Datasets/DFC_Public_Dataset/sensors/train/s2/ROIs0000_autumn_s2_BandarAnzali_p163.tif`.
- Mixup partner: `Datasets/DFC_Public_Dataset/sensors/train/s2/ROIs0000_autumn_s2_BandarAnzali_p136.tif`.
- Image dimensions: 256 × 256; channels are the existing stored RGB order.
- Masks use eight training classes (0–7); `255` is ignored.
- The figures reproduce the current dataset standardization, then undo it for display using the same fixed clipping threshold (3000). There is no per-panel contrast stretch. Display values outside [0,1] are clipped.
- Mask colors use the shared `DFC2020.LABEL_MAP` (WorldCover-inspired); ignored pixels are black. See the [class legend](figures/dfc_label_legend.png). Colors denote classes, not continuous values.

In the original mask, DFC2020 labels **75.58%** of pixels as cropland (raw ID 6, training ID 4, pink `#f096ff`) and **1.50%** as forest (raw ID 1, training ID 0, dark green `#006400`). The colors in the lower row come directly from the ground-truth labels, so vegetated areas in the RGB image can appear pink in the mask.
- Seeds are fixed. Crop boxes below are `[top, left, height, width]` in source pixels.

## Where each operation happens

```text
Raw patch → existing preprocessing → random resized crop → random rotation
          → horizontal flip → batch assembly → optional mixup → model
          → smoothed, mixup-aware training cross-entropy
```

The spatial transforms run only for the training split when `data.augmentation.enabled` is true. Mixup and label smoothing have separate training settings; disabling spatial augmentation does not disable them. Validation receives neither spatial augmentation nor mixup and uses ordinary unsmoothed cross-entropy. The augmentation functions do not change the cached original arrays.

## Effect of each augmentation

| Operation | Current setting | Effect on image | Effect on segmentation targets |
|---|---|---|---|
| Random resized crop | Area fraction [0.7, 1.0]; width/height ratio [0.9, 1.1] | Selects a subwindow and resizes to the original dimensions; changes framing and apparent object scale. | Applies the identical crop; nearest-exact resizing preserves categorical IDs. Regions outside the crop disappear. |
| Random rotation | Uniform angle in ±30° | Changes orientation at fixed output size; bilinear interpolation blends neighboring pixels; corners may be removed. Exposed image pixels are filled with zero in preprocessed tensor space. | Uses the same angle with nearest-neighbor interpolation; exposed pixels receive `255` and contribute no supervised loss. |
| Horizontal flip | Probability 0.5 per call | Mirrors the image left-to-right without interpolation. | Mirrors the mask identically; class identities remain unchanged. |
| Mixup | Enabled; alpha 0.2; attempt probability 0.5 | Blends two batch images pixelwise; creates superimposed scenes rather than a physically observed image. | Keeps two integer masks and combines their losses by the blending coefficient; never averages class indices. |
| Label smoothing | Epsilon 0.1 | Does not modify images. | Changes the loss's target probabilities from a one-hot distribution to a slightly softer distribution over classes. Ignored pixels remain excluded. |

### Random resized cropping

For a square crop containing 70% of a square patch's area, resizing back enlarges features by approximately `1/sqrt(0.7) ≈ 1.20` in each dimension. The ratio range permits mild stretching when the selected crop is not square. Sampling also obeys image bounds; not every proposed area/ratio is feasible, and torchvision can fall back to a centered crop.

Expected learning objective: reduce dependence on a fixed framing and scale. Data tradeoff: spatial context and small or rare regions can be cropped out; interpolation changes fine image details. Original output dimensions are retained for batching.

### Random rotation and horizontal flip

These vary orientation while preserving land-cover class meaning for this top-down segmentation task. Rotation adds interpolation and boundary loss; flipping only rearranges pixels. Zero image fill is **not necessarily black in displayed RGB**: after standardization, zero corresponds to the preprocessing mean. The associated mask fill is ignored, although those image borders can still influence surrounding predictions through model context.

![Individual spatial transformations](figures/augmentations/individual.png)

The isolated crop and +25° rotation above are fixed demonstrations; the repeated experiments below use the actual random pipeline. No `rot90` operation remains in that pipeline.

### Mixup

For two images A and B, `mixed = lambda*A + (1-lambda)*B`, with `lambda ~ Beta(alpha, alpha)`. Alpha 0.2 produces coefficients often near 0 or 1, so many mixed samples closely resemble one source. A random batch permutation selects partners; self-pairings are possible. The 0.5 setting is the probability of attempting mixup for a batch, not a guarantee that half the samples visibly change. Batches of one are unchanged.

![Mixup example with fixed coefficient](figures/augmentations/mixup.png)

The illustration deliberately fixes lambda=0.65 and swaps the two sources; it demonstrates the operation rather than a typical Beta draw. The mask is not displayed as one hard blended mask because both targets contribute. At a pixel with different valid labels, the unsmoothed target has mass 0.65 on A's class and 0.35 on B's. Equal labels combine into one class. If one source label is ignored, only the other source contributes supervision; if both are ignored, the pixel contributes nothing.

The implemented loss sums the two weighted pixel losses and divides by the weighted number of valid pixels. Entirely ignored batches yield zero loss with zero gradients. This prevents ignored rotation borders from becoming an invented background class.

### Label smoothing

PyTorch-style smoothing uses `q = (1-epsilon)*one_hot(class) + epsilon/C`. With C=8 and epsilon=0.1, the correct-class target is 0.9125; each other class has probability 0.0125. This regularizes the confidence objective; it does not blur boundaries, change saved integer masks, or guarantee better calibration or accuracy.

![Effect of label smoothing on a target distribution](figures/augmentations/label_smoothing.png)

When both mixup labels are valid, the effective distribution is the same weighted combination of their smoothed distributions. Training loss therefore has a different target objective from unsmoothed validation loss; their absolute values should not be treated as directly equivalent.

## Multiple augmentations applied repeatedly to the same original

Each of these five draws starts from the **same original image and mask** and runs crop → rotation → possible flip. This matches repeated dataset access during normal training: a previously augmented output is not written back into the dataset.

![Independent repeated draws from one original](figures/augmentations/repeated_fresh.png)

| Seed | Crop box | Rotation | Ignored pixels after transforms | Valid classes present |
|---|---|---|---|---|
| 101 | [17, 10, 225, 221] | -12.64° | 8.86% | 7 |
| 102 | [14, 11, 209, 220] | 6.97° | 5.39% | 7 |
| 103 | [9, 7, 246, 236] | 15.70° | 10.44% | 7 |
| 104 | [11, 12, 238, 230] | -5.43° | 4.32% | 7 |
| 105 | [2, 19, 245, 222] | 29.70° | 15.41% | 7 |

The original mask contains 0.00% ignored pixels and 7 valid classes. Across 100 independent draws (seeds 1000–1099), ignored pixels average **8.94%**, ranging from **0.00%** to **15.41%**. These percentages combine pre-existing ignored regions with rotation fill; a crop can also remove ignored regions. They are observations for one patch, not whole-dataset estimates. All outputs retain 256 × 256 pixels and mask IDs remain in the original class set plus `255`.

The same semantic scene appears at different positions, scales, and orientations. Individual outputs may lose classes or context. Image and mask undergo the same spatial mapping, but interpolated image edges and nearest-sampled mask edges are not expected to have identical numeric values.

## Interpretation and limitations

These augmentations expand the variation seen during training without collecting new scenes. They do not add new geographic coverage or fix train/validation distribution differences. Improvements must be established by controlled runs with the same split and seeds: baseline, spatial transforms only, then add mixup and smoothing separately. Compare validation IoU, per-class metrics, and qualitative masks; this report does not claim a measured model-performance gain.

The current source has an independent preprocessing inconsistency: `prepare_s2()` standardizes even when the DINO variant requests reflectance preprocessing. The figures use that existing standardization explicitly; the augmentation examples do not validate the DINO input contract. Also, at generation time `dfc.yaml` specifies `learning_rate: 0.0`; a zero learning rate prevents optimizer weight updates and must be corrected before evaluating augmentation benefits.

## Reproduce

From the project root with the `segment` environment active:

```bash
python -m resources.docs.generate_augmentation_report
```

This regenerates this report, the figures under `resources/figures/augmentations/`, and the recorded draw metadata in `measurements.json`. It reads the local dataset and does not start training or modify the dataset or training settings.
