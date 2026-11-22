# Dataset Description and Analysis 

This file describes both datasets used in this project: DFC2020 and SEN12MS.


## DFC2020
DFC dataset is a "cleaner" companion to SEN12MS — same Sentinel 1&2 sensor pairing and same simplified-IGBP class scheme, but with manually curated/higher-quality land cover labels over a smaller set of scenes (a handful of regions vs. SEN12MS's global spread). 

Download the dataset from here: ```https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files``` Use the zip file: DFC_Public_Dataset.zip (Size: 9.63 GB).

The data is restructured as by the file: ```data_structuring/process_data.py```. The purpose of this step is to organize the data files based on the sensor rather than the season. 

The ground truth semantic segmentation labels are located in the dfc directory. They follow the IGBP scheme with 8 land-cover classes:
- 1: Forest
- 2: Shrubland
- 4: Grassland
- 5: Wetlands
- 6: Croplands
- 7: Urban/Built-up
- 9: Barren
- 10: Water


Split the dataset into train and validation:
- Choose a ratio of 85% training samples and 15% validation samples.
- Careful selection of the validation set so that it includes samples of rare classes. 
- Keep the samples from regional areas of: Mexico City and Khabarovsk for validation (they constitute 15.9% of the whole dataset). The distribution of these two regions for each class is:
  
| Class     | Khabarovsk + Mexico City |
| --------- | -----------------------: |
| Forest    |                   ~21.2% |
| Shrubland |                    ~8.4% |
| Grassland |                   ~15.2% |
| Wetlands  |                   ~18.1% |
| Croplands |                    ~9.3% |
| Urban     |                   ~19.9% |
| Barren    |                    ~2.0% |
| Water     |                    ~5.2% |



An thorough analysis of the raw data is done before the processing and can be found at ```src/notebooks/analyse_and_process_dfc.ipynb```.

#### Summary statistics

The raw data exhibits a reflectance range of [1, ~25000]. These very high-reflectance pixels usually represent clouds or bright areas. 
The statistics computed per channel (R, G, B) from the data are as follows:

| Channel | min | max | mean | std | p1 | p99 |
|---|---|---|---|---|---|---|
| R | 1 | 24462.0 | 742.2 | 457.9 | 252 | 2368 |
| G | 1 | 21768.0 | 900.9 | 303.5 | 474 | 1899 |
| B | 1 | 21955.0 | 1001.5 | 261.9 | 656 | 1823 |

<!--
Interpretation:

- The mean is B > G > R. This is expected for L1C data, where atmospheric path radiance raises the blue band the most. 
- The std is R > G > B. Red carries the most surface contrast (vegetation vs. soil/urban/cloud), while blue is dominated by a fairly constant atmospheric offset.
- The std matches the width of the central 98% of the data (p1 to p99 is about 4.65 std), so it reflects real variation and is not driven by outliers.
- The data is strongly right-skewed: p99 is about 1800 to 2400 while the maximum is far higher.
-->

#### Histogram analysis

The per-channel histogram is computed for the DFC dataset. The y-axis of the histogram is visualized in the log-scale. 

The histograms show a dense bulk below about 2500, followed by a shoulder up to about 6000 and then a very sparse tail. The apparently flat tail out to 25000 consists of single pixels.

![histogram_99.9percentile](../../figures/percentile.png)

The brightest pixels of DFC2020 are probably mostly clouds, snow and ice. With RGB only, these pixels are flagged heuristically: 
- bright in all channels (`min(R,G,B) > ~1800`) 
- spectrally flat (`(max-min)/mean < ~0.2`). 

The DFC2020 labels describe land cover and not clouds, so a fully clouded patch still carries a land-cover label that the model cannot see. Masking the cloud patches or removing the samples with heavy cloud presence would be an approach towards cleaning the dataset. Another idea would be to combine the S2 data with the SAR images from Sentinel S1 (a multimodal approach) to overcome the cloud issue for segmentation. (Future work) 

<img src="../../figures/CCDF.png" width="40%">

The complementary CDF function (fraction of pixels above a value, log scale) shows that most pixels are concentrated at relatively low reflectance values. The curves fall very steeply in roughly the first 2000-3000, meaning the number of pixels above this threshold decreases very quickly there.
Between roughly 3000 and 6000, the curves continue to drop strongly. By around 6000, only a few thousand pixels remain above that value, compared with a very large number at low reflectance. Beyond about 6000-7000, the curves flatten considerably. The range 6000 - 25000 contains rare high-reflectance tail of the distribution.


#### Decide on the clipping threshold

Clipping is needed to keep the sparse tail from dominating normalization and producing very large activations. A single shared threshold is used for all channels, so that the R:G:B ratios of bright pixels are preserved. Otherwise, per-channel clipping at p99 would distort them.


There are three options for deciding the clipping threshold: 

- T = 6000: Shown by the histogram, it is the threshold where the majority of pixel's range falls into. 
- T = 3500: Obtained by the 99.9 percentile
- T = 3000: It is the common convention for displaying Sentinel-2 images

The decided threshold value: T = 3000. This is because:
- The number of pixels above this threshold is minimal (R: 0.2%; G: 0.1%, B: 0.1%), proving that the pixels above this value are outliers.
- The tail beyond about 3000 is mostly clouds, snow and saturated surfaces, so it carries little information for land-cover classification.
- And most importantly, it is the display convention for Sentinel S2 images. 

The brightest images (after clipping the data at T=3000): 

![bright_areas](../../figures/bright_areas.png)


#### Processing pipeline

1. Load the raw data and analyse it, check min/max, and the fraction of values at 0, 1 and above 6000.
2. Clip all channels to `[0, T]` after deciding on the the threshold value `T`.
3. Compute the per-channel mean and std on the clipped (and transformed) training set.
4. Standardize: `x = (clip(x, 0, T) - mean) / std`.
5. Apply the identical clip and constants to validation and test, and store them with the model checkpoint.


## Local dataset inventory and split

The processed dataset is in `Datasets/DFC_Public_Dataset/sensors/{train,val}/{s1,s2,lc,dfc}`. A **sample** means one spatial patch with four corresponding modality/label files, not four independent observations. Current training uses only the three-channel S2 RGB TIFF and the DFC segmentation mask; S1 and LC files remain available on disk. RGB is extracted as Sentinel-2 bands B04, B03, B02 (zero-based indices `[3, 2, 1]`) by the restructuring script. The inspected examples have shape 256 × 256 × 3 for RGB and 256 × 256 for masks.

The original target of 85%/15% is approximate. The actual region-based split is **84.13% training / 15.87% validation**. Mexico City and Khabarovsk (`Chabarovsk` in filenames) are reserved for validation. This evaluates transfer to held-out regions rather than a random patch split. The local split has no separate test directory; validation therefore should not be described as an independent final test set.

| Set | Samples | Share of samples | Label pixel positions¹ | RGB + DFC TIFF size² | All four modalities² |
|---|---:|---:|---:|---:|---:|
| Training (`train`) | 5,144 | 84.13% | 337,117,184 | 1.271 GiB | 6.299 GiB |
| Validation (`val`) | 970 | 15.87% | 63,569,920 | 0.210 GiB | 1.171 GiB |
| **Total** | **6,114** | **100%** | **400,687,104** | **1.482 GiB** | **7.470 GiB** |

¹ Nominal pixel positions at 256 × 256 per patch, not a count of valid class labels or unique ground coverage. Patches can contain ignored labels or overlapping geography.

² Sum of stored TIFF file sizes (1 GiB = 2³⁰ bytes), excluding archives, directory metadata, and ancillary files. This is compressed file storage, not model RAM usage or download size. Each set has the same number of files per modality: 20,576 TIFFs in training and 3,880 in validation. Filename identities match across all four modalities, and no exact sample identity occurs in both sets; this does not establish absence of spatial overlap.

| Region (filename spelling) | Set | Samples | Share of its set | Share of all samples |
|---|---|---:|---:|---:|
| BandarAnzali | train | 676 | 13.14% | 11.06% |
| BlackForest | train | 1,444 | 28.07% | 23.62% |
| CapeTown | train | 1,440 | 27.99% | 23.55% |
| KippaRing | train | 684 | 13.30% | 11.19% |
| Mumbai | train | 900 | 17.50% | 14.72% |
| Chabarovsk | val | 486 | 50.10% | 7.95% |
| MexicoCity | val | 484 | 49.90% | 7.92% |

The class percentages copied from the README below describe the reported validation **pixel composition**, not the percentage of samples in each class or the fraction of each class allocated to validation. They sum to about 99.3%; rounding and the original denominator/ignored-label treatment are not documented. They are approximate historical results, not newly verified class counts. Barren is particularly scarce in that summary (~2%), so report per-class IoU/F1 alongside aggregate metrics. Regional holdout can also shift class proportions and image appearance.

## Paired RGB samples and segmentation masks

![Four DFC2020 RGB patches above their paired ground-truth segmentation masks, with a shared class-color legend.](figures/dfc_samples_masks.png)

**Top row:** original stored S2 RGB samples, displayed with the same fixed stretch `clip(RGB, 0, 3000) / 3000` (gamma = 1). **Bottom row:** the matching DFC ground-truth masks, not model predictions. Each column is one spatially aligned image/mask pair. The first two examples are from training and the last two from validation. Images are selected for class diversity among 16 evenly spaced filenames per region; they illustrate classes rather than represent dataset-wide prevalence. Patch identity and season appear above each column.

The following palette matches the notebooks' `tab10` mapping for remapped training IDs 0–7. It is a display convention: water is gray here, for example; RGB colors in the top row are unrelated to the categorical mask colors. Ignored/unmapped labels are explicitly black in this figure.

| Land cover | Raw DFC ID | Training ID | Mask color | Hex |
|---|---:|---:|---|---|
| Forest | 1 | 0 | Blue | `#1f77b4` |
| Shrubland | 2 | 1 | Orange | `#ff7f0e` |
| Grassland | 4 | 2 | Green | `#2ca02c` |
| Wetlands | 5 | 3 | Red | `#d62728` |
| Croplands | 6 | 4 | Purple | `#9467bd` |
| Urban / built-up | 7 | 5 | Brown | `#8c564b` |
| Barren | 9 | 6 | Pink | `#e377c2` |
| Water | 10 | 7 | Gray | `#7f7f7f` |
| Ignored / unmapped | Any other value | 255 | Black | `#000000` |

The label remapping is defined in [`DFC2020.py`](../../src/data/DFC2020.py). Regenerate this figure from the repository root using the project environment:

```bash
python resources/docs/generate_dfc_examples.py
```

## Reading the original analysis

The following analysis is preserved from the README, including its interpretation and processing recommendations. Broken notebook/script references have been updated to existing paths, and figure links are relative to this report. Keep these qualifications in mind:

- The raw RGB summary and clipping percentages below are historical figures. Their exact split and sampling scope are not established here; they should not automatically be treated as training-only estimates.
- High RGB values can indicate clouds, snow, or bright surfaces, but RGB brightness alone does not establish their identity. The proposed cloud test is a heuristic, not a validated cloud mask. Rare values are not automatically invalid observations.
- A shared clipping threshold avoids different per-channel ceilings, but clipping still changes channel ratios when only some channels exceed the threshold. Ratios are preserved only where clipping is inactive (or all channels change proportionally).
- `T = 3000` is the project's selected display/preprocessing threshold; the claim below that it is a convention is not an independently verified universal Sentinel-2 rule. Fixed clipping is an engineering choice that can remove useful bright-surface information.
- The historical CCDF description refers to counts. The notebook actually plots the **number** of pixels above each value; a fraction requires division by the total number of pixels.
- The loader currently clips at 3000 and standardizes using RGB mean `[740.54718457, 900.030319, 1000.71999455]` and standard deviation `[460.21289232, 301.78994503, 260.39637968]`. Verify that these constants were fitted on training data only before relying on validation results; validation/test should reuse the training constants.


## SEN12MS
Inference on SEN12MS (only the S2 - optical data)

1. Obtain the RGB spectrum of the data:
RGB = [B04, B03, B02]
2.




## Supporting project sources

- [Raw-data analysis notebook](../../src/notebooks/analyse_and_process_dfc.ipynb)
- [Dataloader visualization notebook](../../src/notebooks/visualize_dataloader.ipynb)
- [Data restructuring / RGB extraction](../../data_structuring/process_data.py) (select `DFC2020` when processing DFC; its current default is SEN12MS)
- [DFC loader and label remapping](../../src/data/DFC2020.py)
- [DFC experiment configuration](../../configs/dfc.yaml)
