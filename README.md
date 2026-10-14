# Sentinel S2 Semantic Segmentation

A PyTorch implementation of semantic segmentation for the RGB data of Sentinel sensor S2 part of the public dataset SEN12MS.  

## Overview

This projects aims to semantically segment the optical data from Sentinel dataset SEN12MS. This project relies on pre-trained networks, which are fine-tuned to provide segmentation masks on the desired dataset.
For this purpose, the following steps are followed:

- Certain pretrained network suitable for segmentation are chosen:DINOv3, SegFormer, DeepLabV3+. They are fine-tuned on a subset from Sentinel data, which contains labelled information. 
- A small dataset, named DFC2020 is used for fine-tuning the pre-trained networks. The dataset contains more curated labelled information, and is a obtained from the sentinel satellites (S1 and S2). 
- Then the model is used to infer the segmentation masks on the S2 (optical) set of the sentinel SEN12MS dataset.


## Project Structure (to be changed--template taken from another project)

```
├── config/                             # Configuration files for different experiments
│   ├── dfc.yaml                        # Default configuration for the DFC2020 dataset
    └── sen12ms.yaml                    # Default configuration for the SEN12MS dataset
├── experiments/                        # Auto-generated per-run directories (created by train.py at every run)
│   └── exp_droneclassifier_YYYY_MM_DD_HH_MM_SS/   # Timestamped directory isolating one training run
│       ├── checkpoints/                # Model checkpoints saved during training (best model + periodic saves)
│       ├── logs/                       # Text log files produced by the training run
│       ├── results/                    # Evaluation outputs: metrics, predictions, confusion matrices
│       ├── tb_logger/                  # TensorBoard event files for loss and metric visualisation
│       └── spectrogram_config.yaml     # Snapshot of the exact config used for this run (with resolved absolute paths)
├── src/
│   ├── data/                           # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py                  # Custom PyTorch Dataset class
│   │   ├── dataloader.py               # DataLoader utilities
│   │   └── preprocessing.py            # Data preprocessing functions
│   ├── model/                          # Neural network models
│   │   ├── __init__.py
│   │   ├── base.py                     # Base model class
│   │   ├── resnet.py                   # ResNet architectures
│   ├── lib/                            # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── config.py                   # Configuration management (load, merge, save YAML configs)
│   │   ├── logger.py                   # Logging utilities (file + console handlers)
│   │   ├── utils.py                    # General utilities (seeding, device selection, experiment dir creation)
│   │   └── metrics.py                  # Evaluation metrics (accuracy, precision, recall, F1, AUC)
│   ├── training/                       # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Main training loop (forward pass, backprop, validation)
│   │   └── callbacks.py                # Training callbacks (early stopping, checkpointing)
│   ├── notebooks/                      # Jupyter notebooks for data exploration and prototyping
│   └── tests/                          # Unit tests
├── scripts/                            # Standalone entry-point scripts
│   ├── train.py                        # Launch a training run; creates a new experiment directory automatically
│   ├── evaluate.py                     # Evaluate a saved checkpoint against a labelled dataset
│   └── inference.py                    # Run inference on a single raw .e file
├── requirements.txt                    # Python dependencies in .txt file
├── environment.yml                     # Python dependencies in .yml file
└── README.md                           # This file
```



## Installation 
1. Clone the repository

```
git clone https://github.com/your-org/S2_Segmentation.git
cd S2_Segmentation
```

2. Create the conda environment from either of the files:
- environment.yml 

```
conda env create -f environment.yml
conda activate segment
```
- environment.txt
```
conda create -n segment python=3.11 pip
conda activate segment
python -m pip install -r requirements.txt
```


## Dataset Description

### DFC2020
DFC dataset is a "cleaner" companion to SEN12MS — same Sentinel 1&2 sensor pairing and same simplified-IGBP class scheme, but with manually curated/higher-quality land cover labels over a smaller set of scenes (a handful of regions vs. SEN12MS's global spread). 

Download the dataset from here: ```https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files``` Use the zip file: DFC_Public_Dataset.zip (Size: 9.63 GB).

The data is restructured as by the file: ```data_structuring/analyse_and_process_SEN12MS.py```. The purpose of this step is to organize the data files based on the sensor rather than the season. 

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



An thorough analysis of the raw data is done before the processing and can be found at ```src/notebooks/process_dfc.ipynb```.

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

![histogram_99.9percentile](figures/percentile.png)

The brightest pixels of DFC2020 are probably mostly clouds, snow and ice. With RGB only, these pixels are flagged heuristically: 
- bright in all channels (`min(R,G,B) > ~1800`) 
- spectrally flat (`(max-min)/mean < ~0.2`). 

The DFC2020 labels describe land cover and not clouds, so a fully clouded patch still carries a land-cover label that the model cannot see. Masking the cloud patches or removing the samples with heavy cloud presence would be an approach towards cleaning the dataset. Another idea would be to combine the S2 data with the SAR images from Sentinel S1 (a multimodal approach) to overcome the cloud issue for segmentation. (Future work) 

<img src="figures/CCDF.png" width="40%">

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

![bright_areas](figures/bright_areas.png)


#### Processing pipeline

1. Load the raw data and analyse it, check min/max, and the fraction of values at 0, 1 and above 6000.
2. Clip all channels to `[0, T]` after deciding on the the threshold value `T`.
3. Compute the per-channel mean and std on the clipped (and transformed) training set.
4. Standardize: `x = (clip(x, 0, T) - mean) / std`.
5. Apply the identical clip and constants to validation and test, and store them with the model checkpoint.



### SEN12MS
Inference on SEN12MS (only the S2 - optical data)

1. Obtain the RGB spectrum of the data:
RGB = [B04, B03, B02]
2. 


## Training

All the parameters are defined/controlled in the config files.

```
python train.py --model segformer --phase train --config dfc.yaml
```

## Evaluation and Testing

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***



## Results


### Qualitative Results
...

### Metrics
...

## Authors and acknowledgment
This project is developed by: denisaqosja97@gmail.com.



## Future Development
Segmentation from multi-modalities: both S1 and S2.