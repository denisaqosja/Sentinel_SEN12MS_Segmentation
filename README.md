# Sentinel S2 Semantic Segmentation

A PyTorch implementation of semantic segmentation for the RGB data of Sentinel sensor S2 part of the public dataset SEN12MS.  

## Overview

This projects aims to semantically segment the optical data from Sentinel dataset SEN12MS. It relies on pre-trained networks, which are fine-tuned to provide segmentation masks on the desired dataset.
For this purpose, the following steps are followed:

- Certain pretrained network suitable for segmentation are chosen: DINOv3, SegFormer, DeepLabV3+. They are fine-tuned on a subset from Sentinel data, which contains labelled information. 
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
conda create -n segment python=3.13 pip
conda activate segment
python -m pip install -r requirements.txt
```


## Datasets' Description

#### DFC2020
DFC dataset is a "cleaner" companion to SEN12MS — same Sentinel 1&2 sensor pairing and same simplified-IGBP class scheme, but with manually curated/higher-quality land cover labels over a smaller set of scenes (a handful of regions vs. SEN12MS's global spread). 

Download the dataset from here: ```https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files``` Use the zip file: DFC_Public_Dataset.zip (Size: 9.63 GB).

We have restructured the data by using the file: ```data_structuring/process_SEN12MS.py``` so that the data files are organised based on the sensor rather than the season. 

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
- Maintain a ratio of 85% training samples and 15% validation samples.
- Select the validation set so that it includes samples of classes not included in the training set. 
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



#### SEN12MS
Inference on SEN12MS (only the S2 - optical data)

1. Obtain the RGB spectrum of the data:
RGB = [B04, B03, B02]
2. TODO 


## Training

All the parameters are defined/controlled in the config files.

```
python train.py --model segformer --phase train --config dfc.yaml
```

## Evaluation and Testing



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