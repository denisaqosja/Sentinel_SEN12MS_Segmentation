import os, torch 
import tifffile as tiff
import numpy as np    

from torch.utils.data import Dataset
from src.data.augmentations import augment_pair

# statistics after clipping
S2_MEAN = np.array(
    [740.54718457, 900.030319, 1000.71999455],     # [mean_R, mean_G, mean_B]
    dtype=np.float32
)

S2_STD = np.array(
    [460.21289232, 301.78994503, 260.39637968],       # [std_R, std_G, std_B]
    dtype=np.float32
)

S2_CLIP_THRESHOLD = 3000.0
IGNORE_INDEX = 255

# labels indices
RAW_TO_TRAIN = {
            1: 0,
            2: 1,
            4: 2,
            5: 3,
            6: 4,
            7: 5,
            9: 6,
            10: 7,
        }

class Dataset_Base(Dataset):
    def __init__(self, config, split="train", seed=42):
        super().__init__()

        data_path = os.path.join(config["data"]["root_dir"], split)

        self.preprocessing = config["data"].get("preprocessing", "standardize")
        if self.preprocessing not in ("standardize", "reflectance"):
            raise ValueError("data.preprocessing must be standardize or reflectance")

        # load data
        input_modalities = config["data"].get("input_modalities", ["s2"])
        print(f"Using input modalities: {input_modalities}")
        self.preload = config["data"].get("preload", True)
        if self.preload:
            self.s2_samples, self.gt_labels_samples = self.load_tiff(data_path, input_modalities=input_modalities)
        else:
            # Keep paths only; decode and normalize patches as batches request them.
            self.s2_paths = self.patch_paths(data_path, "s2")
            self.label_paths = self.patch_paths(data_path, "dfc")
            image_ids = [os.path.basename(p).replace("_s2_", "_sensor_") for p in self.s2_paths]
            label_ids = [os.path.basename(p).replace("_dfc_", "_sensor_") for p in self.label_paths]
            if image_ids != label_ids:
                raise ValueError("S2 and DFC patch filenames do not match")
            print(f"{split}: reading {len(self.s2_paths)} patches on demand (preload=false)")

    @staticmethod
    def patch_paths(datapath, sensor_type):
        sensor_path = os.path.join(datapath, sensor_type)
        paths = [os.path.join(sensor_path, name) for name in sorted(os.listdir(sensor_path))
                 if sensor_type in name and name.lower().endswith((".tif", ".tiff"))]
        if not paths:
            raise ValueError(f"No TIFF patches found in {sensor_path}")
        return paths

    def resolve_datapath(self, datapath):
        if os.path.basename(os.path.normpath(datapath)) == "sensors":
            datapath = os.path.dirname(os.path.normpath(datapath))

        if os.path.isdir(datapath):
            return datapath

        dataset_path = os.path.join("Datasets", datapath)
        if os.path.isdir(dataset_path):
            return dataset_path

        raise FileNotFoundError(f"DFC2020 dataset path does not exist: {datapath}")


    def load_from_dir(self, datapath, sensor_type="s2"):
        sensor_path = os.path.join(datapath, sensor_type)

        file_dirs = sorted(os.listdir(sensor_path))
        sensor_list = []
        for dir in file_dirs:
            if sensor_type in dir:
                sensor_file_path = os.path.join(sensor_path, dir)
                sensor = tiff.imread(sensor_file_path)
                sensor_list.append(sensor)
       
        return np.stack(sensor_list, axis=0)

        
    def load_tiff(self, datapath, input_modalities=["s2", "dfc"]):
        s2 = self.load_from_dir(datapath, "s2")
        assert s2.ndim == 4 and s2.shape[-1] == 3, (
            f"Expected S2 patches of shape (N, H, W, 3), got {s2.shape}"
        )
        s2 = self.prepare_s2(s2)

        raw_labels = self.load_from_dir(datapath, "dfc")
        assert len(s2) == len(raw_labels), (
            f"{len(s2)} S2 patches vs {len(raw_labels)} label patches"
        )
        # remap all the labels 
        labels = np.stack([self.prepare_labels(m) for m in raw_labels])
        
        return s2, labels   


    def prepare_s2(self, s2, report=True):
        "S2 contains reflectance values from the sensor"
        s2 = s2.astype(np.float32, copy=False)

        # handle very bright areas. Set Threshold=3000 since it is the common value used for displaying S2 images
        np.clip(s2, 0.0, S2_CLIP_THRESHOLD, out=s2)
        # standardization
        s2 = (s2 - S2_MEAN) / S2_STD

        # (N, H, W, C) -> (N, C, H, W)
        s2 = np.moveaxis(s2, -1, 1) if s2.ndim == 4 else np.moveaxis(s2, -1, 0)

        if report:
            print(f"S2 data normalized: min = {s2.min():.4f}, max = {s2.max():.4f}")
        return s2

    def prepare_labels(self, mask):
        # class 3 and 8 are removed from the dataset, so remap the labels to a continuous range of integers
        if mask.ndim == 3:
            mask = mask[..., 0]

        target = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
        for raw_id, train_id in RAW_TO_TRAIN.items():
            target[mask == raw_id] = train_id
 
        return target


class DFC2020(Dataset_Base):
    def __init__(self, config, split="train", seed=42):
        super().__init__(config, split=split, seed=seed)

        # Land-cover labels are invariant to orientation in these top-down patches.
        # Use torch's RNG so set_seed() and DataLoader worker seeding apply.
        self.augmentation = config["data"].get("augmentation", {})
        self.augment = split == "train" and self.augmentation.get("enabled", True)
        self.ignore_index = config["data"].get("ignore_index", IGNORE_INDEX)
        self.crop_scale = tuple(self.augmentation.get("crop_scale", [0.7, 1.0]))
        self.crop_ratio = tuple(self.augmentation.get("crop_ratio", [0.9, 1.1]))
        self.rotation_degrees = self.augmentation.get("rotation_degrees", 30.0)
        if not (0 < self.crop_scale[0] <= self.crop_scale[1] <= 1):
            raise ValueError("augmentation.crop_scale must satisfy 0 < min <= max <= 1")
        if not (0 < self.crop_ratio[0] <= self.crop_ratio[1]):
            raise ValueError("augmentation.crop_ratio must be positive and ordered")
        if not 0 <= self.rotation_degrees <= 180:
            raise ValueError("augmentation.rotation_degrees must be between 0 and 180")
    
    def __getitem__(self, i):
        if self.preload:
            image, mask = self.s2_samples[i], self.gt_labels_samples[i]
        else:
            image = tiff.imread(self.s2_paths[i])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected S2 patch (H, W, 3), got {image.shape}: {self.s2_paths[i]}")
            image = self.prepare_s2(image, report=False)
            mask = self.prepare_labels(tiff.imread(self.label_paths[i]))
        s2_tensor = torch.from_numpy(image)
        target = torch.from_numpy(mask).long()
        
        if self.augment:
            s2_tensor, target = augment_pair(
                s2_tensor, target, crop_scale=self.crop_scale, crop_ratio=self.crop_ratio,
                rotation_degrees=self.rotation_degrees, ignore_index=self.ignore_index,
            )

        return s2_tensor, target

    def __len__(self):
        return len(self.s2_samples) if self.preload else len(self.s2_paths)

