import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.DFC2020 import DFC2020
from src.data.SEN12MS import SEN12MS


class S1Bands(Enum):
	VV = 1
	VH = 2
	ALL = [VV, VH]
	NONE = None


class Sensor(Enum):
	s1 = "s1"
	s2 = "s2"
	lc = "lc"
	dfc = "dfc"


PHASE_DATASETS = {
	"train": "dfc2020",
	"val": "dfc2020",
	"inference": "sen12ms"
}


@dataclass(frozen=True)
class Observation:
	season: str
	scene: str
	patch_id: int


def build_dataloaders(config: dict, phase: str = "train"):
	dataset_mode = dataset_mode_from_phase(phase)
	data_config = config["data"]

	if phase.lower() == "inference":
		inference_dataset = SEN12MS(config, split="inference", mode=dataset_mode)
		inference_loader = DataLoader(
			inference_dataset,
			batch_size=data_config.get("batch_size", data_config["inference"]["batch_size"]),
			shuffle=False,
			num_workers=data_config.get("num_workers", 4),
			pin_memory=data_config.get("pin_memory", True),
		)
		return inference_loader, None

	train_dataset = DFC2020(config, split="train")
	val_dataset = DFC2020(config, split="val")

	train_loader = DataLoader(
		train_dataset,
		batch_size=data_config["batch_size"],
		shuffle=True,
		num_workers=data_config.get("num_workers", 4),
		pin_memory=data_config.get("pin_memory", True),
		drop_last=data_config.get("drop_last", False),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=data_config.get("val_batch_size", data_config["batch_size"]),
		shuffle=False,
		num_workers=data_config.get("num_workers", 4),
		pin_memory=data_config.get("pin_memory", True),
	)
	return train_loader, val_loader


def dataset_mode_from_phase(phase: str):
	try:
		return PHASE_DATASETS[phase.lower()]
	except KeyError as exc:
		raise ValueError(f"Unsupported phase {phase!r}. Use one of {sorted(PHASE_DATASETS)}") from exc


""" Taken from another work with SEN12MS and DFC2020"""
"""
class SEN12MS(Dataset):
	def __init__(self,
		root_dir: str,
		mode: str,
		split: str,
		image_size: int = 256,
		input_modalities: Iterable[str] = ("s1", "s2"),
		use_dfc_labels: bool = True,
		normalize: bool = True,
		clip_values: bool = True,
		used_data_fraction: float = 1.0,
		seed: int = 42,
	):
		self.root_dir = Path(root_dir) 
		self.mode = mode.lower()
		self.split = split
		self.image_size = image_size
		self.input_modalities = tuple(input_modalities)
		self.use_dfc_labels = use_dfc_labels
		self.has_labels = self.mode != "sen12ms"
		self.normalize = normalize
		self.clip_values = clip_values

		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset path does not exist: {self.root_dir}")

		self.observations = self._scan_observations()
		if used_data_fraction < 1.0:
			rng = np.random.default_rng(seed)
			count = max(1, int(len(self.observations) * used_data_fraction))
			indices = sorted(rng.choice(len(self.observations), count, replace=False).tolist())
			self.observations = [self.observations[index] for index in indices]

	@classmethod
	def from_config(cls, config: dict, split: str, mode: Optional[str] = None):
		data_config = config["data"]
		if split not in data_config:
			raise KeyError(f"Missing data.{split}.root_dir in the config")
		split_config = data_config[split]
		return cls(
			root_dir=split_config["root_dir"],
			mode=mode or split_config.get("mode", data_config.get("dataset", "dfc2020")),
			split=split,
			image_size=data_config.get("image_size", 256),
			input_modalities=data_config.get("input_modalities", ["s1", "s2"]),
			use_dfc_labels=data_config.get("use_dfc_labels", True),
			normalize=data_config.get("normalize", True),
			clip_values=data_config.get("clip_values", True),
			used_data_fraction=split_config.get("used_data_fraction", 1.0),
			seed=config.get("seed", 42),
		)

	@property
	def num_input_channels(self) -> int:
		count = 0
		if "s1" in self.input_modalities:
			count += 2
		if "s2" in self.input_modalities:
			count += 13
		return count

	def __len__(self):
		return len(self.observations)

	def __getitem__(self, index):
		observation = self.observations[index]
		window = self._sample_window()

		arrays = []
		if "s1" in self.input_modalities:
			s1 = self._read_patch(observation, S1Bands.ALL, window=window)
			arrays.append(self._prepare_s1(s1))
		if "s2" in self.input_modalities:
			s2 = self._read_patch(observation, S2Bands.ALL, window=window)
			arrays.append(self._prepare_s2(s2))

		image = np.concatenate(arrays, axis=0).astype(np.float32)
		image_tensor = torch.from_numpy(image)
		item = {
			"image": image_tensor,
			"season": observation.season,
			"scene": observation.scene,
			"patch_id": observation.patch_id,
		}
		if self.has_labels:
			mask = self._read_mask(observation, window=window)
			item["mask"] = torch.from_numpy(mask.astype(np.int64))
		return item

	def _scan_observations(self):
		sensor_dir = self._scan_sensor_dir()
		if sensor_dir is None:
			raise RuntimeError(
				f"No modality folders found under {self.root_dir}. Expected folders like s1, s2, lc, or dfc."
			)

		observations = []
		for patch_path in sorted(sensor_dir.glob("*.tif")):
			parsed = self._parse_patch_filename(patch_path.name)
			if parsed is None:
				continue
			observations.append(
				Observation(
					parsed["season"],
					parsed["scene"],
					parsed["patch_id"],
				)
			)
		if not observations:
			raise RuntimeError(f"No TIFF observations found in {sensor_dir}")
		return observations

	def _scan_sensor_dir(self):
		for sensor in self.input_modalities:
			sensor_dir = self.root_dir / sensor
			if sensor_dir.is_dir():
				return sensor_dir
		for sensor in (Sensor.s2.value, Sensor.s1.value, Sensor.dfc.value, Sensor.lc.value):
			sensor_dir = self.root_dir / sensor
			if sensor_dir.is_dir():
				return sensor_dir
		return None

	def _sample_window(self):
		if self.image_size >= 256:
			return None
		if self.split == "train":
			max_offset = 256 - self.image_size
			x_offset, y_offset = np.random.randint(0, max_offset + 1, 2)
			return Window(x_offset, y_offset, self.image_size, self.image_size)
		return Window(0, 0, self.image_size, self.image_size)

	def _read_patch(self, observation: Observation, bands, window=None):
		sensor = self._sensor_from_bands(bands)
		band_indexes = self._band_indexes(bands)
		patch_path = self._patch_path(observation, sensor)

		with rasterio.open(patch_path) as patch:
			data = patch.read(band_indexes, window=window)
		if sensor == Sensor.lc.value:
			data = IGBP2DFC[data]
		return data

	def _read_mask(self, observation: Observation, window=None):
		use_dfc = self.use_dfc_labels and self._mask_exists(observation, Sensor.dfc.value)
		bands = LCBands.DFC if use_dfc else LCBands.LC
		mask = self._read_patch(observation, bands, window=window)[0].astype(np.int64)
		return remap_to_eight_dfc_classes(mask)

	def _mask_exists(self, observation: Observation, sensor: str):
		return self._patch_path(observation, sensor).exists()

	def _patch_path(self, observation: Observation, sensor: str):
		season = observation.season
		scene = f"{sensor}_{observation.scene}"
		filename = f"{season}_{scene}_p{observation.patch_id}.tif"
		return self.root_dir / sensor / filename

	@staticmethod
	def _parse_patch_filename(filename: str):
		match = PATCH_FILENAME_PATTERN.match(filename)
		if not match:
			return None
		return {
			"season": match.group("season"),
			"sensor": match.group("sensor"),
			"scene": match.group("scene"),
			"patch_id": int(match.group("patch_id")),
		}

	def _prepare_s1(self, s1):
		s1 = s1.astype(np.float32)
		if self.clip_values:
			s1 = np.clip(s1, -25.0, 0.0) + 25.0
		if self.normalize:
			s1 = (s1 - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
		return s1

	def _prepare_s2(self, s2):
		s2 = s2.astype(np.float32)
		if self.clip_values:
			s2 = np.clip(s2, 0.0, 10000.0)
		if self.normalize:
			s2 = (s2 - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
		return s2

	@staticmethod
	def _sensor_from_bands(bands):
		first_band = bands[0] if isinstance(bands, list) else bands
		if isinstance(first_band, S1Bands):
			return Sensor.s1.value
		if isinstance(first_band, S2Bands):
			return Sensor.s2.value
		if isinstance(first_band, LCBands):
			return Sensor.dfc.value if first_band == LCBands.DFC else Sensor.lc.value
		raise ValueError(f"Unsupported band selection: {bands}")

	@staticmethod
	def _band_indexes(bands):
		if isinstance(bands, list):
			return [band.value for band in bands]
		value = bands.value
		if isinstance(value, list):
			return [band.value for band in value]
		return 1 if isinstance(bands, LCBands) else value


def remap_to_eight_dfc_classes(mask: np.ndarray) -> np.ndarray:
	mask = mask.copy()
	mask[mask == 3] = 0
	mask[mask == 8] = 0
	mask[mask >= 3] -= 1
	mask[mask >= 8] -= 1
	mask -= 1
	mask[mask == -1] = 255
	return mask
"""