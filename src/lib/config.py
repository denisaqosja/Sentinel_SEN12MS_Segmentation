"""Resolve model-specific experiment settings before saving or loading data."""
from copy import deepcopy
from pathlib import Path

import yaml


MODEL_NAMES = ("segformer", "dino")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_training_config(path, model_name=None):
    path = Path(path).expanduser()
    if not path.is_file() and not path.is_absolute():
        path = PROJECT_ROOT / "configs" / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return resolve_model_config(config, model_name)


def resolve_model_config(config, model_name=None):
    """Return the effective config; preserve flat configs from older checkpoints."""
    config = deepcopy(config)
    model = config["model"]
    name = (model_name or model.get("name", "segformer")).lower()
    if name == "dinovs":
        name = "dino"
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {name!r}. Choose from {MODEL_NAMES}")
    variants = model.pop("variants", None)
    if variants is not None:
        if name not in variants:
            raise ValueError(f"Missing model.variants.{name} configuration")
        selected = deepcopy(variants[name])
        config["data"].update(selected.pop("data", {}))
        model.update(selected)
    model["name"] = name
    if name == "dino":
        if model.get("input_channels", 3) != 3:
            raise ValueError("DINO requires model.input_channels=3 (Sentinel-2 RGB)")
        # DFC is a label modality; only S2 is passed to the model.
        if any(m not in ("s2", "dfc") for m in config["data"].get("input_modalities", ["s2"])):
            raise ValueError("DINO requires S2 RGB input; remove S1 from input_modalities")
        config["data"]["input_modalities"] = ["s2"]
        config["data"].setdefault("preprocessing", "reflectance")
    return config
