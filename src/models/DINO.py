import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.schedulers import Freezer

from transformers import AutoConfig, AutoImageProcessor, AutoModel

class DINO(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        num_classes: int = 8,
        input_channels: int = 15,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_encoder: bool | None = None,
        freeze_decoder: bool = False,
        frozen_epochs: int | None = None,
    ):
        super().__init__()

        if input_channels != 3:
            raise ValueError(
                "The SAT-493M checkpoint expects 3-channel RGB input. "
                "Select Sentinel-2 bands B4, B3, B2 in that order."
            )

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer("image_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = self.backbone.config.num_register_tokens

        # One score per class at each patch location.
        self.segmentation_head = nn.Conv2d(
            self.backbone.config.hidden_size,
            num_classes,
            kernel_size=1,
        )

        # Keep the previous constructor argument as an encoder-freezing alias.
        self.freezer = Freezer(
            self, frozen_epochs=frozen_epochs,
            freeze_encoder=freeze_backbone if freeze_encoder is None else freeze_encoder,
            freeze_decoder=freeze_decoder,
        )

    def set_freezing(self, freeze_encoder=False, freeze_decoder=False):
        self.freezer.set_freezing(freeze_encoder, freeze_decoder)
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "freezer"):
            self.freezer.enforce_training_mode()
        return self

    @property
    def encoder(self):
        return self.backbone

    @property
    def decoder(self):
        return self.segmentation_head

    @property
    def freeze_backbone(self):
        return self.freeze_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects RGB float images scaled to [0, 1], shape [B, 3, H, W].
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(x.shape)}")

        original_size = x.shape[-2:]
        x = (x - self.image_mean) / self.image_std

        # Pad rather than resize so the output retains the original pixel grid.
        patch = self.patch_size
        if isinstance(patch, (tuple, list)):
            patch_h, patch_w = patch
        else:
            patch_h = patch_w = patch

        pad_h = (-x.shape[-2]) % patch_h
        pad_w = (-x.shape[-1]) % patch_w
        x = F.pad(x, (0, pad_w, 0, pad_h))

        grid_h = x.shape[-2] // patch_h
        grid_w = x.shape[-1] // patch_w

        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone(pixel_values=x).last_hidden_state
        else:
            tokens = self.backbone(pixel_values=x).last_hidden_state

        # Token order: CLS, register tokens, then image patch tokens.
        patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]
        b, n, c = patch_tokens.shape
        if n != grid_h * grid_w:
            raise RuntimeError(
                f"Expected {grid_h * grid_w} patch tokens, received {n}"
            )

        features = patch_tokens.transpose(1, 2).reshape(
            b, c, grid_h, grid_w
        )
        logits = self.segmentation_head(features)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return logits[..., : original_size[0], : original_size[1]]


def build_dino(config: dict) -> nn.Module:
    model_config = config["model"]
    data_config = config["data"]

    modalities = data_config.get("input_modalities", ["s2"])
    if modalities != ["s2"]:
        raise ValueError(
            "This RGB DINOv3 setup expects input_modalities=['s2']. "
            )
    return DINO(
		model_name=model_config.get("dino_name", "facebook/dinov3-vitl16-pretrain-sat493m"),
		num_classes=model_config.get("num_classes", 8),
		input_channels=model_config.get("input_channels", _count_input_channels(data_config)),
		pretrained=model_config.get("pretrained", True),
		freeze_encoder=model_config.get("freeze_encoder", model_config.get("freeze_backbone", False)),
		freeze_decoder=model_config.get("freeze_decoder", False),
		frozen_epochs=model_config.get("frozen_epochs"),
	)


def _count_input_channels(data_config: dict) -> int:
	modalities = data_config.get("input_modalities", ["s1", "s2"])
	return (1 if "s1" in modalities else 0) + (3 if "s2" in modalities else 0)
