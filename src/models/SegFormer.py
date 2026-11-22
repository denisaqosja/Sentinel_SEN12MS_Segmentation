import torch
import torch.nn as nn
import torch.nn.functional as F

from .freezing import EncoderDecoderFreezingMixing

from transformers import SegformerForSemanticSegmentation

class SegFormer(EncoderDecoderFreezingMixing, nn.Module):
	def __init__(
		self,
		model_name: str = "nvidia/mit-b0",
		num_classes: int = 8,
		input_channels: int = 15,
		pretrained: bool = True,
		freeze_encoder: bool = False,
		freeze_decoder: bool = False,
	):
		super().__init__()
		
		id2label = {i: f"class_{i}" for i in range(num_classes)}
		label2id = {v: k for k, v in id2label.items()}

		if pretrained:
			self.model = SegformerForSemanticSegmentation.from_pretrained(
						model_name,
						num_labels=num_classes,
						id2label=id2label,
						label2id=label2id,
						num_channels=input_channels,
						ignore_mismatched_sizes=True,
					)
		else:
			from transformers import SegformerConfig

			config = SegformerConfig.from_pretrained(
				model_name,
				num_labels=num_classes,
				num_channels=input_channels,
			)
			self.model = SegformerForSemanticSegmentation(config)

		self.set_freezing(freeze_encoder, freeze_decoder)

	@property
	def encoder(self):
		return self.model.segformer

	@property
	def decoder(self):
		return self.model.decode_head

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		logits = self.model(pixel_values=x).logits
		# SegFormer outputs logits at 1/4 of the input resolution. Therefore, upsample the logits to match the input resolution.
		return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_segformer(config: dict) -> nn.Module:
	model_config = config["model"]
	data_config = config["data"]
	return SegFormer(
		model_name=model_config.get("segformer_name", "nvidia/mit-b0"),
		num_classes=model_config.get("num_classes", 8),
		input_channels=model_config.get("input_channels", _count_input_channels(data_config)),
		pretrained=model_config.get("pretrained", True),
		freeze_encoder=model_config.get("freeze_encoder", False),
		freeze_decoder=model_config.get("freeze_decoder", False),
	)


def _count_input_channels(data_config: dict) -> int:
	modalities = data_config.get("input_modalities", ["s1", "s2"])
	return (1 if "s1" in modalities else 0) + (3 if "s2" in modalities else 0)
