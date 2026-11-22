"""Paired spatial augmentation, batch mixup, and smoothed segmentation loss."""
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop, RandomRotation, InterpolationMode
from torchvision.transforms import functional as TF


def augment_pair(image, target, crop_scale=(0.7, 1.0), crop_ratio=(0.9, 1.1),
                 rotation_degrees=30.0, ignore_index=255):
    """Share crop/angle across RGB and masks; never interpolate class IDs."""
    output_size = list(target.shape[-2:])
    top, left, height, width = RandomResizedCrop.get_params(
        image, scale=crop_scale, ratio=crop_ratio
    )
    image = TF.resized_crop(image, top, left, height, width, output_size,
                            interpolation=InterpolationMode.BILINEAR, antialias=True)
    target = TF.resized_crop(target.unsqueeze(0), top, left, height, width,
                             output_size, interpolation=InterpolationMode.NEAREST_EXACT).squeeze(0)
    angle = RandomRotation.get_params([-rotation_degrees, rotation_degrees])
    image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
    target = TF.rotate(target.unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST,
                       fill=ignore_index).squeeze(0)
    if torch.rand(()).item() < 0.5:
        image = torch.flip(image, dims=(-1,))
        target = torch.flip(target, dims=(-1,))
    return image.contiguous(), target.contiguous()


def mixup_batch(images, masks, alpha=0.2, probability=0.5):
    if alpha <= 0 or images.shape[0] < 2 or torch.rand(()).item() >= probability:
        return images, masks, masks, 1.0
    weight = torch.distributions.Beta(alpha, alpha).sample().item()
    permutation = torch.randperm(images.shape[0], device=images.device)
    return (weight * images + (1 - weight) * images[permutation],
            masks, masks[permutation], weight)


def segmentation_mixup_loss(logits, masks_a, masks_b, weight,
                            ignore_index=255, label_smoothing=0.0):
    """Mix class distributions via weighted losses, never numeric class IDs.

    Normalize by the weighted number of valid pixels. Each ignored target
    contributes no loss, including when only one of the two targets is ignored.
    """
    numerator = logits.sum() * 0.0
    denominator = logits.new_zeros(())
    for masks, factor in ((masks_a, weight), (masks_b, 1 - weight)):
        if factor == 0:
            continue
        losses = F.cross_entropy(logits, masks, reduction="none",
                                 ignore_index=ignore_index, label_smoothing=label_smoothing)
        valid = masks != ignore_index
        numerator = numerator + factor * losses[valid].sum()
        denominator = denominator + factor * valid.sum()
    return numerator / denominator.clamp_min(1e-12)
