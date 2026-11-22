"""Pixel-count metrics for semantic segmentation."""
import torch


@torch.no_grad()
def segmentation_confusion_matrix(logits, targets, num_classes, ignore_index):
    """Count pixels with ground-truth classes in rows and predictions in columns."""
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    indices = targets[valid].to(torch.int64) * num_classes + predictions[valid]
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion_matrix(confusion):
    """Exclude classes absent from both ground truth and predictions from mIoU."""
    counts = confusion.to(torch.float64)
    intersection = counts.diag()
    union = counts.sum(dim=1) + counts.sum(dim=0) - intersection
    total = counts.sum()
    accuracy = (intersection.sum() / total).item() if total.item() else 0.0
    present = union > 0
    mean_iou = (intersection[present] / union[present]).mean().item() if present.any().item() else 0.0
    return accuracy, mean_iou


def compute_metrics(logits, targets, num_classes, ignore_index):
    """Calculate metrics for a single batch using the same pixel-count convention."""
    confusion = segmentation_confusion_matrix(logits, targets, num_classes, ignore_index)
    return metrics_from_confusion_matrix(confusion)
