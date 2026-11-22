import argparse
import random

import os
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.lib import setup_config
from src.lib.config import load_training_config
from src.lib.logger import Logger
from src.lib.tensorboard import TensorboardWriter
from src.data.dataloader import build_dataloaders
from src.models.SegFormer import build_segformer
from src.models.DINO import build_dino


MODELS = {
    "segformer": build_segformer,
    "dinovs": build_dino,
    "dino": build_dino
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer or DINO on satellite segmentation data.")
    parser.add_argument("--config", default="dfc.yaml", help="Config path or filename under configs/.")
    parser.add_argument("--phase", choices=['train', 'val', 'inference'], default='train', 
                        help="Phase: train, val, or inference (use DFC2020/SEN12MS).")
    parser.add_argument("--model", choices=sorted(MODELS), default=None, 
                        help="Override model.name from config.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config):
    model_name = config["model"]["name"].lower()
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from {sorted(MODELS)}")
    return MODELS[model_name](config)


def log_frozen_parameters(tb_logger, model, step):
    """Log frozen scalar parameters as a percentage of each part's total.

    Counts parameter elements (including biases), rather than parameter tensors.
    Non-parameter buffers such as BatchNorm running statistics are excluded.
    """
    parts = {"model": model}
    for name in ("encoder", "decoder"):
        part = getattr(model, name, None)
        if isinstance(part, nn.Module):
            parts[name] = part
    for name, part in parts.items():
        total = sum(parameter.numel() for parameter in part.parameters())
        frozen = sum(
            parameter.numel() for parameter in part.parameters()
            if not parameter.requires_grad
        )
        percentage = 100.0 * frozen / total if total else 0.0
        tb_logger.add_scalar(f"parameters/{name}/frozen_percent", percentage, step)


def compute_metrics(logits, targets, num_classes, ignore_index):
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    correct = ((predictions == targets) & valid).sum().item()
    total = valid.sum().item()
    pixel_accuracy = correct / total if total else 0.0

    ious = []
    for class_index in range(num_classes):
        pred_class = predictions == class_index
        target_class = targets == class_index
        intersection = (pred_class & target_class & valid).sum().item()
        union = ((pred_class | target_class) & valid).sum().item()
        if union > 0:
            ious.append(intersection / union)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return pixel_accuracy, mean_iou

def move_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch(value, device) for value in batch)
    return batch

def reconstruction_figure(s2_image, sample_pred, s2_target, n_items):
    figure, axes = plt.subplots(n_items, 3, figsize=(10, 3 * n_items), squeeze=False)
    for row in range(n_items):
        panels = ((s2_image[row, 0], "S2 image"), (s2_target[row], "target"), (sample_pred[row], "prediction"))
        for col, (image, title) in enumerate(panels):
            axis = axes[row, col]
            mesh = axis.pcolor(image.numpy(), cmap="viridis", shading="auto")
            figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    return figure

@torch.no_grad()
def log_reconstruction(tb_logger, tag, model, loader, device, step):
    sample_batch = move_batch(next(iter(loader)), device)
    s2_image = sample_batch[0].cpu()
    s2_target = sample_batch[1].cpu()
    sample_pred = model(sample_batch[0]).argmax(dim=1).cpu()
    figure = reconstruction_figure(s2_image, sample_pred, s2_target, n_items=min(4, s2_image.shape[0]))
    tb_logger.add_figure(tag, figure, step)
    plt.close(figure)

# def train(model, loader, criterion, optimizer, device, scaler, use_amp):
#     model.train()
#     running_loss = 0.0
#     num_batches = 0

#     for batch in loader:
#         images = batch[0].to(device, non_blocking=True)
#         seg_masks = batch[1].to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)
#         with torch.amp.autocast(device.type, enabled=use_amp):
#             logits = model(images)
#             loss = criterion(logits, seg_masks)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         num_batches += 1

#     return running_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index, desc="Validation"):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    num_batches = 0

    with tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True) as progress:
        for batch in progress:
            images = batch[0].to(device, non_blocking=True)
            masks = batch[1].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            accuracy, mean_iou = compute_metrics(logits, masks, num_classes, ignore_index)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_iou += mean_iou
            num_batches += 1

            progress.set_postfix(
                loss=f"{total_loss / num_batches:.4f}",
                pixel_acc=f"{total_accuracy / num_batches:.2%}",
                mIoU=f"{total_iou / num_batches:.4f}",
                refresh=False,
            )

    if num_batches == 0:
        return {"loss": 0.0, "pixel_accuracy": 0.0, "mean_iou": 0.0}
    return {
        "loss": total_loss / num_batches,
        "pixel_accuracy": total_accuracy / num_batches,
        "mean_iou": total_iou / num_batches,
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, best_mean_iou, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_mean_iou": best_mean_iou,
            "config": config,
        },
        path,
    )


def main():
    # load the configuration 
    args = parse_args()
    config = load_training_config(args.config, args.model)

    # set up the experiment directories
    setup_config.create_exp_directories(config)
    exp_path = config["paths"]["experiment_path"]

    # log the experiment configuration 
    Logger(exp_path=exp_path).log_info("Starting training", message_type="new_exp")
    # save the configuration to the experiment directory
    setup_config.save_config(exp_path, config)

    set_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    tb_logger = TensorboardWriter(logdir=config["paths"]["tb_logger"])

    train_loader, val_loader = build_dataloaders(config, phase=args.phase)
    model = build_model(config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_parameters == 0:
        raise ValueError("No trainable parameters: unfreeze the encoder or decoder before training")
    print(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")

    train_config = config["training"]
    criterion = nn.CrossEntropyLoss(ignore_index=config["data"].get("ignore_index", 255))
    # initialize optimier and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 6e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_config.get("patience", 20),
        factor=train_config.get("lr_factor", 0.5),
    ) if train_config.get("scheduler", "plateau") == "plateau" else None
    
    # initialize automatic mixed precision (AMP) if enabled and using CUDA
    use_amp = train_config.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    reconstruction_frequency = train_config.get("reconstruction_frequency", 5)
    grad_clip = train_config.get("clip_grad_norm", 1.0)

    start_epoch = 0
    best_mean_iou = 0.0
    resume_training = config["training"].get("resume_training", None)
    if resume_training:
        resume_path = config["paths"].get("resume_checkpoint")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_mean_iou = checkpoint.get("best_mean_iou", 0.0)

    log_frozen_parameters(tb_logger, model, start_epoch)

    epochs = train_config.get("n_epochs", 50)
    num_classes = config["model"].get("num_classes", 8)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        train_losses = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} | Train",
                  unit="batch", dynamic_ncols=True) as progress:
            for batch in progress:
                images = batch[0].to(device, non_blocking=True)
                masks = batch[1].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, masks)

                scaler.scale(loss).backward()
                # Clip actual gradients before updating weights.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

                progress.set_postfix(
                    batch_loss=f"{loss.item():.4f}",
                    loss=f"{running_loss / num_batches:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )

        train_loss = running_loss / max(1, num_batches)
        metrics = evaluate(model, 
                           val_loader, 
                           criterion, 
                           device, num_classes, 
                           config["data"].get("ignore_index", 255),
                           desc=f"Epoch {epoch + 1}/{epochs} | Validation")
        
        if scheduler is not None:
            scheduler.step(metrics["loss"])

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['pixel_accuracy']:.4f} "
            f"val_miou={metrics['mean_iou']:.4f}"
        )

        # log lr, losses and metrics to TensorBoard
        log_frozen_parameters(tb_logger, model, epoch + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        tb_logger.add_scalar("training/lr", current_lr, epoch + 1)
        tb_logger.add_scalar("loss/train_total", train_loss, epoch + 1)
        tb_logger.add_scalar("loss/valid", metrics['loss'], epoch + 1)
        tb_logger.add_scalar("metrics/IOU", metrics['mean_iou'], epoch + 1)
        tb_logger.add_scalar("metrics/Accuracy", metrics['pixel_accuracy'], epoch + 1)

        if reconstruction_frequency > 0 and (epoch + 1) % reconstruction_frequency == 0:
            model.eval()
            log_reconstruction(tb_logger, "reconstruction/train",
                               model, train_loader, device, epoch + 1)
            log_reconstruction(tb_logger, "reconstruction/valid",
                               model, val_loader, device, epoch + 1)

        save_checkpoint(os.path.join(config["paths"]["checkpoint"], "last.pt"), 
                        model, optimizer, scaler, epoch, best_mean_iou, config)
        if metrics["mean_iou"] >= best_mean_iou:
            best_mean_iou = metrics["mean_iou"]
            save_checkpoint(os.path.join(config["paths"]["checkpoint"], "checkpoint_best.pth"),
                            model, optimizer, scaler, epoch, best_mean_iou, config)

        if train_config.get("save_frequency", 0) and (epoch + 1) % train_config["save_frequency"] == 0:
            save_checkpoint(os.path.join(config["paths"]["checkpoint"], f"epoch_{epoch + 1}.pt"), 
                            model, optimizer, scaler, epoch, best_mean_iou, config)

if __name__ == "__main__":
    main()
