"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import warnings

import torch

from src.lib.logger import log_function, print_
from src.lib.utils import create_directory



MODELS = ["ConvNet", "UNet", "VGG", "ResNet", "ResNet50", "ResNext", "EfficientNet", "ConvNext", "ConvNext_nano", "ConvNext_pico"]

@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    import torchvision
    from .model_utils import replace_classifier_outsize

    model_name = model_params["model_name"]
    print(model_name)
    pretrained_flag = model_params["pretrained"]
    if model_name not in MODELS:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models; {MODELS}")
    cur_model_params = model_params.get(model_name, {})
    if model_name == "ConvNet":
        from . import ConvNet
        model = ConvNet(**cur_model_params)
    elif model_name == "VGG":
        model = torchvision.models.vgg16(pretrained=pretrained_flag)
        model = replace_classifier_outsize(model_name, model, cur_model_params["num_classes"])
    elif model_name == "ResNet":
        model = torchvision.models.resnet18(pretrained=False)
        model = replace_classifier_outsize(model_name, model, cur_model_params["num_classes"])
    elif model_name == "ResNet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        model = replace_classifier_outsize(model_name, model, cur_model_params["num_classes"])
    elif model_name == "ResNext":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model = replace_classifier_outsize(model_name, model, cur_model_params["num_classes"])
    elif model_name == "EfficientNet":
        model = torchvision.models.efficientnet_v2_s(pretrained=True)
        model = replace_classifier_outsize(model_name, model, cur_model_params["num_classes"])
    elif model_name == "ConvNext":
        from . import ConvNext
        model = ConvNext(**cur_model_params)
    elif model_name == "ConvNext_nano":
        import timm
        model = timm.create_model("convnext_nano", pretrained=True, num_classes=cur_model_params["num_classes"],
                                  drop_rate=0.1, drop_path_rate=0.0)
    elif model_name == "ConvNext_pico":
        import timm
        model = timm.create_model("convnext_pico", pretrained=True, num_classes=cur_model_params["num_classes"],
                                  drop_rate=0.1, drop_path_rate=0.0)
    else:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models; {MODELS}")

    return model


def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                create_directory(self_.exp_path, "models")
                save_checkpoint(
                    path=os.path.join(self_.exp_path, "models", fname),
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=getattr(self_, "scheduler", None),
                    scaler=getattr(self_, "scaler", None),
                    epoch=self_.epoch,
                    best_mean_iou=getattr(self_, "best_mean_iou", 0.0),
                    config=getattr(self_, "config", {}),
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except


def save_checkpoint(path, model, optimizer, scaler, epoch, best_mean_iou, config, scheduler=None):
    """Save model, optimization state, and metadata at the end of an epoch."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_mean_iou": best_mean_iou,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scaler, device, scheduler=None):
    """Restore training state and return the next epoch and best validation mIoU.

    Older checkpoints may omit scaler/scheduler state and best-mIoU metadata.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        else:
            warnings.warn(
                "Checkpoint has no scheduler state. The optimizer learning rate is restored, "
                "but the scheduler restarts its history using the current configuration.",
                UserWarning,
                stacklevel=2,
            )
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("best_mean_iou", 0.0)


@log_function
def setup_optimization(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model
    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """
    from .model_utils import split_model_parameters

    lr = exp_params["training"]["lr"]
    lr_backbone = exp_params["training"]["lr_backbone"]

    # filtering parameters to assign different learning rates
    classifier_params, backbone_params = split_model_parameters(model)
    parameters = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr}
        ]
    print_(f"  --> Model learning rate {lr}")
    print_(f"  --> Encoder learning rate {lr_backbone}")

    # setting up optimizer and LR-scheduler
    optimizer = setup_optimizer(parameters, exp_params)
    scheduler = setup_scheduler(exp_params, optimizer)

    return optimizer, scheduler


def setup_optimizer(parameters, exp_params):
    """ Instanciating a new optimizer """
    lr = exp_params["training"]["lr"]
    momentum = exp_params["training"]["momentum"]
    optimizer = exp_params["training"]["optimizer"]
    nesterov = exp_params["training"]["nesterov"]

    # SGD-based optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optimizer == "adamw":
        print(f"{float(lr) = }")
        optimizer = torch.optim.AdamW(parameters, lr=float(lr))
    else:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)

    return optimizer


def setup_scheduler(exp_params, optimizer):
    """ Instanciating a new scheduler """
    lr = exp_params["training"]["lr"]
    lr_factor = exp_params["training"]["lr_factor"]
    patience = exp_params["training"]["patience"]
    scheduler = exp_params["training"]["scheduler"]

    if scheduler == "plateau":
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"  --> Patience: {patience}")
        print_(f"  --> Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif scheduler == "step":
        print_("Setting up Step LR-Scheduler")
        print_(f"  --> Step Size: {patience}")
        print_(f"  --> Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )
    elif scheduler == "exponential":
        from src.lib.schedulers import ExponentialLRSchedule
        print_("Setting up Exponential LR-Scheduler")
        print_(f"  --> Init LR: {lr}")
        print_(f"  --> Factor:  {lr_factor}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    return scheduler


def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False):
    """
    Updating the learning rate scheduler

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    scheduler_type = exp_params["training"]["scheduler"]
    if scheduler_type == "plateau" and end_epoch:
        scheduler.step(control_metric)
    elif scheduler_type == "step" and end_epoch:
        scheduler.step()
    elif scheduler_type == "exponential":
        scheduler.step(iter)
    return


@log_function
def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    from src.lib.schedulers import LRWarmUp

    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if use_warmup:
        warmup_steps = params["warmup_steps"]
        warmup_epochs = params["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
    return lr_warmup


#
