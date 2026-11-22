"""
Model utils
"""

from time import time
from tqdm import tqdm
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn import init
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

from lib.logger import print_, log_info


def split_model_parameters(model):
    """ Splitting model paramters into convolutional backbone and classifier """
    backbone = list(filter(lambda kv: "head" not in kv[0], model.named_parameters()))
    classifier = list(filter(lambda kv: "head" in kv[0], model.named_parameters()))
    backbone_params = [p[1] for p in backbone]
    classifier_params = [p[1] for p in classifier]
    return classifier_params, backbone_params


def count_model_params(model, verbose=False):
    """Counting number of learnable parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print_(f"  --> Number of learnable parameters: {num_params}")
    return num_params


def replace_classifier_outsize(model_name, model, num_classes):
    if "VGG" in model_name:
        class_layer_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=class_layer_features, out_features=num_classes)
    elif "ResNet" in model_name:
        class_layer_features = model.fc.in_features
        model.fc = nn.Linear(in_features=class_layer_features, out_features=num_classes)
    elif model_name == "ResNext":
        class_layer_features = model.fc.in_features
        model.fc = nn.Linear(in_features=class_layer_features, out_features=num_classes)
    elif model_name == "EfficientNet":
        class_layer_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=class_layer_features, out_features=num_classes)
    else:
        raise ValueError(f"Upsi! Unrecognized {model_name = }...")

    return model


def compute_flops(model, dummy_input, verbose=True, detailed=False):
    """ Computing the number of activations and flops in a forward pass """
    func = print_ if verbose else log_info

    # benchmarking
    fca = FlopCountAnalysis(model, dummy_input)
    act = ActivationCountAnalysis(model, dummy_input)
    if detailed:
        fcs = flop_count_str(fca)
        func(fcs)
    total_flops = fca.total()
    total_act = act.total()

    # logging
    func("  --> Number of FLOPS in a forward pass:")
    func(f"   --> FLOPS = {total_flops}")
    func(f"    --> FLOPS = {round(total_flops / 1e9, 3)}G")
    func("  --> Number of activations in a forward pass:")
    func(f"    --> Activations = {total_act}")
    func(f"    --> Activations = {round(total_act / 1e6, 3)}M")
    return total_flops, total_act


def compute_throughput(model, dataset, device, num_imgs=500, use_tqdm=True, verbose=True):
    """ Computing the throughput of a model in imgs/s """
    times = []
    N = min(num_imgs, len(dataset))
    iterator = tqdm(range(N)) if use_tqdm else range(N)
    model = model.to(device)

    # benchmarking by averaging over N images
    for i in iterator:
        img = dataset[i][0].unsqueeze(0).to(device)
        torch.cuda.synchronize()
        start = time()
        _ = model(img)
        torch.cuda.synchronize()
        times.append(time() - start)
    avg_time_per_img = np.mean(times)
    throughput = 1 / avg_time_per_img

    # logging
    func = print_ if verbose else log_info
    func(f"  --> Average time per image: {round(avg_time_per_img, 3)}s")
    func(f"  --> Throughput: {round(throughput)} imgs/s")
    return throughput, avg_time_per_img


def freeze_params(model):
    """Freezing model params to avoid updates in backward pass"""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_params(model):
    """Unfreezing model params to allow for updates during backward pass"""
    for param in model.parameters():
        param.requires_grad = True
    return model


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    log_info(f"  --> Initialization method: {init_type}")

    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))