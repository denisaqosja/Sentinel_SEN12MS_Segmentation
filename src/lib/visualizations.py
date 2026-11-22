import os, torch
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def plot_loss_curves(path, train_loss, test_loss):
    import seaborn as sns
    sns.set_theme()
    plt.figure()
    plt.plot(train_loss, label="training loss")
    plt.xlabel("Epochs", fontsize=13, fontweight='bold', loc="center")
    plt.ylabel("Loss", fontsize=13, fontweight='bold', loc="center")
    plt.legend(loc="best")
    plt.title("Training loss", fontsize=14, c="green", fontweight='bold')
    plt.savefig(os.path.join(path, "training_loss.png"), dpi=150)

    plt.figure()
    plt.plot(test_loss, label="test loss", c="orange")
    plt.xlabel("Epochs", fontsize=13, fontweight='bold', loc="center")
    plt.ylabel("Loss", fontsize=13, fontweight='bold', loc="center")
    plt.legend(loc="best")
    plt.title("Test loss", fontsize=14, c="green", fontweight='bold')
    plt.savefig(os.path.join(path, "test_loss.png"), dpi=150)

    plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.plot(test_loss, label="test loss")
    plt.xlabel("Epochs", fontsize=13, fontweight='bold', loc="center")
    plt.ylabel("Loss", fontsize=13, fontweight='bold', loc="center")
    plt.legend(loc="best")
    plt.title("Loss curves", fontsize=14, c="green", fontweight='bold')
    plt.savefig(os.path.join(path, "loss_curves.png"))

    # save the data
    with open(os.path.join(path, "train_loss.npy"), "wb") as file:
        np.save(file, train_loss, allow_pickle=True)
    with open(os.path.join(path, "test_loss.npy"), "wb") as file:
        np.save(file, test_loss, allow_pickle=True)

    return


def plot_metrics(path, metric):
    import seaborn as sns
    sns.set_theme()
    plt.figure()
    plt.plot(metric,  label="accuracy", c="red")
    plt.xlabel("Epochs", fontsize=13, fontweight='bold', loc="center")
    plt.ylabel("Accuracy", fontsize=13, fontweight='bold', loc="center")
    plt.legend(loc="best")
    plt.savefig(os.path.join(path, "accuracy.png"), dpi=150)

    # save the data
    with open(os.path.join(path, "accuracy.npy"), "wb") as file:
        np.save(file, metric, allow_pickle=True)

    return


def visualize_confusion_matrix(path, targets, preds, labels):
    confusion_matrix = metrics.confusion_matrix(targets, preds)
    metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot(xticks_rotation="vertical")
    plt.savefig(os.path.join(path, "Confusion_Matrix.png"))

    return


def tensor2img(tensor, out_type=torch.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    
    if n_dim == 5:
        tensor_permuted = torch.permute(tensor, [0, 1, 3, 4, 2])  # HWC, RGB
    elif n_dim == 4:
        tensor_permuted = torch.permute(tensor, [0, 2, 3, 1])  # HWC, RGB
    elif n_dim == 3:
        tensor_permuted = np.transpose(tensor, [1, 2, 0])  # HWC, RGB
    else:
        raise TypeError(
            'Only support 5D, 4D and 3D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == torch.uint8:
        tensor_permuted = (tensor_permuted * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return tensor_permuted


def plot_grid(tensor, img_path, img_idx, show=False):
    batch_size = tensor.shape[0]
        
    plt.figure(figsize=(10,10))
    for i in range(batch_size):
        plt.subplot(4,3,i+1)
        plt.imshow(tensor[i], cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
    
    plt.savefig(f"{img_path}_{img_idx}", dpi=300, bbox_inches='tight')
    plt.close()
    return 


def plot_denoising_grid(tensor, img_path):
    # plot a tensor of shape 5
    n_imgs = tensor.shape[0]
    
    for img_id in range(n_imgs):
        plot_grid(tensor[img_id], img_path, img_id)
    
    return 


def plot_fig(img, img_path, img_idx):
    # plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    
    plt.savefig(f"{img_path}_{img_idx}", bbox_inches='tight')
    plt.close()
    return 


def save_img(tensor, img_path, save_grid=False): 
    # convert tensor to img / numpy
    img = tensor2img(tensor).numpy()  # uint8
    if save_grid:
        plot_denoising_grid(img, img_path)
    else:
        if img.ndim == 4:
            for img_id in range(img.shape[0]):
                plot_fig(img[img_id], img_path, img_id)
        elif img.ndim == 3:
            plot_fig(img, img_path, img_idx=0)
        else:
            raise NotImplementedError(f"Image should be of shape CxWxH!!")
            

def plot_learning_curves(path, iters, loss, phase="train"):
    import seaborn as sns
    sns.set_theme()
    plt.figure()
    plt.plot(iters, loss, label=f"{phase} loss")
    plt.xlabel("Iters", fontsize=13, fontweight='bold', loc="center")
    plt.ylabel("Loss", fontsize=13, fontweight='bold', loc="center")
    plt.legend(loc="best")
    plt.title(f"{phase} loss", fontsize=14, c="green", fontweight='bold')
    plt.savefig(os.path.join(path, f"{phase}_loss.png"), dpi=150)
    plt.close()

    return


#
