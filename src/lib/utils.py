import shutil
import pickle
import os, torch
import datetime
import numpy as np
import matplotlib.pyplot as plt


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way
    """
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def create_directory(dir_path, dir_name=None):
    """
    Creating a folder in given path.
    """
    if dir_name is not None:
        dir_path = os.path.join(dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def delete_directory(dir_path):
    """
    Deleting a directory and all its contents
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return


def split_path(path):
    """ Splitting a path into a list containing the names of all directories to the path """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def load_pickle_file(path):
    """ Loading pickle file """
    with open(path, "rb") as a_file:
        data = pickle.load(a_file)
    return data


def save_pickle_file(path, data):
    """ Saving pickle file """
    with open(path, "wb") as file:
        pickle.dump(data, file)
    return


def save_npy(path, filename, np_array, save_fig=False):
    array_filepath = os.path.join(path["results_tensors"], f"{filename}.npy")
    with open(os.path.join(array_filepath, ".npy"), 'wb') as f:
        np.save(f, np_array)

    if save_fig:
        plt.pcolor(np.squeeze(np_array[0]))
        plt.savefig(os.path.join(path["results_images"], f"{filename}.png"))
    return 


def get_from_dict(params, key_list):
    """ Getting a value from a dictionary given a list with the keys to get there """
    for key in key_list:
        params = params[key]
    return params


def set_in_dict(params, key_list, value):
    """ Updating a dictionary value, indexed by a list of keys to get there """
    for key in key_list[:-1]:
        # params = params.setdefault(key, {})
        params = params[key]
    params[key_list[-1]] = value
    return


def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # getting all_params
    with open(savepath, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if isinstance(layer, torch.nn.Module):
            log_module(module=layer, exp_path=exp_path, fname=fname)
    return


def log_module(module, exp_path, fname="model_architecture.txt", append=True):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # writing from scratch or appending to existing file
    if append is False:
        with open(savepath, "w") as f:
            f.write("")
    else:
        with open(savepath, "a") as f:
            f.write("\n\n")

    # writing info
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return


def set_device(x, device):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None:
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x


#
