import os

import yaml
import src.lib.utils as utils
from src.lib.config import load_training_config
from src.lib.logger import get_datetime


def load_config(args):
    """Load the same resolved configuration used by the training entry point."""
    return load_training_config(args.config, getattr(args, "model", None))



def save_config(exp_path, config):
    # save the config parameters as yaml file 
    with open(os.path.join(exp_path, "config.yaml"), "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    
    return 



def create_exp_directories(exp_params):
    # create the experiment directory
    exp_path = os.path.join("experiments", f"{exp_params['model']['name']}_{exp_params['exp_name']}_{get_datetime()}")
    utils.create_directory(dir_path=exp_path)

    # Only output directories belong under the new run. Input checkpoint paths
    # and other supplied paths must keep their original values.
    output_directories = {"logs", "loss", "results", "checkpoint", "tb_logger", "output_dir"}
    for sub_dir in exp_params["paths"]:
        if sub_dir in output_directories:
            exp_params["paths"][sub_dir] = os.path.join(exp_path, sub_dir)
            utils.create_directory(exp_params["paths"][sub_dir])
    
    # add the experiment path to the config file
    exp_params["paths"]["experiment_path"] = exp_path
    return 
