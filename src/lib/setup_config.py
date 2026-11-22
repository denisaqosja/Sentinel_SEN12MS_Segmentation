import os, yaml
import src.lib.utils as utils
from src.lib.logger import get_datetime


def load_config(args):
    # read from the yaml file
    config_path = os.path.join(os.getcwd(), "src", "configs", args.config)
    print(config_path)
    with open(config_path, "r") as f:
        exp_params = yaml.safe_load(f)
    
    return exp_params



def save_config(exp_path, config):
    # save the config parameters as yaml file 
    with open(os.path.join(exp_path, "config.yaml"), "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    
    return 



def create_exp_directories(exp_params):
    # create the experiment directory
    exp_path = os.path.join("experiments", f"{exp_params['model']['name']}_{exp_params['exp_name']}_{get_datetime()}")
    utils.create_directory(dir_path=exp_path)

    # create the children-directories: results, tb_logger, ...
    for sub_dir in exp_params["paths"]:
        exp_params["paths"][sub_dir] = os.path.join(exp_path, sub_dir)
        utils.create_directory(exp_params["paths"][sub_dir])
    
    # add the experiment path to the config file
    exp_params["paths"]["experiment_path"] = exp_path
    return 