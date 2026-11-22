import os
import json 
import argparse


def get_arguments():
    """
    Obtain the arguments given from console; 
    If phase is val: include exp_directory & checkpoint
    Otherwise:don't include them 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", "-p", choices=['train', 'val', 'test'], type=str, help="Either train(training) or val(generation)", required=True)
    parser.add_argument("--config", "-c", type=str, help="The config file for the experiment", required=True)
    parser.add_argument("--exp_directory", type=str, 
                        help="The path where checkpoint is saved, from which you want to generate/val data")
    parser.add_argument("--checkpoint", type=str, help="The checkpoint to be evaluated: e.g. checkpoint_94.pt")
    
    args = parser.parse_args()
    
    return args

