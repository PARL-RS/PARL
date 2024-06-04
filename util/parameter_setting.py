import argparse
from re import T
import sys
import os
from torch.nn.functional import multi_margin_loss
def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def RL_poison_main_exp():  # intialize paprameters
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--poi_data_path", type=str, default='')
    parser.add_argument("--cuda", type=int, default=0)

 
    
    args = parser.parse_args()
    print(args)

    return args



def train_classifier_main_exp():  # intialize paprameters
    parser = argparse.ArgumentParser("")
    
    
    
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--method_name", type=str, default='')
    parser.add_argument("--poison_data_sort_flag", type=str, default='')
    parser.add_argument("--poison_data_ratio", type=str, default='')
    parser.add_argument("--dataset_tag", type=str, default='')
    
    parser.add_argument("--cuda", type=int, default=0)

 
    args = parser.parse_args()
    print(args)

    return args