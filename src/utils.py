import os
import json
import numpy as np
import torch

def get_train_cases(opts):
    if opts.data_type == 'nnUNet':
        train_cases = []
        with open(opts.splits_dir1, 'r') as file:
            all_plits = json.load(file)
            split_data = all_plits[opts.cases_folds]
            train_cases.append(split_data['train'] + split_data['val'])

        with open(opts.splits_dir2, 'r') as file:
            all_plits = json.load(file)
            split_data = all_plits[opts.cases_folds]
            train_cases.append(split_data['train'] + split_data['val'])

        print("Length of train cases data dir 1: ", len(train_cases[0]))
        print("Length of train cases data dir 2: ", len(train_cases[1]))

    else: 
        if opts.cases_folds == 0:
            train_cases_only = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19]
        elif opts.cases_folds == 1:
            train_cases_only = [0,1,2,4,6,7,9,10,12,13,14,15,16,17,18,19]
        elif opts.cases_folds == 2:
            train_cases_only = [0,1,3,4,5,6,7,8,9,10,11,12,14,15,17,19]
        elif opts.cases_folds == 3:
            train_cases_only = [0,1,2,3,5,6,7,8,10,11,13,14,15,16,17,18]
        elif opts.cases_folds == 4:
            train_cases_only = [0,1,2,3,4,5,8,9,11,12,13,15,16,17,18,19]
        train_cases = [train_cases_only, train_cases_only]

    return train_cases

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


