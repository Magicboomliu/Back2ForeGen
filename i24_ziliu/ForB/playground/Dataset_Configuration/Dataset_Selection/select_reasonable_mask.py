import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("../..")
from PIL import Image
import numpy as np
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

if __name__=="__main__":


    datapath = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset"
    threshold = 0.4
    training_data_all_path = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all.txt"
    training_data_all_selected = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all_selected.txt"
    contents = read_text_lines(training_data_all_path)

    keeped_data = []

    for input_str in tqdm(contents):
        parts = input_str.split(' ', 2)
        image_path = parts[0]
        bg_mask_path = parts[1]

        current_bg_mask = os.path.join(datapath,bg_mask_path)
        bg_mask = np.array(Image.open(current_bg_mask)).astype(np.float32)/255
        fg_mask = np.ones_like(bg_mask) - bg_mask
        current_h = fg_mask.shape[0]
        current_w = fg_mask.shape[1]
        fg_ratio = np.sum(fg_mask) / (current_h * current_w)

        if fg_ratio<threshold:
            keeped_data.append(input_str)

    
    with open(training_data_all_selected,mode='w') as f:
        for idx, line in enumerate(keeped_data):
            if idx!=len(keeped_data)-1:
                f.writelines(str(line) + "\n")
            else:
                f.writelines(str(line))


    print(len(keeped_data))