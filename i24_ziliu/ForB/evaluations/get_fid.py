import os
import torch
import numpy as np
from pytorch_fid import fid_score
import os
from tqdm import tqdm
import json

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Infenrece")

    parser.add_argument(
        "--root_folder",
        type=str,
        default="{}_with_initial",
        required=True,
        help="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/")

    parser.add_argument(
        "--saved_name",
        type=str,
        default=None,
        help="../outputs/evaluation_results/inpainting_pfn_with_initial")

    
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    
    args = parse_args()
    
    root_folder = args.root_folder
    root_folder_example = "{}/ours_adaIN".format(root_folder)
    
    fname_list = os.listdir(root_folder_example)
    
    saved_dict = {}
    for fname in tqdm(fname_list):
        # est_folder 和 GT_folder 的路径
        
        saved_dict[fname] = dict()
        
        # ours adaIN
        ours_adaIN_est_folder = '{}/ours_adaIN/{}/'.format(root_folder,fname)
        GT_folder = '{}/original_images/{}/'.format(root_folder,fname)
        fid_value_ours_adaIN = fid_score.calculate_fid_given_paths([ours_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_adaIN'] = fid_value_ours_adaIN
        
        # ours attn
        ours_attn_est_folder = '{}/ours_attn/{}/'.format(root_folder,fname)
        fid_value_ours_attn = fid_score.calculate_fid_given_paths([ours_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_attn'] = fid_value_ours_attn
        
        # our attn + adaIN
        ours_adaIN_ours_attn_est_folder = '{}/ours_adaIN_ours_attn/{}/'.format(root_folder,fname)
        fid_value_ours_adaIN_ours_attn = fid_score.calculate_fid_given_paths([ours_adaIN_ours_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_attn_adaIN'] = fid_value_ours_adaIN_ours_attn
        
        
        
        # normal inpainting
        normal_inpainting_est_folder = '{}/normal_inpainting/{}/'.format(root_folder,fname)
        fid_value_normal_inpainting = fid_score.calculate_fid_given_paths([normal_inpainting_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['normal_inpainting'] = fid_value_normal_inpainting

        # off adaIN
        official_adaIN_est_folder = '{}/official_adaIN/{}/'.format(root_folder,fname)
        fid_value_official_adaIN = fid_score.calculate_fid_given_paths([official_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_adaIN'] = fid_value_official_adaIN
        
        # off attn
        official_attn_est_folder = '{}/official_attn/{}/'.format(root_folder,fname)
        fid_value_official_attn = fid_score.calculate_fid_given_paths([official_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_attn'] = fid_value_official_attn
        
        # off AdaIN + Attn
        official_attn_adaIN_est_folder = '{}/official_attn_adaIN/{}/'.format(root_folder,fname)
        fid_value_official_attn_adaIN = fid_score.calculate_fid_given_paths([official_attn_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_attn_adain'] = fid_value_official_attn_adaIN
        
    
    with open(args.saved_name, 'w', encoding='utf-8') as f:
        json.dump(saved_dict, f, ensure_ascii=False, indent=4)