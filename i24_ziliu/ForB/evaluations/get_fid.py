import os
import torch
import numpy as np
from pytorch_fid import fid_score
import os
from tqdm import tqdm
import json

if __name__=="__main__":
    
    root_folder = "/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/ours_adaIN"
    
    fname_list = os.listdir(root_folder)
    
    saved_dict = {}
    for fname in tqdm(fname_list):
        # est_folder 和 GT_folder 的路径
        
        saved_dict[fname] = dict()
        
        # ours adaIN
        ours_adaIN_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/ours_adaIN/{}/'.format(fname)
        GT_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/original_images/{}/'.format(fname)
        fid_value_ours_adaIN = fid_score.calculate_fid_given_paths([ours_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_adaIN'] = fid_value_ours_adaIN
        
        # ours attn
        ours_attn_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/ours_attn/{}/'.format(fname)
        fid_value_ours_attn = fid_score.calculate_fid_given_paths([ours_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_attn'] = fid_value_ours_attn
        
        # our attn + adaIN
        ours_adaIN_ours_attn_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/ours_adaIN_ours_attn/{}/'.format(fname)
        fid_value_ours_adaIN_ours_attn = fid_score.calculate_fid_given_paths([ours_adaIN_ours_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['ours_attn_adaIN'] = fid_value_ours_adaIN_ours_attn
        
        
        
        # normal inpainting
        normal_inpainting_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/normal_inpainting/{}/'.format(fname)
        fid_value_normal_inpainting = fid_score.calculate_fid_given_paths([normal_inpainting_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['normal_inpainting'] = fid_value_normal_inpainting

        # off adaIN
        official_adaIN_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/official_adaIN/{}/'.format(fname)
        fid_value_official_adaIN = fid_score.calculate_fid_given_paths([official_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_adaIN'] = fid_value_official_adaIN
        
        # off attn
        official_attn_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/official_attn/{}/'.format(fname)
        fid_value_official_attn = fid_score.calculate_fid_given_paths([official_attn_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_attn'] = fid_value_official_attn
        
        # off AdaIN + Attn
        official_attn_adaIN_est_folder = '/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn/official_attn_adaIN/{}/'.format(fname)
        fid_value_official_attn_adaIN = fid_score.calculate_fid_given_paths([official_attn_adaIN_est_folder, GT_folder], batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
        saved_dict[fname]['offical_attn_adain'] = fid_value_official_attn_adaIN
        
    
    
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(saved_dict, f, ensure_ascii=False, indent=4)
    print("Done!!!!!!")