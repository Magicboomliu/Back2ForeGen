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


def Create_New_Sub_List(flat2d_images_path,flat2d_bg_mask_v1_path,flat2d_descriptions_path,flat2d_selected_idx_path,flat2d_selected_path,flat2d_selected_bg_mask_v1_path,flat2d_selected_descriptions_path):
    
    os.makedirs(flat2d_selected_path,exist_ok=True)
    os.makedirs(flat2d_selected_bg_mask_v1_path,exist_ok=True)

    all_descriptions = read_text_lines(flat2d_descriptions_path)
    assert len(all_descriptions) == 1000
    selected_idx = read_text_lines(flat2d_selected_idx_path)
    saved_descriptions = []
    for idx in selected_idx:
        idx = int(idx)
        descriptions = all_descriptions[idx]
        saved_descriptions.append(descriptions)
        
        selected_images = os.path.join(flat2d_images_path,"{}.png".format(idx))
        selected_backrgound = os.path.join(flat2d_bg_mask_v1_path,"{}.png".format(idx))

        saved_selected_images = os.path.join(flat2d_selected_path,"{}.png".format(idx))
        saved_selected_backrgound = os.path.join(flat2d_selected_bg_mask_v1_path,"{}.png".format(idx))

        assert os.path.exists(selected_images)
        assert os.path.exists(selected_backrgound)

        os.system("cp {} {}".format(selected_images,saved_selected_images))
        os.system("cp {} {}".format(selected_backrgound,saved_selected_backrgound))
    

    with open(flat2d_selected_descriptions_path,'w') as f:
        for idx, line in enumerate(saved_descriptions):
            if idx!=len(saved_descriptions)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    
    readed_selected_descriptions = read_text_lines(flat2d_selected_descriptions_path)
    assert len(readed_selected_descriptions) == len(os.listdir(flat2d_selected_path))


    assert len(os.listdir(flat2d_selected_path)) == len(os.listdir(flat2d_selected_bg_mask_v1_path))
    print("Transfered_OK")






if __name__=="__main__":

    # Flat2D
    flat2d_images_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Gap_Mix/images"
    flat2d_bg_mask_v1_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/bg_mask_v1"
    flat2d_descriptions_path = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Flat2D/Prompts/Generated_From_Template/flat2d.txt"
    flat2d_selected_idx_path = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Flat2D/Prompts/Generated_From_Template/flat2d_selected_list.txt"

    # Saved Path
    flat2d_selected_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/selected_images"
    flat2d_selected_bg_mask_v1_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/selected_bg_mask_v1"
    flat2d_selected_descriptions_path = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Flat2D/Prompts/Generated_From_Template/flat2d_selected_descriptions.txt"


    Create_New_Sub_List(flat2d_images_path=flat2d_images_path,
    flat2d_bg_mask_v1_path=flat2d_bg_mask_v1_path,
    flat2d_descriptions_path=flat2d_descriptions_path,
    flat2d_selected_idx_path=flat2d_selected_idx_path,
    flat2d_selected_path=flat2d_selected_path,
    flat2d_selected_bg_mask_v1_path=flat2d_selected_bg_mask_v1_path,
    flat2d_selected_descriptions_path=flat2d_selected_descriptions_path
    )



    # os.makedirs(flat2d_selected_path,exist_ok=True)
    # os.makedirs(flat2d_selected_bg_mask_v1_path,exist_ok=True)

    # all_descriptions = read_text_lines(flat2d_descriptions_path)
    # assert len(all_descriptions) == 1000
    # selected_idx = read_text_lines(flat2d_selected_idx_path)
    # saved_descriptions = []
    # for idx in selected_idx:
    #     idx = int(idx)
    #     descriptions = all_descriptions[idx]
    #     saved_descriptions.append(descriptions)
        
    #     selected_images = os.path.join(flat2d_images_path,"{}.png".format(idx))
    #     selected_backrgound = os.path.join(flat2d_bg_mask_v1_path,"{}.png".format(idx))

    #     saved_selected_images = os.path.join(flat2d_selected_path,"{}.png".format(idx))
    #     saved_selected_backrgound = os.path.join(flat2d_selected_bg_mask_v1_path,"{}.png".format(idx))

    #     assert os.path.exists(selected_images)
    #     assert os.path.exists(selected_backrgound)

    #     os.system("cp {} {}".format(selected_images,saved_selected_images))
    #     os.system("cp {} {}".format(selected_backrgound,saved_selected_backrgound))
    

    # with open(flat2d_selected_descriptions_path,'w') as f:
    #     for idx, line in enumerate(saved_descriptions):
    #         if idx!=len(saved_descriptions)-1:
    #             f.writelines(line+"\n")
    #         else:
    #             f.writelines(line)
    
    # readed_selected_descriptions = read_text_lines(flat2d_selected_descriptions_path)
    # assert len(readed_selected_descriptions) == len(os.listdir(flat2d_selected_path))


    # assert len(os.listdir(flat2d_selected_path)) == len(os.listdir(flat2d_selected_bg_mask_v1_path))
    # print("Transfered_OK")
    

    





    # Gap_Mix

    # NeverEndDream

    # PhotoMaxUltraV1

    # Real_Doll

    # Real_Puni

    # Screenshot

    # Sketch_Dataset

    # T_animev4

    pass