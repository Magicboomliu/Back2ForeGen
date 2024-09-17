import torch
import torch.nn as nn
import torch.nn.functional as F


from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import os
import sys
sys.path.append("..")
from pipelines.PFN_testing_Reference_Only_Pipeline import PFN_Text2Image_Reference_Only_Pipeline


from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import skimage.io 



def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def resize_image(image,size,is_mask):
    original_shape = image.shape[:2]
    if not is_mask:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        resized_image = resized_image.astype(np.bool_)
    return resized_image, original_shape



if __name__=="__main__":
    
    inference_type_list = ["off_adaIN",'off_attn',"off_attn_adaIN","off_inpainting","ours_attn","ours_adaIN","ous_attn_adaIN"]
    
    inference_type = 'normal_inpainting'
    
    if inference_type=='off_inpainting':
        use_converter = False
        use_adaIN = False
        use_attn = False
        
    elif inference_type=='off_adaIN':
        use_converter = False
        use_adaIN = True
        use_attn = False
        
    elif inference_type=="off_attn":
        use_converter = False
        use_adaIN = False
        use_attn = True
        
    elif inference_type=="off_attn_adaIN":
        use_converter = False
        use_adaIN = True
        use_attn = True
        
    elif inference_type=='ours_attn':
        use_converter = True
        use_adaIN = False
        use_attn = True
        
    elif inference_type=="ours_adaIN":
        use_converter = True
        use_adaIN = True
        use_attn = False
        
    elif inference_type=='ours_attn_adaIN':
        use_converter = True
        use_adaIN = True
        use_attn = True
    
    root_path = "/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
    saved_path = '../outputs/evaluation_results/text2image_inpaint_pfn'
    os.makedirs(saved_path,exist_ok=True)
    
    set_seed(1024)

    base_model_path = "/home/zliu/PFN/pretrained_models/base_models/sd15_anime.safetensors"
    current_stable_diffusion_model = PFN_Text2Image_Reference_Only_Pipeline.from_single_file(pretrained_model_link_or_path=base_model_path,torch_dtype=torch.float16)
    
    noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2",subfolder='scheduler')
    noise_scheduler.config.prediction_type = "epsilon"
    
    


    pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict="/home/zliu/PFN/pretrained_models/VAEs/vae-ft-mse-840000-ema-pruned_fp16.safetensors")
    current_stable_diffusion_model.vae = pretrained_vae # update the VAE
    
    pipe = current_stable_diffusion_model.to("cuda")

    pipe.scheduler = noise_scheduler
    pipe.vae = pipe.vae.half()
    pipe.unet = pipe.unet.half()
    pipe = pipe.to("cuda:0")
    
    print("loaded the base models")
    
    
    
    #------------------------------------------------------------------#
    # fg_images
    saved_images_folder = os.path.join(saved_path,"fg_images")
    os.makedirs(saved_images_folder,exist_ok=True)
    
    # original images
    saved_original_images_folder = os.path.join(saved_path,"original_images")
    os.makedirs(saved_original_images_folder,exist_ok=True)
    
    # bg images
    saved_bg_images_folder = os.path.join(saved_path,"bg_images")
    os.makedirs(saved_bg_images_folder,exist_ok=True)

    # normal inpaintings
    saved_official_inpainting = os.path.join(saved_path,"normal_inpainting")
    os.makedirs(saved_official_inpainting,exist_ok=True)
    
    #------------------------------------------------------------------#
    
    # official attention
    saved_official_attn_folder = os.path.join(saved_path,"official_attn")
    os.makedirs(saved_official_attn_folder,exist_ok=True)
    
    # officail adaIN
    saved_official_adaIN_folder = os.path.join(saved_path,"official_adaIN")
    os.makedirs(saved_official_adaIN_folder ,exist_ok=True)
    
    # official adaIN +attn
    saved_official_attn_adaIN_folder = os.path.join(saved_path,"official_attn_adaIN")
    os.makedirs(saved_official_attn_adaIN_folder,exist_ok=True)
    
    #------------------------------------------------------------------#
    
    # Ours AdaIN
    saved_ours_adaIN_folder = os.path.join(saved_path,"ours_adaIN")
    os.makedirs(saved_ours_adaIN_folder,exist_ok=True)
    
    # Ours AdaIN + Ours Attn
    saved_ours_adaIN_ours_attn_folder = os.path.join(saved_path,"ours_adaIN_ours_attn")
    os.makedirs(saved_ours_adaIN_ours_attn_folder,exist_ok=True)

    # Ours Attn
    saved_ours_attns_folder = os.path.join(saved_path,"ours_attn")
    os.makedirs(saved_ours_attns_folder,exist_ok=True)
    
    #------------------------------------------------------------------#

    # GT Reference Attn
    saved_gt_reference_attn = os.path.join(saved_path,"gt_reference_attn")
    os.makedirs(saved_gt_reference_attn,exist_ok=True)

    # GT refernece AdaIN
    saved_gt_reference_adaIN = os.path.join(saved_path,"gt_reference_adaIN")
    os.makedirs(saved_gt_reference_adaIN,exist_ok=True)
    
    # GT reference Attn + AdaIN
    saved_gt_reference_attn_adain = os.path.join(saved_path,"gt_reference_adaIN_attn")
    os.makedirs(saved_gt_reference_attn_adain,exist_ok=True)

    validation_list_path = "/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/filenames/validation_select_data.txt"
    validation_contents = read_text_lines(validation_list_path)

    for fname in tqdm(validation_contents):
        parts = fname.split(' ', 2)
        image_path = parts[0]
        bg_mask_path = parts[1]
        descriptions = parts[2]
        
        saved_cls_name = image_path.split("/")[0]
        
        # foreground images
        saved_images_folder_specific_folder = os.path.join(saved_images_folder,saved_cls_name)
        os.makedirs(saved_images_folder_specific_folder,exist_ok=True) 
        # original images
        saved_original_images_folder_specific_folder = os.path.join(saved_original_images_folder,saved_cls_name)
        os.makedirs(saved_original_images_folder_specific_folder,exist_ok=True) 
        # background Images
        saved_bg_images_folder_specific_folder = os.path.join(saved_bg_images_folder,saved_cls_name)
        os.makedirs(saved_bg_images_folder_specific_folder,exist_ok=True) 
        
        # normal inpaintings 
        saved_official_inpainting_specific_folder = os.path.join(saved_official_inpainting,saved_cls_name)
        os.makedirs(saved_official_inpainting_specific_folder,exist_ok=True)
        
        # -------------------------------------------------------------------------------------
        # official attn 
        saved_official_attn_folder_specific_folder = os.path.join(saved_official_attn_folder,saved_cls_name)
        os.makedirs(saved_official_attn_folder_specific_folder,exist_ok=True)
        # offical adaIN 
        saved_official_adaIN_folder_specific_folder = os.path.join(saved_official_adaIN_folder,saved_cls_name)
        os.makedirs(saved_official_adaIN_folder_specific_folder,exist_ok=True)
        # offical adaIn anda attn 
        saved_official_attn_adaIN_folder_specific_folder = os.path.join(saved_official_attn_adaIN_folder,saved_cls_name)
        os.makedirs(saved_official_attn_adaIN_folder_specific_folder,exist_ok=True)
        
        #---------------------------------------------------------------------------------------------------#
        # ours adaIN : saved_ours_adaIN_folder
        saved_ours_adaIN_folder_specific_folder = os.path.join(saved_ours_adaIN_folder,saved_cls_name)
        os.makedirs(saved_ours_adaIN_folder_specific_folder,exist_ok=True)
        # ours AdaIn and Ours Attn: saved_ours_adaIN_ours_attn_folder
        saved_ours_adaIN_ours_attn_folder_specific_folder = os.path.join(saved_ours_adaIN_ours_attn_folder,saved_cls_name)
        os.makedirs(saved_ours_adaIN_ours_attn_folder_specific_folder,exist_ok=True)
        # Ours Attns Only
        saved_ours_attns_folder_specific_folder = os.path.join(saved_ours_attns_folder,saved_cls_name)
        os.makedirs(saved_ours_attns_folder_specific_folder,exist_ok=True)
        
        #-----------------------------------------------------------------------------------#

        # gt reference attn
        saved_gt_inference_attn_folder_specific_folder = os.path.join(saved_gt_reference_attn,saved_cls_name)
        os.makedirs(saved_gt_inference_attn_folder_specific_folder,exist_ok=True)

        # gt refernece adaIN
        saved_gt_inference_adaIN_folder_specific_folder = os.path.join(saved_gt_reference_adaIN,saved_cls_name)
        os.makedirs(saved_gt_inference_adaIN_folder_specific_folder,exist_ok=True)
        
        # gt reference adaIN and attention
        saved_gt_inference_attn_adaIN_folder_specific_folder = os.path.join(saved_gt_reference_attn_adain,saved_cls_name)
        os.makedirs(saved_gt_inference_attn_adaIN_folder_specific_folder,exist_ok=True)
        



       # original image 
        input_image= Image.open(os.path.join(root_path,image_path)).convert("RGB")
        origianl_size = input_image.size
        saved_input = input_image
        
        # bg image
        bg_mask = Image.open(os.path.join(root_path,bg_mask_path)).convert("RGB")
        original_bg_mask = bg_mask


        input_image = np.array(input_image).astype(np.float32)
        bg_mask = np.array(bg_mask).astype(np.float32)/255
        input_image,_ = resize_image(input_image,size=(512,512),is_mask=False)
        bg_mask,_ = resize_image(bg_mask,size=(512,512),is_mask=False)

        resize_input_image_pil = Image.fromarray((input_image).astype(np.uint8))


        # saved initial foreground image
        fg_mask = np.ones_like(bg_mask) -bg_mask
        fg_mask = fg_mask[:,:,0:1] #[H,W,1]
        fg_mask_np = fg_mask
        fg_image = input_image * fg_mask
        initial_fg_image = fg_image
        initial_fg_image = initial_fg_image.astype(np.uint8)
        initial_fg_image = Image.fromarray(initial_fg_image)

        # fg_image
        fg_image = fg_image.astype(np.uint8)
        fg_image = Image.fromarray(fg_image)
        

        quit()
        with torch.no_grad():
            
            without_noise_model = "/home/zliu/PFN/pretrained_models/Converter/Mix_training_Text2Image/ckpt_8001.pt"
            result = pipe(full_image=input_image,prompt=[descriptions],fg_image= fg_image,fg_mask=fg_mask_np,
                pretrained_converter_path = without_noise_model,
                use_converter=use_converter,
                use_adaIN=use_adaIN,
                use_attn=use_attn,
                gudiance_score=10,
                attn_weight=0.1)
            
            quit()

            if inference_type=='off_inpainting':
                saved_name = os.path.join(saved_official_inpainting_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=='off_adaIN':
                saved_name = os.path.join(saved_official_adaIN_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=="off_attn":
                saved_name = os.path.join(saved_official_attn_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=="off_attn_adaIN":
                saved_name = os.path.join(saved_official_attn_adaIN_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=='ours_attn':
                saved_name = os.path.join(saved_ours_attns_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=="ours_adaIN":
                saved_name = os.path.join(saved_ours_adaIN_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
                
            elif inference_type=='ours_attn_adaIN':
                saved_name = os.path.join(saved_ours_adaIN_ours_attn_folder_specific_folder,os.path.basename(image_path))
                skimage.io.imsave(saved_name,result)
            
            