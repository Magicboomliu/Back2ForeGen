import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import sys
sys.path.append("../../..")
from PIL import Image


# from ForB.pipelines.SD_AdaIN_Inpainting import StableDiffusionReferenceInpaintingPipeline
from ForB.pipelines.SD_AdaIN_Attn_Inpainting import StableDiffusionReferenceInpaintingPipeline
from ForB.dataloader.utils.file_io import read_img,read_mask,resize_image,get_resize_foreground_and_mask
from ForB.dataloader.utils.utils import read_text_lines,get_id_and_prompt
import numpy as np
import cv2
from PIL import Image
import random
from diffusers import StableDiffusionInpaintPipeline
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler
)

from tqdm import tqdm
import os

def set_seed(seed):
    random.seed(seed)  # Set seed for the Python random module
    np.random.seed(seed)  # Set seed for numpy
    torch.manual_seed(seed)  # Set seed for torch
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # Ensure deterministic behavior for GPU operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resize_image(image,size,is_mask):
    original_shape = image.shape[:2]
    if not is_mask:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        resized_image = resized_image.astype(np.bool_)
    return resized_image, original_shape


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



if __name__=="__main__":
    
    root_path = "/data1/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
    
    saved_path = "/home/zliu/PFN/PFN24/i24_ziliu/ForB/outputs/Validation_Results"
    
    pretrained_path ="/home/zliu/PFN/PFN24/i24_ziliu/ForB/outputs/Mixed_Dataset/sd15_learnable_VAE_With_Attn/ckpt_8001.pt"
    
    saved_images_folder = os.path.join(saved_path,"images")
    os.makedirs(saved_images_folder,exist_ok=True)
    
    saved_original_images_folder = os.path.join(saved_path,"original_images")
    os.makedirs(saved_original_images_folder,exist_ok=True)
    
    saved_bg_images_folder = os.path.join(saved_path,"bg_images")
    os.makedirs(saved_bg_images_folder,exist_ok=True)
    
    saved_official_attn_folder = os.path.join(saved_path,"official_attn")
    os.makedirs(saved_official_attn_folder,exist_ok=True)
    
    saved_official_adaIN_folder = os.path.join(saved_path,"official_adaIN")
    os.makedirs(saved_official_adaIN_folder ,exist_ok=True)
    
    saved_official_attn_adaIN_folder = os.path.join(saved_path,"official_attn_adaIN")
    os.makedirs(saved_official_attn_adaIN_folder,exist_ok=True)
    
    saved_ours_adaIN_folder = os.path.join(saved_path,"ours_adaIN")
    os.makedirs(saved_ours_adaIN_folder,exist_ok=True)
    
    saved_ours_adaIN_official_attn_folder = os.path.join(saved_path,"ours_adaIN_official_attn")
    os.makedirs(saved_ours_adaIN_official_attn_folder,exist_ok=True)
    
    saved_ours_adaIN_ours_attn_folder = os.path.join(saved_path,"ours_adaIN_ours_attn")
    os.makedirs(saved_ours_adaIN_ours_attn_folder,exist_ok=True)

    saved_ours_attns_folder = os.path.join(saved_path,"ours_ours_attn")
    os.makedirs(saved_ours_attns_folder,exist_ok=True)
    
    
    
    validation_list_path = "/home/zliu/PFN/PFN24/i24_ziliu/ForB/filenames/validation_select_data.txt"
    validation_contents = read_text_lines(validation_list_path)
    
    
    pipe = StableDiffusionReferenceInpaintingPipeline.from_single_file("/home/zliu/PFN/pretrained_models/base_models/sd15_anime.safetensors",
                                                torch_dtype=torch.float16).to('cuda:0')
    
    # which Scheduler shall we use 
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # UPDATE
    pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict="/home/zliu/PFN/pretrained_models/VAEs/vae-ft-mse-840000-ema-pruned_fp16.safetensors")
    pipe.vae = pretrained_vae # update the VAE
    pipe.vae = pipe.vae.half()
    pipe = pipe.to("cuda:0")
    
    
    
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
                
        # official attn 
        saved_official_attn_folder_specific_folder = os.path.join(saved_official_attn_folder,saved_cls_name)
        os.makedirs(saved_official_attn_folder_specific_folder,exist_ok=True)
        
        # offical adaIN 
        saved_official_adaIN_folder_specific_folder = os.path.join(saved_official_adaIN_folder,saved_cls_name)
        os.makedirs(saved_official_adaIN_folder_specific_folder,exist_ok=True)
        
        
        # offical adaIn anda attn 
        saved_official_attn_adaIN_folder_specific_folder = os.path.join(saved_official_attn_adaIN_folder,saved_cls_name)
        os.makedirs(saved_official_attn_adaIN_folder_specific_folder,exist_ok=True)
        
        
        # ours adaIN : saved_ours_adaIN_folder
        saved_ours_adaIN_folder_specific_folder = os.path.join(saved_ours_adaIN_folder,saved_cls_name)
        os.makedirs(saved_ours_adaIN_folder_specific_folder,exist_ok=True)
        
        
        # ours AdaIn and official attn
        saved_ours_adaIN_official_attn_folder_specific_folder = os.path.join(saved_ours_adaIN_official_attn_folder,saved_cls_name)
        os.makedirs(saved_ours_adaIN_official_attn_folder_specific_folder,exist_ok=True)
        
        
        # ours AdaIn and Ours Attn: saved_ours_adaIN_ours_attn_folder
        saved_ours_adaIN_ours_attn_folder_specific_folder = os.path.join(saved_ours_adaIN_ours_attn_folder,saved_cls_name)
        os.makedirs(saved_ours_adaIN_ours_attn_folder_specific_folder,exist_ok=True)
        
        
        # Ours Attns Only
        saved_ours_attns_folder_specific_folder = os.path.join(saved_ours_attns_folder,saved_cls_name)
        os.makedirs(saved_ours_attns_folder_specific_folder,exist_ok=True)
        
        
        
        #...........................................................................................#

        # original image PIL and original Background Image PIL
        input_image= Image.open(os.path.join(root_path,image_path)).convert("RGB")
        origianl_size = input_image.size
        saved_input = input_image
        
        
        bg_mask = Image.open(os.path.join(root_path,bg_mask_path)).convert("RGB")

        # Original image numpy and original Background imageNumpu
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
        # resize operation
        # fg_image, fg_mask = get_resize_foreground_and_mask(image=fg_image,
        #                                                         mask=fg_mask)

        fg_image = fg_image.astype(np.uint8)
        fg_image = Image.fromarray(fg_image)


        # get the results
        result_img = pipe(ref_image=fg_image,
            original_image = resize_input_image_pil,
            original_foreground_mask=fg_mask_np,
            prompt=descriptions,
            num_inference_steps=50,
            reference_attn=True,
            reference_adain=True,
            fg_mask=fg_mask,
            use_converter=True,
            pretrained_model_path=pretrained_path,
            ).images[0]
        
        # initial foreground image
        # initial_fg_image = initial_fg_image.resize(origianl_size)
        # initial_fg_image.save(os.path.join(saved_images_folder_specific_folder,os.path.basename(image_path)))
        
        # # save initial image
        # saved_input.save(os.path.join(saved_original_images_folder_specific_folder,os.path.basename(image_path)))
        
        # result images: Ours Adin
        result_img = result_img.resize(origianl_size)
        result_img.save(os.path.join(saved_ours_adaIN_official_attn_folder_specific_folder,os.path.basename(image_path)))
        
        

        
        
        
        