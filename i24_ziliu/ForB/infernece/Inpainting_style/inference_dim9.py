import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers
from diffusers import StableDiffusionPipeline,AutoencoderKL
from diffusers import DDPMScheduler,DDIMScheduler
from diffusers import DPMSolverMultistepScheduler
import sys
sys.path.append("../..")
from pipelines.SD_inpainting_pipeline_with_learnable_adain_attn import StableDiffusionInpaintPipeline

from PIL import Image
import numpy as np
import cv2
import random
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


def decode_image(latents,vae):
    image = vae.decode(latents /vae.config.scaling_factor, return_dict=False)[0]
    image = torch.clamp(image,min=-1,max=1)
    image_pil = Image.fromarray(((image + 1)/2 *255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))

    return image_pil


if __name__=="__main__":
    
    set_seed(1024)
    
    base_model_path = "/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
    current_stable_diffusion_model = StableDiffusionInpaintPipeline.from_single_file(pretrained_model_link_or_path=base_model_path,torch_dtype=torch.float16)
    

    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("bdsqlsz/stable-diffusion-v1-5",subfolder='scheduler')
    my_pipe = StableDiffusionPipeline.from_pretrained("ckpt/anything-v4.5-vae-swapped")
    current_stable_diffusion_model.vae = my_pipe.vae
    pipe = current_stable_diffusion_model.to("cuda")
    pipe.scheduler = noise_scheduler
    pipe.vae = pipe.vae.half()
    pipe.unet = pipe.unet.half()
    
    pipe = pipe.to("cuda:0")

    
    root_path = "/data1/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
    input_image_path = "Sketch_Style/images/54.png"
    bg_mask_path = "Sketch_Style/bg_mask_v1/54.png"
    
    
    # original image PIL and original Background Image PIL
    input_image= Image.open(os.path.join(root_path,input_image_path)).convert("RGB")
    origianl_size = input_image.size
    saved_input = input_image
    
    
    bg_mask = Image.open(os.path.join(root_path,bg_mask_path)).convert("RGB")
    original_bg_mask = bg_mask

    # Original image numpy and original Background imageNumpu
    input_image = np.array(input_image).astype(np.float32)
    bg_mask = np.array(bg_mask).astype(np.float32)/255

    input_image,_ = resize_image(input_image,size=(512,512),is_mask=False)
    bg_mask,_ = resize_image(bg_mask,size=(512,512),is_mask=False)

    resize_input_image_pil = Image.fromarray((input_image).astype(np.uint8))


    # saved initial foreground image
    fg_mask = np.ones_like(bg_mask) -bg_mask
    original_bg_mask_black = Image.fromarray((np.zeros_like(fg_mask) * 255).astype(np.uint8))
    original_bg_mask_white = Image.fromarray((np.ones_like(fg_mask) * 255).astype(np.uint8))
    
    fg_mask_pil = Image.fromarray((fg_mask*255).astype(np.uint8))
    fg_mask = fg_mask[:,:,0:1] #[H,W,1]
    fg_mask_np = fg_mask
    fg_image = input_image * fg_mask
    initial_fg_image = fg_image
    initial_fg_image = initial_fg_image.astype(np.uint8)
    initial_fg_image = Image.fromarray(initial_fg_image)
    fg_image = fg_image.astype(np.uint8)
    fg_image = Image.fromarray(fg_image)
    
    prompt = "sketch artstyle A girl with honey blonde hair and caramel brown eyes in a shacket is exploring the interior of a pyramid at the Pyramids of Giza, Egypt."    
    image,middle_state_list = pipe(prompt=prompt, image=fg_image, mask_image=original_bg_mask,
                 strength=1.0,reference_image =saved_input,reference_attn =True,reference_adain =False,use_converter=False,
                 fg_mask=fg_mask,
                 pretrained_model_path = "/home/zliu/PFN/PFN24/i24_ziliu/ForB/outputs/Mixed_Dataset/sd15_inpainting_with_attn_and_adaIN_new/ckpt_8001.pt"
                 )
    

    # for i, state in enumerate(middle_state_list):
    #     mid_state_image = decode_image(state,vae=pipe.vae)
    #     mid_state_image.save("{}.png".format(i))
        
    image[0].save("GT_Reference_ATTN_V3.png")
    initial_fg_image.save("fg.png")
    saved_input.save("save_input.png")