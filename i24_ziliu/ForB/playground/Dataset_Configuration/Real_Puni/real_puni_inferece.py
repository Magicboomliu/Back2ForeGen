import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import os
import diffusers
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm

# https://civitai.com/models/124339/real-puni-mix


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

if __name__=="__main__":

    prompt_fname = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Real_Puni/Prompts/Generated_From_Template/real_puni.txt"
    
    base_model_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/base_models/REAL_PUNI_MIX.safetensors"
    vae_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/VAEs/ClearVAE_V2.3_fp16.safetensors"


    current_stable_diffusion_model = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path=base_model_path)
    pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict=vae_path)
    current_stable_diffusion_model.vae = pretrained_vae # update the VAE
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("bdsqlsz/stable-diffusion-v1-5",subfolder='scheduler')

    pipe = current_stable_diffusion_model.to("cuda")
    
    print("Loaded the Pipeline and VAEs")
    prompts = read_text_lines(prompt_fname)


    idx = 0
    for text_prompt in tqdm(prompts):

        image = pipe(text_prompt,negative_prompt="EasyNegative",clip_skip=2,height=768,width=768,num_inference_steps=30).images[0]
        image.save("/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Real_Puni/images/{}.png".format(idx))
        idx = idx +1
