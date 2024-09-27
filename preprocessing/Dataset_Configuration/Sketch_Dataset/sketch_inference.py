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
from PIL import Image
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

# https://civitai.com/models/80357/queratogray-sketch-eddiemauro-mix
if __name__=="__main__":

    prompt_fname = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Sketch_Dataset/Prompts/Generated_From_Template/reasonable_unique_prompts.txt"
    
    base_model_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/base_models/queratogray_sketch.safetensors"
    vae_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/VAEs/kl_f8_anime2_VAE.safetensors"

    current_stable_diffusion_model = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path=base_model_path)
    pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict=vae_path)
    current_stable_diffusion_model.vae = pretrained_vae # update the VAE

    noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("bdsqlsz/stable-diffusion-v1-5",subfolder='scheduler')


    pipe = current_stable_diffusion_model.to("cuda")
    print("Loaded the Pretrained Models and VAEs")


    prompts = read_text_lines(prompt_fname)

    idx = 0
    for text_prompt in tqdm(prompts):

        image = pipe(text_prompt,negative_prompt="EasyNegative",clip_skip=2,
        height=768,width=768,num_inference_steps=30).images[0]

        image.save("/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Sketch_Style/images/{}.png".format(idx))

        idx = idx +1



    
    