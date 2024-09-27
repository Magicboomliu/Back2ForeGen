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
# https://civitai.com/models/35960/flat-2d-animerge

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


if __name__=="__main__":

    prompt_fname = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/Flat2D/Prompts/Generated_From_Template/flat2d.txt"
    base_model_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/base_models/flat2d.safetensors"

    current_stable_diffusion_model = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path=base_model_path)
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("bdsqlsz/stable-diffusion-v1-5",subfolder='scheduler')
    pipe = current_stable_diffusion_model.to("cuda")

    pipe.scheduler = noise_scheduler
    print("Loaded the Pipeline and VAEs")

    prompts = read_text_lines(prompt_fname)

    idx = 0
    for text_prompt in tqdm(prompts):
        image = pipe(text_prompt,negative_prompt="EasyNegative",clip_skip=2,
        height=768,width=768,num_inference_steps=30).images[0]

        image.save("/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/images/{}.png".format(idx))

        idx = idx +1