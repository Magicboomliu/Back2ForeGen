import diffusers
import os
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline

# Explore ControlNet?
from safetensors import safe_open

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from tqdm import tqdm

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

if __name__=="__main__":

    generated_prompts = "../caption_generation/Results/Generated_From_Template/reasonable_unique_prompts.txt"
    readed_contents = read_text_lines(generated_prompts)
    idx = 0

    saved_folder = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images/gen_images"
    os.makedirs(saved_folder,exist_ok=True)

    model_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/animexl_v60LCM.safetensors"
    pipeline = StableDiffusionXLPipeline.from_single_file(pretrained_model_link_or_path=model_path,torch_dtype=torch.float16)
    # 加载 VAE 模型
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to("cuda")

    # 替换 pipeline 中的 VAE
    pipeline.vae = vae
    # 将 pipeline 移动到 GPU 上（如果使用 GPU）
    pipeline = pipeline.to("cuda")
    image_size =1024
    pipeline = pipeline.to("cuda")  # 如果有GPU，切换到GPU加速
    pipeline.enable_model_cpu_offload()


    for text_prompt in tqdm(readed_contents):
        negative_prompt = "bad quality, low resolution, disharmonious, deformed, Out of frame, Poorly drawn, bad anatomy, deformed, ugly, disfigured"
        image = pipeline(prompt=text_prompt,
                    negative_prompt=negative_prompt).images[0]
        
        saved_generated_name = os.path.join(saved_folder,"{}.png".format(idx))
        
        image.save(saved_generated_name)
        idx = idx +1

    print("All Files Generatation Is DONE........")

    



