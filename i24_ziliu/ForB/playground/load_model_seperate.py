import diffusers
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append("..")
from diffusers import StableDiffusionPipeline


if __name__=="__main__":

    use_default_vae = False
    single_sd_pretrained_file_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/sd15_anime.safetensors"
    single_vae_file_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/pretrained_VAE/vae-ft-mse-840000-ema-pruned_fp16.safetensors"

    # stable diffusion pipeline
    loaded_sd_pipeline = StableDiffusionPipeline.from_single_file(
                                            pretrained_model_link_or_path=single_sd_pretrained_file_path)

    # Loaded the UNet
    if not use_default_vae:
        vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict=single_vae_file_path)
        loaded_sd_pipeline.vae = vae
    loaded_sd_pipeline = loaded_sd_pipeline.to("cuda:0")

    print("Loaded the SD Pipeline")

    with torch.no_grad():
        prompt = "A girl with auburn hair and light brown eyes in a bomber jacket is visiting the top level at the Eiffel Tower."
        generator = torch.Generator("cuda").manual_seed(31)
        image = loaded_sd_pipeline(prompt, generator=generator).images[0]
    
    print(image.size)
        
    image.save("original_new_vae.png")