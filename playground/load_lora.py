import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image



if __name__=="__main__":

    safetensors_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/sd15_anime.safetensors"
    saved_lora_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/LoRAs/lorita.safetensors"

    pipe = diffusers.StableDiffusionPipeline.from_single_file(
        safetensors_path
        ).to("cuda:0")
    print("loaded the pipeline successfully.")
    pipe.load_lora_weights(saved_lora_path)
    print("loaded the LoRA weight Successfully")


    prompt = "A high school girl with black hair sitting in the park.jirai fashion"

    lora_scale = 1.0
    image = pipe(
        prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
    ).images[0]
    
    image.save("example_new.png")


    

    
    

    pass