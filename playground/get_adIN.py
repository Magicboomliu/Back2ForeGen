import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from stable_diffusion_reference import StableDiffusionReferencePipeline
from diffusers import DDIMScheduler

import sys
sys.path.append("../..")
from PIL import Image

image_path = "/home/zliu/PFN/PFN24/i24_ziliu/ForB/playground/inpaint-cat.png"
input_image= Image.open(image_path).convert("RGB")


pipe = StableDiffusionReferencePipeline.from_single_file(
       "/home/zliu/PFN/pretrained_models/base_models/sd15_anime.safetensors",
       safety_checker=None,
       torch_dtype=torch.float16
       ).to('cuda:0')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

result_img = pipe(ref_image=input_image,
      prompt="A girl with light grey hair and steel blue eyes wearing a bow tie is jogging in a park.",
      num_inference_steps=50,
      reference_attn=True,
      reference_adain=False).images[0]


input_image.save("init.png")
result_img.save("result.png")