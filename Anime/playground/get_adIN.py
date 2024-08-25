import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from stable_diffusion_reference import StableDiffusionReferencePipeline
from diffusers import DDIMScheduler
# from stable_diffusion_reference_with_noise_inversion import StableDiffusionReferencePipeline



input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

pipe = StableDiffusionReferencePipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       safety_checker=None,
       torch_dtype=torch.float16
       ).to('cuda:0')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

result_img = pipe(ref_image=input_image,
      prompt="1girl",
      num_inference_steps=50,
      reference_attn=True,
      reference_adain=True).images[0]


input_image.save("initial_girl.png")
result_img.save("result_girl.png")