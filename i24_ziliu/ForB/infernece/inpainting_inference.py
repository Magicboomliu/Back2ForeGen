import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import sys
sys.path.append("../..")
from PIL import Image
# from ForB.pipelines.SD_AdaIN import StableDiffusionReferencePipeline
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


image_path = "/data1/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/Flat2D/images/11.png"
background_mask ="/data1/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/Flat2D/bg_mask_v1/11.png"

# original image PIL and original Background Image PIL
input_image= Image.open(image_path).convert("RGB")
origianl_size = input_image.size
saved_input = input_image
bg_mask = Image.open(background_mask).convert("RGB")


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


pipe = StableDiffusionReferenceInpaintingPipeline.from_single_file("/home/zliu/PFN/pretrained_models/base_models/sd15_anime.safetensors",
                                            torch_dtype=torch.float16).to('cuda:0')
# which Scheduler shall we use 
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# UPDATE
pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict="/home/zliu/PFN/pretrained_models/VAEs/vae-ft-mse-840000-ema-pruned_fp16.safetensors")
pipe.vae = pretrained_vae # update the VAE
pipe.vae = pipe.vae.half()
pipe = pipe.to("cuda:0")


# get the results
result_img = pipe(ref_image=fg_image,
      original_image = resize_input_image_pil,
      original_foreground_mask=fg_mask_np,
      prompt="A girl with butterscotch blonde hair and burnt orange eyes in a denim skirt is exploring the narrow streets in Santorini, Greece.",
      num_inference_steps=50,
      reference_attn=True,
      reference_adain=False,
      fg_mask=fg_mask,
      use_converter=True).images[0]



initial_fg_image = initial_fg_image.resize(origianl_size)
initial_fg_image.save("input.png")
saved_input.save("origin.png")
result_img = result_img.resize(origianl_size)
result_img.save("ours_attn.png")