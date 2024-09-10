import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import sys
sys.path.append("../..")
from PIL import Image
from ForB.pipelines.SD_AdaIN import StableDiffusionReferencePipeline
from ForB.dataloader.utils.file_io import read_img,read_mask,resize_image,get_resize_foreground_and_mask
from ForB.dataloader.utils.utils import read_text_lines,get_id_and_prompt
import numpy as np
import cv2
from PIL import Image
import random

from diffusers import StableDiffusionInpaintPipeline

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


# set_seed(0)
test_image_ids = [0,1,10,102,104,106,107,115,123,127,125,147,148,15,157,169,171,173,177,196,203,233,250,252,276,280,
291,369,374,383]

print(len(test_image_ids))
quit()

image_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images/images/10.png"
background_mask = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images/background_masks/10.png"

# original image PIL and original Background Image PIL
input_image= Image.open(image_path).convert("RGB")
bg_mask = Image.open(background_mask).convert("RGB")

# Original image numpy and original Background imageNumpu
input_image = np.array(input_image).astype(np.float32)
bg_mask = np.array(bg_mask).astype(np.float32)/255

input_image,_ = resize_image(input_image,size=(512,512),is_mask=False)
bg_mask,_ = resize_image(bg_mask,size=(512,512),is_mask=False)

# saved initial foreground image
fg_mask = np.ones_like(bg_mask) -bg_mask
fg_mask = fg_mask[:,:,0:1]
fg_image = input_image * fg_mask
initial_fg_image = fg_image
initial_fg_image = initial_fg_image.astype(np.uint8)
initial_fg_image = Image.fromarray(initial_fg_image)


fg_image, fg_mask = get_resize_foreground_and_mask(image=fg_image,
                                                        mask=fg_mask)
fg_image = fg_image.astype(np.uint8)
fg_image = Image.fromarray(fg_image)



pipe = StableDiffusionReferencePipeline.from_single_file("/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/sd15_anime.safetensors",
                                            torch_dtype=torch.float16).to('cuda:0')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# get the results
result_img = pipe(ref_image=fg_image,
      prompt="A girl with auburn hair and light brown eyes in a bomber jacket is visiting the top level at the Eiffel Tower.",
      num_inference_steps=50,
      reference_attn=False,
      reference_adain=True,
      fg_mask=fg_mask,
      use_converter=True).images[0]


initial_fg_image.save("input.png")
result_img.save("result.png")