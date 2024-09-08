import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForInpainting
import os
from tqdm import tqdm
import sys
sys.path.append("..")
from pipelines.custom_sd_inpaint_pipeline import Custom_SD_Inpainting
from pipelines.custom_sdxl_inpaint_pipeline import Custom_SDXL_Inpainting


def saved_generated_images(saved_folder,
                            image_list,
                            init_image,
                            mask_image):
    os.makedirs(saved_folder,exist_ok=True)

    for idx in tqdm(range(len(image_list))):
        saved_name = os.path.join(saved_folder,"{}.png".format(idx))

        image = image_list[idx]
        initial_saved_image = image
        initial_saved_image.save(saved_name)

import argparse

def adjust_resolution(H, W, IR):
    # 确定哪个边是最大边
    max_dim = max(H, W)
    min_dim = min(H, W)

    # 如果最大边大于理想分辨率
    if max_dim > IR:
        # 调整最大边为接近的8的倍数
        if H > W:
            H = (H // 8) * 8
        else:
            W = (W // 8) * 8
    else:
        # 如果最大边小于理想分辨率，调整为IR
        if H > W:
            scale_factor = IR / H
            H = IR
            W = int(W * scale_factor)
        else:
            scale_factor = IR / W
            W = IR
            H = int(H * scale_factor)

        # 调整调整后的 H 和 W 为接近的8的倍数
        H = (H // 8) * 8
        W = (W // 8) * 8

    return H, W



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_type', type=str, default='SD_IM',
                        help='pipeline type')

    parser.add_argument('--weights_type', type=str, default='from_hub',
                        help='pipeline type')

    parser.add_argument('--pretrained_weight', type=str, default='diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
                        help='pipeline type')

    parser.add_argument('--image_folder', type=str, default='SD_IM',
                        help='pipeline type')

    parser.add_argument('--mask_folder', type=str, default='SD_IM',
                        help='pipeline type')

    parser.add_argument('--saved_folder', type=str, default='SD_IM',
                        help='pipeline type')
    parser.add_argument('--seed', type=int, default=114514,
                        help='pipeline type')

    parser.add_argument('--cfg', type=int, default=7.5,
                        help='pipeline type')

    parser.add_argument('--strength', type=int, default=1.0,
                        help='pipeline type')


    args = parser.parse_args()
    ideal_resolution = 512

    '''--------------------- Loading the Pipeline -----------------------------------'''
    if args.weights_type=='from_hub' or args.weights_type=='from_folder':
        if args.pipeline_type=='SD_IM':
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_weight, 
                torch_dtype=torch.float16,
                safety_checker=None)
            ideal_resolution = 512

        elif args.pipeline_type =="SDXL_IM":
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                args.pretrained_weight, 
                torch_dtype=torch.float16)
            ideal_resolution = 1024

        elif args.pipline_type=="Custom_SD_IM":
            pipe = Custom_SD_Inpainting.from_pretrained(
                args.pretrained_weight, 
                torch_dtype=torch.float16)
            ideal_resolution = 512

        elif args.pipline_type=="Custom_SDXL_IM":
            pipe = Custom_SDXL_Inpainting.from_pretrained(
                args.pretrained_weight, 
                torch_dtype=torch.float16)
            ideal_resolution = 1024
    
    else:
        if args.pipeline_type=='SD_IM':
            pass
        elif args.pipeline_type =="SDXL_IM":
            pass

    pipeline = pipe.to("cuda")  # 如果有GPU，切换到GPU加速
    pipeline.enable_model_cpu_offload()
    print("loaded the pretrained pipeline with Name {}".format(args.pipeline_type))

    '''--------------------- Loading Inference Data ---------------------------------------'''
    input_image_folder = args.image_folder
    mask_folder = args.mask_folder
    input_images_pil_list = [Image.open(os.path.join(input_image_folder,fname)).convert("RGB") for fname in os.listdir(input_image_folder)]
    mask_pil_list = [Image.open(os.path.join(mask_folder,fname)).convert("RGB") for fname in os.listdir(input_image_folder)]

    generator = torch.Generator("cuda").manual_seed(args.seed)
    prompt = "A girl in front of The Tokyo Tower."
    negative_prompt = "bad quality, low resolution,Disharmonious ,Deformed, Out of frame, Poorly drawn, bad anatomy, deformed, ugly, disfigured"
    
    print("Setting the data is Done")

    cfg_score = args.cfg
    strength = args.strength

    for idx in tqdm(range(len(input_images_pil_list))):
        old_W, old_H = input_images_pil_list[idx].size
        new_H, new_W = adjust_resolution(H=old_H,W=old_W,
                                    IR=ideal_resolution)

        images = pipeline(prompt=prompt, 
                                image=input_images_pil_list[idx], 
                                mask_image=mask_pil_list[idx], 
                                generator=generator,
                                negative_prompt=negative_prompt,
                                guidance_scale=cfg_score,
                                num_images_per_prompt=1,
                                height=new_H,
                                width=new_W,
                                strength =strength,
                                ).images
        








    
    




    pass