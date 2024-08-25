import diffusers
import os
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline

from diffusers.models import AutoencoderKL,ImageProjection,UNet2DConditionModel
from diffusers import DDIMScheduler,DPMSolverMultistepScheduler

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from text_encoder_without_positional_encoding import CLIPTextModelWithoutPrompt

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# Explore ControlNet?
from safetensors import safe_open

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision import transforms as tfms
from PIL import Image
import cv2


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


# prompt text into latents
def encode_prompt(
    prompt,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
    prompt_embeds=None,
    tokenizers=None,
    text_encoders=None,
    prompt_2=None,
    negative_prompt_2=None,
    clip_skip=None,
    negative_prompt_embeds=None
    ):
    '''
    tokenizers is list
    
    '''
    
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # define the tokenizers
    if len(tokenizers)>1:
        tokenizers = tokenizers
    else:
        tokenizers = tokenizers[-1]
    if len(text_encoders)>1:
        text_encoders = text_encoders
    else:
        text_encoders = text_encoders[-1]


    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    # textual inversion: process multi-vector tokens if necessary
    prompt_embeds_list = []
    prompts = [prompt, prompt_2]
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",)
        

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

        #FIXME 
        # text_input_ids = torch.ones_like(text_input_ids)
        
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # list

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and False
    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            # #FIXME 
            # uncond_input_ids = torch.ones_like(uncond_input_ids)
            
            negative_prompt_embeds = text_encoder(
                uncond_input_ids,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if len(text_encoders)>1:
        text_encoder_2 = text_encoders[1]
    if text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=text_encoders[0].dtype, device=device) # FIXME

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        if text_encoder_2 is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoders[0].dtype, device=device) #FIXME
 
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_add_time_ids(
    original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None,
    unet=None,):

    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


@torch.no_grad()
def ddim_inversion(
    start_latents,
    prompt,
    text_encoder,
    text_tokenizer,
    scheduler,
    unet,
    num_inference_steps,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt="",
    height=None,
    width=None,
    original_size=None,
    target_size=None,
    guidance_scale=1):
    
    original_size =  (height, width)
    target_size = (height, width)

    batch_size = 1
    # 1 get the text embedding
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(prompt=prompt,
                  num_images_per_prompt=num_images_per_prompt,
                  do_classifier_free_guidance=do_classifier_free_guidance,
                  negative_prompt=negative_prompt,
                  tokenizers=text_tokenizer,
                  text_encoders=text_encoder)


    # 1.1 Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if len(text_encoder)>1:
        text_encoder_2 = text_encoder[-1]
    if text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = text_encoder_2.config.projection_dim

    
    crops_coords_top_left = (0,0)
    add_time_ids = get_add_time_ids(
        original_size=original_size,
        crops_coords_top_left=crops_coords_top_left,
        target_size=target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
        unet=unet
    )

    negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    
    # Latents are now the specified start latents
    latents = start_latents.clone()
    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []
    # set the steps of the scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)


        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}        
        timestep_cond = None
        cross_attention_kwargs = None
        
        # predict the noise
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Perform guidance: CFG
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)
    

def decode_latents(latents,vae):

    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = ((image +1)/2).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy().squeeze(0)

    return image



# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    start_latents,
    prompt,
    text_encoder,
    text_tokenizer,
    scheduler,
    unet,
    num_inference_steps,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt="",
    height=None,
    width=None,
    original_size=None,
    target_size=None,
    guidance_scale=1,
    start_step =0,
):

    original_size =  (height, width)
    target_size = (height, width)

    batch_size = 1
    # 1 get the text embedding
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(prompt=prompt,
                  num_images_per_prompt=num_images_per_prompt,
                  do_classifier_free_guidance=do_classifier_free_guidance,
                  negative_prompt=negative_prompt,
                  tokenizers=text_tokenizer,
                  text_encoders=text_encoder)


    # 1.1 Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if len(text_encoder)>1:
        text_encoder_2 = text_encoder[-1]
    if text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = text_encoder_2.config.projection_dim

    
    crops_coords_top_left = (0,0)
    # negative_crops_coords_top_left=(0,0)
    # negative_original_size = original_size
    # negative_crops_coords_top_left= (0, 0),
    # negative_target_size = target_size,
    
    add_time_ids = get_add_time_ids(
        original_size=original_size,
        crops_coords_top_left=crops_coords_top_left,
        target_size=target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
        unet=unet
    )

    negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = scheduler.timesteps[i]
        
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}        
        timestep_cond = None
        cross_attention_kwargs = None
        
        # predict the noise
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt


    return latents



def calculate_psnr(image1, image2):
    # Ensure the images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    
    # Convert images to float32
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity if images are identical
    
    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr



def load_pretrained_weights(source_model, target_model):
    """
    将source_model的权重加载到target_model中。

    Args:
        source_model (torch.nn.Module): 已经加载了预训练权重的源模型。
        target_model (torch.nn.Module): 需要加载预训练权重的目标模型。

    Returns:
        target_model (torch.nn.Module): 加载了预训练权重的目标模型。
    """
    target_model.load_state_dict(source_model.state_dict())
    return target_model


if __name__=="__main__":
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # get the models
    #model_path = "/home/zliu/SDWebUI/stable-diffusion-webui/models/Stable-diffusion/animexlXuebimix_v60LCM.safetensors"
    
    model_path = "/home/zliu/SDWebUI/stable-diffusion-webui/models/Stable-diffusion/animagineXLV31_v31.safetensors"
    pipeline = StableDiffusionXLPipeline.from_single_file(pretrained_model_link_or_path=model_path,torch_dtype=torch.float16)
    # 加载 VAE 模型
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to("cuda")
    # vae = pipeline.vae
    # # 将 pipeline 移动到 GPU 上（如果使用 GPU）
    pipeline = pipeline.to("cuda")

    pipeline.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",subfolder='scheduler')
    #pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",subfolder='scheduler')

    tokenizer = pipeline.tokenizer
    tokenizer_2 =pipeline.tokenizer_2
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    scheduler = pipeline.scheduler
    

    # load the input images
    input_image_filename = "/home/zliu/PFN_Internship/DDIM_Inversion/diffusers_ddim_inversion/anime.jpg"
    input_image = Image.open(input_image_filename).convert("RGB")
    original_size = input_image.size
    input_image = input_image.resize((1024,1024))
    
    width,height = input_image.size
    
    input_image_tensor = tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
    input_image_tensor = input_image_tensor.to(device).half()

    with torch.no_grad():
        latent = vae.encode(input_image_tensor)
    latent_image = 0.18215 * latent.latent_dist.sample()
    
    num_inference_steps = 50 # this more bu
    guidance_scale = 1
    start_step = 30
    prompt = ""
    
    # get the inverted noise
    inverted_latents = ddim_inversion(start_latents=latent_image,
                                      prompt=prompt,
                                      text_encoder=[text_encoder,text_encoder_2],
                                      text_tokenizer=[tokenizer,tokenizer_2],
                                      scheduler=scheduler,
                                      unet=unet,
                                      num_images_per_prompt=1,
                                      num_inference_steps=num_inference_steps,
                                      do_classifier_free_guidance=True,
                                      negative_prompt="",
                                      height=height,
                                      width=width,
                                      guidance_scale=guidance_scale,
                                      )
    
    final_latents = sample(start_latents=inverted_latents[-(start_step + 1)][None],
                        prompt=prompt,
                        text_encoder=[text_encoder,text_encoder_2],
                        text_tokenizer=[tokenizer,tokenizer_2],
                        scheduler=scheduler,
                        unet=unet,
                        num_images_per_prompt=1,
                        num_inference_steps=num_inference_steps,
                        do_classifier_free_guidance=True,
                        negative_prompt="",
                        height=height,
                        width=width,
                        start_step=start_step,
                        guidance_scale=guidance_scale,
    )[0]

    
    # #saved latents images
    with torch.no_grad():
        inverted_latents_image = decode_latents(latents=final_latents.unsqueeze(0),
                                                vae=vae)
        
        inverted_latents_image_pil = Image.fromarray((inverted_latents_image*255).astype(np.uint8))
        inverted_latents_image_pil = inverted_latents_image_pil.resize(original_size)
        inverted_latents_image_pil.save("restored.png")


    image1 = cv2.imread("/home/zliu/PFN_Internship/DDIM_Inversion/diffusers_ddim_inversion/anime.jpg")
    image2 = cv2.imread('restored.png')

    # Convert images to grayscale (optional, depending on your needs)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr_value = calculate_psnr(image1, image2)
    print(f"PSNR: {psnr_value} dB")




    

    

    
    
    
    
    
    
    


