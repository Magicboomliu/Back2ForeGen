from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt

from diffusers.models import AutoencoderKL,ImageProjection,UNet2DConditionModel
from diffusers import DDIMScheduler,DPMSolverMultistepScheduler

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    clip_skip=False,
    negative_prompt_embeds=None):
    
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]


    text_inputs = tokenizers(
        prompt,
        padding="max_length",
        max_length=tokenizers.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizers(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizers.batch_decode(
            untruncated_ids[:, tokenizers.model_max_length - 1 : -1]
        )

    if hasattr(text_encoders.config, "use_attention_mask") and text_encoders.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    if clip_skip is None:
        prompt_embeds = text_encoders(text_input_ids.to(device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]
    else:
        prompt_embeds = text_encoders(
            text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
        )
        prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
        prompt_embeds = text_encoders.text_model.final_layer_norm(prompt_embeds)

        if text_encoders is not None:
            prompt_embeds_dtype = text_encoders.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt


            max_length = prompt_embeds.shape[1]
            uncond_input =tokenizers(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoders.config, "use_attention_mask") and text_encoders.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = text_encoders(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds


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
    guidance_scale=1):
    
    batch_size = 1
    # 1 get the text embedding
    prompt_embeds, negative_prompt_embeds = encode_prompt(prompt=prompt,
                  num_images_per_prompt=num_images_per_prompt,
                  do_classifier_free_guidance=do_classifier_free_guidance,
                  negative_prompt=negative_prompt,
                  tokenizers=text_tokenizer,
                  text_encoders=text_encoder)
    
    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(scheduler.timesteps)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        # Perform guidance
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
    image = (image / 2 + 0.5).clamp(0, 1)
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
    guidance_scale=1,
    start_step =0,
):

    prompt_embeds, negative_prompt_embeds = encode_prompt(prompt=prompt,
                  num_images_per_prompt=num_images_per_prompt,
                  do_classifier_free_guidance=do_classifier_free_guidance,
                  negative_prompt=negative_prompt,
                  tokenizers=text_tokenizer,
                  text_encoders=text_encoder)

    # Set num inference steps
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= scheduler.init_noise_sigma

    latents = start_latents.clone()

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    for i in tqdm(range(start_step, num_inference_steps)):

        t = scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

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




import os

if __name__=="__main__":
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # get the models
    
    model_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/aingdiffusion_v170.safetensors"

    pipeline = StableDiffusionPipeline.from_single_file(pretrained_model_link_or_path=model_path,torch_dtype=torch.float16,
                                                        )
    pipeline = pipeline.to("cuda")
    
    
    pipeline.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder='scheduler')
    #pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder='scheduler')
    
    tokenizer = pipeline.tokenizer
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    scheduler = pipeline.scheduler
    vae = pipeline.vae

    
    resolution = 512
    # load the input images
    input_image_filename = "input_images/anime_example.jpg"
    # input_image_filename = "/home/zliu/PFN_Internship/DDIM_Inversion/diffusers_ddim_inversion/anime.jpg"
    input_image = Image.open(input_image_filename).convert("RGB")
    original_size = input_image.size
    input_image = input_image.resize((resolution,resolution))
    width,height = input_image.size
    input_image_tensor = tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
    input_image_tensor = input_image_tensor.to(device).half()

    with torch.no_grad():
        latent = vae.encode(input_image_tensor)
    latent_image = 0.18215 * latent.latent_dist.sample()
    
    num_inference_steps = 50 # this more bu
    guidance_scale = 1
    start_step = 0
    prompt = ""

    # get the inverted noise
    inverted_latents = ddim_inversion(start_latents=latent_image,
                                      prompt=prompt,
                                      text_encoder=text_encoder,
                                      text_tokenizer=tokenizer,
                                      scheduler=scheduler,
                                      unet=unet,
                                      num_images_per_prompt=1,
                                      num_inference_steps=num_inference_steps,
                                      do_classifier_free_guidance=True,
                                      negative_prompt="",
                                      guidance_scale=guidance_scale,
                                      )
    
    
    #noisy_l = pipeline.scheduler.add_noise(latent_image, torch.randn_like(latent_image), pipeline.scheduler.timesteps[start_step])

    start_latents = inverted_latents[-(start_step+1)][None]
    #start_latents = noisy_l

    final_latents = sample(start_latents=start_latents,
                        prompt=prompt,
                        text_encoder=text_encoder,
                        text_tokenizer=tokenizer,
                        scheduler=scheduler,
                        unet=unet,
                        num_images_per_prompt=1,
                        num_inference_steps=num_inference_steps,
                        do_classifier_free_guidance=True,
                        negative_prompt="",
                        start_step=start_step,
                        guidance_scale=guidance_scale,
    )[0]


    #saved latents images
    with torch.no_grad():
        inverted_latents_image = decode_latents(latents=final_latents.unsqueeze(0),
                                                vae=vae)
        
        inverted_latents_image_pil = Image.fromarray((inverted_latents_image*255).astype(np.uint8))
        inverted_latents_image_pil = inverted_latents_image_pil.resize(original_size)
        inverted_latents_image_pil.save("input_images/anime_example_restored_sd15.jpg")


    # image1 = cv2.imread("/home/zliu/PFN_Internship/DDIM_Inversion/diffusers_ddim_inversion/anime.jpg")
    image1 = cv2.imread(input_image_filename)
    image2 = cv2.imread("input_images/anime_example_restored_sd15.jpg")

    # Convert images to grayscale (optional, depending on your needs)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr_value = calculate_psnr(image1, image2)
    print(f"PSNR: {psnr_value} dB")
    
    