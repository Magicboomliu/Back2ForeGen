import argparse
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import os
import logging
import tqdm
import itertools

from safetensors import safe_open
from transformers.utils import ContextManagers
import accelerate
from accelerate import Accelerator
import transformers
import datasets
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil
import diffusers
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline


from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

import sys
sys.path.append("../..")
import skimage.io
logger = get_logger(__name__, log_level="INFO")
import  matplotlib.pyplot as plt
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D


from networks.learnable_reference_only import ConverterNetwork

from losses.layer_wise_l1_loss import LayerWise_L1_Loss
from losses.attn_loss import Attn_loss 
import matplotlib.pyplot as plt
from diffusers.pipelines.pipeline_utils import DiffusionPipeline,StableDiffusionMixin
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
import cv2
import inspect
from PIL import Image
import PIL
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers, PIL_INTERPOLATION
import skimage.io



def image_normalization(image_tensor):
    image_normalized = image_tensor * 2.0 - 1.0
    return image_normalized

def torch_dfs(model: torch.nn.Module):
    r"""
    Performs a depth-first search on the given PyTorch model and returns a list of all its child modules.

    Args:
        model (torch.nn.Module): The PyTorch model to perform the depth-first search on.

    Returns:
        list: A list of all child modules of the given model.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result
# get all_mean and var and clear
def get_all_mean_and_var_and_hidden_states(gn_modules):
    all_mean = dict()
    all_std = dict()
    all_hidden_states = dict()

    for i, module in enumerate(gn_modules):
        if getattr(module, "original_forward", None) is None:
            module.original_forward = module.forward
        
        if i == 0:
            # mid_block
            all_mean[('mid_block',i)] = module.mean_bank
            all_std[('mid_block',i)] = module.var_bank
            all_hidden_states[('mid_block',i)] = module.feature_bank
            module.mean_bank = []
            module.var_bank = []
            module.feature_bank = []


        elif isinstance(module, CrossAttnDownBlock2D):
            all_mean[('CrossAttnDownBlock2D',i)] = module.mean_bank
            all_std[('CrossAttnDownBlock2D',i)] = module.var_bank
            all_hidden_states[('CrossAttnDownBlock2D',i)] = module.feature_bank
            module.mean_bank = []
            module.var_bank = []
            module.feature_bank = []

        elif isinstance(module, DownBlock2D):
            all_mean[('DownBlock2D',i)] = module.mean_bank
            all_std[('DownBlock2D',i)] = module.var_bank
            all_hidden_states[('DownBlock2D',i)] = module.feature_bank
            module.mean_bank = []
            module.var_bank = []
            module.feature_bank = []

        elif isinstance(module, CrossAttnUpBlock2D):
            all_mean[('CrossAttnUpBlock2D',i)] = module.mean_bank
            all_std[('CrossAttnUpBlock2D',i)] = module.var_bank
            all_hidden_states[('CrossAttnUpBlock2D',i)] = module.feature_bank
            module.mean_bank = []
            module.var_bank = []
            module.feature_bank = []

        elif isinstance(module, UpBlock2D):
            all_mean[('UpBlock2D',i)] = module.mean_bank
            all_std[('UpBlock2D',i)] = module.var_bank
            all_hidden_states[('UpBlock2D',i)] = module.feature_bank
            module.mean_bank = []
            module.var_bank = []
            module.feature_bank = []

    return all_mean,all_std,all_hidden_states
# get attention and clear
def get_all_attn_states(attn_modules):    
    all_attn_banks = []
    for i, module in enumerate(attn_modules):
        all_attn_banks.append(module.attn_bank)
        module.attn_bank = []
    return all_attn_banks

def position_encoding(num, d_model=512, device='cuda:0'):
    position = torch.arange(0, d_model, device=device).float().unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (position // 2)) / d_model)
    angle_rads = num * angle_rates
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.zeros(angle_rads.shape, device=device)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return pos_encoding

def encode_prompt(
            prompt,
            negative_prompt,device,
            prompt_embeds=None,
            do_classifier_free_guidance=True, \
            negative_prompt_embeds=None,
            tokenizer=None,
            text_encoder=None,
            clip_skip=None,
            unet=None,
            num_images_per_prompt=1):
    
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # text input tokenizer
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )

            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)

    if text_encoder is not None:
        prompt_embeds_dtype = text_encoder.dtype
    elif unet is not None:
        prompt_embeds_dtype = unet.dtype
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

        # uncond
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = text_encoder(
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

def resize_image(image,size,is_mask):
    original_shape = image.shape[:2]
    if not is_mask:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        resized_image = resized_image.astype(np.bool_)
    return resized_image, original_shape

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,):

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



def image_normalization(image_tensor):
    image_normalized = image_tensor * 2.0 - 1.0
    return image_normalized

def image_denormalization(image_tensor):
    image_denormalized = (image_tensor + 1.0)/2.0
    image_denormalized = torch.clamp(image_denormalized,min=0,
                                    max=1.0)
    return image_denormalized



def update_all_attn_states(attn_modules,attn_list):
    for i, module in enumerate(attn_modules):
        module.attn_bank = []
        module.attn_bank = [attn_list[i]*1.0]


def update_mean_and_variane_gn_auto(gn_modules,new_mean_bank,new_var_bank):
    for i, module in enumerate(gn_modules):
        if getattr(module, "original_forward", None) is None:
            module.original_forward = module.forward
        
        if i == 0:
            # mid_block
            module.mean_bank = new_mean_bank[('mid_block',i)]
            module.var_bank = new_var_bank[('mid_block',i)]
            
            module.feature_bank = []


        elif isinstance(module, CrossAttnDownBlock2D):
            if i==1 or i==2:
                module.mean_bank = []
                module.var_bank =[]
                module.feature_bank = []
            
            else:
                module.mean_bank = new_mean_bank[('CrossAttnDownBlock2D',i)]
                module.var_bank = new_var_bank[('CrossAttnDownBlock2D',i)]
                module.feature_bank = []


        elif isinstance(module, DownBlock2D):
            module.mean_bank = new_mean_bank[('DownBlock2D',i)]
            module.var_bank = new_var_bank[('DownBlock2D',i)]
            module.feature_bank = []

        elif isinstance(module, CrossAttnUpBlock2D):
            if i==8:
                module.mean_bank = []
                module.var_bank =[]
                module.feature_bank = []
            else:
                module.mean_bank = new_mean_bank[('CrossAttnUpBlock2D',i)]
                module.var_bank = new_var_bank[('CrossAttnUpBlock2D',i)]
                module.feature_bank = []

        elif isinstance(module, UpBlock2D):
            module.mean_bank = new_mean_bank[('UpBlock2D',i)]
            module.var_bank = new_var_bank[('UpBlock2D',i)]
            module.feature_bank = []
            
            


class PFN_AdaIN_Inpainting_SD_Pipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    FromSingleFileMixin):

    def __init__(
        self,
        vae: Union[AutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True):
        super().__init__()
        
        
        self.do_classifier_free_guidance = True

        if unet.config.in_channels != 9:
            logger.info(f"You have loaded a UNet with {unet.config.in_channels} input channels which.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)



    def _encode_image_to_latents(self,image,weight_dtype='cuda:0'):
    
        h_rgb = self.vae.encoder(image.to(weight_dtype))
        moments_rgb = self.vae.quant_conv(h_rgb)
        mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
        latents = mean_rgb * self.vae.config.scaling_factor    #torch.Size([1, 4, 64, 64])
        return latents

    def _decode_latents_to_images(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.vae.config.scaling_factor
        
        depth_latent = depth_latent.half()
        
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        stacked = torch.clamp(stacked,min=-1,max=1.0)
        
        return stacked

    def _load_pretrained_converter(self,pretrained_path):        
        # loaded the pretrained models.
        saved_contents = torch.load(pretrained_path)
        self.converter_network = ConverterNetwork()
        self.converter_network.load_state_dict(saved_contents["model_state"])
        self.converter_network.cuda()
        self.converter_network.half()

    def prepare_inpainted_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            #image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents


    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    
    def __call__(self,full_image,prompt,fg_image,fg_mask,
                 use_converter=False,
                 pretrained_converter_path=None,
                 use_adaIN=True,use_attn=True,
                 num_inference_steps=50,
                 num_images_per_prompt=1,
                 height=512,
                 width=512,
                 is_strength_max= True,
                 gudiance_score = 1.0,
                 attn_weight = 0.1,
                 
                 device='cuda:0'):
        
        if use_converter:
            self._load_pretrained_converter(pretrained_path=pretrained_converter_path)
        
        full_image_np = full_image


        # full imaghe
        full_image = torch.from_numpy(np.array(full_image).astype(np.float32)/255).permute(2,0,1).unsqueeze(0).half()
        fg_image =  torch.from_numpy(np.array(fg_image).astype(np.float32)/255).permute(2,0,1).unsqueeze(0).half()
        fg_mask = torch.from_numpy(fg_mask).permute(2,0,1).unsqueeze(0)

        full_image = image_normalization(full_image)
        # foreground image normalization
        fg_image = image_normalization(fg_image)

        full_image_latents = self._encode_image_to_latents(full_image)
        fore_latents = self._encode_image_to_latents(fg_image)
        

        masked_image = full_image * fg_mask
        masked_image = masked_image.half()
        masked_for_concated = F.interpolate(fg_mask,scale_factor=1./8,
                        mode='nearest')
        
        masked_for_concated = torch.ones_like(masked_for_concated) - masked_for_concated
        
        # masked_for_concated = torch.ones_like(masked_for_concated) - masked_for_concated
        masked_image_latents = self._encode_image_to_latents(masked_image)
        masked_for_concated = masked_for_concated.to(masked_image_latents.device).half()


        
        
        prompt_embeds, negative_prompt_embeds = encode_prompt(prompt=prompt,
                                                              negative_prompt=["" for _ in range(1)],
                                                              device=device,
                                                              prompt_embeds=None,
                                                              tokenizer=self.tokenizer,
                                                              text_encoder=self.text_encoder,
                                                              unet=self.unet)
        
        self.scheduler.config.prediction_type = "epsilon"
        #noise_scheduler.config.prediction_type = "epsilon"
        
        
        if gudiance_score>1:
            # encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = prompt_embeds
        else:
            encoder_hidden_states = prompt_embeds
            


        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]

        latent_timestep = timesteps[:1].repeat(1 * num_images_per_prompt)
        num_channels_latents=4
        return_image_latents = False
        
        latents_outputs = self.prepare_inpainted_latents(
            1 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            image=full_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )
        
    
        if return_image_latents:
            latents_to_be_updated, noise, image_latents = latents_outputs
        else:
            latents_to_be_updated, noise = latents_outputs
        

        
        
        MODE = "write"
        style_fidelity= 0.5
        uc_mask = (
            torch.Tensor([1] * 1 * num_images_per_prompt + [0] * 1 * num_images_per_prompt)
            .type_as(latents_to_be_updated)
            .bool()
        )
        
        attention_auto_machine_weight = 1.0
        gn_auto_machine_weight = 1.0
        
        
        
        
        if use_adaIN:
            def hacked_mid_forward(self, *args, **kwargs):
                eps = 1e-6
                x = self.original_forward(*args, **kwargs)
                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                        self.feature_bank.append(x)
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                        var_acc = sum(self.var_bank) / float(len(self.var_bank))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        # x_uc = (((x - mean) / std) * std_acc) + mean_acc

                        x = (((x - mean) / std) * std_acc) + mean_acc
                        # x_c = x_uc.clone()
                        # if True and style_fidelity > 0:
                        #     x_c[uc_mask] = x[uc_mask]
                        # x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
                    self.mean_bank = []
                    self.var_bank = []
                
                if MODE =='read_old':
                    x= x
                    self.mean_bank = []
                    self.var_bank = []
                
                
                return x

            def hack_CrossAttnDownBlock2D_forward(
                self,
                hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,):
                eps = 1e-6

                # TODO(Patrick, William) - attention mask is not used
                output_states = ()

                for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                            self.feature_bank.append([hidden_states])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            # hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            
                            hidden_states= (((hidden_states - mean) / std) * std_acc) + mean_acc
                            # hidden_states_c = hidden_states_uc.clone()
                            # if True and style_fidelity > 0:
                            #     hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            # hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc
                        
                    if MODE =='read_old':
                        hidden_states = hidden_states
                        self.mean_bank = []
                        self.var_bank = []
                        
                    output_states = output_states + (hidden_states,)

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.downsamplers is not None:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    output_states = output_states + (hidden_states,)

                return hidden_states, output_states

            def hacked_DownBlock2D_forward(
                self,
                hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None,
                **kwargs: Any,) -> Tuple[torch.Tensor, ...]:
                eps = 1e-6

                output_states = ()

                for i, resnet in enumerate(self.resnets):
                    hidden_states = resnet(hidden_states, temb)

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                            self.feature_bank.append([hidden_states])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            # hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc

                            hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            # hidden_states_c = hidden_states_uc.clone()
                            # if True and style_fidelity > 0:
                            #     hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            # hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc
                            
                    if MODE=='read_old':
                        hidden_states = hidden_states
                        self.mean_bank = []
                        self.var_bank = []
                        

                    output_states = output_states + (hidden_states,)

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.downsamplers is not None:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    output_states = output_states + (hidden_states,)

                return hidden_states, output_states

            def hacked_CrossAttnUpBlock2D_forward(
                self,
                hidden_states: torch.Tensor,
                res_hidden_states_tuple: Tuple[torch.Tensor, ...],
                temb: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                upsample_size: Optional[int] = None,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
                eps = 1e-6
                # TODO(Patrick, William) - attention mask is not used
                for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                    # pop res hidden states
                    res_hidden_states = res_hidden_states_tuple[-1]
                    res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                            self.feature_bank.append([hidden_states])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            #hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            
                            hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            # hidden_states_c = hidden_states_uc.clone()
                            # if True and style_fidelity > 0:
                            #     hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            # hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc
                            
                    if MODE =='read_old':
                        hidden_states = hidden_states
                        self.mean_bank = []
                        self.var_bank = []
                        

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.upsamplers is not None:
                    for upsampler in self.upsamplers:
                        hidden_states = upsampler(hidden_states, upsample_size)

                return hidden_states

            def hacked_UpBlock2D_forward(
                self,
                hidden_states: torch.Tensor,
                res_hidden_states_tuple: Tuple[torch.Tensor, ...],
                temb: Optional[torch.Tensor] = None,
                upsample_size: Optional[int] = None,
                **kwargs: Any,) -> torch.Tensor:
                eps = 1e-6
                for i, resnet in enumerate(self.resnets):
                    # pop res hidden states
                    res_hidden_states = res_hidden_states_tuple[-1]
                    res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, temb)

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                            self.feature_bank.append([hidden_states])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            # hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            
                            hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            # hidden_states_c = hidden_states_uc.clone()
                            # if True and style_fidelity > 0:
                            #     hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            # hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc
                            
                    if MODE=='read_old':
                        hidden_states = hidden_states
                        self.mean_bank = []
                        self.var_bank = []
                        

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.upsamplers is not None:
                    for upsampler in self.upsamplers:
                        hidden_states = upsampler(hidden_states, upsample_size)

                return hidden_states


            #FIXME
            # all the mid bloacks
            if use_converter:
                if MODE == 'write':
                    gn_auto_machine_weight = 10000
                    
            
            gn_modules = [self.unet.mid_block]
            down_blocks = self.unet.down_blocks
            self.unet.mid_block.gn_weight = 0
            
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)


            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            # totally is 9 layers.
            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid_block
                    module.forward = hacked_mid_forward.__get__(module, torch.nn.Module)
                elif isinstance(module, CrossAttnDownBlock2D):
                    module.forward = hack_CrossAttnDownBlock2D_forward.__get__(module, CrossAttnDownBlock2D)
                elif isinstance(module, DownBlock2D):
                    module.forward = hacked_DownBlock2D_forward.__get__(module, DownBlock2D)
                elif isinstance(module, CrossAttnUpBlock2D):
                    module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                elif isinstance(module, UpBlock2D):
                    module.forward = hacked_UpBlock2D_forward.__get__(module, UpBlock2D)
                
                # initial set it to zero
                module.mean_bank = []
                module.var_bank = []
                module.feature_bank = []
                module.gn_weight *= 2

        
        if use_attn:
            if MODE=='read':
                attention_auto_machine_weight = 1.0
                
            def hacked_basic_transformer_inner_forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None):  
                
                # perform layer noramlzaition.
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero:
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                else:
                    norm_hidden_states = self.norm1(hidden_states)

                # 1. Self-Attention
                cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
                if self.only_cross_attention:
                    # update the v
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                else:
                    # when writing, without text prmpt
                    if MODE == "write":
                        self.attn_bank.append(norm_hidden_states.detach().clone())
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )

                    alpha = attn_weight
                    if MODE == "read":
                        # maybe alayws true: just add the
                        if attention_auto_machine_weight > self.attn_weight:
                            
                            if len(self.attn_bank)!=0:
                                current_attn_bank = [self.attn_bank[0] * alpha + (1-alpha) *norm_hidden_states]
                            else:
                                current_attn_bank = self.attn_bank
                            
                            attn_output = self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=torch.cat([norm_hidden_states] + current_attn_bank, dim=1),
                                #encoder_hidden_states=torch.cat([norm_hidden_states]+[norm_hidden_states], dim=1),
                                attention_mask=attention_mask,
                                **cross_attention_kwargs,
                            )
                            
                            # attn_output = attn_output.view(1,4096,320)
                            # #[2,4096,320]: 1:c,2: Noc
                            # # print(attn_output_uc.shape)
                            # # quit()
                            # attn_output_c = attn_output_uc.clone()
                            # if True and style_fidelity > 0:
                            #     # select the uncond regions
                            #     attn_output_c[uc_mask] = self.attn1(
                            #         norm_hidden_states[uc_mask],
                            #         encoder_hidden_states=norm_hidden_states[uc_mask],
                            #         **cross_attention_kwargs,
                            #     )
                            # attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                            self.attn_bank.clear()
                        else:
                            
                            attn_output = self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                attention_mask=attention_mask,
                                **cross_attention_kwargs,
                            )

                    if MODE == "read_old":                           
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        self.attn_bank.clear()
                
                
                if self.use_ada_layer_norm_zero:
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                
                # update hidden state
                hidden_states = attn_output + hidden_states

                if self.attn2 is not None:
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )
                    # 2. Cross-Attention
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )

                    
                    attn_output = attn_output.view(1,-1,attn_output.shape[-1])
                    hidden_states = attn_output + hidden_states
                    


                # 3. Feed-forward
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                ff_output = self.ff(norm_hidden_states)

                if self.use_ada_layer_norm_zero:
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states

                return hidden_states


            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.attn_bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
        
        
        MODE = "write"
        # Denoising loop
        iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )

        for i, t in iterable:
            
            if use_adaIN or use_attn:    
                noisy_latents = self.scheduler.add_noise(full_image_latents, noise, t)
                
                if not use_converter:
                    fore_latents_for_input = self.scheduler.add_noise(fore_latents, noise, t)
                else:
                    fore_latents_for_input = fore_latents
                    
                #-------------------------------------------------------------------------------------------#
                # Input Foreground Images.
                unet_input = torch.cat([fore_latents_for_input, masked_for_concated, masked_image_latents], dim=1)
                MODE = "write"
                self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                
                if use_attn:
                    fg_attn_banks = get_all_attn_states(attn_modules)
                    fg_attn_banks_list = [fg[0] for fg in fg_attn_banks]
                
                if use_adaIN:
                    mean_bank_fore,variance_bank_fore,feature_bank_fore = get_all_mean_and_var_and_hidden_states(gn_modules=gn_modules)



                if MODE=='write':
                    gn_auto_machine_weight = 1000
                    
                if use_converter:
                    if MODE=='write':
                        gn_auto_machine_weight = 1000
                        
                        t_embed = position_encoding(t,device=encoder_hidden_states.device)
                        t_embed = t_embed.half()
                        fg_mask = fg_mask.to(encoder_hidden_states.device)
                        fg_mask = fg_mask.half()
                        


                        if use_adaIN and not use_attn:
                            fg_attn_banks_list = None
                            fg2all_mean_bank, fg2all_variance_bank,converted_fg_attn_banks_list = self.converter_network(mean_bank = mean_bank_fore,
                                            var_bank=variance_bank_fore,
                                            feat_bank=feature_bank_fore,
                                            time_embed=t_embed,
                                            text_embed=prompt_embeds,
                                            foreground_mask=fg_mask,
                                            inputs = fg_attn_banks_list,
                                            use_seperate = 'adaIN'
                                            )
                        if use_attn and not use_adaIN:
                            variance_bank_fore = None
                            feature_bank_fore = None
                            mean_bank_fore = None
                            
                            fg2all_mean_bank, fg2all_variance_bank,converted_fg_attn_banks_list = self.converter_network(mean_bank = mean_bank_fore,
                                            var_bank=variance_bank_fore,
                                            feat_bank=feature_bank_fore,
                                            time_embed=t_embed,
                                            text_embed=prompt_embeds,
                                            foreground_mask=fg_mask,
                                            inputs = fg_attn_banks_list,
                                            use_seperate = 'attn'
                                            )
                        if use_adaIN and use_attn:
                            
                            fg2all_mean_bank, fg2all_variance_bank,converted_fg_attn_banks_list = self.converter_network(mean_bank = mean_bank_fore,
                                            var_bank=variance_bank_fore,
                                            feat_bank=feature_bank_fore,
                                            time_embed=t_embed,
                                            text_embed=prompt_embeds,
                                            foreground_mask=fg_mask,
                                            inputs = fg_attn_banks_list,
                                            use_seperate = 'all'
                                            )
                        
                
                if use_attn:                    
                    if not use_converter:
                        converted_fg_attn_banks_list = fg_attn_banks_list
                    update_all_attn_states(attn_modules=attn_modules,attn_list=converted_fg_attn_banks_list)
                
                if use_adaIN:
                    if not use_converter:
                        fg2all_mean_bank = mean_bank_fore
                        fg2all_variance_bank = variance_bank_fore
                        
                    
                    update_mean_and_variane_gn_auto(gn_modules=gn_modules,new_mean_bank=fg2all_mean_bank,
                                                    new_var_bank=fg2all_variance_bank)

                
                
            # using CFG Gudiance
            if gudiance_score>1:
                MODE = "read"
                # Prompt Denosing
                latent_model_input_cond = self.scheduler.scale_model_input(latents_to_be_updated, t)
                cond_latent_input_for_unet = torch.cat([latent_model_input_cond, masked_for_concated, masked_image_latents], dim=1)

                noise_pred_text = self.unet(
                        cond_latent_input_for_unet,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]
                
            
                # update_all_attn_states(attn_modules=attn_modules,attn_list=converted_fg_attn_banks_list)
                # Unconditional Prompt Denosing
                MODE = "read"
                latent_model_input_uncond = self.scheduler.scale_model_input(latents_to_be_updated, t)
                uncond_latent_input_for_unet = torch.cat([latent_model_input_uncond, masked_for_concated, masked_image_latents], dim=1)

                noise_pred_uncond = self.unet(
                        uncond_latent_input_for_unet,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]


            #  Without CFG Gudiance
            else:
                
                MODE = "read"
                latent_model_input_cond = self.scheduler.scale_model_input(latents_to_be_updated, t)
                cond_latent_input_for_unet = torch.cat([latent_model_input_cond, masked_for_concated, masked_image_latents], dim=1)

                noise_pred_text = self.unet(
                        cond_latent_input_for_unet,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]
                
        

    
            if gudiance_score>1:
                noise_pred = noise_pred_uncond + gudiance_score * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_text
            
            latents_to_be_updated = self.scheduler.step(noise_pred, t, latents_to_be_updated, return_dict=False)[0]
                

        

        torch.cuda.empty_cache()
        recovered_image = self._decode_latents_to_images(depth_latent=latents_to_be_updated)
        recovered_image = (recovered_image + 1.0)/2
        recovered_image = torch.clamp(recovered_image,min=0,max=1)
        recovered_image_np = recovered_image.squeeze(0).permute(1,2,0).cpu().numpy()
        recovered_image_for_save = (recovered_image_np * 255.).astype(np.uint8)
        
        return recovered_image_for_save
        # skimage.io.imsave("normal_Mix2.png",recovered_image_for_save)
        

        

        

if __name__=="__main__":

    set_seed(1024)
    
    base_model_path = "/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
    current_stable_diffusion_model = PFN_AdaIN_Inpainting_SD_Pipeline.from_single_file(pretrained_model_link_or_path=base_model_path,torch_dtype=torch.float16)
    
    noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2",subfolder='scheduler')
    noise_scheduler.config.prediction_type = "epsilon"
    my_pipe = StableDiffusionPipeline.from_pretrained("ckpt/anything-v4.5-vae-swapped")
    current_stable_diffusion_model.vae = my_pipe.vae
    pipe = current_stable_diffusion_model.to("cuda")
    pipe.scheduler = noise_scheduler
    pipe.vae = pipe.vae.half()
    pipe.unet = pipe.unet.half()
    pipe = pipe.to("cuda:0")
    


    # Loading the data
    root_path = "/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
    input_image_path = "SD_Gen_Images/images/5.png"
    bg_mask_path = "SD_Gen_Images/background_masks/5.png"
    # original image PIL and original Background Image PIL
    input_image= Image.open(os.path.join(root_path,input_image_path)).convert("RGB")
    origianl_size = input_image.size
    saved_input = input_image
    bg_mask = Image.open(os.path.join(root_path,bg_mask_path)).convert("RGB")
    original_bg_mask = bg_mask
    # Original image numpy and original Background imageNumpu
    input_image = np.array(input_image).astype(np.float32)
    bg_mask = np.array(bg_mask).astype(np.float32)/255
    input_image,_ = resize_image(input_image,size=(512,512),is_mask=False)
    bg_mask,_ = resize_image(bg_mask,size=(512,512),is_mask=False)
    resize_input_image_pil = Image.fromarray((input_image).astype(np.uint8))
    # saved initial foreground image
    fg_mask = np.ones_like(bg_mask) -bg_mask
    original_bg_mask_black = Image.fromarray((np.zeros_like(fg_mask) * 255).astype(np.uint8))
    original_bg_mask_white = Image.fromarray((np.ones_like(fg_mask) * 255).astype(np.uint8))
    
    fg_mask_pil = Image.fromarray((fg_mask*255).astype(np.uint8))
    fg_mask = fg_mask[:,:,0:1] #[H,W,1]
    fg_mask_np = fg_mask
    fg_image = input_image * fg_mask
    initial_fg_image = fg_image
    initial_fg_image = initial_fg_image.astype(np.uint8)
    initial_fg_image = Image.fromarray(initial_fg_image)
    fg_image = fg_image.astype(np.uint8)
    fg_image = Image.fromarray(fg_image)


    #without_noise_model = "/home/zliu/PFN/PFN24/i24_ziliu/ForB/outputs/Mixed_Dataset/sd15_inpainting_single_V4/ckpt_10001.pt"
    
    without_noise_model = "/home/zliu/PFN/pretrained_models/Converter/Mix_Inpainting/ckpt_11001.pt"
    
    prompt = ["A girl with sandy brown hair and deep blue eyes wearing a kimono is taking a tour in a vineyard."]
    #prompt = ["A girl with butterscotch blonde hair and topaz brown eyes wearing a kimono is relaxing on a bench on a beachside boardwalk."]
    with torch.no_grad():
        pipe(full_image=input_image,prompt=prompt,fg_image= fg_image,fg_mask=fg_mask_np,
            pretrained_converter_path = without_noise_model,use_converter=False,use_adaIN=False,use_attn=True,
            gudiance_score=10,
            attn_weight=0.1
            )
    
    pass