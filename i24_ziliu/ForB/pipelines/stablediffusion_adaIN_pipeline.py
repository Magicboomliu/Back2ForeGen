import argparse
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Union
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import logging


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

from diffusers import StableDiffusionPipeline
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
sys.path.append("..")
from trainer.dataset_configuration import prepare_dataset,image_normalization,image_denormalization
check_min_version("0.26.0.dev0")
import skimage.io
logger = get_logger(__name__, log_level="INFO")
import  matplotlib.pyplot as plt
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
# from ForB.networks.f2all_converter import F2All_Converter
from ForB.networks.f2all_converter_tpm import F2All_Converter
import os
import skimage.io
from diffusers.utils import load_image


def from_tensor_to_image(tensor,saved_name):
    saved_image_data = (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    skimage.io.imsave(saved_name,saved_image_data)



def parse_args():
    parser = argparse.ArgumentParser(description="Foregroud2Background-Inference")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--pretrained_single_file",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--pretrained_converter_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sceneflow",
        required=True,
        help="Specify the dataset name used for training/validation.")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data1/liu",
        required=True,
        help="The Root Dataset Path.")
    
    parser.add_argument(
        "--trainlist",
        type=str,
        default="",
        required=True,
        help="train file listing the training files")
    
    parser.add_argument(
        "--vallist",
        type=str,
        default="",
        required=True,
        help="validation file listing the validation files")
    

    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--use_adIN', action='store_true', help="Increase output verbosity")

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    # get the local rank
    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank
    return args

def encode_prompt(prompt,negative_prompt,device,prompt_embeds=None,do_classifier_free_guidance=True, \
            negative_prompt_embeds=None,tokenizer=None,text_encoder=None,clip_skip=None,unet=None,num_images_per_prompt=1):
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

def update_all_mean_and_var(gn_modules,fg2all_mean_bank, fg2all_variance_bank):
    for i, module in enumerate(gn_modules):
        if getattr(module, "original_forward", None) is None:
            module.original_forward = module.forward

        if i == 0:
            # mid_block
            module.mean_bank = fg2all_mean_bank[('mid_block',i)]
            module.var_bank = fg2all_variance_bank[('mid_block',i)]


        elif isinstance(module, CrossAttnDownBlock2D):
            module.mean_bank = fg2all_mean_bank[('CrossAttnDownBlock2D',i)]
            module.var_bank = fg2all_variance_bank[('CrossAttnDownBlock2D',i)]
  


        elif isinstance(module, DownBlock2D):
            module.mean_bank = fg2all_mean_bank[('DownBlock2D',i)]
            module.var_bank = fg2all_variance_bank[('DownBlock2D',i)]



        elif isinstance(module, CrossAttnUpBlock2D):
            module.mean_bank = fg2all_mean_bank[('CrossAttnUpBlock2D',i)]
            module.var_bank = fg2all_variance_bank[('CrossAttnUpBlock2D',i)]



        elif isinstance(module, UpBlock2D):
            module.mean_bank = fg2all_mean_bank[('UpBlock2D',i)]
            module.var_bank = fg2all_variance_bank[('UpBlock2D',i)]




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


def inference():
    args = parse_args()
    logging_dir = "inference_logs"
    logging_dir = os.path.join(args.output_dir, logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    
    num_inference_steps = 50

    '''------------------------------- Models Part Configuration ---------------------------------------'''
    if args.pretrained_single_file!="none":
        current_stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                                            pretrained_model_link_or_path=args.pretrained_single_file)
    else:
        current_stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                        safety_checker=None,
        )
    
    # get the tokenizer/text_encoder/unet/scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    tokenizer = current_stable_diffusion_model.tokenizer
    logger.info("loading the noise scheduler and the tokenizer from {}".format(args.pretrained_model_name_or_path),main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = current_stable_diffusion_model.vae
        text_encoder = current_stable_diffusion_model.text_encoder
        unet = current_stable_diffusion_model.unet
        # define the network here
        converter_network = F2All_Converter(mid_channels=1280)
    

    # Loaded the Pretrained-Model Here
    pretrained_saved_dict = torch.load(args.pretrained_converter_path)['model_state']
    converter_network.load_state_dict(pretrained_saved_dict)
    print("loaded the model successfully........ with the Name of {}".format(args.pretrained_converter_path))



    # Freeze vae and text_encoder and set unet to trainable.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # only make the unet-trainable
    # unet.requires_grad_(False)
    converter_network.requires_grad_(False) # train the converter
    converter_network.eval()


    # get the dataloder
    with accelerator.main_process_first():
        (train_loader,test_loader), num_batches_per_epoch = prepare_dataset(
                datapath=args.dataset_path,
                trainlist=args.trainlist,
                vallist=args.vallist,
                batch_size=1,
                logger=logger,
                test_size=1,
                datathread=args.dataloader_num_workers,
                target_resolution=(512,512),
                use_foreground=True)
        logger.info("Loaded the DataLoader....") # only the main process show the logs


    # Prepare everything with our `accelerator`.
    unet, train_loader, test_loader,converter_network = accelerator.prepare(
        unet, train_loader, test_loader,converter_network
    )

    # scale factor.
    rgb_latent_scale_factor = 0.18215

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)



    with torch.no_grad():
        for step, batch in enumerate(test_loader):


            '''Real Image '''
            image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
            mask = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")


            image = np.array(image).astype(np.float32)/255
            mask = np.array(mask).astype(np.float32)
            mask = mask>127
            mask = mask.astype(np.float32)
            mask = mask[:,:,0]
            prompt = ["mountain in the wild."]
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            image =  torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

            fg_image = mask * image
            fg_mask = mask

            fg_image = fg_image.cuda()
            image = image.cuda()
            fg_mask = fg_mask.cuda()


            # # get the images
            # image = batch['image']
            # # saved_image
            # saved_original_image = (image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)


            # # get the background masks
            # bg_mask = batch['bg_mask']
            # # get the text embeddings
            # prompt= batch['prompt']
            # # get the resize foreground image
            # fg_image = batch['fg_image']
            # # get the resize foreground mask
            # fg_mask = batch['fg_mask']

            # # foregound_image
            # fg_for_save = image * (torch.ones_like(bg_mask.float())- bg_mask.float())
            fg_for_save = (fg_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)



            # image normalization
            image = image_normalization(image)
            # foreground image normalization
            fg_image = image_normalization(fg_image)

            # # encode all_image into latent space
            # h_rgb = vae.encoder(image.to(weight_dtype))
            # moments_rgb = vae.quant_conv(h_rgb)
            # mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
            # latents = mean_rgb *rgb_latent_scale_factor    #torch.Size([1, 4, 64, 64])


            # encode all foreground images into latent space.
            fore_h_rgb = vae.encoder(fg_image.to(weight_dtype))
            fore_moments_rgb = vae.quant_conv(fore_h_rgb)
            fore_mean_rgb, fore_logvar_rgb = torch.chunk(fore_moments_rgb, 2, dim=1)
            fore_latents = fore_mean_rgb * rgb_latent_scale_factor #torch.Size([1, 4, 64, 64])

            foreground_latents_mask = F.interpolate(fg_mask,scale_factor=1./8,
                            mode='nearest')
            foreground_latents_mask = torch.clamp(foreground_latents_mask,min=0,max=1.0) #torch.Size([1, 1, 64, 64])



            # Get the text embedding for conditioning
            #[1,77,768]
            prompt_embeds, negative_prompt_embeds = encode_prompt(prompt=prompt,negative_prompt=[""],device=accelerator.device,prompt_embeds=None,tokenizer=tokenizer,text_encoder=text_encoder,unet=unet)
            encoder_hidden_states = prompt_embeds


            # sample a time and then do the "AdaIn" Operation

            # 5. Prepare timesteps
            noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
            timesteps = noise_scheduler.timesteps


            # Initial latents map (Guassian noise)
            latents = torch.randn(
                fore_latents.shape, device=accelerator.device, dtype=fore_latents.dtype
            )  # [B, 4, H/8, W/8]

            if args.use_adIN:
                # get the current time step's foreground statics
                def hacked_mid_forward(self, *args, **kwargs):
                    eps = 1e-6
                    x = self.original_forward(*args, **kwargs)
                    if MODE == "write":
                        # if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append(mean) # add mean
                        self.var_bank.append(var)   # add var
                        self.feature_bank.append(x) # add feature bank
                    if MODE =="read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            # get current mean and current value
                            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            ref_mean = self.mean_bank[0]
                            ref_var = self.var_bank[0]
                            std_ref_var= torch.maximum(ref_var, torch.zeros_like(ref_var) + eps) ** 0.5
                            # adaIN Here
                            x = (((x - mean) / std) * std_ref_var) + ref_mean
                        
                        self.mean_bank = []
                        self.var_bank = []
                        self.feature_bank
                    
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
                            # if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean]) # add mean
                            self.var_bank.append([var])   # add var
                            self.feature_bank.append([hidden_states]) # add hidden feature
                        if MODE =='read':
                            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                                var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                                mean_acc = self.mean_bank[i]
                                var_acc = self.var_bank[i]
                                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                                # perform "AdaIN" Here
                                hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc


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
                            # if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean]) # add mean
                            self.var_bank.append([var])   # add var
                            self.feature_bank.append([hidden_states]) # add feature bank
                        if MODE=='read':
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = self.mean_bank[i]
                            var_acc = self.var_bank[i]
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            # "AdaIN" Here
                            hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc


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
                            # if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean]) # add mean
                            self.var_bank.append([var])   # add var
                            self.feature_bank.append([hidden_states]) # add hidden states
                        elif MODE=='read':
                            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                                var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                                mean_acc = self.mean_bank[i]
                                var_acc = self.var_bank[i]
                                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                                # AdaIN Here
                                hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc

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
                            # if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean]) # add mean
                            self.var_bank.append([var])   # add var
                            self.feature_bank.append([hidden_states]) # add hidden states
                        if MODE =='read':
                            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                                var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                                mean_acc = self.mean_bank[i]
                                var_acc = self.var_bank[i]
                                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                                hidden_states = (((hidden_states - mean) / std) * std_acc) + mean_acc


                    if MODE == "read":
                        self.mean_bank = []
                        self.var_bank = []

                    if self.upsamplers is not None:
                        for upsampler in self.upsamplers:
                            hidden_states = upsampler(hidden_states, upsample_size)

                    return hidden_states



                    pass
                
                # get the foreground statics
                # all the mid bloacks
                gn_modules = [unet.mid_block]
                # # 归一化的强度 gn weight
                # unet.mid_block.gn_weight = 0
                # 这种递减的 gn_weight 允许模型在下采样过程中对前几层进行更强的归一化，而对后几层的归一化处理逐渐减弱。
                down_blocks = unet.down_blocks
                for w, module in enumerate(down_blocks):
                    # module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                    gn_modules.append(module)

                #通过遍历 U-Net 的上采样块（up_blocks），代码为每个块设置了一个递增的 gn_weight，从 0.0 逐渐增加到接近 1.0。
                # 这种策略允许模型在上采样的不同阶段应用不同强度的归一化，在高分辨率输出的生成过程中平衡特征的恢复和细节的增强。
                up_blocks = unet.up_blocks
                for w, module in enumerate(up_blocks):
                    # module.gn_weight = float(w) / float(len(up_blocks))
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

                show_pbar = True
                if show_pbar:
                    iterable = tqdm(
                        enumerate(timesteps),
                        total=len(timesteps),
                        leave=False,
                        desc=" " * 4 + "Diffusion denoising",
                    )
                else:
                    iterable = enumerate(timesteps)
                
                for inner_i, t in iterable:

                    # write the current rf-attentions into forward functions.
                    MODE = "write"
                    unet(
                        fore_latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    # get all the feature banks / mean banks/ feature banks
                    mean_bank_fore,variance_bank_fore,feature_bank_fore = get_all_mean_and_var_and_hidden_states(gn_modules=gn_modules)
                    t_embed = position_encoding(t)
                    # use the network here
                    fg2all_mean_bank, fg2all_variance_bank = converter_network(mean_bank = mean_bank_fore,
                                    var_bank=variance_bank_fore,
                                    feat_bank=feature_bank_fore,
                                    time_embed=t_embed,
                                    text_embed=encoder_hidden_states,
                                    foreground_mask=fg_mask)
                    # update the store mean and variance
                    update_all_mean_and_var(gn_modules=gn_modules,fg2all_mean_bank=fg2all_mean_bank,fg2all_variance_bank=fg2all_variance_bank)
                    
                    MODE = "read"
                    noise_pred = unet(
                        latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            else:
                show_pbar = True
                if show_pbar:
                    iterable = tqdm(
                        enumerate(timesteps),
                        total=len(timesteps),
                        leave=False,
                        desc=" " * 4 + "Diffusion denoising",
                    )
                else:
                    iterable = enumerate(timesteps)

                for i, t in iterable:
                    noise_pred = unet(
                        latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                

            torch.cuda.empty_cache()
            assert latents.shape[0]==1

            # decode to image
            latents = latents.half()
            image_tensor = decode_image(latent=latents,vae=vae)
            image_tensor = image_tensor.float()
            image_tensor = torch.clip(image_tensor, -1.0, 1.0)
            # shift to [0, 1]
            image_tensor = (image_tensor + 1.0) / 2.0



            from_tensor_to_image(image_tensor,os.path.join(args.output_dir,"real_adin.png".format(step)))
            # skimage.io.imsave(os.path.join(args.output_dir,"{}_original.png".format(step)),saved_original_image)
            skimage.io.imsave(os.path.join(args.output_dir,"real_fore.png".format(step)),fg_for_save)
            # if step>10:
            #     quit()
            quit()
                


def decode_image(latent,vae,scale_factor=0.18215) -> torch.Tensor:
    """
    Decode depth latent into depth map.

    Args:
        depth_latent (`torch.Tensor`):
            Depth latent to be decoded.

    Returns:
        `torch.Tensor`: Decoded depth map.
    """
    # scale latent
    latent = latent / scale_factor

    
    # decode
    z = vae.post_quant_conv(latent)
    stacked = vae.decoder(z)

    
    return stacked



if __name__=="__main__":
    inference()