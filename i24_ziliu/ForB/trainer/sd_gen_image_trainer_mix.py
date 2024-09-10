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

from trainer.dataset_configurationV2 import prepare_dataset,image_denormalization,image_normalization
# from trainer.dataset_configuration import prepare_dataset,image_normalization,image_denormalization
check_min_version("0.26.0.dev0")
import skimage.io
logger = get_logger(__name__, log_level="INFO")
import  matplotlib.pyplot as plt
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
# from ForB.networks.f2all_converter import F2All_Converter
# from ForB.networks.f2all_converter_tpm import F2All_Converter
from ForB.networks.f2all_conveter_ver2 import F2All_Converter
from ForB.losses.layer_wise_l1_loss import LayerWise_L1_Loss


def parse_args():
    parser = argparse.ArgumentParser(description="Foregroud2Background")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
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
        "--pretrained_VAE_single_file",
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
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ))
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.")
    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    
    parser.add_argument("--num_train_epochs", type=int, default=70)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # using xformers for efficient training 
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # noise offset?::: #TODO HERE
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # validations every 5 Epochs
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )

    parser.add_argument('--use_adIN', action='store_true', help="Increase output verbosity")

    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
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


def main():
    ''' ------------------------Configs Preparation----------------------------'''
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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

    '''------------------------------- Models Part Configuration ---------------------------------------'''
    if args.pretrained_single_file!="none":
        current_stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                                            pretrained_model_link_or_path=args.pretrained_single_file)
        logger.info("----------------------------------------------------------------------------------",main_process_only=True)
        logger.info("Training From Single File!!!, Files is {}".format(args.pretrained_single_file),main_process_only=True)
        logger.info("----------------------------------------------------------------------------------",main_process_only=True)
    else:
        current_stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                        safety_checker=None,
        )
    
    if args.pretrained_VAE_single_file!='none':
        pretrained_vae = AutoencoderKL.from_single_file(pretrained_model_link_or_path_or_dict=args.pretrained_VAE_single_file)
        current_stable_diffusion_model.vae = pretrained_vae # update the VAE
        
        logger.info("----------------------------------------------------------------------------------",main_process_only=True)
        logger.info("VAE From Single File!!!, Files is {}".format(args.pretrained_VAE_single_file),main_process_only=True)
        logger.info("----------------------------------------------------------------------------------",main_process_only=True)

    
    # get the tokenizer/text_encoder/unet/scheduler
    noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2",subfolder='scheduler')
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

    # Freeze vae and text_encoder and set unet to trainable.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # only make the unet-trainable
    # unet.requires_grad_(False)
    converter_network.train() # train the converter

    # using xformers for efficient attentions.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    #FIXME: define how to save model
    # using checkpint  for saving the memories
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    #FIXME : Fixed Here for Optimizer Definition.
    #optimizer settings 
    optimizer = optimizer_cls(
        converter_network.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        (train_loader,test_loader), num_batches_per_epoch = prepare_dataset(
                datapath=args.dataset_path,
                trainlist=args.trainlist,
                vallist=args.vallist,
                batch_size=args.train_batch_size,
                logger=logger,
                test_size=1,
                datathread=args.dataloader_num_workers,
                target_resolution=(512,512),
                use_foreground=True)
        logger.info("Loaded the DataLoader....") # only the main process show the logs


    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_loader, test_loader,lr_scheduler,converter_network = accelerator.prepare(
        unet, optimizer, train_loader, test_loader,lr_scheduler,converter_network
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
    # unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        converter_network.train() # FIXME: Specific the Model we want to fixed.
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(converter_network): # FIXME:Specific the Model we want to train
                # get the images
                image = batch['image']
                # get the background masks
                bg_mask = batch['bg_mask']
                # get the text embeddings
                prompt= batch['prompt']
                # get the resize foreground image
                fg_image = batch['fg_image']
                # get the resize foreground mask
                fg_mask = batch['fg_mask']
                # image normalization
                image = image_normalization(image)
                # foreground image normalization
                fg_image = image_normalization(fg_image)

                # encode all_image into latent space
                h_rgb = vae.encoder(image.to(weight_dtype))
                moments_rgb = vae.quant_conv(h_rgb)
                mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                latents = mean_rgb *rgb_latent_scale_factor    #torch.Size([1, 4, 64, 64])


                # encode all foreground images into latent space.
                fore_h_rgb = vae.encoder(fg_image.to(weight_dtype))
                fore_moments_rgb = vae.quant_conv(fore_h_rgb)
                fore_mean_rgb, fore_logvar_rgb = torch.chunk(fore_moments_rgb, 2, dim=1)
                fore_latents = fore_mean_rgb * rgb_latent_scale_factor #torch.Size([1, 4, 64, 64])

                foreground_latents_mask = F.interpolate(fg_mask,scale_factor=1./8,
                                mode='nearest')
                foreground_latents_mask = torch.clamp(foreground_latents_mask,min=0,max=1.0) #torch.Size([1, 1, 64, 64])
                


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                #[1,77,768]
                prompt_embeds, negative_prompt_embeds = encode_prompt(prompt=prompt,negative_prompt=["" for _ in range(args.train_batch_size)],device=accelerator.device,prompt_embeds=None,tokenizer=tokenizer,text_encoder=text_encoder,unet=unet)
                encoder_hidden_states = prompt_embeds
                
                noise_scheduler.config.prediction_type = "epsilon"

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(disp_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                if args.use_adIN:
                    # gn_auto_machine_weight=2.0
                    # style_fidelity = 0.5
                    # do_classifier_free_guidance=False
                    def hacked_mid_forward(self, *args, **kwargs):
                        eps = 1e-6
                        x = self.original_forward(*args, **kwargs)
                        if MODE == "write":
                            # if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append(mean) # add mean
                            self.var_bank.append(var)   # add var
                            self.feature_bank.append(x) # add feature bank
                        
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
 
                            output_states = output_states + (hidden_states,)

                        
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

                            output_states = output_states + (hidden_states,)
                        
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

                        if self.upsamplers is not None:
                            for upsampler in self.upsamplers:
                                hidden_states = upsampler(hidden_states, upsample_size)

                        return hidden_states


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


                # write the current rf-attentions into forward functions.
                MODE = "write"
                unet(
                    fore_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                # get all the feature banks / mean banks/ feature banks
                mean_bank_fore,variance_bank_fore,feature_bank_fore = get_all_mean_and_var_and_hidden_states(gn_modules=gn_modules)
                t_embed = position_encoding(timesteps,device=encoder_hidden_states.device)
                # use the network here
                fg2all_mean_bank, fg2all_variance_bank = converter_network(mean_bank = mean_bank_fore,
                                var_bank=variance_bank_fore,
                                feat_bank=feature_bank_fore,
                                time_embed=t_embed,
                                text_embed=encoder_hidden_states,
                                foreground_mask=fg_mask
                                )
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                mean_bank_all,variance_bank_all,feature_bank_all = get_all_mean_and_var_and_hidden_states(gn_modules=gn_modules)
                
                # est_mean_bank,est_variance_bank,gt_mean_bank,gt_variance_bank
                mean_loss,var_loss = LayerWise_L1_Loss(est_mean_bank=fg2all_mean_bank,est_variance_bank=fg2all_variance_bank,
                                    gt_mean_bank=mean_bank_all,gt_variance_bank=variance_bank_all)

                loss = mean_loss + var_loss
                loss = loss.float()

                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                 # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # currently the EMA is not used.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwarped_model = accelerator.unwrap_model(converter_network)
                        unwarped_optimizer = accelerator.unwrap_model(optimizer)
                        unwarped_lr = accelerator.unwrap_model(lr_scheduler)
                        torch.save(
                            {"model_state":unwarped_model.state_dict(),
                            'optim_state': unwarped_optimizer.state_dict(),
                            'lr_state': unwarped_lr.state_dict()
                            },
                            os.path.join(args.output_dir,f"ckpt_{global_step+1}.pt")
                            # model_config['saved_path'] + f'ckpt_{epoch+1}.pt'
                        )
                        logger.info(f'checkpoint ckpt_{global_step+1}.pt is saved...')

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    accelerator.end_training()

                    
if __name__=="__main__":
    main()