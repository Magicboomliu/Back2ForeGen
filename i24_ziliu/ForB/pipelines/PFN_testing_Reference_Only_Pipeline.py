# Inspired by: https://github.com/Mikubill/sd-webui-controlnet/discussions/1236 and https://github.com/Mikubill/sd-webui-controlnet/discussions/1280
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict, deprecate
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor

import sys
import os
sys.path.append("../..")
from ForB.networks.f2all_conveter_ver2 import F2All_Converter
import torch
import torch.nn as nn
import torch.nn.functional as F

from ForB.networks.learnable_reference_only import ConverterNetwork


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




# 这个 torch_dfs 函数通过深度优先搜索（DFS）的方式遍历给定的 PyTorch 模型，并返回该模型及其所有子模块的列表。

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


# get all_mean and var
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

def update_mean_and_variane(gn_modules,new_mean_bank,new_var_bank):
    for i, module in enumerate(gn_modules):
        if getattr(module, "original_forward", None) is None:
            module.original_forward = module.forward
        
        if i == 0:
            # mid_block
            module.mean_bank = new_mean_bank[('mid_block',i)]
            module.var_bank = new_var_bank[('mid_block',i)]
            
            module.feature_bank = []


        elif isinstance(module, CrossAttnDownBlock2D):

            module.mean_bank = new_mean_bank[('CrossAttnDownBlock2D',i)]
            module.var_bank = new_var_bank[('CrossAttnDownBlock2D',i)]
            module.feature_bank = []


        elif isinstance(module, DownBlock2D):
            module.mean_bank = new_mean_bank[('DownBlock2D',i)]
            module.var_bank = new_var_bank[('DownBlock2D',i)]
            module.feature_bank = []

        elif isinstance(module, CrossAttnUpBlock2D):
            module.mean_bank = new_mean_bank[('CrossAttnUpBlock2D',i)]
            module.var_bank = new_var_bank[('CrossAttnUpBlock2D',i)]
            module.feature_bank = []



        elif isinstance(module, UpBlock2D):
            module.mean_bank = new_mean_bank[('UpBlock2D',i)]
            module.var_bank = new_var_bank[('UpBlock2D',i)]
            module.feature_bank = []
# pass
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


def get_all_attn_states(attn_modules):
    
    all_attn_banks = []
    for i, module in enumerate(attn_modules):
        all_attn_banks.append(module.attn_bank)
        module.attn_bank = []
    
    return all_attn_banks


def update_all_attn_states(attn_modules,attn_list):
    for i, module in enumerate(attn_modules):
        module.attn_bank = [attn_list[i]]



class PFN_Text2Image_Reference_Only_Pipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    r"""
    Pipeline for Stable Diffusion Reference.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
    - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
    - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
    - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
    - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
    - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "skip_prk_steps not set",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        # Check shapes, assume num_channels_latents == 4, num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 4:
            logger.warning(
                f"You have loaded a UNet with {unet.config.in_channels} input channels, whereas by default,"
                f" {self.__class__} assumes that `pipeline.unet` has 4 input channels: 4 for `num_channels_latents`,"
                ". If you did not intend to modify"
                " this behavior, please check whether you have loaded the right checkpoint."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    #这个函数 _default_height_width 用于计算给定图像的默认高度和宽度。
    # 如果用户没有指定高度或宽度，
    # 则该函数根据输入图像的大小自动确定这些值，
    # 并将它们调整为 8 的倍数（这通常用于确保图像的尺寸与某些深度学习模型的输入要求兼容）。
    def _default_height_width(
        self,
        height: Optional[int],
        width: Optional[int],
        image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image]],
    ) -> Tuple[int, int]:
        r"""
        Calculate the default height and width for the given image.

        Args:
            height (int or None): The desired height of the image. If None, the height will be determined based on the input image.
            width (int or None): The desired width of the image. If None, the width will be determined based on the input image.
            image (PIL.Image.Image or torch.Tensor or list[PIL.Image.Image]): The input image or a list of images.

        Returns:
            Tuple[int, int]: A tuple containing the calculated height and width.

        """
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        callback_steps: Optional[int],
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[torch.Tensor] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ) -> None:
        """
        Check the validity of the input arguments for the diffusion model.

        Args:
            prompt (Optional[Union[str, List[str]]]): The prompt text or list of prompt texts.
            height (int): The height of the input image.
            width (int): The width of the input image.
            callback_steps (Optional[int]): The number of steps to perform the callback on.
            negative_prompt (Optional[str]): The negative prompt text.
            prompt_embeds (Optional[torch.Tensor]): The prompt embeddings.
            negative_prompt_embeds (Optional[torch.Tensor]): The negative prompt embeddings.
            ip_adapter_image (Optional[torch.Tensor]): The input adapter image.
            ip_adapter_image_embeds (Optional[torch.Tensor]): The input adapter image embeddings.
            callback_on_step_end_tensor_inputs (Optional[List[str]]): The list of tensor inputs to perform the callback on.

        Raises:
            ValueError: If `height` or `width` is not divisible by 8.
            ValueError: If `callback_steps` is not a positive integer.
            ValueError: If `callback_on_step_end_tensor_inputs` contains invalid tensor inputs.
            ValueError: If both `prompt` and `prompt_embeds` are provided.
            ValueError: If neither `prompt` nor `prompt_embeds` are provided.
            ValueError: If `prompt` is not of type `str` or `list`.
            ValueError: If both `negative_prompt` and `negative_prompt_embeds` are provided.
            ValueError: If both `prompt_embeds` and `negative_prompt_embeds` are provided and have different shapes.
            ValueError: If both `ip_adapter_image` and `ip_adapter_image_embeds` are provided.

        Returns:
            None
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Encodes the prompt into embeddings.

        Args:
            prompt (Union[str, List[str]]): The prompt text or a list of prompt texts.
            device (torch.device): The device to use for encoding.
            num_images_per_prompt (int): The number of images per prompt.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.
            negative_prompt (Optional[Union[str, List[str]]], optional): The negative prompt text or a list of negative prompt texts. Defaults to None.
            prompt_embeds (Optional[torch.Tensor], optional): The prompt embeddings. Defaults to None.
            negative_prompt_embeds (Optional[torch.Tensor], optional): The negative prompt embeddings. Defaults to None.
            lora_scale (Optional[float], optional): The LoRA scale. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The encoded prompt embeddings.
        """
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        if do_classifier_free_guidance:
            # concatenate for backwards comp
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
        else:
            prompt_embeds = prompt_embeds_tuple[0]


        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Optional[str],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it


        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
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

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
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

        # if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
        #     # Retrieve the original scale by scaling back the LoRA layers
        #     unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Prepare the latent vectors for diffusion.

        Args:
            batch_size (int): The number of samples in the batch.
            num_channels_latents (int): The number of channels in the latent vectors.
            height (int): The height of the latent vectors.
            width (int): The width of the latent vectors.
            dtype (torch.dtype): The data type of the latent vectors.
            device (torch.device): The device to place the latent vectors on.
            generator (Union[torch.Generator, List[torch.Generator]]): The generator(s) to use for random number generation.
            latents (Optional[torch.Tensor]): The pre-existing latent vectors. If None, new latent vectors will be generated.

        Returns:
            torch.Tensor: The prepared latent vectors.
        """
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

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(
        self, generator: Union[torch.Generator, List[torch.Generator]], eta: float
    ) -> Dict[str, Any]:
        r"""
        Prepare extra keyword arguments for the scheduler step.

        Args:
            generator (Union[torch.Generator, List[torch.Generator]]): The generator used for sampling.
            eta (float): The value of eta (η) used with the DDIMScheduler. Should be between 0 and 1.

        Returns:
            Dict[str, Any]: A dictionary containing the extra keyword arguments for the scheduler step.
        """
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_image(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[Union[torch.Tensor, PIL.Image.Image]]],
        width: int,
        height: int,
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        do_classifier_free_guidance: bool = False,
        guess_mode: bool = False,
    ) -> torch.Tensor:
        r"""
        Prepares the input image for processing.

        Args:
            image (torch.Tensor or PIL.Image.Image or list): The input image(s).
            width (int): The desired width of the image.
            height (int): The desired height of the image.
            batch_size (int): The batch size for processing.
            num_images_per_prompt (int): The number of images per prompt.
            device (torch.device): The device to use for processing.
            dtype (torch.dtype): The data type of the image.
            do_classifier_free_guidance (bool, optional): Whether to perform classifier-free guidance. Defaults to False.
            guess_mode (bool, optional): Whether to use guess mode. Defaults to False.

        Returns:
            torch.Tensor: The prepared image for processing.
        """
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    
    # This is the main 
    def prepare_ref_latents(
        self,
        refimage: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[int, List[int]],
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        r"""
        Prepares reference latents for generating images.

        Args:
            refimage (torch.Tensor): The reference image.
            batch_size (int): The desired batch size.
            dtype (torch.dtype): The data type of the tensors.
            device (torch.device): The device to perform computations on.
            generator (int or list): The generator index or a list of generator indices.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.

        Returns:
            torch.Tensor: The prepared reference latents.
        """
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(
        self, image: Union[torch.Tensor, PIL.Image.Image], device: torch.device, dtype: torch.dtype
    ) -> Tuple[Union[torch.Tensor, PIL.Image.Image], Optional[bool]]:
        r"""
        Runs the safety checker on the given image.

        Args:
            image (Union[torch.Tensor, PIL.Image.Image]): The input image to be checked.
            device (torch.device): The device to run the safety checker on.
            dtype (torch.dtype): The data type of the input image.

        Returns:
            (image, has_nsfw_concept) Tuple[Union[torch.Tensor, PIL.Image.Image], Optional[bool]]: A tuple containing the processed image and
            a boolean indicating whether the image has a NSFW (Not Safe for Work) concept.
        """
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept


    # Load the converter model's pre-trained weight
    def _load_converter_pre_train(self,pretrained_path):
        # pretrained path
        saved_contents = torch.load(pretrained_path)
        # define the converter
        self.converter_network = ConverterNetwork()
        # load the converter's pertrained weight
        self.converter_network.load_state_dict(saved_contents["model_state"])
        self.converter_network.cuda()
        self.converter_network.half()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        original_image = None,
        original_foreground_mask = None,
        ref_image: Union[torch.Tensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 0.5,
        reference_attn: bool = True,
        reference_adain: bool = True,
        fg_mask = None,
        use_converter=False,
        pretrained_model_path=None,
    ):

        if use_converter:
            gn_auto_machine_weight=10000000

        # load the foreground mask
        fg_mask = torch.from_numpy(fg_mask) # (H,W,1)
        fg_mask = fg_mask.permute(2,0,1).unsqueeze(0)
        fg_mask = fg_mask.float()
        fg_mask = fg_mask.cuda()

        self._load_converter_pre_train(pretrained_model_path)
        print("Loaded the Pretrained Converter Successfully....")
        
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, ref_image) # default is 512,512
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0  # default is 7.5
        do_classifier_free_guidance = True
        

  
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=None,
        )



        ref_image = self.prepare_image(
            image=ref_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )


        original_image = self.prepare_image(
            image=original_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )

        

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) # pure guassin noise, or input latents, which is pure guassian noise.


        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        ) #[1,4,H/8,W/8]

        # 7.1 Prepare the Original latents : using VAE
        original_image_latents = self.prepare_ref_latents(
            original_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        ) #[1,4,H/8,W/8]

        # foreground mask 
        original_foreground_mask = torch.from_numpy(original_foreground_mask).permute(2,0,1).unsqueeze(0)
        resize_original_foreground_mask = F.interpolate(original_foreground_mask,scale_factor=1./8,
                                                        mode='nearest').type_as(ref_image_latents)



        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) # here is the NULL        
        # 9. Modify self attention and group norm
        MODE = "write"
        uc_mask = (
            torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt)
            .type_as(ref_image_latents)
            .bool()
        )


 
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,):  
            
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


                if MODE == "read":
                    # maybe alayws true: just add the

                    if attention_auto_machine_weight > self.attn_weight:
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=torch.cat([norm_hidden_states] + self.attn_bank, dim=1),
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        #[2,4096,320]: 1:c,2: Noc
                        # print(attn_output_uc.shape)
                        # quit()
                        attn_output_c = attn_output_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            # select the uncond regions
                            attn_output_c[uc_mask] = self.attn1(
                                norm_hidden_states[uc_mask],
                                encoder_hidden_states=norm_hidden_states[uc_mask],
                                **cross_attention_kwargs,
                            )
                        attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                        self.attn_bank.clear()
                    else:
                        
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
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
                    x_uc = (((x - mean) / std) * std_acc) + mean_acc
                    x_c = x_uc.clone()
                    if do_classifier_free_guidance and style_fidelity > 0:
                        x_c[uc_mask] = x[uc_mask]
                    x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
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
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

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
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

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
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

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
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        # reference attn
        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.attn_bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

        # reference adaIN
        if reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0
            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)


            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)
            
            

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
                    
                module.mean_bank = []
                module.var_bank = []
                module.feature_bank = []
                module.gn_weight *= 2


        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ref only part
                noise = randn_tensor(
                    ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                )
                
                # this is the reference
                ref_xt = self.scheduler.add_noise(
                    ref_image_latents,
                    noise,
                    t.reshape(
                        1,
                    ),
                )
                ref_xt = torch.cat([ref_xt] * 2) if do_classifier_free_guidance else ref_xt
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)  # already two, but all contains.

                
                # write the current rf-attentions into forward functions.
                MODE = "write"
                self.unet(
                    ref_xt,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )
                
                if use_converter:
                    # get input information
                    if reference_adain:
                        mean_bank_fore,variance_bank_fore,feature_bank_fore = get_all_mean_and_var_and_hidden_states(gn_modules=gn_modules)
                    
                    if reference_attn:
                        fg_attn_banks = get_all_attn_states(attn_modules)
                        fg_attn_banks_list = [fg[0] for fg in fg_attn_banks]
                    
                    
                    t_embed = position_encoding(t)
                    # t_embed = t_embed.repeat(2,1)
                    t_embed = t_embed.half()
                    fg_mask = fg_mask.half()
                    fg_mask = fg_mask[0:1,:,:,:]
                    t_embed = t_embed[0:1,:]
                    # use the network here
                    if do_classifier_free_guidance:
                        fg_mask  = fg_mask.repeat(2,1,1,1)
                        # t_embed = t_embed.repeat(2,1)

                    
                    if reference_adain and not reference_attn:
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
                    if reference_attn and not reference_adain:
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
                    if reference_adain and reference_attn:
                        
                        fg2all_mean_bank, fg2all_variance_bank,converted_fg_attn_banks_list = self.converter_network(mean_bank = mean_bank_fore,
                                        var_bank=variance_bank_fore,
                                        feat_bank=feature_bank_fore,
                                        time_embed=t_embed,
                                        text_embed=prompt_embeds,
                                        foreground_mask=fg_mask,
                                        inputs = fg_attn_banks_list,
                                        use_seperate = 'all'
                                        )
                    
                    if reference_adain:
                        update_mean_and_variane_gn_auto(gn_modules=gn_modules,new_mean_bank=fg2all_mean_bank,
                                                new_var_bank=fg2all_variance_bank)
                    
                    
                    if reference_attn:    
                        update_all_attn_states(attn_modules=attn_modules,attn_list=fg_attn_banks_list)



                # predict the noise residual
                MODE = "read"
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                '''Implement the inpainting logic here'''
                # perform the inpainting Here
                init_latents_proper = original_image_latents
                # add noise 
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )
                
                latents = resize_original_foreground_mask * init_latents_proper + (1-resize_original_foreground_mask) * latents
                
       
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)