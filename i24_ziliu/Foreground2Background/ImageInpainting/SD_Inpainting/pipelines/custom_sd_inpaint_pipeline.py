import torch
import inspect
import diffusers
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    StableDiffusionMixin,
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models import (
    AsymmetricAutoencoderKL, 
    AutoencoderKL, 
    ImageProjection, 
    UNet2DConditionModel)

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.loaders import (
    FromSingleFileMixin, 
    IPAdapterMixin, 
    StableDiffusionLoraLoaderMixin, 
    TextualInversionLoaderMixin)

from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.utils.torch_utils import randn_tensor
import skimage.io

# ------------------------------------------------------------------#

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
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







class Custom_SD_Inpainting(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,):

    # some setting for GPU/CPU Loading.
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,):
        super().__init__() 

      # Check shapes, assume num_channels_latents == 4, num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 9:
            logger.info(f"You have loaded a UNet with {unet.config.in_channels} input channels which.")


        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        # VAE Scale Factor---> Basically Equals to 8 
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, 
            do_binarize=True, do_convert_grayscale=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # make sure all the inputs are valid.
    def check_inputs(
        self,
        prompt,
        image,
        mask_image,
        height,
        width,
        strength,
        callback_steps,
        output_type,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        padding_mask_crop=None,):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
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
        if padding_mask_crop is not None:
            if not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"The image should be a PIL image when inpainting mask crop, but is of type" f" {type(image)}."
                )
            if not isinstance(mask_image, PIL.Image.Image):
                raise ValueError(
                    f"The mask image should be a PIL image when inpainting mask crop, but is of type"
                    f" {type(mask_image)}."
                )
            if output_type != "pil":
                raise ValueError(f"The output type should be PIL when inpainting mask crop, but is" f" {output_type}.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # encode the text prompt:  negative_prompt_embeds,prompt_embeds,
    # Concated as one tensor.
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,):
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
            **kwargs,)

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds
   
   # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,):
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
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        # prompt can be a string
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        # prompt can be a list
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        # prompt can be direcly as a embedding vetctor
        else:
            batch_size = prompt_embeds.shape[0]

       # -------------------- Get the Input Text Embedding ------------------------#   

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            # convert the text into tokenizer
            # 将输出转换为 PyTorch 张量（tensor），以便与 PyTorch 模型兼容。
            # 输出：text_inputs 是一个字典，包含了分词后的输入张量以及其他相关信息，
            # 其中 text_inputs.input_ids 是分词后的文本ID序列。
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # 生成未截断的文本 ID
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            # 查文本在被截断前后的差异，或者用于需要保留完整输入序列的任务
            
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                # 作用：从未截断的 untruncated_ids 中提取被截断的部分。
                # 具体来说，这段代码提取的是从模型最大长度位置开始到结尾的部分，
                # 因为这些内容在 text_input_ids 中被截断了。 batch_decode：将分词后的 ID 转换回可读的文本
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                # 使用日志系统发出警告，告知用户由于输入文本超过了模型的最大处理能力，
                # 部分文本被截断了。警告信息包括模型能处理的最大 token 数量
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            # 这段代码片段的目的是根据 text_encoder 的配置来决定是否生成并使用注意力掩码（attention mask）。
            # 注意力掩码在处理文本输入时非常重要，
            # 特别是在处理可变长度的序列时，能够帮助模型忽略填充（padding）的部分。
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            # 生成嵌入向量：如果 clip_skip 为 None，则直接通过 text_encoder 生成文本嵌入。text_encoder 接受分词后的文本 ID（text_input_ids）和注意力掩码（attention_mask），然后返回一个嵌入向量。
            # 获取嵌入向量：prompt_embeds[0] 访问返回的嵌入向量列表中的第一个元素，通常是最终的输出嵌入向量
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                # clip_skip 的使用：clip_skip 是一种技术，用于跳过某些层的计算，以减少计算成本或避免过拟合。
                # 它常用于推理阶段，以减少模型的计算量，并加速生成过程。
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
                # 在对模型隐藏层输出进行操作时，使用 LayerNorm 确保输出嵌入的稳定性和一致性是很重要的，这有助于保持生成结果的质量
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        # 作用：获取 prompt_embeds 的形状信息，
        # bs_embed 是批量大小（即有多少个文本提示），
        # seq_len 是序列长度（即每个文本提示有多少个 token），
        # _ 通常代表特征维度（例如每个 token 的嵌入维度）
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # 根据 num_images_per_prompt（每个提示生成的图像数量），对 prompt_embeds 进行复制，使得每个文本提示都有对应数量的嵌入向量。
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # -------------------- Get the Negative (Uncond) Text Embedding ------------------------# 

        # get unconditional embeddings for classifier free guidance
        # 启动CFG 但是如果没有提供 negative Prompt的词嵌入
        # 意思就是如果存在negative prompt，
        # 那么CFG的uncond_tokens就是negative prompt,否则CFG的uncond_tokens就是空的字符串
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                # 没用负样本
                uncond_tokens = [""] * batch_size
            # 这段代码确保 prompt 和 negative_prompt 的类型一致。如果类型不一致，则抛出 TypeError。
            # 要不都是str, 要不都是list
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
            # self.maybe_convert_prompt 是一个用于处理和转换提示词（prompts）的方法。
            # 当 Textual Inversion 使用多个嵌入向量表示一个 token 时，可能需要特别处理这些 tokens。
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

        # DO CFG Here
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # encode the images.
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):

        # 使用 next 的原因是为了快速获取 self.image_encoder 中的第一个参数的 dtype（数据类型）。
        dtype = next(self.image_encoder.parameters()).dtype

        # if not feature extractor
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds


    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    @property
    def clip_skip(self):
        return self._clip_skip
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # inference at here 
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.Tensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,):

        ##------------------------ Begin Logic Here ---------------------------------#
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 0. Default height and width to unet
        # print(self.unet.config.sample_size) default is equals to 64.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            mask_image,
            height,
            width,
            strength,
            callback_steps,
            output_type,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device # CPU or GPU in cuda:0


        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            # [2*batch_size, 77,768]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # using ip adapter for decouple cross attention
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            # TODO Not Implemented yet
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. set timesteps: adaptive settings
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # timesteps is the length of `num_inference_steps` list begin from [1000 to 0]
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # 这行代码从 timesteps 序列中取出第一个时间步长（timesteps[:1]），
        # 并将其复制，数量为 batch_size * num_images_per_prompt。这样做的目的是为每个生成图像的过程（包括批量中的所有图像）设置相同的初始时间步长
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        # process the input image here
        original_image = image
        # normalized from -1.0 to 1.0 after normalization
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        # initial input image
        init_image = init_image.to(dtype=torch.float32) # default is [1,3,512,512]

       # 6. Prepare latent variables
        # VAE Input Channels is 4. 
        num_channels_latents = self.vae.config.latent_channels # 4
        num_channels_unet = self.unet.config.in_channels # 9
        return_image_latents = num_channels_unet == 4 # Fasle

        # latents
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        ) #print(latents_outputs[0].shape) # already the [1,4,64,64] resolution.and

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )   #[1,1,512,512]

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5) # mask out image is the foreground characture
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        ) # [2,1,H,W],[2,4,H,W] 

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1] # 1
            num_channels_masked_image = masked_image_latents.shape[1] # 4
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 9.2 Optionally get Guidance Scale Embedding:defalt is the nan.
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)


        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # if self.interrupt:
                #     continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                #cont togther
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        print("Denosing is Finished........")

        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
            )[0]

        image = torch.clip(image , -1.0, 1.0)
        # shift to [0, 1]
        image  = (image  + 1.0) / 2.0

        returned_image_list = []
        nums_of_images = image.shape[0]
        for idx in range(nums_of_images):
            current_saved_image = (image[idx].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            returned_image_list.append(current_saved_image)
        # Offload all models
        self.maybe_free_model_hooks()

        return returned_image_list


    # get the image lantents
    def prepare_latents(
        self,batch_size,
        num_channels_latents,
        height,width,dtype,device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,):
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
            # 在扩散模型中，latents 的初始值可以是纯噪声，也可以是输入图像加上噪声。这两种初始化方式取决于生成任务的具体需求和 strength 参数的设置
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

    # get the mask latents
    def prepare_mask_latents(
        self, 
        mask, 
        masked_image, 
        batch_size, height, width, dtype, device, generator, do_classifier_free_guidance):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        # 首先将输入的蒙版（mask）调整到与潜在表示相同的尺寸。这是通过 torch.nn.functional.interpolate 实现的，
        # 该函数可以调整图像的尺寸。调整后的尺寸基于图像的高度和宽度，并使用 vae_scale_factor 进行缩放。
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # here the image that saveds
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

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
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

    def run_safety_checker(self, image, device, dtype):
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

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, 
    sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
def saved_generated_images(saved_folder,
                            image_list,
                            init_image,
                            mask_image):
    os.makedirs(saved_folder,exist_ok=True)

    for idx in tqdm(range(len(image_list))):
        saved_name = os.path.join(saved_folder,"{}.png".format(idx))

        image = image_list[idx]
        initial_saved_image = Image.fromarray(image.astype(np.uint8))
        initial_saved_image.save(saved_name)


    


import os
from tqdm import tqdm
if __name__=="__main__":

    # diffusers.utils.logging.set_verbosity_info()
    diffusers.utils.logging.set_verbosity_error()
    model_path = "runwayml/stable-diffusion-inpainting"
    # loaded the unet 
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        variant="fp16",  # 指定使用的变体，这里选择 "fp16"
        torch_dtype=torch.float16,  # 确保数据类型匹配
        filename="diffusion_pytorch_model.fp16.bin"  # 明确指定 .bin 文件
    )
    # loaded the vae
    pretrained_vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        variant="fp16",  # 同样地，指定 "fp16" 变体
        torch_dtype=torch.float16,
        filename="diffusion_pytorch_model.fp16.bin"  # 明确指定 .bin 文件
    )

    # Load the text encoder
    pretrained_text_encoder = CLIPTextModel.from_pretrained(model_path,
                                            subfolder="text_encoder",
                                            variant="fp16",  # 同样地，指定 "fp16" 变体
                                            torch_dtype=torch.float16
                                            )

    # Load the tokenizer
    pretrained_tokenizer = CLIPTokenizer.from_pretrained(model_path,
                                                    subfolder="tokenizer")

    # loaded the scheduler
    predefined_scheduler = DDIMScheduler.from_pretrained(model_path,
                                                            subfolder="scheduler")
    # loaded the feature extractor
    pretrained_feature_extractor = CLIPImageProcessor.from_pretrained(model_path,
                                                    subfolder="feature_extractor")

    # 
    predefined_safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path,
                                                    subfolder="safety_checker")



    pfn_custom_pipeline = PFN_Inpainting_Pipeline(vae=pretrained_vae,
                        text_encoder=pretrained_text_encoder,
                        tokenizer=pretrained_tokenizer,
                        unet=pretrained_unet,
                        scheduler=predefined_scheduler,
                        safety_checker=predefined_safety_checker,
                        feature_extractor=pretrained_feature_extractor,
                        image_encoder=None,
                        requires_safety_checker=False
    )   
    
    pipeline = pfn_custom_pipeline.to("cuda")  # 如果有GPU，切换到GPU加速
    pipeline.enable_model_cpu_offload()
    print("loaded the pretrained pipeline")


    # inputs images
    init_image = Image.open("/mnt/nfs-mnj-home-43/i24_ziliu/Foreground2Background/AnimeSegmentation/anime-segmentation/out/000001.png").convert("RGB")
    mask_image = Image.open("/mnt/nfs-mnj-home-43/i24_ziliu/Foreground2Background/AnimeSegmentation/anime-segmentation/out/background_mask.png").convert("RGB")    
    original_size = init_image.size
    
    # image_size =512
    # init_image = init_image.resize((image_size,image_size))
    # mask_image = mask_image.resize((image_size,image_size))

    generator = torch.Generator("cuda").manual_seed(92)
    prompt = "A girl sitting in front of Tokyo Tower."


    # reference from here
    #https://civitai.com/articles/3885/negative-prompts
    negative_prompt = "bad quality, low resolution,Disharmonious ,Deformed, Out of frame, Poorly drawn, bad anatomy, deformed, ugly, disfigured"
    

    personalized_guidance_score = 15

    image_list = pipeline(prompt=prompt, 
                        negative_prompt= negative_prompt,
                        image=init_image, 
                        strength=0.999,
                        mask_image=mask_image, generator=generator,
                        num_images_per_prompt=5,
                        guidance_scale=personalized_guidance_score,
                        height=int(original_size[1]/8)*8,
                        width=original_size[0]
                        
                        )

    
    image = image_list[0]
    saved_generated_images(saved_folder="results/new_results/sd15_non/SD_no_Strength0999".
                                        format(personalized_guidance_score),
                                                image_list=image_list,
                                                init_image = init_image,
                                                mask_image=mask_image)
