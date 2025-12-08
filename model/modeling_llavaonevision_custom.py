from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from transformers.utils import logging
from transformers.processing_utils import Unpack, TensorType
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


from transformers.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessor, LlavaOnevisionProcessorKwargs
from transformers.models.llava_onevision.image_processing_llava_onevision import (
    divide_to_patches, get_size_dict, make_flat_list_of_images, valid_images, validate_preprocess_arguments,
    convert_to_rgb, is_scaled_image, infer_channel_dimension_format
)
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array, PILImageResampling, ChannelDimension
from transformers.video_utils import VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_processing_utils import select_best_resolution
from transformers.image_transforms import resize, to_channel_dimension_format

from transformers.modeling_flash_attention_utils import (
    flash_attn_supports_top_left_mask, _flash_attention_forward,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionCausalLMOutputWithPast, KwargsForCausalLM,
    LlavaOnevisionModel, LlavaOnevisionModelOutputWithPast, FlashAttentionKwargs, 
    image_size_to_num_patches, get_anyres_image_grid_shape, unpad_image,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb, repeat_kv, Qwen2Model, Qwen2DecoderLayer, Qwen2Attention, Qwen2Config,
)


logger = logging.get_logger(__name__)


"""
Here, we modify the code to remove grid tokens.
"""

def llava_imageprocessor_preprocess(
    self,
    images: ImageInput,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    image_grid_pinpoints: Optional[List] = None,
    resample: PILImageResampling = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    do_convert_rgb: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    do_resize = do_resize if do_resize is not None else self.do_resize
    size = size if size is not None else self.size
    size = get_size_dict(size, default_to_square=False)
    image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
    resample = resample if resample is not None else self.resample
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else self.do_normalize
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    do_pad = do_pad if do_pad is not None else self.do_pad
    do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

    images = make_flat_list_of_images(images)

    if not valid_images(images):
        raise ValueError(
            "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
            "torch.Tensor, tf.Tensor or jax.ndarray."
        )

    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )

    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    # All transformations expect numpy arrays.
    images = [to_numpy_array(image) for image in images]

    if do_rescale and is_scaled_image(images[0]):
        logger.warning_once(
            "It looks like you are trying to rescale already rescaled images. If the input"
            " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        )

    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(images[0])

    new_images, num_tiles_hw_batch = [], []
    image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
    for image in images:
        # convert image into a list of patches
        # we intentionally use the same data format as the input data format
        size_tuple = (
            (size["height"], size["width"])
            if "height" in size and "width" in size
            else (size["shortest_edge"], size["shortest_edge"])
        )
        image_patches, num_tiles_hw = self.get_image_patches(
            image,
            image_grid_pinpoints,
            size=size_tuple,
            patch_size=size_tuple[0],
            resample=resample,
            data_format=input_data_format,
            input_data_format=input_data_format,
        )

        # preprocess patches
        pixel_values = self._preprocess(
            image_patches,
            do_resize=do_resize,
            size=size_tuple,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        pixel_values = np.array(pixel_values)
        new_images.append(pixel_values)
        num_tiles_hw_batch.append(num_tiles_hw)
    
    if do_pad:
        processed_images = self._pad_for_batching(new_images)

    return BatchFeature(
        data={"pixel_values": processed_images, "image_sizes": image_sizes, "num_tiles_hw": num_tiles_hw_batch}, tensor_type=return_tensors
    )

def llavaov_processor_call(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    audio=None,
    videos: VideoInput = None,
    **kwargs: Unpack[LlavaOnevisionProcessorKwargs],
) -> BatchFeature:

    output_kwargs = self._merge_kwargs(
        LlavaOnevisionProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )

    if isinstance(text, str):
        text = [text]
    elif not isinstance(text, list) and not isinstance(text[0], str):
        raise ValueError("Invalid input text. Please provide a string, or a list of strings")

    image_inputs = video_inputs = {}

    if images is not None:
        ##### Start (JS): assume without grid_pinpoints - only single scale
        # image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        num_tiles_hw = image_inputs.pop("num_tiles_hw", None) # TODO. _get_unpadded_features
        assert num_tiles_hw is not None
        ##### End

        image_sizes = iter(image_inputs["image_sizes"])
        height, width = get_image_size(
            to_numpy_array(image_inputs["pixel_values"][0][0]),
            channel_dim=output_kwargs["images_kwargs"].get("data_format"),
        )
        _text = text
        text, num_image_tokens, unpadded_hw = self._expand_image_tokens(text, image_sizes, height, width, self.image_token)
        ##### Start (JS): assume without grid_pinpoints - only single scale
        # patches_height_width = int(math.sqrt(self.num_image_tokens))
        # image_inputs["image_grid_thw"] = torch.tensor([[1, patches_height_width, patches_height_width]], dtype=torch.long)
        # image_inputs["image_grid_thw"] = torch.tensor([[1, patches_height_width * num_tiles_hw[0, 0], patches_height_width * num_tiles_hw[0, 1]]], dtype=torch.long) # suppose same frames per video
        ## supports multi-scale tiling 
        image_grid_thw = torch.tensor(unpadded_hw)
        image_grid_thw = torch.cat([torch.ones(len(image_grid_thw), 1, dtype=torch.long), image_grid_thw], dim=-1)
        image_inputs["image_grid_thw"] = image_grid_thw
        ##### End
    if videos is not None:
        video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])
        one_video = video_inputs.get("pixel_values_videos")[0]
        if isinstance(video_inputs.get("pixel_values_videos")[0], (list, tuple)):
            one_video = np.array(one_video)
        else:
            one_video = to_numpy_array(one_video)
        height, width = get_image_size(one_video[0], channel_dim=output_kwargs["images_kwargs"].get("data_format"))
        num_frames = one_video.shape[0]  # frame dim is always after batch dim
        patches_height_width = int(math.sqrt(self.num_image_tokens))
        pooled_height_width = math.ceil(patches_height_width / 2)
        ##### Start (JS)
        if self.use_newline:
            num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token
        else:
            num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) # no newline token
        video_inputs["video_grid_thw"] = torch.tensor([[num_frames, pooled_height_width, pooled_height_width]], dtype=torch.long)
        ##### End
        text = [sample.replace(self.video_token, self.video_token * num_video_tokens) for sample in text]

    return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
    text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
    self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

    return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

def llavaov_expand_image_tokens(
    self,
    text: List[TextInput],
    image_sizes: Iterable[Union[List[int], int]],
    height: int,
    width: int,
    special_token: str,
    num_frames: int = 1,
):
    prompt_strings = []
    max_num_vision_tokens = 0
    num_tiles_hw_list = []
    for sample in text:
        while special_token in sample:
            image_size_list = next(image_sizes)
            original_size = image_size_list[0] if num_frames != 1 else image_size_list
            if not isinstance(original_size, (list, tuple)):
                # cast to list to avoid numerical precision errors when calculating unpadding
                original_size = original_size.tolist()
            orig_height, orig_width = original_size
            # num_image_tokens, unpadded_hw = self._get_number_of_features(orig_height, orig_width, height, width)
            unpadded_hw = self._get_number_of_features(orig_height, orig_width, height, width)
            num_tiles_hw_list.append(unpadded_hw)
            num_image_tokens = unpadded_hw[0] * unpadded_hw[1]
            max_num_vision_tokens = max(max_num_vision_tokens, num_image_tokens)
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1
            sample = sample.replace(special_token, "<placeholder>" * num_image_tokens * num_frames, 1)
        prompt_strings.append(sample)
    text = [sample.replace("<placeholder>", special_token) for sample in prompt_strings]
    return text, max_num_vision_tokens, num_tiles_hw_list

# def llavaov_get_number_of_image_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
#     ## only get the single-scale image features
#     ## ignoring grid pinpoints
#     assert not self.use_newline, "This function does not support newline token."
#     return self.num_image_tokens

def llavaov_get_number_of_image_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
    image_grid_pinpoints = self.image_processor.image_grid_pinpoints

    height_best_resolution, width_best_resolution = select_best_resolution(
        [orig_height, orig_width], image_grid_pinpoints
    )
    scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

    patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
    # num_image_tokens = patches_height * patches_width * scale_height * scale_width
    unpadded_features, newline_features, unpadded_hw = self._get_unpadded_features(
        orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
    )
    # The base patch covers the entire image (no CLS for SigLIP)
    # base_features = self.num_image_tokens
    # num_image_tokens = unpadded_features + newline_features + base_features
    # num_image_tokens = unpadded_features
    return unpadded_hw

def llavaov_get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
    """
    Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
    because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
    patches an image is divided into and get the number of features from that.
    """
    current_height = patches_height * scale_height
    current_width = patches_width * scale_width

    original_aspect_ratio = width / height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(round(height * (current_width / width), 7))
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = int(round(width * (current_height / height), 7))
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height

    max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
    ratio = math.sqrt(current_height * current_width / (max_num_patches * patches_height**2))
    if ratio > 1.1:
        resized_h, resized_w = int(current_height // ratio), int(current_width // ratio)
        unpadded_features = resized_h * resized_w
        newline_features = int(current_height // ratio)
        current_height, current_width = resized_h, resized_w

    return (unpadded_features, newline_features, (current_height, current_width))

def llavaov_get_image_patches(
    self,
    image: np.array,
    grid_pinpoints,
    size: tuple,
    patch_size: int,
    resample: PILImageResampling,
    data_format: ChannelDimension,
    input_data_format: ChannelDimension,
) -> List[np.array]:
    if grid_pinpoints is not None and not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints must be a list of possible resolutions.")

    ##### Start (JS) - support without grid pinpoints mode
    patches = []
    if grid_pinpoints is not None:
        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)
        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)
        resized_h, resized_w, _ = padded_image.shape
        num_tile_h, num_tile_w = resized_h // patch_size, resized_w // patch_size
        
        # resized_image = resize(
        #     image,
        #     size=best_resolution,
        #     resample=resample,
        #     data_format=data_format,
        #     input_data_format=input_data_format,
        # )
        # patches = divide_to_patches(resized_image, patch_size=patch_size, input_data_format=input_data_format)
        # resized_h, resized_w, _ = resized_image.shape
        # num_tile_h, num_tile_w = resized_h // patch_size, resized_w // patch_size
        # patches = patches

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]
        image_patches = patches
    else:
        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        num_tile_h, num_tile_w = 1, 1
        image_patches = [resized_original_image]
    num_tiles_hw = [num_tile_h, num_tile_w]
    # self.num_tiles_hw = num_tiles_hw
    ##### End

    # resized_original_image = resize(
    #     image,
    #     size=size,
    #     resample=resample,
    #     data_format=data_format,
    #     input_data_format=input_data_format,
    # )

    # image_patches = [resized_original_image] + patches

    return image_patches, num_tiles_hw

class Qwen2AttentionCusom(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        ##### Start: Added by JS
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = None 
        if output_attentions and self.attn_layer_flag:
            with torch.no_grad():
                _query_states, _key_states = query_states.to(torch.float32), key_states.to(torch.float32)
                attn_weights = torch.matmul(_query_states, _key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights.to(torch.float32)
                L = attn_weights.shape[-1]
                temp_causal_mask = torch.tril(torch.ones((L, L), device=attn_weights.device)).bool()
                attn_weights = attn_weights.masked_fill(~temp_causal_mask, float("-inf"))
                attn_weights = F.softmax(attn_weights, dim=-1)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        ##### End: Added by JS

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DecoderLayerCustom(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2AttentionCusom(config=config, layer_idx=layer_idx)


class Qwen2ModelCusom(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerCustom(config, layer_idx) for 
            layer_idx in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        ##### Start: Added by JS
        prev_attn, visual_token_mask = None, None
        L = hidden_states.shape[1]  # sequence length
        if L == 1: # for decoding phase
            output_attentions = False
        if hasattr(self, "visual_token_mask") and self.visual_token_mask is not None:
            visual_token_mask = self.visual_token_mask[0]
            visual_token_idxs = torch.nonzero(visual_token_mask, as_tuple=False).squeeze(1)
            visual_token_start_idx, visual_token_end_idx = visual_token_idxs[0].item(), visual_token_idxs[-1].item()
        ##### End: Added by JS

        ##### Start: Added by JS
        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            ##### Start: Added by JS
            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)
            if output_attentions and (layer_idx >= self.attn_layer_idx_start and layer_idx < self.attn_layer_idx_end):
                with torch.no_grad():
                    curr_attn = layer_outputs[1]
                    B, H, L, _ = curr_attn.shape  # batch_size, num_heads, seq_len, seq_len

                    ## head aggregation
                    if visual_token_mask is not None:
                        head_weights = curr_attn[:, :, :, visual_token_start_idx:visual_token_end_idx+1].max(dim=-1).values # (B, H, L)
                    else:
                        head_weights = curr_attn.max(dim=-1).values # (B, H, L)
                    head_weights = head_weights.mean(dim=-1)  # (B, H)
                    head_weights = head_weights / head_weights.max(dim=-1, keepdim=True).values
                    head_weights = head_weights.reshape(B, H, 1, 1) # (B, H, 1, 1)
                    curr_attn = (curr_attn * head_weights).mean(dim=1)  # (B, L, L)

                    ## apply identity matrix
                    identity_matrix = torch.eye(L, device=curr_attn.device, dtype=curr_attn.dtype).view(1, L, L)
                    curr_attn = curr_attn * 0.5 + identity_matrix * 0.5

                    ## accumulate attention map with previous layer
                    prev_attn = torch.matmul(curr_attn, prev_attn) if prev_attn is not None else curr_attn

                    ## for the last layer, we only keep the last token's attention
                    if layer_idx == self.attn_layer_idx_end - 1:
                        prev_attn_last_token = prev_attn[:, -1]
                        all_self_attns += (prev_attn_last_token,)
            ##### End: Added by JS
        ##### End: Added by JS

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlavaOnevisionModelCustom(LlavaOnevisionModel):
    def __init__(self, config):
        super().__init__(config)

        self.language_model = Qwen2ModelCusom(config.text_config)
        self.post_init()
        # self.language_model = Qwen2_5_VLTextModelCustom._from_config(config.text_config)
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, LlavaOnevisionModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both `pixel_values`/`pixel_values_videos` and `inputs_embeds` at the same time, "
                "and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Images are processed with Anyres
        if pixel_values is not None:
            image_features = self.get_image_features( ## image_sizes needs to be changed
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline if self.use_newline else None,
                vision_aspect_ratio=vision_aspect_ratio,
            )

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_features.shape[0]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Video are simply embedded and further pooled to decrease seq len
        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            if self.use_newline:
                image_newline = (
                    self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
                )
                video_features = torch.cat((video_features, image_newline), dim=1)
            
            video_features = video_features.flatten(0, 1)

            special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_video_mask].numel() != video_features.numel():
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_features.shape[0]
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return LlavaOnevisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            video_hidden_states=video_features if pixel_values_videos is not None else None,
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        ## Start: Added by JS
        # ! infer image_num_patches from image_sizes
        if self.use_newline:
            image_num_patches = [
                image_size_to_num_patches(image_size=imsize, grid_pinpoints=self.config.image_grid_pinpoints,
                            patch_size=self.config.vision_config.image_size) for imsize in image_sizes
            ]
        else:
            ## suppose the same video frames in a batch
            image_num_patches = [pixel_values.size(1) for _ in range(pixel_values.size(0))]
        ## End: Added by JS
        
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features
    
    def pack_image_features(self, image_features, image_sizes, image_newline=None, vision_aspect_ratio="anyres_max_9"):
        new_image_features = []
        feature_lens = []
        image_features
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1 or len(self.config.image_grid_pinpoints) > 1: # when (1) multiple tiles, (2) tiling configs
                ##### Start: Added by JS - exclude base image features
                # base_image_feature = image_feature[0]
                # image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                # if height * width != base_image_feature.shape[0]:
                #     raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    resized_h, resized_w = int(curr_height // ratio), int(curr_width // ratio)
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(image_feature, [resized_h, resized_w], mode="bilinear")[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                # image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

class LlavaOnevisionForConditionalGenerationCustom(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaOnevisionModelCustom(config)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, LlavaOnevisionCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_sizes=image_sizes,
            image_sizes_videos=image_sizes_videos,
            vision_aspect_ratio=vision_aspect_ratio,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LlavaOnevisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )
