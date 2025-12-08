import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import matplotlib.pyplot as plt
from tqdm import trange

from transformers.modeling_flash_attention_utils import (
    flash_attn_supports_top_left_mask,
    _flash_attention_forward,
)
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2DecoderLayer,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.models.internvl.modeling_internvl import (
    InternVLForConditionalGeneration,
    InternVLModel,
)
from transformers.models.internvl.configuration_internvl import InternVLConfig

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import SizeDict
from transformers.utils import TensorType
from transformers.image_processing_utils_fast import group_images_by_shape, reorder_images

logger = logging.get_logger(__name__)


def _got_ocr2_fast_preprocess_custom(
    self,
    images: List["torch.Tensor"],
    do_resize: bool,
    size: SizeDict,
    crop_to_patches: bool,
    min_patches: int,
    max_patches: int,
    interpolation: Optional["F.InterpolationMode"],
    do_center_crop: bool,
    crop_size: SizeDict,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: Optional[Union[float, List[float]]],
    image_std: Optional[Union[float, List[float]]],
    return_tensors: Optional[Union[str, TensorType]],
) -> BatchFeature:
    if crop_to_patches:
        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}
        num_patches = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.crop_image_to_patches(
                stacked_images,
                min_patches,
                max_patches,
                use_thumbnail=False,
                patch_size=size,
                interpolation=interpolation,
            )
            processed_images_grouped[shape] = stacked_images
            num_patches[shape] = [stacked_images.shape[1]] * stacked_images.shape[0]
        images = reorder_images(processed_images_grouped, grouped_images_index)
        images = [image for images_list in images for image in images_list]
        num_patches = reorder_images(num_patches, grouped_images_index)
    else:
        num_patches = [1] * len(images)

    # Group images by size for batched resizing
    grouped_images, grouped_images_index = group_images_by_shape(images)
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        if do_resize:
            stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
        resized_images_grouped[shape] = stacked_images
    resized_images = reorder_images(resized_images_grouped, grouped_images_index)

    # Group images by size for further processing
    # Needed in case do_resize is False, or resize returns images with different sizes
    grouped_images, grouped_images_index = group_images_by_shape(resized_images)
    processed_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        if do_center_crop:
            stacked_images = self.center_crop(stacked_images, crop_size)
        # Fused rescale and normalize
        stacked_images = self.rescale_and_normalize(
            stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        processed_images_grouped[shape] = stacked_images

    processed_images = reorder_images(processed_images_grouped, grouped_images_index)
    processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

    return BatchFeature(
        data={"pixel_values": processed_images, "num_patches": num_patches}, tensor_type=return_tensors
    )


class Qwen2AttentionCustom(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
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

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        ##### Start: Add by SH - attention weights extraction
        output_attentions = kwargs.get("output_attentions", False)
        if output_attentions and self.attn_layer_flag:
            with torch.no_grad():
                _query_states, _key_states = query_states.to(torch.float32), key_states.to(torch.float32)
                _key_states = repeat_kv(_key_states, self.num_key_value_groups)
                attn_weights = torch.matmul(_query_states, _key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                L = attn_weights.shape[-1]
                temp_causal_mask = torch.tril(torch.ones((L, L), device=attn_weights.device)).bool()
                attn_weights = attn_weights.masked_fill(~temp_causal_mask, float("-inf"))
                attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_weights = None

        # attn_output, attn_weights = attention_interface(
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        ##### End: Add by SH

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DecoderLayerCustom(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2AttentionCustom(config=config, layer_idx=layer_idx)

class Qwen2ModelCustom(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerCustom(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
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

        ### Start: Add by SH - for visual token masking
        if hidden_states.shape[1] == 1:
            output_attentions = False

        prev_attn=None
        visual_token_mask = None
        if hasattr(self, "visual_token_mask"):
            visual_token_mask = self.visual_token_mask
        ### End: Add by SH

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

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

            ### Start: Add by SH - for extracting attention weights
            if output_attentions and (layer_idx >= self.attn_layer_idx_start and layer_idx < self.attn_layer_idx_end):
                with torch.no_grad():
                    curr_attn = layer_outputs[1]
                    B, H, L, _ = curr_attn.shape    # batch size, num_heads, seq_len, seq_len

                    if visual_token_mask is not None:
                        # visual_token_mask: (B, L), bool
                        assert visual_token_mask.shape[0] == B and visual_token_mask.shape[1] == L
                        # check visual token number in the batch are same
                        assert visual_token_mask.sum(1).unique().numel() == 1

                        # get the visual token indices
                        visual_token_idx = torch.where(visual_token_mask)[1].view(B, -1)   # (B, num_visual_tokens)

                        # reshape the visual token mask for gathering
                        gather_visual_token_idx = visual_token_idx[:, None, None, :].expand(B, H, L, visual_token_idx.shape[1])  # (B, H, L, num_visual_tokens)

                        # gather the attention weights corresponding to visual tokens in the last dimension
                        head_weights = curr_attn.gather(dim=-1, index=gather_visual_token_idx)  # (B, H, L, num_visual_tokens)
                        # get max weight of visual tokens per each query
                        head_weights = head_weights.max(dim=-1).values  # (B, H, L)
                    else:
                        # get max weight of all tokens per each query
                        head_weights = curr_attn.max(dim=-1).values  # (B, H, L)

                    head_weights = head_weights.mean(dim=-1)  # (B, H)

                    ## version 1
                    ## normalize head weights to one as maximum
                    head_weights = head_weights / head_weights.max(dim=-1, keepdim=True).values  # (B, H)
                    head_weights = head_weights.reshape(B, H, 1, 1) # (B, H, 1, 1)
                    curr_attn_weighted = (curr_attn * head_weights).mean(dim=1)  # (B, L, L)

                    ## apply identity matrix
                    W_attn, W_res = 0.5, 0.5
                    identity_matrix = torch.eye(L, device=curr_attn.device, dtype=curr_attn.dtype).view(1, L, L)
                    curr_attn_with_residual = curr_attn_weighted * W_attn + identity_matrix * W_res

                    ## accumulate attention map with previous layer
                    prev_attn = torch.matmul(curr_attn_with_residual, prev_attn) if prev_attn is not None else curr_attn_with_residual

                    ## for the last layer, we only keep the last token's attention
                    if layer_idx == self.attn_layer_idx_end - 1:
                        prev_attn_last_token = prev_attn[:, -1]
                        all_self_attns += (prev_attn_last_token,)

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


class Qwen2ForCausalLMCustom(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ModelCustom(config)
        self.post_init()


class InternVLModelCustom(InternVLModel):
    def __init__(self, config: InternVLConfig):
        super().__init__(config)
        # self.language_model = Qwen2ForCausalLMCustom._from_config(config.text_config)
        self.language_model = Qwen2ModelCustom._from_config(config.text_config)
        self.post_init()

class InternVLForConditionalGenerationCustom(InternVLForConditionalGeneration):
    def __init__(self, config: InternVLConfig):
        super().__init__(config)
        self.model = InternVLModelCustom(config)

        self.post_init()