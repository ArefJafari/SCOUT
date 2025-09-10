# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the
# HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is based on the original Qwen2 model configuration implementation,
# with modifications for the SCOUT-SWA architecture.

from typing import Any, List, Optional, Tuple, Union, Dict, Callable
import os

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_scout_swa import ScoutSWAConfig

from fla.modules import GatedMLP as ScoutSWAMLP
from fla.modules import RMSNorm as ScoutSWARMSNorm

# ---- Triton SCOUT kernel import ----
# Expects: flash_attn_scout(q_btHD, ksel_btHD, vsel_btHD, kself_btHD, vself_btHD,
#                           bias=None, self_bias=None, causal=True, sel_positions=...)
# Shapes: [B,T,H,D] in/out
from .scout_attention_triton_v2 import flash_attn_scout

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ScoutSWAConfig"


# =========================
# Cache helpers
# =========================
def maybe_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    selection_window_size: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]

    save_cache = self._seen_tokens % selection_window_size == 0

    # Update the cache
    if key_states is not None:
        if save_cache:
            if len(self.key_cache) <= layer_idx:
                # Fill skipped
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif len(self.key_cache[layer_idx]) == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append([])
                self.value_cache.append([])
                return key_states, value_states
            elif len(self.key_cache[layer_idx]) == 0:
                return key_states, value_states
            else:
                return (
                    torch.cat([self.key_cache[layer_idx], key_states], dim=-2),
                    torch.cat([self.value_cache[layer_idx], value_states], dim=-2),
                )

    return self.key_cache[layer_idx], self.value_cache[layer_idx]


def slide_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    sliding_window_size: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep only last `sliding_window_size` tokens."""
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]

    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return key_states, value_states

    cached_key_states = self.key_cache[layer_idx]
    cached_value_states = self.value_cache[layer_idx]

    full_key_states = torch.cat([cached_key_states, key_states], dim=-2)
    full_value_states = torch.cat([cached_value_states, value_states], dim=-2)

    if full_key_states.shape[-2] > sliding_window_size:
        key_states = full_key_states[:, :, -sliding_window_size:, :]
        value_states = full_value_states[:, :, -sliding_window_size:, :]
    else:
        key_states = full_key_states
        value_states = full_value_states

    self.key_cache[layer_idx] = key_states
    self.value_cache[layer_idx] = value_states
    return key_states, value_states


DynamicCache.maybe_update = maybe_update
DynamicCache.slide_update = slide_update
Cache.maybe_update = None
Cache.slide_update = None


# =========================
# RoPE helpers
# =========================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =========================
# Attention cores
# =========================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_sparse_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    sel = module.selection_window_size
    selected_key_states = key_states[:, :, sel - 1 :: sel, :]
    selected_value_states = value_states[:, :, sel - 1 :: sel, :]

    # query -> selected kv
    selected_attn_weights = torch.matmul(query, selected_key_states.transpose(2, 3))

    # handle mask + drop diagonal in grid path
    if attention_mask is not None:
        attention_mask_cloned = attention_mask[:, :, :, : attention_mask.size(-2)].clone()
        min_val = torch.finfo(attention_mask.dtype).min
        diag_mask = torch.eye(attention_mask.size(-2), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_cloned[:, 0] = attention_mask_cloned[:, 0].masked_fill(diag_mask.bool(), min_val)

        selected_attention_mask = attention_mask_cloned[:, :, :, sel - 1 :: sel]
        causal_mask = selected_attention_mask[:, :, :, : selected_key_states.shape[-2]]
        selected_attn_weights = selected_attn_weights + causal_mask

    # self term (diagonal)
    query_self_attn_weights = torch.einsum("...bi,...bi->...b", query, key_states).unsqueeze(-1)

    # concat + softmax
    attn_weights = torch.cat((query_self_attn_weights, selected_attn_weights), dim=-1)
    attn_weights = attn_weights * scaling

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # split back
    self_w, sel_w = torch.split(attn_weights, [1, selected_attn_weights.size(3)], dim=-1)

    sel_out = torch.matmul(sel_w, selected_value_states)
    self_out = self_w * value_states

    attn_output = sel_out + self_out
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# -------------------------
# Triton wrapper for SCOUT
# -------------------------
def _has_padding(attn_mask: Optional[torch.Tensor]) -> bool:
    if attn_mask is None:
        return False
    # attn_mask is 4D inverted in this model. We only detect zeros presence in original 2D,
    # but since here we get 4D, detect zeros in the provided tensor.
    # If user passed a classic 2D mask earlier, the model converted it already.
    try:
        return (attn_mask == 0).any().item()
    except Exception:
        return False


# def triton_sparse_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,  # [B,H,T,D]
#     key: torch.Tensor,    # [B,HKV,T,D]
#     value: torch.Tensor,  # [B,HKV,T,D]
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     # If output_attentions required or padding exists -> fallback outside
#     if kwargs.get("output_attentions", False):
#         raise RuntimeError("triton_sparse_attention_forward does not return attn weights")

#     # Expand KV heads (MQA/GQA -> MHA)
#     key_states = repeat_kv(key, module.num_key_value_groups)     # [B,H,T,D]
#     value_states = repeat_kv(value, module.num_key_value_groups) # [B,H,T,D]

#     # To [B,T,H,D] for kernel
#     q_btHD = query.transpose(1, 2).contiguous()
#     k_btHD = key_states.transpose(1, 2).contiguous()
#     v_btHD = value_states.transpose(1, 2).contiguous()

#     B, T, H, D = q_btHD.shape
#     sel = module.selection_window_size

#     # Selected grid kv
#     if T >= sel:
#         ksel_btHD = k_btHD[:, sel - 1 :: sel, :, :].contiguous()
#         vsel_btHD = v_btHD[:, sel - 1 :: sel, :, :].contiguous()
#         sel_positions = torch.arange(sel - 1, T, sel, device=q_btHD.device, dtype=torch.int32)
#     else:
#         ksel_btHD = k_btHD[:, 0:0, :, :]
#         vsel_btHD = v_btHD[:, 0:0, :, :]
#         sel_positions = torch.empty(0, device=q_btHD.device, dtype=torch.int32)

#     # Self kv are full (for the explicit diagonal term)
#     kself_btHD = k_btHD
#     vself_btHD = v_btHD
#     # Causal always True in decoder
#     out_btHD = flash_attn_scout(
#         q_btHD, ksel_btHD, vsel_btHD, kself_btHD, vself_btHD,
#         causal=True, sel_positions=sel_positions, drop_diagonal=True
#     )  # [B,T,H,D]


#     out_bHtD = out_btHD.transpose(1, 2).contiguous()  # [B,H,T,D]


#     return out_bHtD.transpose(1, 2).contiguous().transpose(1, 2), None




def triton_sparse_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, H, T, D]
    key: torch.Tensor,    # [B, HKV, T, D]
    value: torch.Tensor,  # [B, HKV, T, D]
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Triton sparse attention forward using BHTD layout end-to-end.
    Returns (out, None) where out is [B, H, T, D].
    """
    # Not supported in this forward-only path
    if kwargs.get("output_attentions", False):
        raise RuntimeError("triton_sparse_attention_forward does not return attn weights")
    if dropout and dropout > 0:
        raise RuntimeError("Dropout > 0 not supported in this Triton path")
    if attention_mask is not None:
        # This kernel path ignores attention_mask; caller should have baked causal/padding into selection.
        # If you need it, integrate masking inside the kernel.
        pass

    # Expand KV heads (MQA/GQA -> MHA) to [B, H, T, D]
    key_states   = repeat_kv(key,   module.num_key_value_groups)     # [B, H, T, D]
    value_states = repeat_kv(value, module.num_key_value_groups)     # [B, H, T, D]

    # Shapes
    B, H, T, D = query.shape
    assert key_states.shape   == (B, H, T, D)
    assert value_states.shape == (B, H, T, D)

    sel = module.selection_window_size

    # Selected grid along time (Tk positions): every `sel` steps starting at sel-1
    if T >= sel:
        ksel_bhKD = key_states[:, :, sel - 1 :: sel, :]    # [B, H, Tk, D]
        vsel_bhKD = value_states[:, :, sel - 1 :: sel, :]  # [B, H, Tk, D]
        sel_positions = torch.arange(sel - 1, T, sel, device=query.device, dtype=torch.int32)  # [Tk]
    else:
        # Empty selection (Tk=0)
        ksel_bhKD = key_states[:, :, 0:0, :]               # [B, H, 0, D]
        vsel_bhKD = value_states[:, :, 0:0, :]             # [B, H, 0, D]
        sel_positions = torch.empty(0, device=query.device, dtype=torch.int32)

    # Self KV are the full sequence (explicit diagonal term)
    kself_bhTD = key_states    # [B, H, T, D]
    vself_bhTD = value_states  # [B, H, T, D]

    # Call the BHTD variant of the kernel (returns [B, H, T, D])
    out_bhTD = flash_attn_scout(
        q=query,
        ksel=ksel_bhKD,
        vsel=vsel_bhKD,
        kself=kself_bhTD,
        vself=vself_bhTD,
        sel_positions=sel_positions,
        causal=True,
        drop_diagonal=True,
    )

    return out_bhTD, None


# =========================
# Modules
# =========================
class ScoutSWAAttention(nn.Module):
    """SCOUT-SWA attention with switchable prefill backend."""

    def __init__(self, config: ScoutSWAConfig, layer_idx: int, attn: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.selection_window_size = config.selection_window_size
        self.sliding_window_size = config.sliding_window
        self.attn = attn

        # Prefill backend: "eager" (default) or "triton"
        cfg_backend = getattr(config, "scout_prefill_backend", "triton")
        env_backend = os.environ.get("SCOUT_PREFILL_BACKEND", "").strip().lower()
        if env_backend in ("eager", "triton"):
            cfg_backend = env_backend
        self.prefill_backend = cfg_backend

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        fa_attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        _, q_len, _ = hidden_states.size()
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,T,D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [B,HKV,T,D]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,HKV,T,D]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.attn == "scout":

            if q_len > 1:
                # ----- Prefill path -----
                # Update selected KV cache (unchanged behavior)
                sel = self.selection_window_size
                selected_key_states = key_states[:, :, sel - 1 :: sel, :]
                selected_value_states = value_states[:, :, sel - 1 :: sel, :]

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    # Store selected-only cache for scout
                    _ = past_key_value.update(
                        selected_key_states, selected_value_states, self.layer_idx * 2 + 1, cache_kwargs
                    )

                # Choose backend (safe fallback if padding or output_attentions=True)
                want_triton = self.prefill_backend == "triton"
                has_padding = _has_padding(attention_mask)
                wants_weights = kwargs.get("output_attentions", False)

                use_triton = want_triton #and (not has_padding) and (not wants_weights)

                if use_triton:
                    # Triton SCOUT (no attn weights output)
                    
                    attn_output, _ = triton_sparse_attention_forward(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        **kwargs,
                    )
                    attn_weights = None
                else:
                    # Eager sparse reference
                    attn_output, attn_weights = eager_sparse_attention_forward(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        scaling=self.scaling,
                        **kwargs,
                    )

            else:
                # ----- Decode path (q_len == 1): keep original behavior -----
                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.maybe_update(
                        key_states, value_states, self.layer_idx * 2 + 1, self.selection_window_size, cache_kwargs
                    )

                attention_interface: Callable = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        logger.warning_once(
                            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                            "Falling back to eager attention. Consider `attn_implementation='eager'`."
                        )
                    else:
                        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

        elif self.attn == "slide":
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.slide_update(
                    key_states, value_states, self.layer_idx * 2, self.sliding_window_size, cache_kwargs
                )

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                        "Falling back to eager attention. Consider `attn_implementation='eager'`."
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            sliding_window = self.sliding_window_size
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                fa_attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid attention type: {self.attn}")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ScoutSWADecoderLayer(nn.Module):
    def __init__(self, config: ScoutSWAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ScoutSWAAttention(config=config, layer_idx=layer_idx, attn="scout")
        self.mlp = ScoutSWAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=True,
        )
        self.input_layernorm = ScoutSWARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ScoutSWARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.swa_attn = ScoutSWAAttention(config=config, layer_idx=layer_idx, attn="slide")
        self.post_swa_layernorm = ScoutSWARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attention_layernorm = ScoutSWARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sec_mlp = ScoutSWAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fa_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # SWA block
        hidden_states, self_attn_weights = self.swa_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            fa_attention_mask=fa_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_swa_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # SCOUT block
        residual = hidden_states
        hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            fa_attention_mask=fa_attention_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.sec_mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class ScoutSWARotaryEmbedding(nn.Module):
    def __init__(self, config: ScoutSWAConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


ScoutSWA_START_DOCSTRING = r"""
Bare ScoutSWA model.
"""


@add_start_docstrings(
    "The bare ScoutSWA Model outputting raw hidden-states without any specific head on top.",
    ScoutSWA_START_DOCSTRING,
)
class ScoutSWAPreTrainedModel(PreTrainedModel):
    config_class = ScoutSWAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ScoutSWADecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


ScoutSWA_INPUTS_DOCSTRING = r"""... (omitted for brevity in this header) ..."""


@add_start_docstrings(
    "The bare ScoutSWA Model outputting raw hidden-states without any specific head on top.",
    ScoutSWA_START_DOCSTRING,
)
class ScoutSWAModel(ScoutSWAPreTrainedModel):
    config_class = ScoutSWAConfig

    def __init__(self, config: ScoutSWAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ScoutSWADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = ScoutSWARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ScoutSWARotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(ScoutSWA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Keep eager during prefill for slide attn; scout path decides internally between eager/triton.
        if input_ids.size(1) > 1:
            self.config._attn_implementation = "eager"
        else:
            self.config._attn_implementation = "flash_attention_2"

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # SWA mask for FA path
        fa_causal_mask = self._update_fa_causal_mask(attention_mask, inputs_embeds, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    fa_attention_mask=fa_causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_fa_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: Cache,
    ):
        if attention_mask is not None and past_key_values is not None:
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            if is_padding_right:
                raise ValueError(
                    "Batched generation with padding_side='right' is not supported for FlashAttention path. "
                    "Use left padding."
                )
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "Batched generation with padding_side='right' is not supported for FlashAttention path. "
                        "Use left padding."
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # Build 4D causal mask if needed
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


class ScoutSWAForCausalLM(ScoutSWAPreTrainedModel, GenerationMixin):
    config_class = ScoutSWAConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = ScoutSWAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if "past_key_values" in str(exception):
                raise AttributeError(
                    "You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. Try a different strategy."
                )
            else:
                raise exception

    # @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    # @add_start_docstrings_to_model_forward(ScoutSWA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
