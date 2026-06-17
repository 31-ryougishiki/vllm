# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# Weight name prefixes for split attention/MoE mode weight filtering.
# Used in load_weights to determine which weights belong to each sub-group.
_ATTN_WEIGHT_PREFIXES = (
    ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
    ".self_attn.o_proj", ".attn.q_proj", ".attn.k_proj",
    ".attn.v_proj", ".attn.o_proj", "qkv_proj",
    ".self_attn.k_norm", ".self_attn.q_norm",
)
_MOE_WEIGHT_PREFIXES = (
    ".mlp.experts.", "experts.gate", "experts.up_proj",
    "experts.down_proj", ".mlp.gate.", ".mlp.gate.weight",
    ".mlp.gate_up_proj", ".mlp.shared_experts.",
)


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        # NOTE: [split] In split mode, MoE ranks don't have TP group,
        # so disable sequence parallel to avoid all_gather errors
        from vllm.distributed import is_split_attn_enabled
        if is_split_attn_enabled():
            self.is_sequence_parallel = False
        else:
            self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            routing_method_type=RoutingMethodType.Renormalize,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        split_tp_size: int = 0,  # NOTE: lqf
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # NOTE: [split] 如果指定了split_tp_size，使用它；否则使用全局TP size
        # tp_size = get_tensor_model_parallel_world_size()
        tp_size = split_tp_size if split_tp_size > 0 else get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if positions.numel() > 0 and q.shape[0] > 0:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config # NOTE: lqf

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # NOTE: Get split attn/moe mode
        self.is_split_attn_mode = False
        self.is_split_moe_mode = False
        if parallel_config.split_tp_size > 0 and parallel_config.split_ep_size > 0:
            from vllm.distributed import is_split_attn_rank, is_split_moe_rank
            self.is_split_attn_mode = is_split_attn_rank()
            self.is_split_moe_mode = is_split_moe_rank()
        if self.is_split_attn_mode or (parallel_config.split_tp_size == 0):
            self.self_attn = Qwen3MoeAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_parameters=config.rope_parameters,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                dual_chunk_attention_config=dual_chunk_attention_config,
                split_tp_size=parallel_config.split_tp_size,  # NOTE: 新增
            )
        else:
            self.self_attn = None

        # NOTE: Only create MoE module in MOE mode (or when not using split mode)
        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        # NOTE: mlp should be None only when using split mode AND current rank is attn rank
        # In normal mode (split_tp_size=0 or split_ep_size=0), mlp should always be created
        if (parallel_config.split_tp_size > 0 and parallel_config.split_ep_size > 0
                and not self.is_split_moe_mode):
            self.mlp = None
        else:
              if (layer_idx not in mlp_only_layers) and (
                  config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
              ):
                  self.mlp = Qwen3MoeSparseMoeBlock(
                      vllm_config=vllm_config, prefix=f"{prefix}.mlp"
                  )
              else:
                  self.mlp = Qwen3MoeMLP(
                      hidden_size=config.hidden_size,
                      intermediate_size=config.intermediate_size,
                      hidden_act=config.hidden_act,
                      quant_config=quant_config,
                      prefix=f"{prefix}.mlp",
                  )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # NOTE: [split] For MoE rank, input_layernorm is not needed because
        # the data received from attn group has already been processed through
        # post_attention_layernorm. Skip input_layernorm to avoid computing on
        # placeholder/dummy data.
        if self.is_split_moe_mode:
            # MoE rank: skip input_layernorm, data will come from recv_from_attn
            if residual is None:
                residual = hidden_states
        else:
            # Attn rank or non-split mode: normal input_layernorm
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # NOTE: In split mode, handle cross-group communication
        if self.is_split_attn_mode and self.self_attn is not None:
            # Attn group: compute attention -> send to moe -> recv from moe
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            # Import and call cross-group communication
            from vllm_ascend.distributed.split_attn_moe_communicator import (
                send_to_moe,
                recv_from_moe,
            )
            # Send to moe group (directly uses hidden_states)
            send_to_moe(hidden_states)
            # Receive from moe group directly into hidden_states
            hidden_states = recv_from_moe(hidden_states)

            # [FIX] After recv_from_moe, need all_reduce to combine outputs from both moe ranks
            # In split mode, attn_rank 0 gets moe_rank 0 output, attn_rank 1 gets moe_rank 1 output
            # They need to be combined via all_reduce
            from vllm.distributed import tensor_model_parallel_all_reduce
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        elif self.is_split_moe_mode and self.mlp is not None:
            # Moe group: recv from attn -> compute moe -> send to attn
            # NOTE: [split] Data received from attn has already been processed through
            # post_attention_layernorm, so we should NOT apply it again.
            # (2026-04-15 fix: removed redundant post_attention_layernorm)
            from vllm_ascend.distributed.split_attn_moe_communicator import (
                recv_from_attn,
                send_to_attn,
            )
            # Receive from attn group directly into hidden_states
            hidden_states = recv_from_attn(hidden_states)
            # Skip post_attention_layernorm - data is already normalized
            # hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

            # Compute MoE
            hidden_states = self.mlp(hidden_states)
            # Send back to attn group
            send_to_attn(hidden_states)

            # Set hidden_states to None as result is already sent
            hidden_states = None
        else:
            # Non-split mode: normal forward
            if self.self_attn is not None:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )

            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            if self.mlp is not None:
                hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config


        # NOTE: Get split attn/moe mode for model structure
        self.split_tp_size = parallel_config.split_tp_size
        self.split_ep_size = parallel_config.split_ep_size
        self.is_split_attn_mode = False
        self.is_split_moe_mode = False

        if self.split_tp_size > 0 and self.split_ep_size > 0:
            from vllm.distributed import is_split_attn_rank, is_split_moe_rank
            self.is_split_attn_mode = is_split_attn_rank()
            self.is_split_moe_mode = is_split_moe_rank()

            # Initialize cross-group communication
            from vllm_ascend.distributed.split_attn_moe_communicator import (
                init_cross_group,
            )
            init_cross_group(self.split_tp_size, self.split_ep_size)

            logger.info(
                f"[Rank {torch.distributed.get_rank()}] Split attn-moe mode: "
                f"split_tp_size={self.split_tp_size}, split_ep_size={self.split_ep_size}, "
                f"is_attn_mode={self.is_split_attn_mode}, is_moe_mode={self.is_split_moe_mode}"
            )

        # NOTE: [split] Create embed_tokens for attn ranks only.
        # MoE ranks don't need embed_tokens during inference (they receive data from attn).
        # For warmup, we'll handle moe rank specially in model_runner.
        if self.is_split_attn_mode:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        elif self.split_tp_size == 0:
            # Non-split mode: create embed_tokens normally
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            # Split mode but not attn rank (moe rank): create a simple dummy embedding
            # that returns zeros. This is used during warmup only.
            # (2026-04-15 fix - create dummy to avoid crash)
            self.embed_tokens = None

        import time as _time
        _rank = torch.distributed.get_rank()
        def _make_layer(prefix):
            _t0 = _time.time()
            layer = Qwen3MoeDecoderLayer(vllm_config=vllm_config, prefix=prefix)
            _dt = _time.time() - _t0
            return layer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            _make_layer,
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()


    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # NOTE: [split] For moe rank during warmup, embed_tokens is None.
        # Return dummy zeros with the right shape for recv_from_attn to work.
        if self.embed_tokens is None:
            # Create dummy hidden_states for warmup - the actual data will come
            # from attn rank via recv_from_attn in each layer
            return input_ids.new_zeros(
                (input_ids.shape[0], self.config.hidden_size),
                dtype=torch.bfloat16
            )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # TODO: [split] 计算流程需要适配, moe-rank不进行embed计算
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            # NOTE: [split] In split mode (tp=2, ep=2), we need additional all_reduce
            # because VocabParallelEmbedding's internal all_reduce may not work correctly
            # when the two ranks have different token embeddings (each rank has partial vocab).
            # After embed, both ranks should have the same hidden_states.
            if self.is_split_attn_mode and hidden_states is not None:
                from vllm.distributed import tensor_model_parallel_all_reduce
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # NOTE: [split] For MoE rank, after each layer's forward sends result,
        # hidden_states becomes None. We need a placeholder for the next layer's
        # recv_from_attn to get the shape. This is only needed during warmup.
        # Store dummy_shape for regenerating placeholder if needed.
        dummy_shape = None
        # Flag to track if we're in MoE placeholder mode (skip norm at the end)
        is_moe_placeholder_mode = False
        if self.is_split_moe_mode and hidden_states is not None:
            dummy_shape = hidden_states.shape

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)

            # TODO: [split] 计算流程需要适配, layer分为layer-attn和layer-moe
            # 对于只有layer-moe的rank，等待接收->计算->发送
            # 对于只有layer-attn的rank，计算->发送->等待接收
            hidden_states, residual = layer(positions, hidden_states, residual)

            # NOTE: [split] After MoE layer sends result, hidden_states becomes None.
            # Need placeholder for next layer's recv_from_attn shape.
            if self.is_split_moe_mode and hidden_states is None and dummy_shape is not None:
                # Use the stored shape to create placeholder; actual data will come from recv_from_attn
                # Must use same device as the original input (NPU), not CPU
                placeholder_device = input_ids.device if input_ids is not None else "cpu"
                hidden_states = torch.empty(dummy_shape, dtype=torch.bfloat16, device=placeholder_device)
                residual = None
                is_moe_placeholder_mode = True
        # NOTE: [split] For MoE rank, hidden_states is None after all layers
        # (each layer sends result back to attn group). Skip final norm.
        # Also skip if we created a placeholder tensor during warmup.
        if self.is_split_moe_mode and (hidden_states is None or is_moe_placeholder_mode):
            return None

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        # TODO: [split] moe-rank 负责返回最后结果(外部是怎么搞的?)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        # Determine what weights to load based on split mode
        load_attention = not self.is_split_moe_mode
        load_moe = not self.is_split_attn_mode

        for name, loaded_weight in weights:
            # Skip attention weights in MoE mode
            if not load_attention and any(p in name for p in _ATTN_WEIGHT_PREFIXES):
                continue

            # Skip MoE weights in attn mode
            if not load_moe and any(p in name for p in _MOE_WEIGHT_PREFIXES):
                continue

            # MoE rank does not need embed_tokens
            if not load_attention and "embed_tokens" in name:
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # NOTE: [split]
        if not(load_moe and load_attention):
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] Weight loading done: "
                f"attention={load_attention}, moe={load_moe}, "
                f"loaded_params={loaded_params}"
            )

        return loaded_params


class Qwen3MoeForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # NOTE: split模式下，MoE rank不初始化lm_head
        parallel_config = vllm_config.parallel_config
        is_split_moe_mode = False
        if parallel_config.split_tp_size > 0 and parallel_config.split_ep_size > 0:
            from vllm.distributed import is_split_moe_rank
            is_split_moe_mode = is_split_moe_rank()

        if not is_split_moe_mode:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

            # TODO: 后续split_ep_group算完之后，把数据发回split_tp_group, 依然由split_tp_group来负责lm_head计算
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            # TODO: moe_mode 不需要lm_head
            self.lm_head = None
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)


        if example_layer is None:
            # NOTE: [split] early escape
            if vllm_config.parallel_config.split_tp_size > 0:
                self.num_moe_layers = len(self.moe_layers)
                self.num_expert_groups = 1
                self.num_shared_experts = 0
                return
            else:
                raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # NOTE: [split] split模式下，MoE rank没有lm_head
        if self.lm_head is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # NOTE: [split] MoE rank 没有 lm_head，需要过滤掉相关权重
        if self.lm_head is None:
            weights = [(n, w) for n, w in weights if not n.startswith("lm_head.")]
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()