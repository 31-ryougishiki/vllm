# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Proxy entry point for Qwen3MoE model.
This module forwards all exports to the compiled binary module.
"""

from .qwen3_moe_core import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)

from vllm.compilation.decorators import support_torch_compile

__all__ = [
    "Qwen3MoeMLP",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeAttention",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeModel",
    "Qwen3MoeForCausalLM",
    "support_torch_compile",
]
