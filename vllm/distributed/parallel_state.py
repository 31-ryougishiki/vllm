# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Proxy entry point for vLLM distributed parallel state.
This module forwards all exports to the compiled binary module.
"""

from .parallel_state_core import *

# Private symbols are NOT exported by ``import *`` above (they start with ``_``).
# The following are explicitly re-exported because upstream and vllm-ascend
# patches import them directly.
#   - _get_unique_name, _register_group  → vllm_ascend.patch.worker.patch_distributed
#   - _ENABLE_CUSTOM_ALL_REDUCE          → vllm.distributed.device_communicators.cuda_communicator
from .parallel_state_core import (
    _ENABLE_CUSTOM_ALL_REDUCE,
    _get_unique_name,
    _register_group,
)

# Re-init logger with correct module name (compiled module's logger
# uses "parallel_state_core" internally, which differs only cosmetically).
from vllm.logger import init_logger as _init_logger
logger = _init_logger(__name__)
