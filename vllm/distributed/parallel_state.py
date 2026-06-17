# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Proxy entry point for vLLM distributed parallel state.
This module forwards all exports to the compiled binary module.
"""

from .parallel_state_core import *

# Re-init logger with correct module name (compiled module's logger
# uses "parallel_state_core" internally, which differs only cosmetically).
from vllm.logger import init_logger as _init_logger
logger = _init_logger(__name__)
