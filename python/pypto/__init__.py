#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO
"""

# torch/torch_npu may use cxxabi=0 or cxxabi=1, while pypto only support cxxabi=0
# if pypto load first, torch/torch_cpu may crash, force load torch first
try:
    import torch
except ImportError:
    pass

# shared lib should be loaded first
from . import _loader

from . import experimental
from .config import *  # noqa
from ._controller import *  # noqa
from .converter import from_torch
from .enum import *  # noqa
from .op import *  # noqa
from .operation import *  # noqa
from .operator import *  # noqa
from .pass_config import *  # noqa
from .cost_model import *  # noqa
from ._utils import ceildiv, bytes_of
from .platform import platform
from .runtime import verify, set_verify_golden_data, RunMode
from .symbolic_scalar import SymbolicScalar, SatStatus
from .tensor import Tensor
from .functions import Function, get_last_function, get_current_function
from ._element import Element
from .logging import *  # noqa

# Import frontend after all other imports to avoid circular imports
from . import frontend


tensor = Tensor
element = Element
symbolic_scalar = SymbolicScalar
