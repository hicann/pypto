#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
'''
'''
from typing import Callable, List
from pypto.pypto_impl import ir
import pypto


class BlockCallHelper:
    @staticmethod
    def call(block_func: Callable, input_tensors: List[pypto.tensor], \
             output_tensors: List[pypto.tensor], indices: List[pypto.symbolic_scalar]):
        args = []
        impl_in_tensors = []
        for in_tensor in input_tensors:
            args.append(pypto.ir_from_tensor(in_tensor))
            impl_in_tensors.append(in_tensor.base())

        impl_out_tensors = []
        for out_tensor in output_tensors:
            args.append(pypto.ir_from_tensor(out_tensor))
            impl_out_tensors.append(out_tensor.base())

        if not callable(block_func):
            raise TypeError("func must be callable")
        try:
            ir_func_ptr = block_func(args)
        except Exception as e:
            raise RuntimeError(f"Error in block function: {e}") from e
        impl_sym_scalar = []
        for index in indices:
            impl_sym_scalar.append(index.base())
        return ir.call_block(ir_func_ptr, impl_in_tensors, impl_out_tensors, impl_sym_scalar)