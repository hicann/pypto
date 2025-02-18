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

from ml_dtypes import bfloat16
import numpy as np


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == "fp16":
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == "fp32":
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == "fp64":
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == "int8":
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == "int16":
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == "int32":
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == "int64":
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == "uint8":
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == "uint16":
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == "uint32":
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == "uint64":
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == "complex64":
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == "complex128":
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == "bool":
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == "bf16":
        np.array(data_pool).astype(bfloat16).tofile(data_path)
