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

""" AttentionPost 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import math
import time
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from ml_dtypes import bfloat16


class PostConfig:
    """Configuration class for AttentionPost processing parameters.

    This class encapsulates the configuration parameters used for generating
    test data and performing attention post processing computations.

    Attributes:
        params: Tuple containing [b, n, s, h, kv_lora_rank, v_head_dim] parameters
        dtype: Data type for computations (np.float16, bfloat16, etc.)
        is_quant_w_uv: Whether to use quantization for w_uv weights (default: True)
        is_quant_w_o: Whether to use quantization for w_o weights (default: True)
        has_smooth_w_uv: Whether to use smoothing weights for w_uv (default: True)
        has_smooth_w_o: Whether to use smoothing weights for w_o (default: True)
        is_nz: Whether to use non-zero format for weight storage (default: True)
    """

    def __init__(self,
                params: Tuple[int, int, int, int, int, int],
                dtype: Union[np.dtype, type],
                is_quant_w_uv: bool = True,
                has_smooth_w_uv: bool = True,
                is_quant_w_o: bool = True,
                has_smooth_w_o: bool = True,
                is_nz: bool = True):
        """
        Initialize PostConfig with the specified parameters.

        Args:
            params: Tuple of [b, n, s, h, kv_lora_rank, v_head_dim] where:
                b: batch size
                n: number of heads
                s: sequence length
                h: hidden dimension
                kv_lora_rank: KV LoRA rank
                v_head_dim: value head dimension
            dtype: Data type for computations
            is_quant_w_uv: Whether to use quantization for w_uv weights
            has_smooth_w_uv: Whether to use smoothing weights for w_uv
            is_quant_w_o: Whether to use quantization for w_o weights
            has_smooth_w_o: Whether to use smoothing weights for w_o
            is_nz: Whether to use non-zero format for weight storage
        """
        self.params = params
        self.dtype = dtype
        self.is_quant_w_uv = is_quant_w_uv
        self.has_smooth_w_uv = has_smooth_w_uv
        self.is_quant_w_o = is_quant_w_o
        self.has_smooth_w_o = has_smooth_w_o
        self.is_nz = is_nz

    @property
    def b(self) -> int:
        """Batch size."""
        return self.params[0]

    @property
    def n(self) -> int:
        """Number of heads."""
        return self.params[1]

    @property
    def s(self) -> int:
        """Sequence length."""
        return self.params[2]

    @property
    def h(self) -> int:
        """Hidden dimension."""
        return self.params[3]

    @property
    def kv_lora_rank(self) -> int:
        """KV LoRA rank."""
        return self.params[4]

    @property
    def v_head_dim(self) -> int:
        """Value head dimension."""
        return self.params[5]

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (f"PostConfig(params={self.params}, dtype={self.dtype}, "
                f"is_quant_w_uv={self.is_quant_w_uv}, has_smooth_w_uv={self.has_smooth_w_uv}, "
                f"is_quant_w_o={self.is_quant_w_o}, has_smooth_w_o={self.has_smooth_w_o}, "
                f"is_nz={self.is_nz})")

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()


if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


fp32 = np.float32


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_w=None):
    input_fp32 = input_t.astype(np.float32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_w
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def post_compute(inputs):
    dtype = inputs.get("dtype")
    is_quant_w_uv = inputs.get("is_quant_w_uv", False)
    has_smooth_w_uv = inputs.get("has_smooth_w_uv", False)
    is_quant_w_o = inputs.get("is_quant_w_o")
    has_smooth_w_o = inputs.get("has_smooth_w_o")
    x = inputs.get("x")
    w_uv = inputs.get("w_uv")
    w_o = inputs.get("w_o")
    if is_quant_w_uv:
        w_uv_scale = inputs.get("w_uv_scale")
        if has_smooth_w_uv:
            smooth_w_uv = inputs.get("smooth_w_uv")
    if is_quant_w_o:
        w_o_scale = inputs.get("w_o_scale")
        if has_smooth_w_o:
            smooth_w_o = inputs.get("smooth_w_o")

    b, s, n, kv_lora_rank = x.shape
    v_head_dim = w_uv.shape[2]
    h = w_o.shape[1]

    x_reshape = x.reshape(b * s, n, kv_lora_rank)
    x_trans = np.transpose(x_reshape, (1, 0, 2))  # [n, b*s, kv_lora_rank]
    if is_quant_w_uv:
        if has_smooth_w_uv:
            x_trans, scale_dequant = quant(x_trans, True, True, smooth_w_uv)
        else:
            x_trans, scale_dequant = quant(x_trans, True)

        bmm = np.matmul(x_trans.astype(np.int32), w_uv.astype(np.int32))

        # dequant
        bmm_fp32 = bmm.astype(fp32)  # [b*s, h]
        bmm_fp32_dequant = bmm_fp32 * scale_dequant
        bmm = bmm_fp32_dequant * w_uv_scale
    else:
        # [n, b*s, kv_lora_rank] @ [n, kv_lora_rank, v_head_dim] -> [n, b*s, v_head_dim]
        bmm = np.matmul(x_trans.astype(np.float32), w_uv.astype(np.float32))
    bmm = bmm.astype(dtype)

    bmm_trans = np.transpose(bmm, (1, 0, 2))  # [b*s, n, v_head_dim]
    bmm_reshape = bmm_trans.reshape(b * s, n * v_head_dim)  # [b*s, n*v_head_dim]
    if is_quant_w_o:
        # quant, per_token
        # scale_dequant: [b*s, 1]
        if has_smooth_w_o:
            bmm_reshape, scale_dequant = quant(bmm_reshape, True, True, smooth_w_o)
        else:
            bmm_reshape, scale_dequant = quant(bmm_reshape, True)  # int8, fp32

        mm = np.matmul(bmm_reshape.astype(np.int32), w_o.astype(np.int32))

        # dequant
        mm_fp32 = mm.astype(fp32)  # [b*s, h]
        mm_fp32_dequant = mm_fp32 * scale_dequant
        mm = mm_fp32_dequant * w_o_scale
    else:
        mm = np.matmul(bmm_reshape.astype(fp32), w_o.astype(fp32))
    mm = mm.astype(dtype)

    output = mm.reshape(b, s, h)
    return output


# params: [b, n, s, h, kv_lora_rank, v_head_dim]
def gen_post_input_data(output_dir: Path, config: PostConfig):
    """
    Generate input data for attention post processing using PostConfig.

    Args:
        output_dir: Output directory for generated files
        config: PostConfig object containing all configuration parameters

    Returns:
        List containing [w_uv, w_uv_scale, smooth_w_uv, w_o, w_o_scale, smooth_w_o]
    """
    b, n, s, h, kv_lora_rank, v_head_dim = config.params
    w_uv_shape = [n, kv_lora_rank, v_head_dim]
    w_uv_scale_shape = [n, 1, v_head_dim]
    smooth_w_uv_shape = [1, kv_lora_rank]
    w_o_shape = [n * v_head_dim, h]
    w_o_scale_shape = [1, h]
    smooth_w_o_shape = [1, n * v_head_dim]

    logging.debug("w_uv shape is %s", w_uv_shape)
    logging.debug("w_o shape is %s", w_o_shape)
    logging.debug("smooth_w_o shape is %s", smooth_w_o_shape)
    logging.debug("smooth_w_uv shape is %s", smooth_w_uv_shape)

    w_uv_path = Path(output_dir, 'w_uv.bin')
    w_uv_scale_path = Path(output_dir, 'w_uv_scale.bin')
    smooth_w_uv_path = Path(output_dir, 'smooth_w_uv.bin')
    w_o_path = Path(output_dir, 'w_o.bin')
    w_o_scale_path = Path(output_dir, 'w_o_scale.bin')
    smooth_w_o_path = Path(output_dir, 'smooth_w_o.bin')

    res = [None] * 6
    w_uv = np.random.uniform(-0.1, 0.1, w_uv_shape).astype(config.dtype)
    if config.is_quant_w_uv:
        w_uv, w_uv_scale = quant(w_uv, False)
        w_uv.tofile(w_uv_path)
        w_uv_scale.tofile(w_uv_scale_path)
        res[0] = w_uv
        res[1] = w_uv_scale

        if config.has_smooth_w_uv:
            smooth_w_uv = np.random.uniform(-1, 1, smooth_w_uv_shape).astype(np.float32)
            smooth_w_uv.tofile(smooth_w_uv_path)
            res[2] = smooth_w_uv
    else:
        w_uv.tofile(w_uv_path)
        res[0] = w_uv

    w_o = np.random.uniform(-0.1, 0.1, w_o_shape).astype(config.dtype)
    if config.is_quant_w_o:
        # per_channel, w_o_scale: [1, h]
        w_o, w_o_scale = quant(w_o, False)
        if config.is_nz:
            w_o.reshape(w_o_shape[0], w_o_shape[1] // 32, 32).transpose(1,0,2).tofile(w_o_path)
        else:
            w_o.tofile(w_o_path)
        w_o_scale.tofile(w_o_scale_path)
        res[3] = w_o
        res[4] = w_o_scale

        if config.has_smooth_w_o:
            smooth_w_o = np.random.uniform(-1, 1, smooth_w_o_shape).astype(np.float32)
            smooth_w_o.tofile(smooth_w_o_path)
            res[5] = smooth_w_o
    else:
        if config.is_nz:
            w_o.reshape(w_o_shape[0], w_o_shape[1] // 16, 16).transpose(1,0,2).tofile(w_o_path)
        else:
            w_o.tofile(w_o_path)
        res[3] = w_o

    return res


def gen_post_test_data(output_dir: Path, config: PostConfig):
    """
    Generate test data for attention post processing using PostConfig.

    Args:
        output_dir: Output directory for generated files
        config: PostConfig object containing all configuration parameters
    """
    b, n, s, h, kv_lora_rank, v_head_dim = config.params
    x_shape = [b, s, n, kv_lora_rank]
    logging.debug("x shape is %s", x_shape)

    x_path = Path(output_dir, 'x.bin')
    output_path = Path(output_dir, 'golden_output.bin')

    np.random.seed(int(time.time()))

    x = np.random.uniform(-10, 10, x_shape).astype(config.dtype)
    x.tofile(x_path)
    w_uv, w_uv_scale, smooth_w_uv, w_o, w_o_scale, smooth_w_o = gen_post_input_data(output_dir, config)

    inputs = {"dtype": config.dtype, "is_quant_w_uv": config.is_quant_w_uv,
              "has_smooth_w_uv": config.has_smooth_w_uv, "is_quant_w_o": config.is_quant_w_o,
              "has_smooth_w_o": config.has_smooth_w_o}
    inputs["x"] = x
    inputs["w_uv"] = w_uv
    inputs["w_o"] = w_o
    if config.is_quant_w_uv:
        inputs["w_uv_scale"] = w_uv_scale
        if config.has_smooth_w_uv:
            inputs["smooth_w_uv"] = smooth_w_uv
    if config.is_quant_w_o:
        inputs["w_o_scale"] = w_o_scale
        if config.has_smooth_w_o:
            inputs["smooth_w_o"] = smooth_w_o

    output = post_compute(inputs)
    output.tofile(output_path)

    return output


@GoldenRegister.reg_golden_func(
    case_names = [
        # fp16
        "AttentionPostSTest.b16_s1_nz_fp16_quant",
        "AttentionPostSTest.b16_s2_nz_fp16_quant",
        "AttentionPostSTest.b32_s1_nz_fp16_quant",
        "AttentionPostSTest.b32_s2_nz_fp16_quant",
        "AttentionPostSTest.b64_s1_nz_fp16_quant",
        "AttentionPostSTest.b64_s2_nz_fp16_quant",
        "AttentionPostSTest.b24_s1_nz_fp16_quant",
        "AttentionPostSTest.b24_s2_nz_fp16_quant",
        "AttentionPostSTest.b48_s1_nz_fp16_quant",
        "AttentionPostSTest.b48_s2_nz_fp16_quant",
        "AttentionPostSTest.b96_s1_nz_fp16_quant",
        "AttentionPostSTest.b96_s2_nz_fp16_quant",
        # bf16
        "AttentionPostSTest.b16_s1_nz_bf16_quant",
        "AttentionPostSTest.b16_s2_nz_bf16_quant",
        "AttentionPostSTest.b32_s1_nz_bf16_quant",
        "AttentionPostSTest.b32_s2_nz_bf16_quant",
        "AttentionPostSTest.b64_s1_nz_bf16_quant",
        "AttentionPostSTest.b64_s2_nz_bf16_quant",
        "AttentionPostSTest.b24_s1_nz_bf16_quant",
        "AttentionPostSTest.b24_s2_nz_bf16_quant",
        "AttentionPostSTest.b48_s1_nz_bf16_quant",
        "AttentionPostSTest.b48_s2_nz_bf16_quant",
        "AttentionPostSTest.b96_s1_nz_bf16_quant",
        "AttentionPostSTest.b96_s2_nz_bf16_quant",
        # fp16, nd, quant
        "AttentionPostSTest.b16_s1_nd_fp16_quant",
        "AttentionPostSTest.b16_s2_nd_fp16_quant",
        "AttentionPostSTest.b32_s1_nd_fp16_quant",
        "AttentionPostSTest.b32_s2_nd_fp16_quant",
        # fp16, nz, no quant
        "AttentionPostSTest.b32_s1_nz_fp16",
        "AttentionPostSTest.b32_s2_nz_fp16",
        # fp16, nd, no quant
        "AttentionPostSTest.b16_s1_nd_fp16",
        "AttentionPostSTest.b16_s2_nd_fp16",
        "AttentionPostSTest.b32_s1_nd_fp16",
        "AttentionPostSTest.b32_s2_nd_fp16",
        # fp16, nz, quant all
        "AttentionPostSTest.b16_s1_nz_fp16_quant_all",
        "AttentionPostSTest.b32_s2_nz_bf16_quant_all",
        "AttentionPostSTest.b48_s1_nz_fp16_quant_all",
    ]
)


def gen_post_date(case_name: str, output: Path) -> bool:
    # b, n, s, h, kv_lora_rank, v_head_dim
    # fp16, nz, quant
    fp16_nz_quant_cases = {
        "AttentionPostSTest.b16_s1_nz_fp16_quant": (16, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b16_s2_nz_fp16_quant": (16, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b32_s1_nz_fp16_quant": (32, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nz_fp16_quant": (32, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b64_s1_nz_fp16_quant": (64, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b64_s2_nz_fp16_quant": (64, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b24_s1_nz_fp16_quant": (24, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b24_s2_nz_fp16_quant": (24, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b48_s1_nz_fp16_quant": (48, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b48_s2_nz_fp16_quant": (48, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b96_s1_nz_fp16_quant": (96, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b96_s2_nz_fp16_quant": (96, 128, 2, 7168, 512, 128),
    }
    config = PostConfig((16, 128, 1, 7168, 512, 128), np.float16, False, False, True, True, True)
    if fp16_nz_quant_cases.get(case_name):
        config.params = fp16_nz_quant_cases[case_name]
        gen_post_test_data(output, config)
        return True

    # bf16, nz, quant
    bf16_nz_quant_cases = {
        "AttentionPostSTest.b16_s1_nz_bf16_quant": (16, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b16_s2_nz_bf16_quant": (16, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b32_s1_nz_bf16_quant": (32, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nz_bf16_quant": (32, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b64_s1_nz_bf16_quant": (64, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b64_s2_nz_bf16_quant": (64, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b24_s1_nz_bf16_quant": (24, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b24_s2_nz_bf16_quant": (24, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b48_s1_nz_bf16_quant": (48, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b48_s2_nz_bf16_quant": (48, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b96_s1_nz_bf16_quant": (96, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b96_s2_nz_bf16_quant": (96, 128, 2, 7168, 512, 128),
    }
    config = PostConfig((16, 128, 1, 7168, 512, 128), bfloat16, False, False, True, True, True)
    if bf16_nz_quant_cases.get(case_name):
        config.params = bf16_nz_quant_cases[case_name]
        gen_post_test_data(output, config)
        return True

    # fp16, nd, quant
    fp16_nd_quant_cases = {
        "AttentionPostSTest.b16_s1_nd_fp16_quant": (16, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b16_s2_nd_fp16_quant": (16, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b32_s1_nd_fp16_quant": (32, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nd_fp16_quant": (32, 128, 2, 7168, 512, 128),
    }
    config = PostConfig((16, 128, 1, 7168, 512, 128), np.float16, False, False, True, True, False)
    if fp16_nd_quant_cases.get(case_name):
        config.params = fp16_nd_quant_cases[case_name]
        gen_post_test_data(output, config)
        return True

    # fp16, nz, no quant
    fp16_nz_no_quant_cases = {
        "AttentionPostSTest.b32_s1_nz_fp16": (32, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nz_fp16": (32, 128, 2, 7168, 512, 128),
    }
    config = PostConfig((32, 128, 1, 7168, 512, 128), np.float16, False, False, False, False, True)
    if fp16_nz_no_quant_cases.get(case_name):
        config.params = fp16_nz_no_quant_cases[case_name]
        gen_post_test_data(output, config)
        return True

    # fp16, nd, no quant
    fp16_nd_no_quant_cases = {
        "AttentionPostSTest.b16_s1_nd_fp16": (16, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b16_s2_nd_fp16": (16, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b32_s1_nd_fp16": (32, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nd_fp16": (32, 128, 2, 7168, 512, 128),
    }
    config = PostConfig((16, 128, 1, 7168, 512, 128), np.float16, False, False, False, False, False)
    if fp16_nd_no_quant_cases.get(case_name):
        config.params = fp16_nd_no_quant_cases[case_name]
        gen_post_test_data(output, config)
        return True

    # fp16, nz, quant all
    fp16_nz_quant_all_cases = {
        "AttentionPostSTest.b16_s1_nz_fp16_quant_all": (16, 128, 1, 7168, 512, 128),
        "AttentionPostSTest.b32_s2_nz_bf16_quant_all": (32, 128, 2, 7168, 512, 128),
        "AttentionPostSTest.b48_s1_nz_fp16_quant_all": (48, 128, 1, 7168, 512, 128),
    }
    config = PostConfig((16, 128, 1, 7168, 512, 128), np.float16, True, True, True, True, True)
    if fp16_nz_quant_all_cases.get(case_name):
        config.params = fp16_nz_quant_all_cases[case_name]
        if case_name == "AttentionPostSTest.b32_s2_nz_bf16_quant_all":
            config.dtype = bfloat16
        gen_post_test_data(output, config)
        return True

    logging.error("Can't get func to gen golden, Case(%s)", case_name)
    return False

def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "AttentionPostSTest.b16_s1_nd_fp16_quant",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_post_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    exit(0 if main() else 1)
