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
"""
GLM-4.5 FFN Router Expert Quantization Module

This module implements the quantized FFN computation for router experts in MoE architecture.
Router experts are dynamically selected based on input token features, allowing the model
to use only a subset of experts while maintaining a large parameter count.

Main Functions:
    - ffn_router_expert_quant: Main function for router expert FFN quantization
    - moe_router_expert_main: JIT compiled kernel for router expert computation
    - expert_infer_base: Base inference function for a single expert
"""
import os
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def symmetric_quantization_per_token(input_tensor):
    """
    Perform symmetric quantization per token (per row).
    
    Args:
        input_tensor: Input tensor to quantize
        
    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_int32 = pypto.cast(x_mul, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    x_fp16 = pypto.cast(x_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    x_int8 = pypto.cast(x_fp16, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
    return x_int8, x_scale_quant


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.
    
    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor
        
    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


def swiglu(up_proj):
    """
    Apply SwiGLU activation function: x * sigmoid(x) * right_half.
    
    Args:
        up_proj: Input tensor with shape [batch, intermediate_size * 2]
        
    Returns:
        SwiGLU activated tensor with shape [batch, intermediate_size]
    """
    intermediate_size = up_proj.shape[1] // 2
    up_proj_left = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, 0])
    up_proj_right = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, intermediate_size])
    swiglu_mul = pypto.mul(up_proj_left, -1.0)
    swiglu_exp = pypto.exp(swiglu_mul)
    swiglu_add = pypto.add(swiglu_exp, 1.0)
    swiglu_div = pypto.div(up_proj_left, swiglu_add)
    swiglu_out = pypto.mul(swiglu_div, up_proj_right)
    return swiglu_out


def check_args(
        hidden_states: torch.Tensor,
        pertoken_scale: torch.Tensor,
        group_list: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor
) -> None:
    """
    Validate input arguments for router expert FFN quantization operation.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        pertoken_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]

    Raises:
        AssertionError: If any input argument doesn't meet the required format or dtype.
    """
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.int8

    assert pertoken_scale.dim() == 1
    assert get_format(pertoken_scale) == 'ND'
    assert pertoken_scale.dtype == torch.float32

    assert group_list.dim() == 1
    assert get_format(group_list) == 'ND'
    assert group_list.dtype == torch.int32

    assert w13.dim() == 3
    assert w13.shape[1] == 5120
    assert w13.shape[2] == 3072
    assert get_format(w13) == 'NZ'
    assert w13.dtype == torch.int8

    assert w13_scale.dim() == 2
    assert w13_scale.shape[1] == 3072
    assert get_format(w13_scale) == 'ND'
    assert w13_scale.dtype == torch.float32

    assert w2.dim() == 3
    assert w2.shape[1] == 1536
    assert w2.shape[2] == 5120
    assert get_format(w2) == 'NZ'
    assert w2.dtype == torch.int8

    assert w2_scale.dim() == 2
    assert w2_scale.shape[1] == 5120
    assert get_format(w2_scale) == 'ND'
    assert w2_scale.dtype == torch.bfloat16


def main():
    test_ffn_router()


def ffn_router_torch_npu(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)
    group_list_cumsum = group_list.cumsum(dim=0)
    output_dtype = w2_scale.dtype
    w13_int8_nz = torch_npu.npu_format_cast(w13, 29)
    hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w13_int8_nz,
                bias=None,
                group_list=group_list_cumsum,
                weight_scale=w13_scale,
                x_scale=hidden_states_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype)[0]
    return hidden_states


def get_token_acc_table(group_list):
    assert len(group_list.shape) == 1
    return (torch.cumsum(group_list, dim=0) - group_list).to(group_list.dtype)


def ffn_golden_quan_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per token (per row).

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per channel (per column) for 3D tensors.

    Note: This function currently uses per-token quantization (dim=1)
    but is kept for backward compatibility.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def gen_input(
    b: int,
    s: int,
    topk: int,
    per_expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    hidden_states = torch.randn((b * s * topk, hidden_size), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    hidden_states, hidden_states_scale = ffn_golden_quan_per_token(hidden_states)
    hidden_states_scale = hidden_states_scale.reshape(-1).to(torch.float32)

    group_list = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        dtype=torch.int32, device=f'npu:{device_id}')
    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    w13 = torch.randn((per_expert_num, hidden_size, intermediate_size * 2), \
        dtype=dtypes, device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    w2 = torch.randn((per_expert_num, intermediate_size, hidden_size), dtype=dtypes, \
        device=f'npu:{device_id}') * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(dtypes)

    ffn_res = torch.empty(hidden_states.shape, dtype=w2_scale.dtype, device=f'npu:{device_id}')
    return hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res


def expert_infer_base(
        hidden_states_params,
        group_list_params,
        w13_params,
        w2_params,
        offset_params,
        tiling_params,
        ffn_res):
    """
    Base inference function for a single expert computation.

    This function performs FFN computation for a specific expert:
    1. Quantized matrix multiplication: up_proj = MatMul(hidden_states, w13)
    2. Dequantization: up_proj_dequant = Dequantize(up_proj, w13_scale, hidden_states_scale)
    3. SwiGLU activation: swiglu_out = SwiGLU(up_proj_dequant)
    4. Quantization: down_proj_quant = Quantize(swiglu_out)
    5. Quantized matrix multiplication: down_proj = MatMul(down_proj_quant, w2)
    6. Dequantization: output = Dequantize(down_proj, w2_scale, down_proj_scale)

    Args:
        hidden_states_params: Tuple of (hidden_states, hidden_states_scale)
        group_list_params: Tuple of (group_list, group_list_cumsum)
        w13_params: Tuple of (w13, w13_scale)
        w2_params: Tuple of (w2, w2_scale)
        offset_params: Tuple of (exp_idx, token_loop_idx, loop_base)
        tiling_params: Tuple of (mm1_cube_tile_shape, mm2_cube_tile_shape)
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function processes tokens in tiles of size loop_base (typically 8)
        to support efficient computation on NPU.
    """
    # 入参信息获取
    hidden_states, hidden_states_scale = hidden_states_params
    group_list, group_list_cumsum = group_list_params
    w13, w13_scale = w13_params
    w2, w2_scale = w2_params
    exp_idx, unroll_offset, unroll_level = offset_params
    mm1_cube_tile_shape, mm2_cube_tile_shape = tiling_params

    hidden_size = hidden_states.shape[1]
    intermediate_size = w13.shape[1] // 2
    x_dtype = w2_scale.dtype

    # 计算对应激活专家的偏移地址
    pypto.set_vec_tile_shapes(32)

    # 获取该激活专家在当前loop参与计算部分，有效token的偏移地址和scale偏移地址
    hidden_states_offset_start = group_list_cumsum[exp_idx, ]
    hidden_states_offset = [hidden_states_offset_start + unroll_offset, 0]
    x_scale_offset = [hidden_states_offset_start + unroll_offset, 0]

    # 获取该激活专家权重和scale的偏移地址
    weight_13_offset = [exp_idx * hidden_size, 0]
    w13_scale_offset = [exp_idx, 0]
    weight_2_offset = [exp_idx * intermediate_size, 0]
    w2_scale_offset = [exp_idx, 0]

    # 获取当前专家的实际token数和scale
    x = pypto.view(hidden_states, [unroll_level, hidden_size], hidden_states_offset)
    x_scale = pypto.view(hidden_states_scale, [unroll_level, 1], x_scale_offset)

    # 获取当前专家的weght_13和scale
    w13_weight_2d = pypto.view(w13, [hidden_size, intermediate_size * 2], weight_13_offset)
    w13_scale_valid = pypto.view(w13_scale, [1, intermediate_size * 2], w13_scale_offset)

    # # 获取当前专家的weght_2和scale
    w2_weight_2d = pypto.view(w2, [intermediate_size, hidden_size], weight_2_offset)
    w2_scale_valid = pypto.view(w2_scale, [1, hidden_size], w2_scale_offset)

    # up_proj的matmul计算
    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm1_cube_tile_shape[1], mm1_cube_tile_shape[1] * 2], \
                               [mm1_cube_tile_shape[2], mm1_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w13_weight_2d.shape[0], w13_weight_2d.shape[1]])
    up_proj = pypto.matmul(x, w13_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(4, intermediate_size * 2)
    up_proj_out = dequant_dynamic(up_proj, w13_scale_valid, x_scale)
    swiglu_out = swiglu(up_proj_out)

    # down_proj
    # quant
    down_proj_quant, down_proj_scale = symmetric_quantization_per_token(swiglu_out)

    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [mm2_cube_tile_shape[1], mm2_cube_tile_shape[1] * 2], \
                               [mm2_cube_tile_shape[2], mm2_cube_tile_shape[2]], True, True)
    pypto.set_matrix_size([unroll_level, w2_weight_2d.shape[0], w2_weight_2d.shape[1]])
    down_proj = pypto.matmul(down_proj_quant, w2_weight_2d, pypto.DT_INT32)

    # dequant
    pypto.set_vec_tile_shapes(4, hidden_size)
    down_proj_dequant = dequant_dynamic(down_proj, w2_scale_valid, down_proj_scale)
    out = pypto.cast(down_proj_dequant, x_dtype)
    pypto.assemble(out, hidden_states_offset, ffn_res)


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"device_sched_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2}
)
def moe_router_expert_main(hidden_states, hidden_states_scale,
                           group_list, group_list_cumsum, w13,
                           w13_scale, w2, w2_scale, ffn_res):
    """
    JIT compiled kernel for router expert FFN quantization.

    This kernel processes multiple experts in a loop, where each expert processes
    a subset of tokens based on the group_list. The computation is done in tiles
    to support efficient execution on NPU.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        hidden_states_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        group_list_cumsum: Cumulative sum of group list [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function uses cube L1 reuse mode 2 for better memory efficiency.
        Each expert processes tokens in tiles of size 8.
    """
    pypto.experimental.set_operation_config(combine_axis=True)

    # tiling config
    mm1_cube_tile_shape = (8, 256, 256)
    mm2_cube_tile_shape = (8, 256, 256)

    # 获取当前device上专家总数
    expert_num = group_list.shape[0]

    # 输入Tensor shape转换为2维
    w13_2d_shape = (w13.shape[0] * w13.shape[1], w13.shape[2])
    w2_2d_shape = (w2.shape[0] * w2.shape[1], w2.shape[2])
    hidden_states_scale_shape = (hidden_states_scale.shape[0], 1)

    w13_2d = pypto.reshape(w13, w13_2d_shape, inplace=True)
    w2_2d = pypto.reshape(w2, w2_2d_shape, inplace=True)
    hidden_states_scale_2d = pypto.reshape(hidden_states_scale, hidden_states_scale_shape, inplace=True)

    for exp_idx in pypto.loop(expert_num, name="LOOP_FFN_ROUTER_MLP_L0", idx_name="exp_idx"):
        # 获取激活专家的token数
        token_num = group_list[exp_idx, ]
        for token_loop_idx, loop_base in pypto.loop_unroll(
            token_num,
            unroll_list=[1, 2, 4, 8, 16, 32],
            name="LOOP_FFN_ROUTER_MLP_L1",
            idx_name="token_loop_idx"):
            expert_infer_base(
                hidden_states_params=[hidden_states, hidden_states_scale_2d],
                group_list_params=[group_list, group_list_cumsum],
                w13_params=[w13_2d, w13_scale],
                w2_params=[w2_2d, w2_scale],
                offset_params=[exp_idx, token_loop_idx, loop_base],
                tiling_params=[mm1_cube_tile_shape, mm2_cube_tile_shape],
                ffn_res=ffn_res
            )


@allow_in_graph
def ffn_router_expert_quant(hidden_states: torch.Tensor,
                            pertoken_scale: torch.Tensor,
                            group_list: torch.Tensor,
                            w13: torch.Tensor,
                            w13_scale: torch.Tensor,
                            w2: torch.Tensor,
                            w2_scale: torch.Tensor,
                            ffn_res: torch.Tensor
) -> None:
    """
    Quantized FFN computation for router experts in MoE architecture.

    This function computes FFN output for router experts using quantized operations.
    Router experts are dynamically selected based on input token features, allowing
    the model to use only a subset of experts while maintaining a large parameter count.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        pertoken_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph. The computation uses grouped matrix
        multiplication to efficiently process multiple experts.
    """
    group_list_int32 = group_list.to(torch.int32)

    group_list_cumsum = (torch.cumsum(group_list_int32, dim=0) - group_list_int32).to(torch.int32)

    inputs = {
        hidden_states: [0],
        pertoken_scale: [0],
        group_list_int32: [],
        group_list_cumsum: [],
        w13: [],
        w13_scale: [],
        w2: [],
        w2_scale: []
    }
    outputs = {
        ffn_res: [0]
    }
    if not isinstance(hidden_states, FakeTensor):
        check_args(hidden_states, pertoken_scale, group_list_int32, w13, w13_scale, w2, w2_scale)
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用


def test_ffn_router() -> None:
    dtype = torch.bfloat16
    # parameter config
    s = 1
    intermediate_size = 1536
    hidden_size = 5120
    per_expert_num = 20
    topk = 8
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    # Test with different batch sizes
    for b in [1]:
        # hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res
        hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtype, device_id)

        inputs = {
            hidden_states: [0],
            hidden_states_scale: [0],
            group_list: [],
            group_list_cumsum: [],
            w13: [],
            w13_scale: [],
            w2: [],
            w2_scale: []
        }
        outputs = {
            ffn_res: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用

        # golden
        golden = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)

        # calc valid token num for compare
        vaild_token_cumsum = group_list.cumsum(dim=0)
        valid_size = vaild_token_cumsum[vaild_token_cumsum.shape[0] - 1] * hidden_size
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()[0: valid_size]), \
                        np.array(golden.cpu().flatten().tolist()[0: valid_size]), rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()