# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for L0C->UB scalar dequant with a DYNAMIC (runtime) scale on A5 CCE.

Dynamic-scale counterpart of test_deq_scalar_move_fp32_to_int8.py. The scale is a runtime kernel
argument `scale_bits: pl.DT_INT32` passed at launch. This is the only test that exercises
the MOVE path's own runtime sign-bit bit-op emit AND the move operand type-dispatch with a runtime
scalar coexisting with acc_to_vec_mode (store's dynamic test does not cover the move codegen).

The scalar carries the low 32 bits = the float32 bit-pattern of the scale; CCE codegen adds the
signed-int8 bit46 flag from the output tile dtype at runtime, so negatives keep their sign.
A5 only (the L0C->UB quant path lives in arch35/copy_l0c_to_ub_impl.h, gated by PTO_NPU_ARCH_A5).
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


FP_SCALE_VALUE = 2.0


def _scale_bits(scale: float) -> int:
    # Low 32 bits = float32 bit-pattern of the scale (scale=2.0 -> 0x40000000).
    return struct.unpack("!I", struct.pack("!f", scale))[0]


def _make_q(device: str) -> torch.Tensor:
    # Includes negatives to validate signed-int8 dequant keeps the sign (-3*2=-6, not 0).
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


@pl.jit()
def deq_scalar_move_dynamic_cce_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    move_out: pl.Tensor[[64, 64], pl.DT_INT8],
    scale_bits: pl.DT_INT32,
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        )
        acc = pl.make_tile(acc_type, addr=0x0000, size=16384)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        # DYNAMIC scale on the L0C->UB move: pre_quant_scalar is the runtime kernel arg, coexisting
        # with acc_to_vec_mode (exercises move's operand type-dispatch + runtime sign-bit emit).
        pl.move(vec_tile, acc, pre_quant_scalar=scale_bits, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(move_out, vec_tile, [0, 0])
        pl.system.bar_all()



@pytest.mark.soc("950")
def test_deq_scalar_move_dynamic_cce():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    move_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    scale_bits = _scale_bits(FP_SCALE_VALUE)  # runtime value, computed on host

    deq_scalar_move_dynamic_cce_kernel(q, k, move_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_out = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    logging.info("***********cce q input (top-left 8x8)***********")
    logging.info("%s", q[:8, :8])
    logging.info("***********cce dynamic scale bits (hex)***********")
    logging.info("0x%X (scale=%s, passed at runtime)", scale_bits, FP_SCALE_VALUE)
    logging.info("***********cce expected move deq int8 output***********")
    logging.info("%s", expected_out[:8, :8])
    logging.info("***********cce dynamic-scale move deq int8 output (top-left 8x8)***********")
    logging.info("%s", move_out[:8, :8])

    torch.testing.assert_close(move_out.to(torch.int32), expected_out.to(torch.int32), rtol=0, atol=0)
    logging.info("cce dynamic-scale move deq_scalar output matches expectation.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_deq_scalar_move_dynamic_cce()
    logging.info("\nAll tests passed!")
