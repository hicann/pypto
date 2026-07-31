# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for pl.move(..., fp_tile=...) on A5 CCE."""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


FP_SCALE_VALUE = 2.0
FP_SCALE_SIGNED_INT8_FLAG = 1 << 46
FP_SCALE_BITS = FP_SCALE_SIGNED_INT8_FLAG | 0x40000000  # signed-int8 + float32 bit-pattern for 2.0f


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_fp_params(device: str) -> torch.Tensor:
    return torch.full((1, 64), FP_SCALE_BITS, device=device, dtype=torch.int64)


def _decode_fp_param(value: int) -> tuple[float, int, int]:
    raw = int(value) & ((1 << 64) - 1)
    scale_bits = raw & 0xFFFFFFFF
    scale = struct.unpack("!f", struct.pack("!I", scale_bits))[0]
    signed_flag = (raw >> 46) & 0x1
    return scale, signed_flag, raw


@pl.jit()
def move_fp_cce_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    fp_params: pl.Tensor[[1, 64], pl.DT_INT64],
    move_fp_out: pl.Tensor[[64, 64], pl.DT_INT8],
):
    vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)

    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        fp_mat_type = pl.TileType(
            shape=[1, 64],
            dtype=pl.DT_INT64,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ND,
        )
        fp_mat = pl.make_tile(fp_mat_type, addr=0x8000, size=512)

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

        fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)

        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.load(fp_mat, fp_params, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc, q_left, k_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

        pl.move(vec_tile, acc, fp_tile=fp_tile, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(move_fp_out, vec_tile, [0, 0])
        pl.system.bar_all()



@pytest.mark.soc("950")
def test_move_fp_cce():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device)
    move_fp_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    move_fp_cce_kernel(q, k, fp_params, move_fp_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_out = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)
    decoded_fp = [_decode_fp_param(v) for v in fp_params[0, :8].tolist()]

    logging.info("***********cce q input (top-left 8x8)***********")
    logging.info("%s", q[:8, :8])
    logging.info("***********cce fp tile source bits (top-left 1x8)***********")
    logging.info("%s", fp_params[:, :8])
    logging.info("***********cce decoded fp tile payload (scale, signed_int8_flag, raw_u64)***********")
    logging.info("%s", decoded_fp)
    logging.info("***********cce expected move_fp int8 output***********")
    logging.info("%s", expected_out[:8, :8])
    logging.info("***********cce move_fp int8 output (top-left 8x8)***********")
    logging.info("%s", move_fp_out[:8, :8])
    logging.info("***********cce move_fp summary***********")
    logging.info(
        "%s",
        {'fp_params_dtype': str(
            fp_params.dtype,
        ), 'move_fp_out_dtype': str(
            move_fp_out.dtype,
        ), 'scale_value': FP_SCALE_VALUE, 'min': int(move_fp_out.min().item()), 'max': int(move_fp_out.max().item())},
    )

    torch.testing.assert_close(move_fp_out.to(torch.int32), expected_out.to(torch.int32), rtol=0, atol=0)
    logging.info("cce move_fp output matches expectation.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_move_fp_cce()
    logging.info("\nAll tests passed!")
