# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for pl.store(..., pre_quant_scalar=...) on A5 CCE.

Exercises the *scalar* fixpipe dequant path (deqScalar) — the counterpart to the
per-channel ``fp_tile`` path in test_store_fp_cce.py. Until now no kernel test
drove ``pre_quant_scalar``; this UT validates L0C(FP32) -> GM(int8) quant with a
scalar scale.

Encoding note (copy_l0c_to_gm_impl.h:44-47): the framework clears bit46 and
auto-sets the signed-int8 flag from the *output* dtype, so the caller passes only
the float32 bit-pattern of the scale (low 32 bits). For scale=2.0f that is
0x40000000, which fits the 32-bit IR attr.
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
# deqScalar low 32 bits = float32 bit-pattern of the scale. For scale=2.0f -> 0x40000000.
#
# Signed-int8 flag (bit 46): the CCE codegen now re-derives bit46 from the OUTPUT dtype
# (backend_cce_block_out_ops.cpp MakeBlockOutStoreCodegenCCE): when the destination is
# signed INT8 it emits preQuant = (scale & ~(1<<46)) | (1<<46), mirroring TStoreExecute in
# copy_l0c_to_gm_impl.h. So the caller passes only the low-32-bit scale here and the sign flag
# is added automatically — this is why negative L0C values now quantize correctly to signed int8
# instead of saturating to 0 (the bug this fix addresses). The 32-bit IR attr / pybind boundary
# truncation no longer matters because the high bit is reconstructed in codegen.
DEQ_SCALAR_BITS = struct.unpack("!I", struct.pack("!f", FP_SCALE_VALUE))[0]  # 0x40000000


def _make_q(device: str) -> torch.Tensor:
    # Includes negatives on purpose: validates that signed-int8 dequant keeps the sign.
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


@pl.jit()
def deq_scalar_store_fp32_to_int8_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    raw_out: pl.Tensor[[64, 64], pl.DT_FP32],
    deq_out: pl.Tensor[[64, 64], pl.DT_INT8],
):
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
        # Reference path: raw FP32 acc -> GM (no quant).
        pl.store(raw_out, acc, [0, 0])
        # Scalar dequant path: acc -> GM int8 with deqScalar = float32(2.0) bit-pattern.
        pl.store(deq_out, acc, [0, 0], pre_quant_scalar=DEQ_SCALAR_BITS)
        pl.system.bar_all()



@pytest.mark.soc("950")
def test_deq_scalar_store_fp32_to_int8():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    raw_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    deq_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    deq_scalar_store_fp32_to_int8_kernel(q, k, raw_out, deq_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_deq = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)

    logging.info("***********cce q input (top-left 8x8)***********")
    logging.info("%s", q[:8, :8])
    logging.info("***********cce deq scalar bits (hex)***********")
    logging.info("0x%X (scale=%s)", DEQ_SCALAR_BITS, FP_SCALE_VALUE)
    logging.info("***********cce raw acc->gm output (top-left 8x8)***********")
    logging.info("%s", raw_out[:8, :8])
    logging.info("***********cce expected deq_scalar int8 output***********")
    logging.info("%s", expected_deq[:8, :8])
    logging.info("***********cce deq_scalar int8 output (top-left 8x8)***********")
    logging.info("%s", deq_out[:8, :8])

    torch.testing.assert_close(raw_out, raw_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(deq_out.to(torch.int32), expected_deq.to(torch.int32), rtol=0, atol=0)
    logging.info("cce raw store and store deq_scalar outputs both match expectations.")


if __name__ == "__main__":
    test_deq_scalar_store_fp32_to_int8()
    logging.info("\nAll tests passed!")
