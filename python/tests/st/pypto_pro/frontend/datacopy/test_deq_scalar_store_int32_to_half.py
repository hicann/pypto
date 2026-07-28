# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for int32->half fixpipe dequant (DEQF16) on A5 CCE.

This is the mainstream dequant scenario: int8 x int8 matmul accumulates to int32 in L0C
(the quant-domain integer), then fixpipe multiplies by a scalar scale during the L0C->GM
copy-out, producing half. Mirrors mainline test_operator_fixpipe.cpp 's322f16_scalar'
(QuantMode_t::DEQF16); PyPTO supports it via copy_l0c_to_gm_impl.h supportedQuantMode
(tileData int32 && globalData half).

deqScalar encoding: reuse the float32 bit-pattern of the scale (scale=2.0f -> 0x40000000),
same low-32-bit form that the validated FP32->int8 path uses, since both go through the
same pto::TSTORE(dst, src, preQuantScalar). If int32->half (DEQF16) turns out to need a
different scalar encoding, the on-device output will reveal it and we adjust here.
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
DEQ_SCALAR_BITS = struct.unpack("!I", struct.pack("!f", FP_SCALE_VALUE))[0]  # 0x40000000


def _make_q(device: str) -> torch.Tensor:
    # int8 inputs incl. negatives; matmul with int8 identity keeps values, so int32 L0C == q.
    row = torch.tensor([-3, -2, -1, 0, 1, 2, 3, 4], device=device, dtype=torch.int8).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.int8)


@pl.jit()
def deq_scalar_store_int32_to_half_kernel(
    q: pl.Tensor[[64, 64], pl.DT_INT8],
    k: pl.Tensor[[64, 64], pl.DT_INT8],
    deq_out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=4096)
        k_mat = pl.make_tile(mat_type, addr=0x2000, size=4096)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=4096)

        right_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
        )
        k_right = pl.make_tile(right_type, addr=0x0000, size=4096)

        acc_type = pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_INT32,
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
        # int32 L0C -> half GM with scalar dequant (DEQF16). Caller passes low-32-bit scale.
        pl.store(deq_out, acc, [0, 0], pre_quant_scalar=DEQ_SCALAR_BITS)
        pl.system.bar_all()



@pytest.mark.soc("950")
def test_deq_scalar_store_int32_to_half():
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device)
    k = _make_k(device)
    deq_out = torch.zeros((64, 64), device=device, dtype=torch.float16)

    deq_scalar_store_int32_to_half_kernel(q, k, deq_out)
    torch.npu.synchronize()

    # int32 matmul (quant-domain), then dequant by scale -> half.
    # NOTE: torch matmul on NPU rejects int32 (cube only supports float types), so compute the
    # golden on CPU. k is the identity matrix, so matmul(q, k) == q; we still do the explicit
    # matmul on CPU to keep the reference faithful rather than assuming the identity shortcut.
    raw_ref = torch.matmul(q.cpu().to(torch.int32), k.cpu().to(torch.int32))
    expected_deq = (raw_ref.to(torch.float32) * FP_SCALE_VALUE).to(torch.float16).to(device)

    logging.info("***********cce q input int8 (top-left 8x8)***********")
    logging.info("%s", q[:8, :8])
    logging.info("***********cce deq scalar bits (hex)***********")
    logging.info("0x%X (scale=%s)", DEQ_SCALAR_BITS, FP_SCALE_VALUE)
    logging.info("***********cce int32 matmul (quant-domain, top-left 8x8)***********")
    logging.info("%s", raw_ref[:8, :8])
    logging.info("***********cce expected deq half output***********")
    logging.info("%s", expected_deq[:8, :8])
    logging.info("***********cce deq half output (top-left 8x8)***********")
    logging.info("%s", deq_out[:8, :8])

    torch.testing.assert_close(deq_out, expected_deq, rtol=1e-3, atol=1e-3)
    logging.info("cce int32->half deq_scalar output matches expectation.")


if __name__ == "__main__":
    test_deq_scalar_store_int32_to_half()
    logging.info("\nAll tests passed!")
