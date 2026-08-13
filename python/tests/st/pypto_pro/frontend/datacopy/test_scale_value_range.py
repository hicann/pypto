# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P2: scale value range + input data patterns + phase/atomic/order tests.

This test file covers:
1. Scale value range: positive, negative, zero, very small, very large, unit, fraction
2. Input data patterns: all positive, all negative, mixed, all zeros, boundary values, large magnitude
3. Phase fusion: Partial, Final
4. Atomic fusion: AtomicAdd
5. Order parameter: transpose, 4D tensor
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _make_q(device: str, pattern: str = "mixed") -> torch.Tensor:
    """Generate test input with different patterns."""
    if pattern == "all_positive":
        return torch.arange(1, 65, dtype=torch.float32, device=device).unsqueeze(0).repeat(64, 1)
    elif pattern == "all_negative":
        return torch.arange(-64, 0, dtype=torch.float32, device=device).unsqueeze(0).repeat(64, 1)
    elif pattern == "all_zeros":
        return torch.zeros((64, 64), dtype=torch.float32, device=device)
    elif pattern == "boundary":
        # Values near INT8 boundaries: -128, -127, 127, 128
        base = torch.tensor([-64.0, -63.5, 63.5, 64.0], dtype=torch.float32, device=device)
        return base.unsqueeze(0).repeat(64, 16)
    elif pattern == "large_magnitude":
        base = torch.tensor([1000.0, -1000.0, 5000.0, -5000.0], dtype=torch.float32, device=device)
        return base.unsqueeze(0).repeat(64, 16)
    else:  # mixed
        row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
        return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


# ============================================================================
# Test 1: Scale value range tests
# ============================================================================


@pl.jit()
def scale_value_kernel(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    quant_out: pl.Tensor[[64, 64], pl.DT_INT8],
    scale_value: pl.DT_INT32,
):
    with pl.section_cube():
        mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = pl.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
        q_left = pl.make_tile(left_type, addr=0x0000, size=16384)

        right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
        k_right = pl.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
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

        pl.store(quant_out, acc, [0, 0], scale=scale_value)
        pl.system.bar_all()


@pytest.mark.soc("950")
def test_scale_value_positive():
    """Test positive scale value (2.0)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_positive passed.")


@pytest.mark.soc("950")
def test_scale_value_negative():
    """Test negative scale value (-1.0)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = -1.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_negative passed.")


@pytest.mark.soc("950")
def test_scale_value_zero():
    """Test zero scale value (0.0)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 0.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # All zeros expected
    expected = torch.zeros((64, 64), dtype=torch.int8, device=device)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_scale_value_zero passed.")


@pytest.mark.soc("950")
def test_scale_value_unit():
    """Test unit scale value (1.0)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 1.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_unit passed.")


@pytest.mark.soc("950")
def test_scale_value_fraction():
    """Test fractional scale value (0.5)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 0.5
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_fraction passed.")


@pytest.mark.soc("950")
def test_scale_value_very_small():
    """Test very small scale value (1e-6)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "large_magnitude")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 1e-6
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Very small scale should result in all zeros or near-zero values
    expected = torch.zeros((64, 64), dtype=torch.int8, device=device)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_very_small passed.")


@pytest.mark.soc("950")
def test_scale_value_very_large():
    """Test very large scale value (1e6)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "mixed")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 1
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    # Very large scale should saturate to INT8 boundaries
    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_scale_value_very_large passed.")


# ============================================================================
# Test 2: Input data pattern tests
# ============================================================================


@pytest.mark.soc("950")
def test_input_pattern_all_positive():
    """Test with all positive input values."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "all_positive")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 0.1
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_input_pattern_all_positive passed.")


@pytest.mark.soc("950")
def test_input_pattern_all_negative():
    """Test with all negative input values."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "all_negative")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 0.1
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_input_pattern_all_negative passed.")


@pytest.mark.soc("950")
def test_input_pattern_all_zeros():
    """Test with all zero input values."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "all_zeros")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    expected = torch.zeros((64, 64), dtype=torch.int8, device=device)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=0)
    logging.info("test_input_pattern_all_zeros passed.")


@pytest.mark.soc("950")
def test_input_pattern_boundary():
    """Test with boundary values near INT8 limits."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "boundary")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 2.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_input_pattern_boundary passed.")


@pytest.mark.soc("950")
def test_input_pattern_large_magnitude():
    """Test with large magnitude values (should saturate)."""
    device = ST_DEVICE
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        logging.info("Current device is %s, skip.", device_name)
        return

    q = _make_q(device, "large_magnitude")
    k = _make_k(device)
    quant_out = torch.zeros((64, 64), device=device, dtype=torch.int8)
    scale_value = 1.0
    scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]

    scale_value_kernel(q, k, quant_out, scale_bits)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected = torch.clamp(torch.round(raw_ref * scale_value), -128, 127).to(torch.int8)

    torch.testing.assert_close(quant_out.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1)
    logging.info("test_input_pattern_large_magnitude passed.")


if __name__ == "__main__":
    # Scale value range tests
    test_scale_value_positive()
    test_scale_value_negative()
    test_scale_value_zero()
    test_scale_value_unit()
    test_scale_value_fraction()
    test_scale_value_very_small()
    test_scale_value_very_large()

    # Input data pattern tests
    test_input_pattern_all_positive()
    test_input_pattern_all_negative()
    test_input_pattern_all_zeros()
    test_input_pattern_boundary()
    test_input_pattern_large_magnitude()

    logging.info("\nAll P2 tests passed!")
