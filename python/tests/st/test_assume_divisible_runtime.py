#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime checks emitted for ``assume_divisible`` in runtime_debug_mode=4."""

import os
import signal
import subprocess
import sys
import tempfile

DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))

_RUNTIME_CASE_SCRIPT = """
import os
import sys

import torch
import pypto

torch.npu.set_device({device_id})

@pypto.frontend.jit(
    debug_options={{"runtime_debug_mode": 4}},
    runtime_options={{"run_mode": pypto.RunMode.NPU}},
)
def kernel(
    source: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    destination: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    for row in pypto.loop(source.shape[0], name="ASSUME_DIVISIBLE_LOOP", idx_name="row"):
        pypto.experimental.assume_divisible(source.shape[0], 128)
        tile = pypto.view(source, [1, 8], [row, 0])
        pypto.assemble(tile, [row, 0], destination)

device = "npu:{device_id}"
source = torch.arange({rows} * 8, dtype=torch.float32, device=device).reshape({rows}, 8)
destination = torch.zeros_like(source)
kernel(source, destination)
torch.npu.synchronize()
assert torch.equal(destination.cpu(), source.cpu())
"""


def _run_isolated(rows):
    script = _RUNTIME_CASE_SCRIPT.format(device_id=DEVICE_ID, rows=rows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name
    try:
        return subprocess.run(
            [sys.executable, script_path],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=300,
        )
    finally:
        os.unlink(script_path)


def test_mode4_assume_divisible_accepts_divisible_runtime_value():
    result = _run_isolated(128)
    assert result.returncode == 0, result.stderr


def test_mode4_assume_divisible_traps_on_violation():
    result = _run_isolated(130)
    assert result.returncode in {-signal.SIGILL, -signal.SIGTRAP}, result.stderr
