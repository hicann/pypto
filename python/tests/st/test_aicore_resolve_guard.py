#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file with in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Guard test for AICORE dependency resolve (DRCO) scheduling path.

Enabled by env var ENABLE_AICORE_RESOLVE=true, this path switches the kernel
entry from KernelEntry to KernelEntryDrco, which resolves cross-core
dependencies at runtime via ready queues instead of static scheduling.

The DRCO logic is executed in a subprocess to avoid polluting other cases:
ENABLE_AICORE_RESOLVE is read by C++ as a static-const flag (common.h)
and as a compile cflag (aicore_compiler.cpp), so once import pypto completes
the flag is frozen for the entire process.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

import pypto

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """\
    import os

    os.environ["ENABLE_AICORE_RESOLVE"] = "true"

    from numpy.testing import assert_allclose
    import torch

    import pypto


    @pypto.options(pass_options={"enable_slice": True})
    def run_aicore_resolve_sigmoid_fp32():
        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)
        x_shape = [4, 4]
        dtype = pypto.DT_FP32
        pypto.runtime._device_init()
        x = pypto.tensor(x_shape, dtype)
        res = pypto.tensor(x_shape, dtype)

        with pypto.function("AICORE_RESOLVE_SIGMOID_FP32", x, res):
            for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
                pypto.set_vec_tile_shapes(4, 4)
                res.move(pypto.sigmoid(x))

        x_tensor = torch.rand(4, 4, dtype=torch.float32) * 200 - 100
        res_tensor = torch.zeros(4, 4, dtype=torch.float32)
        pto_x_tensor = pypto.from_torch(x_tensor, "x_tensor")
        pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
        pypto.runtime._device_run_once_data_from_host(pto_x_tensor, pto_res_tensor)

        expected = torch.sigmoid(x_tensor)
        assert_allclose(res_tensor.flatten(), expected.flatten(), atol=1e-3, verbose=True)

        pypto.runtime._device_fini()

    run_aicore_resolve_sigmoid_fp32()
    print("SUBPROCESS_OK")
    """
)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": True})
def test_aicore_resolve_sigmoid_fp32():
    """Guard: sigmoid produces correct result under DRCO scheduling path"""
    env = os.environ.copy()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_SUBPROCESS_SCRIPT)
        script_path = f.name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as err_f:
        err_path = err_f.name
    try:
        with open(err_path, "w") as err_file:
            result = subprocess.run(
                [sys.executable, script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                timeout=300,
            )
        if result.returncode != 0:
            with open(err_path) as ef:
                err_tail = ef.read()[-8000:]
            raise AssertionError(
                f"subprocess failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr(tail):\n{err_tail}"
            )
    finally:
        os.unlink(script_path)
        os.unlink(err_path)
    assert "SUBPROCESS_OK" in result.stdout, f"unexpected output:\n{result.stdout}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-p", "no:cacheprovider"]))
