#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-bundle scenario worker: the pack half and the consume half of each .pyptokb case.

Run as a subprocess by ``test_kernel_bundle.py``; never imported by the pytest process itself.
Two phases must be separate processes because the consume half plays the role of a downstream that only
has ``libtile_fwk_bundle.so`` + a ``.pyptokb`` -- no compiler front end, no recorded Program state.

    python3 bundle_cases.py pack    <case> <bundle_path>
    python3 bundle_cases.py consume <case> <bundle_path> [--api path|memory]

Cases:
  static_add       non-value-dependent, static workspace (0 B); packed through the emulation hook, so the
                   bundle carries a NON-EMPTY CtrlFlowCache snapshot.
   dyn_cellmatch    non-value-dependent but dynamically shaped: SymbolMeta carries real symbolic trees plus
                   the dynamic cell-match launch metas, so workspace is re-evaluated per launch shape.
"""

import argparse
import ctypes
import os
import sys

# This worker runs as a standalone script (python3 bundle_cases.py ...), not under pytest, so the tests root
# has to reach sys.path before the st.* imports below -- hence the E402 waivers, as elsewhere in the tree.
_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401  -- registers the npu backend

import pypto  # noqa: E402
from st.bundle.bundle_abi import BundleClient, make_desc_array  # noqa: E402
from st.test_cellmatch_case import B_STATIC, D_STATIC, H_STATIC, k_tmp_to_d_emb  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------
def _device():
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(dev_id)
    return dev_id, f"npu:{dev_id}"


def _alloc_workspace(size: int, device: str):
    """Allocate the caller-owned workspace. Returns (keepalive_tensor, void_ptr); (None, None) when 0."""
    if size == 0:
        return None, None
    buf = torch.empty(size, dtype=torch.uint8, device=device)
    return buf, ctypes.c_void_p(buf.data_ptr())


def _max_diff(actual: torch.Tensor, golden: torch.Tensor) -> float:
    return (actual.detach().cpu().float() - golden.detach().cpu().float()).abs().max().item()


def _check(label: str, actual: torch.Tensor, golden: torch.Tensor, tol: float):
    diff = _max_diff(actual, golden)
    print(f"[ST] {label}: max diff = {diff:.6f} (tol {tol})")
    assert diff < tol, f"{label}: output mismatch, max_diff={diff} >= {tol}"


# ---------------------------------------------------------------------------------------------------
# case: static_add -- non-value-dependent, static, zero workspace
# ---------------------------------------------------------------------------------------------------
ADD_SHAPE = (1, 4, 1, 64)


def _make_add_kernel():
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = x + y

    return add_kernel


def _add_inputs(device: str):
    torch.manual_seed(7)
    x = torch.rand(ADD_SHAPE, dtype=torch.float32, device=device)
    y = torch.rand(ADD_SHAPE, dtype=torch.float32, device=device)
    out = torch.zeros(ADD_SHAPE, dtype=torch.float32, device=device)
    return x, y, out


def pack_static_add():
    _, device = _device()
    x, y, out = _add_inputs(device)
    _make_add_kernel()(x, y, out)
    torch.npu.synchronize()
    _check("pack-run add", out, x + y, 3e-3)


def consume_static_add(bundle_path: str, api: str):
    _, device = _device()
    client = BundleClient(bundle_path, api)
    x, y, out = _add_inputs(device)

    # Unified operand list: outputs are folded into the input list, addressed by position.
    descs = make_desc_array([x, y, out])

    # 1) workspace query -- pure host, must not touch the device.
    ws_size = client.workspace(descs)
    print(f"[ST] workspace size = {ws_size}")
    assert ws_size == 0, f"static add should need no workspace, got {ws_size}"

    # 2) launch.
    ws_keep, ws_ptr = _alloc_workspace(ws_size, device)
    rc = client.launch(descs, ws_ptr, None, 1)
    assert rc == 0, f"PyptoLaunch rc={rc}"
    torch.npu.synchronize()
    del ws_keep

    _check(f"add ({api} API)", out, x + y, 3e-3)


# ---------------------------------------------------------------------------------------------------
# case: dyn_cellmatch -- non-value-dependent, dynamic shapes + dynamic cell-match workspace
# ---------------------------------------------------------------------------------------------------
# Reuses the kernel and the shape constants from the dynamic cell-match ST so the two cannot drift apart.
CM_PACK_L = 125
# Host-only workspace probes. The v1 CtrlFlowCache is a single static-shape snapshot, so only the packed
# shape may actually be launched -- see the note in test_kernel_bundle.py.
CM_PROBE_LENGTHS = [64, 125, 2000]


def _cm_inputs(seq_len: int, device: str):
    torch.manual_seed(44 + seq_len)
    dy = torch.randn(B_STATIC, seq_len, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    weight = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)
    d_emb = torch.zeros(B_STATIC, seq_len, D_STATIC, dtype=torch.float32, device=device)
    return dy, weight, d_emb


def _cm_golden(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (dy[:, :, 0, :].reshape(-1, dy.shape[-1]) @ weight[0].T).reshape(dy.shape[0], dy.shape[1], dy.shape[-1])


def pack_dyn_cellmatch():
    _, device = _device()
    dy, weight, d_emb = _cm_inputs(CM_PACK_L, device)
    k_tmp_to_d_emb(dy, weight, d_emb)
    torch.npu.synchronize()
    _check(f"pack-run cellmatch L={CM_PACK_L}", d_emb, _cm_golden(dy, weight), 1e-3)


def consume_dyn_cellmatch(bundle_path: str, api: str):
    _, device = _device()
    client = BundleClient(bundle_path, api)

    # 1) Dynamic workspace evaluation: the same bundle, queried at several shapes. This exercises the
    #    SymbolMeta symbolic trees (RUNTIME_GetInputShapeDim over ARG_dy) and the dynamic cell-match
    #    stride re-patching. A bundle with a baked constant would return the same number every time.
    sizes = []
    for seq_len in CM_PROBE_LENGTHS:
        dy, weight, d_emb = _cm_inputs(seq_len, device)
        ws = client.workspace(make_desc_array([dy, weight, d_emb]))
        print(f"[ST] workspace(L={seq_len}) = {ws}")
        sizes.append(ws)
        del dy, weight, d_emb
    assert all(s > 0 for s in sizes), f"dynamic workspace must be non-zero, got {sizes}"
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), (
        f"workspace must grow strictly with the sequence length, got "
        f"{dict(zip(CM_PROBE_LENGTHS, sizes))} -- symbol evaluation likely fell back to a baked constant"
    )

    # 2) Launch at the packed shape and check the numerics end to end.
    dy, weight, d_emb = _cm_inputs(CM_PACK_L, device)
    descs = make_desc_array([dy, weight, d_emb])
    ws_size = client.workspace(descs)
    ws_keep, ws_ptr = _alloc_workspace(ws_size, device)
    assert ws_size > 0, "expected a non-zero workspace at the packed shape"
    rc = client.launch(descs, ws_ptr, None, 1)
    assert rc == 0, f"PyptoLaunch rc={rc}"
    torch.npu.synchronize()
    del ws_keep

    _check(f"cellmatch L={CM_PACK_L} ({api} API)", d_emb, _cm_golden(dy, weight), 1e-3)


# ---------------------------------------------------------------------------------------------------
# registry + CLI
# ---------------------------------------------------------------------------------------------------
CASES = {
    "static_add": (pack_static_add, consume_static_add),
    "dyn_cellmatch": (pack_dyn_cellmatch, consume_dyn_cellmatch),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pack", "consume"])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("bundle_path")
    parser.add_argument("--api", choices=["path", "memory"], default="path")
    args = parser.parse_args()

    pack_fn, consume_fn = CASES[args.case]
    if args.phase == "pack":
        # Packing is gated by these two env vars; the driver sets them, assert rather than silently no-op.
        assert os.environ.get("PYPTO_ENABLE_KERNEL_BUNDLE") == "1", "PYPTO_ENABLE_KERNEL_BUNDLE must be 1"
        assert os.environ.get("PYPTO_KERNEL_BUNDLE_PATH") == args.bundle_path, (
            "PYPTO_KERNEL_BUNDLE_PATH must match the requested bundle path"
        )
        pack_fn()
        assert os.path.exists(args.bundle_path), f"no bundle produced at {args.bundle_path}"
        print(f"[ST] packed {args.case} -> {args.bundle_path} ({os.path.getsize(args.bundle_path)} B)")
    else:
        consume_fn(args.bundle_path, args.api)
        print(f"[ST] consumed {args.case} via the {args.api} API")


if __name__ == "__main__":
    main()
    # The runtime singletons (DeviceRunner/DevicePerf) are torn down at interpreter exit, AFTER torch_npu has
    # already dropped the NPU context -- that races into a double free in ~DevicePerf. Everything under test
    # has been asserted by this point, so flush and exit hard rather than run the static destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
