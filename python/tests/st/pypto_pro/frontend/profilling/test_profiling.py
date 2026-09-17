#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collect one PyPTO trace and validate both kernel details and trace events."""

import csv
import glob
import json
import logging
import os
import sys

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

pytestmark = pytest.mark.skip_jit_discovery(
    reason="Profiler collection and analysis need real launches and run once in the execution phase"
)


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _build_experimental_config():
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    return experimental_config_cls(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


# =============================================================================
# 被采集的 kernel — 简单 add 操作
# =============================================================================
@pl.jit()
def prof_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tt, addr=0x0000)
    tb = pl.make_tile(tt, addr=0x4000)
    tc = pl.make_tile(tt, addr=0x8000)
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_profiler_api_outputs(tmp_path):
    """One real collection supplies kernel_details.csv and a nonempty JSON trace.

    Warm up the JIT before opening the profiler, then capture one completed launch.
    The CSV still identifies the custom kernel; the trace must contain valid events.
    """
    _check_npu()
    output_dir = str(tmp_path / "profiling_output")
    x = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    y = torch.randn(64, 64, device=ST_DEVICE, dtype=torch.float32)
    z = torch.zeros(64, 64, device=ST_DEVICE, dtype=torch.float32)
    prof_add_kernel(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=_build_experimental_config(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir, analyse_flag=True),
    ) as prof:
        prof_add_kernel(x, y, z)
        torch.npu.synchronize()
        prof.step()
    torch.testing.assert_close(z, x + y)

    kernel_detail_files = glob.glob(os.path.join(output_dir, "**", "kernel_details.csv"), recursive=True)
    assert kernel_detail_files, f"No kernel_details.csv under {output_dir}"
    matches = []
    for csv_file in kernel_detail_files:
        with open(csv_file, "r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if any("prof_add_kernel" in row.get(column, "") for column in ("Type", "Name")):
                    matches.append(row)
    assert matches, f"No prof_add_kernel record in {kernel_detail_files}"
    logging.info("PyPTO kernel details: %s", matches)

    trace_files = glob.glob(os.path.join(output_dir, "**", "trace_view.json"), recursive=True)
    if not trace_files:
        trace_files = glob.glob(os.path.join(output_dir, "**", "trace_result.json"), recursive=True)
    assert trace_files, f"No trace_view.json or trace_result.json under {output_dir}"
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as stream:
            trace = json.load(stream)
        events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        assert isinstance(events, list) and events, f"No trace events in {trace_file}"
        logging.info("Trace file: %s, events=%d", trace_file, len(events))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
