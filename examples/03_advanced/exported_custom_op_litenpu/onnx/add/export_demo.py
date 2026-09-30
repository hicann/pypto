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
"""ONNX export demo — elementwise add: the simplest pypto custom-op export.

The export entry point: ``kernel.py`` -> ``op.py`` -> ``model.py`` -> here, which sets up the import
path, traces the model and writes the ``.onnx``. Running the whole model is the sibling entry point
``run_demo.py``. Nothing may import this file: it runs as ``__main__``.
"""
import argparse
import os
import pathlib as _pl
import sys as _sys
import tempfile

import torch

import pypto

# The exported_custom_op_litenpu helper library lives under the repo-root tools/ directory.
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))  # <repo root>/tools
from exported_custom_op_litenpu.export.onnx_export import onnx_export_session, resolve_pypto_export_opset
from model import SHAPE, CustomModel


# trace the model and run torch.onnx.export inside the session.
def export_demo(model, path: str):
    pypto.set_codegen_options(support_dynamic_aligned=True)
    input_data0 = torch.rand(SHAPE, dtype=torch.float16)
    input_data1 = torch.rand(SHAPE, dtype=torch.float16)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # base ai.onnx opset to export with
    opset = resolve_pypto_export_opset(model=model, example_inputs=(input_data0, input_data1))
    # register the custom ops (safe to call more than once), then
    # validate/save the model on exit.
    with onnx_export_session(path):
        torch.onnx.export(
            model, (input_data0, input_data1), path,
            input_names=["x0", "x1"], output_names=["y"],
            opset_version=opset, do_constant_folding=False, dynamo=False, report=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs="?", help="onnx model path (default: a temp dir)")
    args = parser.parse_args()
    # Optional so the repo examples sweep can run this bare (a fresh temp dir when omitted).
    path = args.path or os.path.join(tempfile.mkdtemp(prefix="pypto_add_"), "add.onnx")
    export_demo(CustomModel(), path=path)
