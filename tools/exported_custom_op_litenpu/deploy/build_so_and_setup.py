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
"""Build the custom-op .so from an ONNX model, then install it where ONNX can find it.

Two steps, both thin wrappers over the deploy API (scripting callers should use
that API directly rather than shelling out to this file):

  1. build_so_from_model  — finds every pypto custom-op node in the model and builds one
                            libcust_opapi.so against $ASCEND_HOME_PATH.
  2. setup_onnx_custom_op_so — installs the .so where ONNX looks for it and prints the
                            ASCEND_CUSTOM_OPP_PATH hint.
"""
import argparse
import logging
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo root>/tools
from exported_custom_op_litenpu.deploy import build_so_from_model, setup_onnx_custom_op_so

# Console output for this script: its own stdout handler at INFO, no root propagation.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _console = logging.StreamHandler(_sys.stdout)
    _console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_console)
logger.setLevel(logging.INFO)
logger.propagate = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .onnx model")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output project directory; default is next to input model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug symbols + exported symbols (PYPTO_DEBUG_SYMBOLS=ON) "
             "so a crash inside the produced .so (e.g. a fault during dlopen) gives "
             "a resolvable backtrace. Default off (optimized, stripped build).",
    )
    args = parser.parse_args()

    # Builds one libcust_opapi.so from every pypto node in the model (clean rebuild). More options
    # (op subset / output name / Ascend-home / incremental builds) are available on build_so_from_model.
    model_path = Path(args.path).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else model_path.parent / "build_op_kernel_lib"
    )

    logger.info("\nLoading model from: %s", model_path)
    so, kernel_py_paths = build_so_from_model(
        model_path,
        out_dir=out_dir,
        clean=True,
        debug=args.debug,
    )

    logger.info("\n[DONE BUILD]")
    logger.info("Project directory: %s", out_dir)
    logger.info("Built module:")
    logger.info("  %s", so)

    # ONNX setup: install the built .so where ONNX looks for it + ship each op's dev-editable snippet.
    logger.info("\n[SETUP]")
    setup_onnx_custom_op_so(so, kernel_py_paths=kernel_py_paths)


if __name__ == "__main__":
    main()
