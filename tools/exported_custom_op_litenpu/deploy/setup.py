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
"""Post-build setup: place a built custom-op ``.so`` where ATC/GE can discover it.

Separate from the build (``codegen`` generates, ``build`` compiles, ``setup`` deploys) -
the build stays artifact-only and target-agnostic; this module owns the deploy layout.

The ONNX flow needs a setup step (hence the single ONNX-specific helper below).
"""
import importlib.util
import logging
import os
from pathlib import Path
import shutil

from .embed import check_pypto_package_version

logger = logging.getLogger(__name__)

__all__ = ("setup_onnx_custom_op_so",)


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy *src* onto *dest* via a sibling ``.tmp`` + ``os.replace``, so *dest* is never partial.

    GE scans the deploy tree by path, so a reader either sees the previous file or the complete new
    one, never a half-written library or snippet.
    """
    tmp_dest = dest.with_name(dest.name + ".tmp")
    shutil.copy2(str(src), str(tmp_dest))
    os.replace(str(tmp_dest), str(dest))


def _smoke_import_kernel_snippet(stem: str, py_path: Path, py_dest: Path) -> None:
    """Import *py_path* and assert it exposes a callable ``__pypto_compile``.

    The infer funcs live in the same module by construction, so that one symbol is a sufficient
    smoke. Run on the build's copy before anything is placed, so a syntax error / bad snippet fails
    at INSTALL with nothing deployed, rather than at ATC; *py_dest* is the install location the
    failure message names.
    """
    try:
        spec = importlib.util.spec_from_file_location(f"_pypto_install_smoke_{stem}", py_path)
        smoke_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_mod)
    except Exception as exc:  # noqa: BLE001 - surface any import failure as a clear install error
        raise RuntimeError(
            f"[SETUP] kernel snippet for {py_dest} failed to import: {py_path} ({exc})"
        ) from exc
    if not callable(getattr(smoke_mod, "__pypto_compile", None)):
        raise RuntimeError(
            f"[SETUP] kernel snippet for {py_dest} does not expose a callable "
            f"__pypto_compile: {py_path}"
        )


def setup_onnx_custom_op_so(so_path, *, kernel_py_paths, verify: bool = True) -> Path:
    """Place a built ONNX custom-op ``.so`` for ATC/GE's ONNX conversion flow.

    A full ATC conversion of an ONNX graph with our custom op needs the library
    discoverable by *three* GE mechanisms, each scanning a different subdir of an
    ``ASCEND_CUSTOM_OPP_PATH`` entry (verified against the GE sources):

    1. ``<entry>/op_proto/``, the op-proto registry. ``OpsProtoManager`` (driven
       from ``GetOpsProtoPath`` -> ``GetPluginPathFromCustomOppPath("op_proto/", …)``)
       dlopens every ``.so`` under ``op_proto/`` early, during
       ``aclgrphBuildInitialize`` -> ``LoadOpsProto()``, *before* graph build. This is
       what runs our ``REG_OP`` static registrar in time for the compile-time
       InferShape pass to find the executor's inference methods. **Without this copy the
       output shape is never inferred and ``CheckStaticShape`` fails with
       "output[0] is unknown shape, not supported".**
    2. ``<entry>/framework/onnx/``, the ONNX parser plugin (parse phase): matches
       the custom op_type while parsing the ``.onnx``.
    3. ``<entry>`` (root), ``OpLibRegistry`` loads ``<entry>/libcust_opapi.so`` directly
       at op-execution/kernel-API time. Note this load happens *late* (inside the custom
       engine's ``GenerateTask``, after ``CheckStaticShape``), so it cannot substitute for
       the ``op_proto/`` copy, hence (1) is required even though the same registrars also
       live in the root copy.

    The single combined ``.so`` serves all three roles, so it is copied into ``op_proto/``
    and ``framework/onnx/`` while the original stays at the root for ``OpLibRegistry``.

    Parameters
    ----------
    so_path
        Path to the built library (e.g. ``<out_dir>/build/libcust_opapi.so`` as returned
        by ``deploy.build_so_from_model``). Its parent directory is treated
        as the ``ASCEND_CUSTOM_OPP_PATH`` entry root, and must carry the ``pypto_version.info`` the
        build writes there: an absent file, or one recording a pypto newer than the installing one,
        fails the install before anything is placed. A plain ``version.info`` there also fails it — GE
        version-gates the entry on that filename and silently drops the package when it is out of range.
    kernel_py_paths
        Iterable of ``(op_type, stem, src_py)`` (as returned by
        ``deploy.build_so_from_model``). Each op's dev-editable kernel snippet ``src_py`` is
        COPIED (never moved, the build's source tree keeps its copy) to the single canonical
        ``<opp_root>/op_kernel/<stem>.py``, where ``PtoCustomOp::Compile`` finds it (via the
        ``ASCEND_CUSTOM_OPP_PATH`` entries, ``<entry>/op_kernel/<stem>.py``, with a ``dladdr``
        upward-walk fallback) and loads it by path so a developer can edit it in place, the change
        takes effect on the next op-compile, NO ``.so`` rebuild. This ``.py`` is the single source of
        truth and is REQUIRED (the ``.so`` errors fatally at op-compile if it is unresolved), so each is
        install-verified below: a hard existence check at ``<opp_root>/op_kernel/<stem>.py`` (the
        resolver's search location) plus (when *verify*) an import smoke asserting a callable
        ``__pypto_compile``.
    verify
        Run the import smoke on each installed snippet (import it + assert a callable ``__pypto_compile``),
        so a syntax error / bad snippet fails at INSTALL, not at ATC. Default True. Set False on a
        pypto-less box to skip the import (the existence check still runs).

    Returns
    -------
    pathlib.Path
        The deployed path under ``framework/onnx/`` (the parse-phase plugin location).
    """
    so_path = Path(so_path).resolve()
    if not so_path.is_file():
        raise FileNotFoundError(f"custom-op .so not found: {so_path}")

    opp_root = so_path.parent  # the ASCEND_CUSTOM_OPP_PATH entry (the build dir)
    op_kernel_dir = opp_root / "op_kernel"
    kernel_py_paths = [(op_type, stem, Path(src_py)) for op_type, stem, src_py in kernel_py_paths]

    # An older pypto must not install a package a newer one built. Part of the same pre-placement
    # verification as the snippet checks below: a refusal here leaves the deploy tree untouched.
    check_pypto_package_version(opp_root, what="install")

    # Verify every kernel snippet BEFORE anything is placed: a failed verification then leaves the
    # deploy tree untouched instead of a deployed-but-snippetless package, which is exactly the
    # late ATC-time failure this check exists to prevent. The build's copy is byte-identical to what
    # is installed below, so verifying it verifies the install.
    for _op_type, stem, src_py in kernel_py_paths:
        if not src_py.is_file():
            raise FileNotFoundError(f"kernel snippet .py not found for {stem}: {src_py}")
        if verify:
            _smoke_import_kernel_snippet(stem, src_py, op_kernel_dir / f"{stem}.py")

    # (1) op_proto/ - compile-phase InferShape: OpsProtoManager scans <entry>/op_proto/
    # early (during aclgrphBuildInitialize), so our REG_OP registrar runs before the InferShape
    # pass. Always a copy: so_path must remain at the root for OpLibRegistry.
    op_proto_dir = opp_root / "op_proto"
    op_proto_dir.mkdir(parents=True, exist_ok=True)
    op_proto_dest = op_proto_dir / so_path.name
    _atomic_copy(so_path, op_proto_dest)

    # (2) framework/onnx/, parse-phase ONNX parser plugin.
    dest_dir = opp_root / "framework" / "onnx"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / so_path.name

    _atomic_copy(so_path, dest)

    # (4) op_kernel/, the shipped dev-editable kernel snippet(s). One canonical copy per op under
    # <opp_root>/op_kernel/<stem>.py; PtoCustomOp::Compile resolves it via the ASCEND_CUSTOM_OPP_PATH
    # entries (<entry>/op_kernel/<stem>.py, dladdr upward-walk fallback) and loads it by path (dev edits
    # take effect on the next op-compile, no .so rebuild). Always a copy.
    for _op_type, stem, src_py in kernel_py_paths:
        op_kernel_dir.mkdir(parents=True, exist_ok=True)
        py_dest = op_kernel_dir / f"{stem}.py"
        _atomic_copy(src_py, py_dest)
        logger.info("[SETUP] dev-editable kernel snippet at: %s", py_dest)

        # Install-time resolvability check. The .py is REQUIRED, so verify
        # it landed where the resolver looks, <opp_root>/op_kernel/<stem>.py is BOTH what the
        # env-probe finds once ASCEND_CUSTOM_OPP_PATH=<opp_root> is exported AND what the dladdr
        # upward-walk finds with the env unset. Hard existence check catches a silently-failed copy.
        if not py_dest.is_file():
            raise RuntimeError(
                f"[SETUP] kernel snippet was not installed at the resolver location: {py_dest}"
            )

        if verify:
            logger.info("[SETUP] verified resolvable + importable: %s", py_dest)

    logger.info("[SETUP] op-proto .so placed at:      %s  (compile-phase InferShape)", op_proto_dest)
    logger.info("[SETUP] ONNX parser .so placed at:   %s  (parse-phase)", dest)
    logger.info("[SETUP] point ATC/GE at it with:\n    export ASCEND_CUSTOM_OPP_PATH=%s", opp_root)
    return dest
