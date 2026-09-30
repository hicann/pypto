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
"""Gate a discovered node's recorded pypto release before it is built into a ``.so``.

``build_so_from_model`` calls ``check_pypto_node_package_version`` on every node ``find_pypto_nodes``
returns: a node exported by a newer pypto than the one running here cannot be built (an older pypto
cannot reconstruct a newer package's kernel-compile snippet), so it is refused before any codegen or
compile work starts. The node-attribute layout is a separate concern, gated inside
``find_pypto_nodes`` itself via ``exported_custom_op_litenpu.common.discovery`` -> pypto core's
``check_pypto_node_layout``;
this module never touches it.

The comparison primitives (``is_newer``, ``parse_version_tuple``, ``installed_pypto_version``) are
``exported_custom_op_litenpu.deploy.embed``'s: that module owns the comparison logic in this package, so
this one imports them rather than holding a second copy.
"""

from typing import Any
import warnings

from pypto.extensions.torch_custom_op_litenpu import extract_pypto_package_version

from .embed import installed_pypto_version, is_newer, parse_version_tuple

__all__ = ("check_pypto_node_package_version",)


def check_pypto_node_package_version(node: Any) -> None:
    """Raise if *node* was produced by a newer pypto than this one can build.

    Warns instead of raising when either side's version cannot be parsed as a release number, since an
    unparseable release number is no evidence of an unbuildable artifact.
    """
    try:
        recorded_raw = extract_pypto_package_version(node)
    except KeyError as exc:
        # The marker and this key are written together, so a discoverable node lacking it is malformed,
        # not old. Re-raised as RuntimeError, the type every other node-side rejection here uses.
        raise RuntimeError(
            "pypto node has no pypto_package_version; it was not produced by a pypto that records one."
        ) from exc
    local_raw = installed_pypto_version()
    recorded = parse_version_tuple(recorded_raw)
    local = parse_version_tuple(local_raw)
    if recorded is None:
        warnings.warn(
            f"pypto node records pypto_package_version={recorded_raw!r}, which cannot be compared with "
            f"this pypto ({local_raw}); skipping the package-version check.",
            # stacklevel=2: warn() is called from this public entry itself, with no helper frame in between,
            # so the warning attributes to whoever called this function (build_so_from_model's node loop).
            stacklevel=2,
        )
        return
    if local is None:
        warnings.warn(
            f"this pypto reports version {local_raw!r}, which cannot be compared with the "
            f"pypto_package_version recorded on the node ({recorded_raw}); skipping the package-version "
            "check.",
            # stacklevel=2: attributes to this function's caller, as above
            stacklevel=2,
        )
        return
    if is_newer(recorded=recorded, local=local):
        raise RuntimeError(
            f"pypto node was produced with pypto_package_version={recorded_raw}, but this pypto is "
            f"{local_raw}. Older pypto cannot build a newer artifact. Either upgrade pypto here to "
            f">= {recorded_raw}, or re-export the model with pypto {local_raw}."
        )
