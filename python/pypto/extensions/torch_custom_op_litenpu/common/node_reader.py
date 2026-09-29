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
"""Schema-driven node reading: the node-type dispatcher, the ``extract_<node-meta-key>`` surface, and the gate.

This is the one coupled unit that ties the onnx ``node_reader`` to the node-meta schema. It carries
one extractor per readable node-meta key and the public node-layout-version gate
``check_pypto_node_layout``.
"""

from typing import Any, Iterator

from .node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _META_KEY__KERNEL_COMPILE_SNIPPET,
    _META_KEY__OP_EXPORT_RECORD,
    _META_KEY__OP_TYPE,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)

__all__ = (
    "check_pypto_node_layout",
    "extract_kernel_compile_snippet",
    "extract_op_export_record",
    "extract_op_type",
    "extract_pypto_package_version",
    "iter_pypto_nodes",
)


def check_pypto_node_layout(node: Any) -> None:
    """Raise if *node*'s attribute layout is newer than this reader supports."""
    _check_node_layout_version(node)


def _check_node_layout_version(node: Any) -> None:
    """Raise if the node's attribute layout is newer than this reader supports."""
    raw = _extract_node_meta_string(node, _META_KEY__PYPTO_NODE_LAYOUT_VERSION)
    try:
        node_fmt = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"pypto node has an unparseable pypto_node_layout_version={raw!r} "
            "(expected an integer string)."
        ) from exc
    if node_fmt > int(_LOCAL_NODE_LAYOUT_VERSION):
        raise RuntimeError(
            f"pypto node was produced with pypto_node_layout_version={raw}, but this pypto "
            f"supports up to {_LOCAL_NODE_LAYOUT_VERSION}. Upgrade pypto to read this artifact."
        )


def _extract_node_meta_string(node: Any, meta_key: str) -> str:
    """Extract a string-valued node-meta entry from an ONNX node."""
    # Function-scope so this module carries no module-scope edge to onnx: importing pypto must not
    # require the onnx wheel (the same rule finalize.py follows for ``onnx.export``).
    import onnx  # noqa: PLC0415 - optional dependency: only the onnx reader path needs it

    # layering: the onnx subpackage back-imports common, so this edge must stay lazy. Below that seam
    # ``attr`` is ONNX's own word for ``NodeProto.attribute`` and carries no provenance, so the callee's
    # ``attr_name`` parameter is correct as it stands and is deliberately not renamed.
    from ..onnx.node_reader import _extract_string_attr_from_onnx_node  # noqa: PLC0415

    if isinstance(node, onnx.NodeProto):
        return _extract_string_attr_from_onnx_node(node, meta_key)
    raise TypeError(f"Unsupported argument type for node-meta extraction: {type(node).__name__}")


def extract_op_type(node) -> str:
    """Return *node*'s qualified op identifier (``op_type``), stripped and non-empty; anything else raises."""
    return _extract_node_meta_string(node, _META_KEY__OP_TYPE)


def extract_kernel_compile_snippet(node) -> str:
    """Return *node*'s kernel source (``kernel_compile_snippet``), stripped and non-empty; anything else raises."""
    return _extract_node_meta_string(node, _META_KEY__KERNEL_COMPILE_SNIPPET)


def extract_op_export_record(node) -> str:
    """Return *node*'s export record JSON (``op_export_record``), stripped and non-empty; anything else raises."""
    return _extract_node_meta_string(node, _META_KEY__OP_EXPORT_RECORD)


def extract_pypto_package_version(node) -> str:
    """Return *node*'s exporting pypto release (``pypto_package_version``), stripped and non-empty; anything
    else raises."""
    return _extract_node_meta_string(node, _META_KEY__PYPTO_PACKAGE_VERSION)


def iter_pypto_nodes(model) -> Iterator[Any]:
    """Yield every node in *model* carrying pypto's node-layout marker; an unsupported type raises."""
    # Function-scope so this module carries no module-scope edge to onnx, and dispatch on the model type
    # exactly as _extract_node_meta_string does -- a second export format adds a branch here, not a name.
    import onnx  # noqa: PLC0415 - optional dependency: only the onnx reader path needs it

    # Plain return, never `yield from`: a generator body would defer this type check to first iteration.
    if isinstance(model, onnx.ModelProto):
        from ..onnx.node_reader import _iter_pypto_nodes_from_onnx_model  # noqa: PLC0415

        return _iter_pypto_nodes_from_onnx_model(model)
    raise TypeError(f"Unsupported argument type for node iteration: {type(model).__name__}")
