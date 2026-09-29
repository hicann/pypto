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
"""The node-meta contract: which key/value entries pypto writes onto a node, and how each is treated.

The authoring path (``exported_custom_op``) writes the node meta by this schema, and every reader resolves
it against the same schema. Adding a node-meta key means adding its constant here, listing it in the set
below, and adding an extractor to the node reader when it must be readable back off a node.
"""

import importlib.metadata

__all__ = ()

_META_KEY__OP_TYPE = "op_type"
# Self-contained Python that rebuilds the kernel: importing it yields the decorated capture set
# (create_kernel/infer_*/kernel_body) with original names + inlined constants, plus ``__pypto_compile``.
# Packed by ``kernel_snippet``; the source every consumer reconstructs the live functions from.
_META_KEY__KERNEL_COMPILE_SNIPPET = "kernel_compile_snippet"
# JSON of the per-op export record, fixed by which authoring flow made the node: input dtypes, the
# (domain, domain_opset_version) coordinates, the framework kind, the capture-set function names (to
# fetch the reconstructed callables), and the factory call shape.
_META_KEY__OP_EXPORT_RECORD = "op_export_record"
# Marks a node as pypto (discovery keys on its presence) and gates layout compatibility: a newer value
# is refused on read. Bumped ONLY when the node attribute layout changes.
_META_KEY__PYPTO_NODE_LAYOUT_VERSION = "pypto_node_layout_version"
# _LOCAL_: the layout THIS install supports, never the one read off a node.
_LOCAL_NODE_LAYOUT_VERSION = "2"
# The installed pypto release that exported the node, read once at import so it matches the loaded code.
# Reports the INSTALLED distribution: a source tree shadowing an install records THAT install's version.
_META_KEY__PYPTO_PACKAGE_VERSION = "pypto_package_version"


def _installed_pypto_version() -> str:
    try:
        return importlib.metadata.version("pypto")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# _LOCAL_: the release THIS install is, never the one recorded on a node (``recorded`` vs ``local``).
_LOCAL_PYPTO_VERSION = _installed_pypto_version()

# Every node-meta key ExportedCustomOp writes itself. A declared attr may not take one of these names:
# both share the node's attribute namespace, so a clash would resolve silently to the node-meta value.
_RESERVED_META_KEYS = frozenset({
    _META_KEY__OP_TYPE,
    _META_KEY__KERNEL_COMPILE_SNIPPET,
    _META_KEY__OP_EXPORT_RECORD,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
})
