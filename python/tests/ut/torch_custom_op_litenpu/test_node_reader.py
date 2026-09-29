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
"""Node-reader guard: the ``pypto_node_layout_version`` gate and the per-field extractor surface.

Needs onnx + torch and nothing else — no CANN, no device, no examples tree. Most cases read nodes
synthesized here under the names ``node_meta`` declares; one reads back a real ``torch.onnx.export``.
The gate's two version rejections raise a plain ``RuntimeError``, so each is pinned by the substring only
it emits — ``unparseable pypto_node_layout_version`` for a value that is not an integer string, and
``supports up to`` for one this reader is too old for.
"""
import json

import onnx
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import (
    ExportedCustomOp,
    OnnxSymbolicSpec,
    check_pypto_node_layout,
    extract_kernel_compile_snippet,
    extract_op_export_record,
    extract_op_type,
    extract_pypto_package_version,
    finalize_pending_ops,
)
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__KERNEL_COMPILE_SNIPPET,
    _META_KEY__OP_EXPORT_RECORD,
    _META_KEY__OP_TYPE,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _qualify_op_type

_QUALNAME = "pypto::node_probe"
_SPEC = OnnxSymbolicSpec(op_type="NodeProbe")
_OP_TYPE = _qualify_op_type(_SPEC.op_type)
_SHAPE = (1, 4, 1, 64)         # 256 elements; only the node meta is under test, so one shape is enough
_DTYPE = torch.float16


# ── authoring functions (module scope: the tracer reads their source off disk) ────────────────────────
def create_node_probe_kernel(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def node_probe_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                         out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b)

    return node_probe_inner


def _node_probe_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _node_probe_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _node_probe_torch(a, b):
    return a + b


class _NodeProbeModel(nn.Module):
    def forward(self, a, b):
        return torch.ops.pypto.node_probe(a, b)


def _meta_node(**attrs) -> onnx.NodeProto:
    """An ONNX node carrying each of *attrs* under its real node-meta key.

    Values are ``str``, so ``make_attribute`` emits ONNX STRING attributes — the one type readers accept.
    """
    node = onnx.helper.make_node(_OP_TYPE, inputs=["a", "b"], outputs=["y"], domain=_SPEC.domain)
    attrs = dict(attrs)
    # Every node carries this key (a real export always stamps it), but setdefault not append: attributes
    # stack and the reader takes the FIRST match, so a hardcoded stamp would win over a case's own value.
    attrs.setdefault(_META_KEY__PYPTO_PACKAGE_VERSION, _LOCAL_PYPTO_VERSION)
    for name, value in attrs.items():
        node.attribute.append(onnx.helper.make_attribute(name, value))
    return node


@pytest.fixture()
def exported_node(tmp_path):
    """The single pypto node of a real ``torch.onnx.export``, reloaded from disk."""
    _reset_export_state()
    ExportedCustomOp(
        kernel=create_node_probe_kernel,
        infer_shape=_node_probe_infer_shape,
        infer_dtype=_node_probe_infer_dtype,
        torch_defn=_node_probe_torch,
        torch_op_qualname=_QUALNAME,
        onnx_spec=_SPEC,
    )
    finalize_pending_ops()
    path = str(tmp_path / "node_probe.onnx")
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    with exporting_scope():
        torch.onnx.export(
            _NodeProbeModel(), (a, b), path,
            input_names=["a", "b"], output_names=["y"],
            opset_version=_SPEC.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={_SPEC.domain: _SPEC.domain_opset_version},
        )
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if n.op_type == _OP_TYPE]
    assert len(nodes) == 1, (
        f"expected exactly one {_OP_TYPE} node — a second means another declaration's symbolic is still "
        f"registered and the reads below would resolve its node meta — got "
        f"{[(n.op_type, n.domain) for n in model.graph.node]}"
    )
    yield nodes[0]
    _reset_export_state()


def test_node_layout_version_current_is_accepted():
    node = _meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: _LOCAL_NODE_LAYOUT_VERSION})
    check_pypto_node_layout(node)   # this reader must accept the version this pypto itself writes


def test_node_layout_version_newer_is_refused_as_unsupported():
    # A readable version above this reader's: the artifact may carry attributes this pypto cannot parse,
    # so the gate refuses instead of mis-parsing downstream.
    newer = str(int(_LOCAL_NODE_LAYOUT_VERSION) + 1)
    with pytest.raises(RuntimeError) as excinfo:
        check_pypto_node_layout(_meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: newer}))
    message = str(excinfo.value)
    assert f"pypto_node_layout_version={newer}" in message
    assert f"supports up to {_LOCAL_NODE_LAYOUT_VERSION}" in message
    assert "unparseable" not in message      # the value parsed fine; only the comparison failed


def test_node_layout_version_unparseable_is_refused_as_malformed():
    # Same exception type, different diagnosis: nothing was compared here because the value never
    # became an int, so the message must not claim a version range.
    with pytest.raises(RuntimeError) as excinfo:
        check_pypto_node_layout(_meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: "1.0-beta"}))
    message = str(excinfo.value)
    assert "unparseable pypto_node_layout_version='1.0-beta'" in message
    assert "supports up to" not in message


def test_extractors_read_their_own_meta_keys():
    # One extractor per readable node-meta key, each resolving the name node_meta declares for it — a rename or a
    # crossed wiring shows up here rather than as a wrong field somewhere downstream.
    node = _meta_node(**{
        _META_KEY__OP_TYPE: _OP_TYPE,
        _META_KEY__KERNEL_COMPILE_SNIPPET: "def __pypto_compile():\n    return None\n",
        _META_KEY__OP_EXPORT_RECORD: json.dumps({"framework_kind": "onnx", "n_inputs": 2}),
    })
    assert extract_op_type(node) == _OP_TYPE
    # Surrounding whitespace is stripped off every string field, so the snippet's trailing newline is gone.
    assert extract_kernel_compile_snippet(node) == "def __pypto_compile():\n    return None"
    assert json.loads(extract_op_export_record(node)) == {"framework_kind": "onnx", "n_inputs": 2}


def test_extract_pypto_package_version_reads_the_recorded_value():
    # The public wrapper resolves the same key the private helper does, under the extract_* naming every
    # other readable field uses.
    node = _meta_node(**{_META_KEY__PYPTO_PACKAGE_VERSION: "1.2.3"})
    assert extract_pypto_package_version(node) == "1.2.3"


def test_read_back_an_exported_node(exported_node):
    # Write-then-read over the real boundary: what the export path serialized to disk is what the
    # extractors resolve off the reloaded node, under the same keys the synthesized cases use.
    check_pypto_node_layout(exported_node)   # the export path writes a version this reader accepts
    assert extract_op_type(exported_node) == _OP_TYPE
    # The snippet is the kernel source of truth: the factory arrives under its authored name.
    assert f"def {create_node_probe_kernel.__name__}(" in extract_kernel_compile_snippet(exported_node)
    record = json.loads(extract_op_export_record(exported_node))
    assert record["framework_kind"] == "onnx"
    assert record["mode"] == "factory"
    assert record["create_kernel_name"] == create_node_probe_kernel.__name__
    assert record["n_inputs"] == 2
    assert extract_pypto_package_version(exported_node) == _LOCAL_PYPTO_VERSION
