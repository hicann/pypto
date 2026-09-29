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
"""Export-stage guard: declare a :class:`ExportedCustomOp`, ``torch.onnx.export`` it, reload it and check the node.

Needs only torch + onnx: no CANN, no device, no compiler, and the kernel itself is never built.

Checks the reloaded node's ``op_type``/``domain``, arity and tensor element types, and every declared
:class:`AttrSpec`'s value against its ``onnx.AttributeProto`` type -- an Int arrives as INT (never FLOAT)
and a whole Float such as ``2.0`` arrives as FLOAT (never INT). Also checks that
``torch.ops.pypto.<op>(...)`` equals ``torch_defn(...)`` element-wise on CPU.
"""
import onnx
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp, OnnxSymbolicSpec, finalize_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _qualify_op_type

# The declaration under test. Two attrs of two different types, in declaration order: an Int and a Float
# whose value is a whole number (2.0) — the case that must not degrade to an ONNX INT attribute.
_QUALNAME = "pypto::scale_bias"
_SPEC = OnnxSymbolicSpec(op_type="ScaleBias", opset_version=12)
_ATTRS = (AttrSpec("bias", "Int", 0), AttrSpec("scale", "Float", 1.0))
_BIAS_VALUE = 3
_SCALE_VALUE = 2.0
_SHAPE = (1, 8, 1, 64)
_DTYPE = torch.float16
_ONNX_ELEM_TYPE = onnx.TensorProto.FLOAT16


# ── authoring functions (module scope: the tracer reads their source off disk) ────────────────────────
def _scale_bias_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    bias = int(attrs["bias"])       # attr values reach the factory as strings
    scale = float(attrs["scale"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def scale_bias_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                             out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move((a + b) * scale + bias)

    return scale_bias_inner


def _scale_bias_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _scale_bias_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _scale_bias_torch(a, b, bias, scale):
    return (a + b) * scale + bias


class _ScaleBiasModel(nn.Module):
    def forward(self, a, b):
        return torch.ops.pypto.scale_bias(a, b, bias=_BIAS_VALUE, scale=_SCALE_VALUE)


@pytest.fixture()
def declared_op():
    """Declare the op, finalize its synthesized ONNX symbolic, and clear the export state around it."""
    _reset_export_state()
    op = ExportedCustomOp(
        kernel=_scale_bias_factory,
        infer_shape=_scale_bias_infer_shape,
        infer_dtype=_scale_bias_infer_dtype,
        torch_defn=_scale_bias_torch,
        torch_op_qualname=_QUALNAME,
        onnx_spec=_SPEC,
        attrs=list(_ATTRS),
    )
    finalize_pending_ops()
    yield op
    _reset_export_state()


@pytest.fixture()
def exported_model(declared_op, tmp_path):
    """The reloaded ``onnx.ModelProto`` produced by a real ``torch.onnx.export`` of the declared op."""
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    path = str(tmp_path / "scale_bias.onnx")
    with exporting_scope():
        torch.onnx.export(
            _ScaleBiasModel(), (a, b), path,
            input_names=["a", "b"], output_names=["y"],
            opset_version=_SPEC.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={_SPEC.domain: _SPEC.domain_opset_version},
        )
    return onnx.load(path)


def _pypto_node(model):
    """The single node the declaration produced (matched on the qualified op_type)."""
    op_type = _qualify_op_type(_SPEC.op_type)
    nodes = [n for n in model.graph.node if n.op_type == op_type]
    assert len(nodes) == 1, (
        f"expected exactly one {op_type} node, got {[(n.op_type, n.domain) for n in model.graph.node]}"
    )
    return nodes[0]


def test_node_op_type_and_domain(exported_model):
    # The graph node carries the author's op name under the PyptoCustomOp namespace, in the spec's domain
    # — the <domain>::<domain_opset_version>::<op_type> triple GE looks the executor up by.
    node = _pypto_node(exported_model)
    assert node.op_type == "PyptoCustomOpScaleBias"
    assert node.op_type == _qualify_op_type(_SPEC.op_type)
    assert node.domain == _SPEC.domain == "pypto"
    opset = {o.domain: o.version for o in exported_model.opset_import}
    assert opset["pypto"] == 1
    assert opset[_SPEC.domain] == _SPEC.domain_opset_version


def test_node_and_graph_arity(exported_model):
    # Two tensor inputs (the attrs are node ATTRIBUTES, not operands) and the one output infer_shape
    # declares, on the node and on the graph alike.
    node = _pypto_node(exported_model)
    assert len(node.input) == 2
    assert len(node.output) == 1
    graph = exported_model.graph
    assert [i.name for i in graph.input] == ["a", "b"]
    assert [o.name for o in graph.output] == ["y"]
    assert list(node.input) == [i.name for i in graph.input]
    assert list(node.output) == [o.name for o in graph.output]


def test_graph_declared_tensor_dtypes(exported_model):
    # The graph's declared element types match the exported tensors and infer_dtype's promise.
    graph = exported_model.graph
    assert len(graph.input) == 2      # the loop below says nothing about an empty input list
    for value_info in graph.input:
        assert value_info.type.tensor_type.elem_type == _ONNX_ELEM_TYPE
    assert [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim] == list(_SHAPE)
    # Element type only: torch.onnx cannot shape-infer a custom-domain op, so the output's DIMS are
    # placeholders and asserting on them would be asserting on a value pypto never sets.
    assert graph.output[0].type.tensor_type.elem_type == _ONNX_ELEM_TYPE


def test_declared_attrs_present_with_value_and_onnx_type(exported_model):
    # Every declared AttrSpec reaches the node, and the ONNX AttributeProto TYPE matches the declared
    # kind: an Int must not arrive as a FLOAT, and a whole-number Float (2.0) must not arrive as an INT.
    node = _pypto_node(exported_model)
    by_name = {at.name: at for at in node.attribute}
    assert set(s.name for s in _ATTRS) <= set(by_name), (
        f"declared attrs missing from the node; node attrs = {sorted(by_name)}"
    )

    bias = by_name["bias"]
    assert bias.type == onnx.AttributeProto.INT
    assert bias.type != onnx.AttributeProto.FLOAT
    assert bias.i == _BIAS_VALUE

    scale = by_name["scale"]
    assert scale.type == onnx.AttributeProto.FLOAT
    assert scale.type != onnx.AttributeProto.INT   # 2.0 is a whole number and must still be a FLOAT attr
    assert scale.f == pytest.approx(_SCALE_VALUE, abs=1e-6)


def test_torch_op_matches_torch_defn_on_cpu(declared_op):
    # With no dispatch override active, the op's implementation IS the torch reference: the registered
    # torch.ops.pypto.<op> and torch_defn must agree element-wise on the same operands.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    out = torch.ops.pypto.scale_bias(a, b, bias=_BIAS_VALUE, scale=_SCALE_VALUE)
    golden = _scale_bias_torch(a, b, _BIAS_VALUE, _SCALE_VALUE)
    assert out.shape == golden.shape == torch.Size(_SHAPE)
    assert out.dtype == golden.dtype == _DTYPE
    # Identical arithmetic on identical operands: agreement is exact, so no tolerance is allowed to hide
    # a divergence between the registered op and the declared reference.
    torch.testing.assert_close(out, golden, rtol=0, atol=0)
    assert declared_op._torch_defn_fn is _scale_bias_torch


@pytest.mark.parametrize("short,qualified", [("Add", "PyptoCustomOpAdd"), ("Softmax", "PyptoCustomOpSoftmax")])
def test_qualify_prefixes_a_short_name(short, qualified):
    assert _qualify_op_type(short) == qualified


def test_qualify_is_idempotent():
    once = _qualify_op_type("Add")
    assert _qualify_op_type(once) == once
