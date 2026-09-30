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
"""Tests for the version gates ``build.build_so_from_model`` applies to a model's pypto nodes."""

from __future__ import annotations

from exported_custom_op_litenpu.deploy import build, node_gate
import onnx
from onnx import TensorProto, helper
import pytest

from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)


def _onnx_pypto_node(
    op_type: str,
    *,
    name: str,
    fmt: str = _LOCAL_NODE_LAYOUT_VERSION,
    pkg: str = _LOCAL_PYPTO_VERSION,
) -> onnx.NodeProto:
    """Build a minimal ONNX node with the pypto marker attributes (+ op_type for dedup).

    *pkg* stays a parameter: attributes stack and the reader takes the FIRST, so a hardcoded stamp wins.
    """
    node = helper.make_node(
        op_type=op_type,
        inputs=["x"],
        outputs=[f"{name}_out"],
        domain="ai.onnx.contrib",
        name=name,
    )
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_NODE_LAYOUT_VERSION, fmt))
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_PACKAGE_VERSION, pkg))
    node.attribute.append(helper.make_attribute("op_type", op_type))
    return node


def _build_onnx_model(*nodes: onnx.NodeProto) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=list(nodes),
        name="g",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    # opset_imports kept minimal — find_pypto_nodes only walks graph.node.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_build_so_from_model_rejects_a_node_with_a_newer_package_version(monkeypatch, tmp_path):
    # The build is where a model exported by a newer pypto is turned away: every discovered node's
    # pypto_package_version is gated before any codegen or compile work starts.
    monkeypatch.setattr(build, "_build_so_from_nodes", lambda *a, **k: pytest.fail(
        "the package-version gate must refuse the node before the build starts"
    ))
    model = _build_onnx_model(_onnx_pypto_node("OpA", name="a1", pkg="999.0.0"))
    with pytest.raises(RuntimeError, match="pypto_package_version=999.0.0"):
        build.build_so_from_model(model, out_dir=tmp_path)


def test_build_so_from_model_diagnoses_the_layout_before_the_package_version(monkeypatch, tmp_path):
    # A node failing BOTH rules reports the layout version, the rule that decides whether its attributes
    # can be read at all: find_pypto_nodes refuses it before the build reaches its package-version gate.
    monkeypatch.setattr(node_gate, "check_pypto_node_package_version", lambda node: pytest.fail(
        "a node with an unsupported layout must be refused before the package-version gate"
    ))
    newer = str(int(_LOCAL_NODE_LAYOUT_VERSION) + 1)
    model = _build_onnx_model(_onnx_pypto_node("OpA", name="a1", fmt=newer, pkg="999.0.0"))
    with pytest.raises(RuntimeError) as excinfo:
        build.build_so_from_model(model, out_dir=tmp_path)
    assert "pypto_node_layout_version" in str(excinfo.value)
    assert "pypto_package_version" not in str(excinfo.value)
