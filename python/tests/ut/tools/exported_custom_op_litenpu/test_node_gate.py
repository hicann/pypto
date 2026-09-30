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
"""Tests for the node-side package-version gate (exported_custom_op_litenpu.deploy.node_gate).

Needs onnx and pypto. The LOCAL side is a callable, ``installed_pypto_version``, read at call time; node_gate
imports it by name from ``exported_custom_op_litenpu.deploy.embed``, so the tests patch node_gate's own binding of it.
"""

from __future__ import annotations

from exported_custom_op_litenpu.deploy import node_gate
from onnx import helper
import pytest

from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _META_KEY__PYPTO_PACKAGE_VERSION,
)


def _node_with_package_version(version):
    """A minimal ONNX node carrying only the ``pypto_package_version`` marker under test.

    ``node_gate.check_pypto_node_package_version`` reads exactly one field, so no other pypto node-meta
    entry is needed to exercise it.
    """
    node = helper.make_node(op_type="OpA", inputs=["x"], outputs=["y"], name="a1")
    if version is not None:
        node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_PACKAGE_VERSION, version))
    return node


def test_missing_key_is_refused_as_runtime_error():
    # extract_pypto_package_version's underlying raise is a KeyError; this gate must re-raise it as a
    # RuntimeError, the type every other node-side rejection in this package uses.
    with pytest.raises(RuntimeError, match="no pypto_package_version") as excinfo:
        node_gate.check_pypto_node_package_version(_node_with_package_version(None))
    # Chained from the extractor's KeyError, so the traceback still shows the attribute lookup that failed.
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_package_version_newer_is_refused_naming_both_versions(monkeypatch):
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: "0.2.1")
    with pytest.raises(RuntimeError) as excinfo:
        node_gate.check_pypto_node_package_version(_node_with_package_version("0.3.0"))
    message = str(excinfo.value)
    assert "pypto_package_version=0.3.0" in message
    assert "0.2.1" in message


def test_package_version_uncomparable_on_the_node_warns_and_passes(monkeypatch):
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: "0.2.1")
    with pytest.warns(UserWarning, match="records pypto_package_version"):
        node_gate.check_pypto_node_package_version(_node_with_package_version("unknown"))


def test_package_version_uncomparable_locally_warns_and_passes(monkeypatch):
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: "unknown")
    with pytest.warns(UserWarning, match="this pypto reports version"):
        node_gate.check_pypto_node_package_version(_node_with_package_version("0.3.0"))


def test_equal_version_passes_without_warning(monkeypatch, recwarn):
    # An identical recorded value is neither refused nor warned about.
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: "0.2.1")
    node_gate.check_pypto_node_package_version(_node_with_package_version("0.2.1"))
    assert not recwarn


def test_older_recorded_version_passes(monkeypatch):
    # Only a NEWER recorded release is refused: an older pypto's artifact still builds here.
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: "0.3.0")
    node_gate.check_pypto_node_package_version(_node_with_package_version("0.2.1"))


@pytest.mark.parametrize("recorded,local", [("unknown", "0.2.1"), ("0.3.0", "unknown")])
def test_stacklevel_attributes_the_warning_to_the_direct_caller(monkeypatch, recorded, local):
    # Both warnings attribute to the caller of check_pypto_node_package_version (here, this file), not to a
    # frame inside node_gate.py: the location is also the key warnings dedups a repeated warning on.
    monkeypatch.setattr(node_gate, "installed_pypto_version", lambda: local)
    with pytest.warns(UserWarning) as record:
        node_gate.check_pypto_node_package_version(_node_with_package_version(recorded))
    assert record[0].filename == __file__
