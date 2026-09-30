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
"""Tests for the shared demo run helper ``tools/exported_custom_op_litenpu/run/demo_run.py``.

* ``test_run_cli_*``: the valid ``run_demo.py`` flag forms (bare / --device=cpu / --device=npu /
  --soc_version alone / --device=cpu --soc_version) pass ``validate_args`` and produce the expected
  ``scenario_label``, while ``--device=npu`` with ``--soc_version`` raises a clean ``RunCliError``.
* ``test_npu_device_id_*``: ``--device=npu`` reads the device id from ``TILE_FWK_DEVICE_ID`` (mirrors
  examples 00/01/02) and raises ``RunEnvError`` when it is unset or not an integer.
* ``test_enable_eslmodel_*``: the ``enable_eslmodel()`` env set/restore toggle leaks nothing.

No backend needed: only the helper module's pure argument/env logic is exercised.
"""

import os

from exported_custom_op_litenpu.run import demo_run
import pytest


class _Args:
    """The flags ``add_run_args`` puts on ``run_demo.py``."""

    def __init__(self, device=None, soc_version=None):
        self.device = device
        self.soc_version = soc_version


@pytest.mark.parametrize("args, label_check", [
    (_Args(), lambda lab:lab == "cpu (torch_defn)"),                     # bare run_demo.py -> cpu
    (_Args(device="cpu"), lambda lab:lab == "cpu (torch_defn)"),         # explicit cpu
    (_Args(device="npu"), lambda lab:lab.startswith("npu")),             # real NPU, auto-soc
    (_Args(soc_version="Kirin9030"), lambda lab:"soc=Kirin9030" in lab),  # soc alone -> sim (--device optional)
    (_Args(device="cpu", soc_version="Kirin9030"), lambda lab:"soc=Kirin9030" in lab),  # cpu + soc -> sim
    (_Args(device="cpu", soc_version="A2A3"), lambda lab:"soc=A2A3" in lab),
])
def test_run_cli_valid_forms(args, label_check):
    # validate_args must not raise, and the formatting-only scenario_label reflects the
    # args-to-scenario mapping the demo branches on directly.
    demo_run.validate_args(args)  # must not raise
    assert label_check(demo_run.scenario_label(args))


@pytest.mark.parametrize("args", [
    _Args(device="npu", soc_version="Kirin9030"),  # --soc_version with --device=npu (npu auto-detects it)
])
def test_run_cli_rejects(args):
    with pytest.raises(demo_run.RunCliError):
        demo_run.validate_args(args)


# ---- --device=npu device id from TILE_FWK_DEVICE_ID (mirrors examples/00|01|02) ------------------------

def _fake_torch_npu(monkeypatch):
    """Give torch a stub ``npu`` attr so npu_device_id()'s hasattr guard passes on a non-NPU host."""
    import torch
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", object(), raising=False)


def test_npu_device_id_from_env(monkeypatch):
    _fake_torch_npu(monkeypatch)
    monkeypatch.setenv("TILE_FWK_DEVICE_ID", "3")
    assert demo_run.npu_device_id() == 3


def test_npu_device_id_unset_raises(monkeypatch):
    _fake_torch_npu(monkeypatch)
    monkeypatch.delenv("TILE_FWK_DEVICE_ID", raising=False)
    with pytest.raises(demo_run.RunEnvError):
        demo_run.npu_device_id()


def test_npu_device_id_non_integer_raises(monkeypatch):
    _fake_torch_npu(monkeypatch)
    monkeypatch.setenv("TILE_FWK_DEVICE_ID", "notanint")
    with pytest.raises(demo_run.RunEnvError):
        demo_run.npu_device_id()


# ---- enable_eslmodel(): sets ENABLE_ESLMODEL=TRUE inside the block, restores/unsets after (no leak) -----

def test_enable_eslmodel_sets_and_unsets_when_previously_unset(monkeypatch):
    monkeypatch.delenv("ENABLE_ESLMODEL", raising=False)
    with demo_run.enable_eslmodel():
        assert os.environ["ENABLE_ESLMODEL"] == "TRUE"
    # previously unset -> must be popped on exit (no leak into the process env)
    assert "ENABLE_ESLMODEL" not in os.environ


def test_enable_eslmodel_restores_prior_value(monkeypatch):
    monkeypatch.setenv("ENABLE_ESLMODEL", "FALSE")
    with demo_run.enable_eslmodel():
        assert os.environ["ENABLE_ESLMODEL"] == "TRUE"
    # previously set -> restored to the prior value on exit
    assert os.environ["ENABLE_ESLMODEL"] == "FALSE"
