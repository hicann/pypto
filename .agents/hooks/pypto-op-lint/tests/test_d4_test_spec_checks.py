# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

"""D4 测试规范规则（OL19-OL22）核心测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def test_ol19_fail_when_no_assert_allclose(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    y = x.float().numpy()
    assert abs(y.max() - y.max()) < 0.01
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL19")
    assert finding.status == "FAIL"


def test_ol19_pass_when_assert_allclose_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL19")
    assert finding.status == "PASS"


def test_ol20_fail_when_no_device_id_and_no_set_device(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_fail_when_device_id_but_no_set_device(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import os
import torch
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_fail_when_set_device_but_no_device_id(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch

def test_level0_basic():
    torch.npu.set_device(0)
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "FAIL"


def test_ol20_pass_when_both_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import os
import torch
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))

def test_level0_basic():
    torch.npu.set_device(device_id)
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL20")
    assert finding.status == "PASS"


def test_ol21_fail_when_missing_level1(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "FAIL"


def test_ol21_pass_when_both_levels(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL21")
    assert finding.status == "PASS"


def test_ol22_fail_when_no_manual_seed(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
from numpy.testing import assert_allclose

def test_level0_basic():
    x = torch.randn(1024, dtype=torch.bfloat16)
    assert_allclose(x.numpy(), x.numpy())
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL22")
    assert finding.status == "WARN"


def test_ol22_pass_when_manual_seed_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    test = """import torch
torch.manual_seed(42)

def test_level0_basic():
    pass
"""
    write_file(op_dir / "test_demo.py", test)
    finding = run_rule(mod, op_dir, "OL22")
    assert finding.status == "PASS"
