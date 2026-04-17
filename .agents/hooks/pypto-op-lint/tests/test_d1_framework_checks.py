# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

"""D1 框架约束合规规则（OL01-OL08）核心测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def test_ol01_pass_when_jit_decorator_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL01")
    assert finding.status == "PASS"


def test_ol01_fail_when_no_jit_decorator(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL01")
    assert finding.status == "FAIL"


def test_ol02_fail_on_bare_assignment(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y = x  # 违规: 应该用 y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL02")
    assert finding.status == "FAIL"


def test_ol02_pass_on_slice_assignment(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL02")
    assert finding.status == "PASS"


def test_ol03_fail_when_return_in_jit(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x
    return y  # 违规

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL03")
    assert finding.status == "FAIL"


def test_ol03_pass_when_no_return(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL03")
    assert finding.status == "PASS"


def test_ol07_fail_when_no_import_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import torch
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "FAIL"


def test_ol07_pass_when_import_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL07")
    assert finding.status == "PASS"


def test_ol15_fail_when_golden_imports_pypto(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    golden = """import pypto
def demo_golden(x, y):
    return x
"""
    write_file(op_dir / "demo_golden.py", golden)
    finding = run_rule(mod, op_dir, "OL15", stage=3)
    assert finding.status == "FAIL"


def test_ol15_pass_when_golden_clean(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL15", stage=3)
    assert finding.status == "PASS"
