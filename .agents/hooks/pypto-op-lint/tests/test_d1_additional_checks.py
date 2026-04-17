# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

"""D1 框架约束合规规则（OL04-OL06, OL25, OL26）补充测试。"""
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# ── OL04: 必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes ──


def test_ol04_pass_when_tile_shapes_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "PASS"


def test_ol04_fail_when_no_tile_shapes(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL04")
    assert finding.status == "FAIL"


# ── OL05: kernel 张量参数必须有 pypto.Tensor 类型注解 ──

def test_ol05_pass_when_tensor_annotated(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL05")
    assert finding.status == "PASS"


def test_ol05_fail_when_no_annotation(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x, y):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL05")
    assert finding.status == "FAIL"


# ── OL06: kernel 内禁用 Python 原生 min()/max() ──

def test_ol06_pass_when_no_native_minmax(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL06")
    assert finding.status == "PASS"


def test_ol06_fail_when_native_max_used(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = max(x, 0)

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL06")
    assert finding.status == "FAIL"


# ── OL25: 禁止 pypto.Tensor() 空参数注解 ──

def test_ol25_pass_when_tensor_has_args(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL25")
    assert finding.status == "PASS"


def test_ol25_fail_when_tensor_empty_args(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor(), y: pypto.Tensor()):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL25")
    assert finding.status == "WARN"


# ── OL26: 张量参数必须在非张量参数之前 ──

def test_ol26_pass_when_order_correct(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL26")
    assert finding.status == "PASS"


def test_ol26_fail_when_scalar_before_tensor(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl = """import pypto
@pypto.frontend.jit
def demo_kernel(n: int, x: pypto.Tensor([1024], pypto.DT_FP32), y: pypto.Tensor([1024], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(n, x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", impl)
    finding = run_rule(mod, op_dir, "OL26")
    assert finding.status == "FAIL"
