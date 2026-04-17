# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file


def test_ol30_uses_supported_dtypes_from_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL30")
    assert finding.status == "PASS"


def test_ol31_checks_dynamic_axes_from_design_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL31")
    assert finding.status == "PASS"


def test_ol32_checks_tolerance_from_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL32")
    assert finding.status in {"PASS", "WARN"}


def test_ol34_recognizes_shapes_from_run_and_check(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL34")
    assert finding.status == "PASS"


def test_ol31_focuses_on_primary_kernel_called_by_wrapper(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")

    impl = """import pypto
_N = pypto.DYNAMIC
_M = pypto.STATIC

@pypto.frontend.jit
def marker_kernel(x: pypto.Tensor([_N, _M], pypto.DT_BF16), y: pypto.Tensor([_N, _M], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(1, 1)
    y[:] = x

@pypto.frontend.jit
def demo_kernel(x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
                y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(32, 128)
    y[:] = x

def demo_wrapper(x, y):
    demo_kernel(x, y)
"""
    write_file(op_dir / "demo_impl.py", impl)

    finding = run_rule(mod, op_dir, "OL31")
    assert finding.status == "FAIL"


def test_ol41_detects_lint_output_pollution(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")

    polluted = (op_dir / "demo_impl.py").read_text(encoding="utf-8") + "\n[pypto-op-lint] 以下规则违规\n"
    write_file(op_dir / "demo_impl.py", polluted)

    finding = run_rule(mod, op_dir, "OL41")
    assert finding.status == "FAIL"
