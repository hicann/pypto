#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""D2 工件完整性与流程合规规则（OL09-OL14, OL24）核心测试。"""
import json
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, write_file

# ── OL09: SPEC.md 结构化章节校验 ──


def test_ol09_pass_when_spec_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec = """---
schema_version: 1
op_name: demo
supported_dtypes: [bfloat16]
p0_shapes: [[1024, 128]]
tolerance: {'atol': 0.001, 'rtol': 0.001}
---
# SPEC
## 数学公式
$$ y = x $$
## 输入输出规格
shape: [N, M], dtype: bfloat16
## 精度要求
atol: 0.001, rtol: 0.001
"""
    write_file(op_dir / "SPEC.md", spec)
    finding = run_rule(mod, op_dir, "OL09", stage=2)
    assert finding.status == "PASS"


def test_ol09_fail_when_spec_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "SPEC.md").unlink()
    finding = run_rule(mod, op_dir, "OL09", stage=2)
    assert finding.status == "FAIL"


# ── OL10: API_REPORT.md 结构化章节校验 ──

def test_ol10_pass_when_api_report_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    api = """---
schema_version: 1
op_name: demo
---
# API REPORT
## API 映射
mapping here
## 约束
constraints here
## Tiling
tiling here
"""
    write_file(op_dir / "API_REPORT.md", api)
    finding = run_rule(mod, op_dir, "OL10", stage=3)
    assert finding.status == "PASS"


def test_ol10_fail_when_missing_sections(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "API_REPORT.md", "# API REPORT\nsome text\n")
    finding = run_rule(mod, op_dir, "OL10", stage=3)
    assert finding.status == "FAIL"


# ── OL11: golden 文件可导入 ──

def test_ol11_pass_when_golden_importable(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL11", stage=4)
    assert finding.status == "PASS"


def test_ol11_fail_when_golden_has_syntax_error(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / "demo_golden.py", "def broken(\n")
    finding = run_rule(mod, op_dir, "OL11", stage=4)
    assert finding.status == "FAIL"


# ── OL12: DESIGN.md 结构化章节校验 ──

def test_ol12_pass_when_design_complete(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    design = """---
schema_version: 1
op_name: demo
dynamic_axes: [N]
---
# DESIGN
## API 映射
mapping here
## 数据切分
tiling here
## 验证方案
validation here
"""
    write_file(op_dir / "DESIGN.md", design)
    finding = run_rule(mod, op_dir, "OL12", stage=5)
    assert finding.status == "PASS"


def test_ol12_fail_when_design_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "DESIGN.md").unlink()
    finding = run_rule(mod, op_dir, "OL12", stage=5)
    assert finding.status == "FAIL"


# ── OL13: 三件套完整 ──

def test_ol13_pass_when_all_three_present(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL13", stage=5)
    assert finding.status == "PASS"


def test_ol13_fail_when_impl_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "demo_impl.py").unlink()
    finding = run_rule(mod, op_dir, "OL13", stage=5)
    assert finding.status == "FAIL"


# ── OL14: 精度通过 ──

def test_ol14_pass_when_stage5_completed(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "current_stage": 7,
        "stage_status": {"5": "completed", "6": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL14", stage=7)
    assert finding.status == "PASS"


def test_ol14_fail_when_no_state_file(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL14", stage=7)
    assert finding.status == "FAIL"


# ── OL24: 状态文件结构合法 ──

def test_ol24_pass_when_state_valid(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    state = {
        "operator_name": "demo",
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "PASS"


def test_ol24_fail_when_state_missing(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"


def test_ol24_fail_when_state_invalid_json(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    write_file(op_dir / ".orchestrator_state.json", "not json")
    finding = run_rule(mod, op_dir, "OL24", stage=5)
    assert finding.status == "FAIL"
