# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess

from ..core import API_REPORT_FILE, DESIGN_FILE, PYTHON_BIN, SPEC_FILE, CheckContext, Finding, register
from ..utils import (
    _extract_markdown_headings,
    _extract_section_text,
    _has_heading_like,
    _parse_front_matter,
    _validate_doc_schema,
)


@register("OL09")
def check_ol09(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL09", "FAIL", f"{SPEC_FILE} 不存在")
    content = ctx.read_file(SPEC_FILE)
    if ctx.op_name not in content:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 中未包含算子名 '{ctx.op_name}'", file=SPEC_FILE)

    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "数学公式"):
        missing.append("数学公式")
    if not _has_heading_like(headings, "输入输出规格"):
        missing.append("输入输出规格")
    if not _has_heading_like(headings, "精度要求"):
        missing.append("精度要求")
    if missing:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 缺少必需内容: {', '.join(missing)}", file=SPEC_FILE)

    # 数学/算法章节：尝试多个可能的章节名
    math_text = _extract_section_text(content, "数学")
    if not math_text:
        math_text = _extract_section_text(content, "算法")
    if not math_text:
        math_text = _extract_section_text(content, "基础信息")
    # 公式建议统一使用 $$...$$ 块公式；为兼容少量纯文本公式，仍保留 '=' 判定。
    if not math_text or not any(token in math_text for token in ("$$", "=", "\\begin{equation}", "round", "clamp")):
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 数学定义章节内容不足（缺少可解析公式特征）", file=SPEC_FILE)

    # 输入输出/数据规格章节
    io_text = _extract_section_text(content, "输入输出规格")
    if not io_text:
        io_text = _extract_section_text(content, "数据规格")
    if not io_text or "dtype" not in io_text.lower() or "shape" not in io_text.lower():
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 输入输出规格章节内容不足（需包含 shape/dtype）", file=SPEC_FILE)

    precision_text = _extract_section_text(content, "精度")
    lower_prec = precision_text.lower() if precision_text else ""
    _has_tolerance = (
        precision_text and
        any(kw in lower_prec for kw in ("atol", "rtol", "mare"))
    )
    if not _has_tolerance:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 精度要求章节内容不足（需包含 atol/rtol 或指标阈值）", file=SPEC_FILE)

    # Front matter schema 校验：在 SPEC 生成时即拦截格式问题（如 p0_shapes 非 list）
    spec_meta, _ = _parse_front_matter(content)
    if not spec_meta:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} 缺少 front matter"
            "（必须以 --- 开头，含 schema_version/op_name/supported_dtypes/p0_shapes/tolerance）",
            file=SPEC_FILE)
    schema_errors = _validate_doc_schema("SPEC", spec_meta)
    if schema_errors:
        return ctx.make_finding("OL09", "FAIL",
            f"{SPEC_FILE} front matter schema 非法: {'; '.join(schema_errors)}",
            file=SPEC_FILE)

    return ctx.make_finding("OL09", "PASS",
        f"{SPEC_FILE} 含算子名、公式、输入输出规格与精度要求，front matter schema 合法", file=SPEC_FILE)


@register("OL10")
def check_ol10(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(API_REPORT_FILE):
        return ctx.make_finding("OL10", "FAIL", f"{API_REPORT_FILE} 不存在")
    content = ctx.read_file(API_REPORT_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "API 映射"):
        missing.append("API 映射")
    if not _has_heading_like(headings, "约束"):
        missing.append("约束")
    if not _has_heading_like(headings, "Tiling"):
        missing.append("Tiling")
    if missing:
        return ctx.make_finding("OL10", "FAIL",
            f"{API_REPORT_FILE} 缺少必需内容: {', '.join(missing)}",
            file=API_REPORT_FILE)
    return ctx.make_finding("OL10", "PASS",
        f"{API_REPORT_FILE} 含 API 映射、约束与 Tiling 说明", file=API_REPORT_FILE)


@register("OL11")
def check_ol11(ctx: CheckContext) -> Finding:
    """进入 Stage 4 需 {op}_golden.py 可导入"""
    golden_file = f"{ctx.op_name}_golden.py"
    if not ctx.file_exists(golden_file):
        return ctx.make_finding("OL11", "FAIL", f"{golden_file} 不存在")
    probe_code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {json.dumps(ctx.op_dir)})\n"
        f"importlib.import_module({json.dumps(f'{ctx.op_name}_golden')})\n"
    )
    try:
        result = subprocess.run(
            [PYTHON_BIN, "-c", probe_code],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入超时（>10s）", file=golden_file)
    except OSError as e:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入探测失败: {e}", file=golden_file)
    if result.returncode != 0:
        return ctx.make_finding("OL11", "FAIL",
            f"{golden_file} 导入失败: {result.stderr[:200]}", file=golden_file)
    return ctx.make_finding("OL11", "PASS", f"{golden_file} 可导入", file=golden_file)


@register("OL12")
def check_ol12(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL12", "FAIL", f"{DESIGN_FILE} 不存在")
    content = ctx.read_file(DESIGN_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "计算图") and not _has_heading_like(headings, "API 映射"):
        missing.append("计算图")
    if not _has_heading_like(headings, "Tiling") and not _has_heading_like(headings, "数据切分"):
        missing.append("Tiling")
    if not _has_heading_like(headings, "验证方案"):
        missing.append("验证方案")
    if missing:
        return ctx.make_finding("OL12", "FAIL",
            f"{DESIGN_FILE} 缺少必需内容: {', '.join(missing)}", file=DESIGN_FILE)
    return ctx.make_finding("OL12", "PASS",
        f"{DESIGN_FILE} 含计算图、Tiling 与验证方案", file=DESIGN_FILE)


@register("OL13")
def check_ol13(ctx: CheckContext) -> Finding:
    """生成文件三件套完整"""
    files = [
        f"{ctx.op_name}_impl.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    missing = [f for f in files if not ctx.file_exists(f)]
    if missing:
        return ctx.make_finding("OL13", "FAIL",
            f"缺少文件: {', '.join(missing)}")
    return ctx.make_finding("OL13", "PASS", "三件套文件完整")


@register("OL14")
def check_ol14(ctx: CheckContext) -> Finding:
    """进入 Stage 7 需精度通过"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL14", "FAIL", "状态文件不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("stage_status", {})
        if status.get("5") == "completed" or status.get("6") == "completed":
            return ctx.make_finding("OL14", "PASS", "精度已通过")
    except ValueError:
        pass
    return ctx.make_finding("OL14", "FAIL", "精度未通过（Stage 5/6 未 completed）")


@register("OL24")
def check_ol24(ctx: CheckContext) -> Finding:
    """.orchestrator_state.json 结构合法"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL24", "FAIL", ".orchestrator_state.json 不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ctx.make_finding("OL24", "FAIL", f"JSON 解析失败: {e}")
    required = ["operator_name", "current_stage", "stage_status"]
    missing = [k for k in required if k not in data]
    if missing:
        return ctx.make_finding("OL24", "FAIL",
            f"缺少必需字段: {', '.join(missing)}")
    return ctx.make_finding("OL24", "PASS", "状态文件结构合法")
