# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import json
import os
import re as _re_ol54
import subprocess

from ..core import (
    API_REPORT_FILE,
    DESIGN_FILE,
    PYTHON_BIN,
    SPEC_FILE,
    CheckContext,
    Finding,
    register,
)
from ..utils import (
    _extract_markdown_headings,
    _extract_section_text,
    _has_heading_like,
    _parse_front_matter,
    _phase_to_module_suffix,
    _validate_doc_schema,
)


@register("OL09")
def check_ol09(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("OL09", "FAIL", f"{SPEC_FILE} 不存在")
    content = ctx.read_file(SPEC_FILE)
    if ctx.op_name not in content:
        return ctx.make_finding(
            "OL09",
            "FAIL",
            f"{SPEC_FILE} 中未包含算子名 '{ctx.op_name}'",
            file=SPEC_FILE,
        )

    issues: list[str] = []
    headings = _extract_markdown_headings(content)
    missing_headings: list[str] = []
    if not _has_heading_like(headings, "数学公式"):
        missing_headings.append("数学公式")
    if not _has_heading_like(headings, "输入输出规格"):
        missing_headings.append("输入输出规格")
    if not _has_heading_like(headings, "精度要求"):
        missing_headings.append("精度要求")
    if missing_headings:
        issues.append(
            f"缺少必需章节: {', '.join(missing_headings)}"
            f"（关键词可能因 heading 使用英文等价词而未匹配）"
        )

    math_text = _extract_section_text(content, "数学")
    if not math_text:
        math_text = _extract_section_text(content, "算法")
    if not math_text:
        math_text = _extract_section_text(content, "基础信息")
    if not math_text or not any(
        token in math_text
        for token in ("$$", "=", "\\begin{equation}", "round", "clamp")
    ):
        issues.append("数学定义章节内容不足（缺少可解析公式特征）")

    io_text = _extract_section_text(content, "输入输出规格")
    if not io_text:
        io_text = _extract_section_text(content, "数据规格")
    if not io_text or "dtype" not in io_text.lower() or "shape" not in io_text.lower():
        issues.append("输入输出规格章节内容不足（需包含 shape/dtype）")

    precision_text = _extract_section_text(content, "精度")
    lower_prec = precision_text.lower() if precision_text else ""
    _has_tolerance = precision_text and any(
        kw in lower_prec for kw in ("atol", "rtol", "mare")
    )
    if not _has_tolerance:
        issues.append("精度要求章节内容不足（需包含 atol/rtol 或指标阈值）")

    spec_meta, _ = _parse_front_matter(content)
    if not spec_meta:
        issues.append(
            "缺少 front matter"
            "（必须以 --- 开头，含 schema_version/op_name/supported_dtypes/p0_shapes/tolerance）"
        )
    else:
        schema_errors = _validate_doc_schema("SPEC", spec_meta)
        if schema_errors:
            issues.append(f"front matter schema 非法: {'; '.join(schema_errors)}")

    if issues:
        return ctx.make_finding(
            "OL09",
            "FAIL",
            f"{SPEC_FILE} 存在 {len(issues)} 个问题:\n" + "\n".join(f"  - {i}" for i in issues),
            file=SPEC_FILE,
        )

    return ctx.make_finding(
        "OL09",
        "PASS",
        f"{SPEC_FILE} 含算子名、公式、输入输出规格与精度要求，front matter schema 合法",
        file=SPEC_FILE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OL54 — Phase M_k 自我评审证据（MEMORY.md）
# ─────────────────────────────────────────────────────────────────────────────


_SELF_REVIEW_HEADING_TMPL = _re_ol54.compile(
    r"^\s*##\s+Phase\s+(M\d+)\s+self-review.*$",
    _re_ol54.IGNORECASE | _re_ol54.MULTILINE,
)
_CHECKLIST_ITEM_RE = _re_ol54.compile(
    r"^\s*[-*]\s+\[([ xX✓✗❌])\]\s+(.+?)(?:\s*$)",
    _re_ol54.MULTILINE,
)


_REQUIRED_SELF_REVIEW_KEYWORDS = [
    "signature",        # host_wrapper signature == module_interfaces.yaml
    "output",           # outputs written via assemble / slice
    "view",             # pypto.view shape/offsets rank
    "inventory",        # SPEC golden inventory cross-check
    "for ... in range", # Layer K Python for-range absent
    "exactly once",     # Layer K JIT call exactly once
]


@register("OL54")
def check_ol54(ctx: CheckContext) -> Finding:
    """complete_phase 时, `MEMORY.md` 必须存在 `## Phase M_k self-review` 章节,
    且 6 个必填检查项均已 ✅ 标记。

    必填项 (按子串匹配):
      1. host_wrapper signature 与 module_interfaces.yaml 一致
      2. 所有 output 通过 pypto.assemble / `[:] =` 写回
      3. 所有 pypto.view 的 shape/offsets/valid_shape rank 一致
      4. SPEC golden inventory 每行均含 impl 侧 line ref
      5. Layer K 内不存在 `for ... in range(...)`
      6. Layer K 的 JIT call 恰好一次

    `phase_scope` 未设置 (即 complete_stage / general check) 时 SKIP。
    """
    phase_scope = getattr(ctx, "phase_scope", None)
    if not phase_scope:
        return ctx.make_finding(
            "OL54", "SKIP",
            "phase_scope 未设置 — 该规则仅在 complete_phase 时生效",
        )
    memory_file = "MEMORY.md"
    if not ctx.file_exists(memory_file):
        return ctx.make_finding(
            "OL54", "FAIL",
            f"{memory_file} 不存在 — Phase {phase_scope} self-review 为必填项",
            file=memory_file,
        )
    text = ctx.read_file(memory_file)
    expected_heading = f"## Phase {phase_scope} self-review"
    headings = list(_SELF_REVIEW_HEADING_TMPL.finditer(text))
    target = None
    for m in headings:
        if m.group(1).upper() == phase_scope.upper():
            target = m
            break
    if target is None:
        return ctx.make_finding(
            "OL54", "FAIL",
            f"{memory_file} 缺少 `{expected_heading}` 章节。"
            f"complete_phase 之前必须填写 6 项必填检查清单。"
            f"模板见 skill `pypto-memory-template` SKILL.md。",
            file=memory_file,
        )
    start = target.end()
    next_h_match = _re_ol54.search(r"^\s*##\s+", text[start:], _re_ol54.MULTILINE)
    body = text[start: start + next_h_match.start()] if next_h_match else text[start:]

    items = _CHECKLIST_ITEM_RE.findall(body)

    missing_keywords: list[str] = []
    unchecked_keywords: list[str] = []
    for kw in _REQUIRED_SELF_REVIEW_KEYWORDS:
        found = None
        for mark, desc in items:
            if kw.lower() in desc.lower():
                found = (mark, desc)
                break
        if found is None:
            missing_keywords.append(kw)
            continue
        mark = found[0].strip().lower()
        if mark not in ("x", "✓"):
            unchecked_keywords.append(kw)

    problems: list[str] = []
    if missing_keywords:
        problems.append(
            f"必填项缺失 ({len(missing_keywords)} 个): {', '.join(missing_keywords)}"
        )
    if unchecked_keywords:
        problems.append(
            f"未勾选 ({len(unchecked_keywords)} 个): {', '.join(unchecked_keywords)}"
        )
    if problems:
        return ctx.make_finding(
            "OL54", "FAIL",
            f"[S1] {memory_file} `{expected_heading}` 章节的自我评审未完成。\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\n修正方针: 每个条目按 `- [x] <说明>` 填写, 必要时附 evidence "
              f"(impl 行号或对应代码片段)。只要存在一个 `- [ ]` / 缺失项, "
              f"complete_phase 都不会通过。",
            file=memory_file,
        )
    return ctx.make_finding(
        "OL54", "PASS",
        f"Phase {phase_scope} self-review 6 项均已 ✅",
        file=memory_file,
    )


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
        return ctx.make_finding(
            "OL10",
            "FAIL",
            f"{API_REPORT_FILE} 缺少必需内容: {', '.join(missing)}\n"
            f"提示: 以上关键词可能因 heading 使用了英文等价词（如 API Mapping）"
            f"而未匹配，请检查对应章节的 heading 是否包含上述中文关键词。",
            file=API_REPORT_FILE,
        )
    return ctx.make_finding(
        "OL10",
        "PASS",
        f"{API_REPORT_FILE} 含 API 映射、约束与 Tiling 说明",
        file=API_REPORT_FILE,
    )


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
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ctx.make_finding(
            "OL11", "FAIL", f"{golden_file} 导入超时（>10s）", file=golden_file
        )
    except OSError as e:
        return ctx.make_finding(
            "OL11", "FAIL", f"{golden_file} 导入探测失败: {e}", file=golden_file
        )
    if result.returncode != 0:
        return ctx.make_finding(
            "OL11",
            "FAIL",
            f"{golden_file} 导入失败: {result.stderr[:200]}",
            file=golden_file,
        )
    return ctx.make_finding("OL11", "PASS", f"{golden_file} 可导入", file=golden_file)


@register("OL12")
def check_ol12(ctx: CheckContext) -> Finding:
    if not ctx.file_exists(DESIGN_FILE):
        return ctx.make_finding("OL12", "FAIL", f"{DESIGN_FILE} 不存在")
    content = ctx.read_file(DESIGN_FILE)
    headings = _extract_markdown_headings(content)
    missing: list[str] = []
    if not _has_heading_like(headings, "计算图") and not _has_heading_like(
        headings, "API 映射"
    ):
        missing.append("计算图")
    if not _has_heading_like(headings, "Tiling") and not _has_heading_like(
        headings, "数据切分"
    ):
        missing.append("Tiling")
    if not _has_heading_like(headings, "验证方案"):
        missing.append("验证方案")
    if missing:
        return ctx.make_finding(
            "OL12",
            "FAIL",
            f"{DESIGN_FILE} 缺少必需内容: {', '.join(missing)}\n"
            f"提示: 以上关键词可能因 heading 使用了英文等价词（如 Compute Graph、Verification Plan）"
            f"而未匹配，请检查对应章节的 heading 是否包含上述中文关键词。",
            file=DESIGN_FILE,
        )
    return ctx.make_finding(
        "OL12", "PASS", f"{DESIGN_FILE} 含计算图、Tiling 与验证方案", file=DESIGN_FILE
    )


@register("OL13")
def check_ol13(ctx: CheckContext) -> Finding:
    """Stage 5 cleanup 三件套：{op}_impl.py + test_{op}.py + README.md。"""
    files = [
        f"{ctx.op_name}_impl.py",
        f"test_{ctx.op_name}.py",
        "README.md",
    ]
    missing = [f for f in files if not ctx.file_exists(f)]
    if missing:
        return ctx.make_finding(
            "OL13",
            "FAIL",
            f"Stage 5 cleanup 三件套不完整，缺少: {', '.join(missing)}",
        )
    return ctx.make_finding("OL13", "PASS", "Stage 5 cleanup 三件套完整")


@register("OL44")
def check_ol44(ctx: CheckContext) -> Finding:
    """Stage 5 当前 Phase 三件套：modules/<op>_module<k>_impl.py +
    modules/<op>_module<k>_golden.py + modules/test_<op>_module<k>.py。

    从 .orchestrator_state.json 读取当前活跃 Phase
    （stage5_phases.active_phase），解析后缀后验证三件套是否存在。
    """
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding(
            "OL44",
            "SKIP",
            ".orchestrator_state.json 不存在（无状态运行），无法判断 Stage 5 modules/ 状态",
        )
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (ValueError, OSError) as e:
        return ctx.make_finding(
            "OL44", "FAIL", f"无法解析 .orchestrator_state.json: {e}"
        )

    # 解析活跃 Phase。Schema v2.0 中存放在 stage5_phases.active_phase。
    active_phase = None
    s5 = state.get("stage5_phases")
    if isinstance(s5, dict):
        active_phase = s5.get("active_phase")
    if not active_phase:
        return ctx.make_finding("OL44", "SKIP", "stage5_phases 中未记录活跃 Phase M_k")

    # active_phase 是 phase 标识符 (M1 / M2 / M3 / …)。impl/golden/test
    # 文件命名遵循累积后缀约定 (M1→"1", M2→"12", M3→"123", …)；用
    # _phase_to_module_suffix 转换以匹配实际 staged 文件名。直接 active_phase[1:]
    # 会得到 phase 编号 ("2", "3" 等), 与累积命名 ("12", "123") 不匹配, 历史上
    # 这导致 agent 被迫复制累积文件产生冗余的 module2/module3 同名副本。
    if not isinstance(active_phase, str):
        return ctx.make_finding(
            "OL44", "FAIL", f"格式异常的 active_phase: {active_phase!r}"
        )
    try:
        suffix = _phase_to_module_suffix(active_phase)
    except ValueError as e:
        return ctx.make_finding(
            "OL44", "FAIL", f"格式异常的 active_phase: {active_phase!r} ({e})"
        )

    modules_dir = ctx.file_path("modules")
    if not os.path.isdir(modules_dir):
        return ctx.make_finding(
            "OL44",
            "FAIL",
            "Stage 5 已激活但 custom/<op>/modules/ 目录不存在",
        )

    expected = [
        f"modules/{ctx.op_name}_module{suffix}_impl.py",
        f"modules/{ctx.op_name}_module{suffix}_golden.py",
        f"modules/test_{ctx.op_name}_module{suffix}.py",
    ]
    missing = [p for p in expected if not ctx.file_exists(p)]
    if missing:
        return ctx.make_finding(
            "OL44",
            "FAIL",
            f"活跃 Phase {active_phase} 三件套不完整，缺少: {', '.join(missing)}",
        )
    return ctx.make_finding(
        "OL44",
        "PASS",
        f"活跃 Phase {active_phase} 三件套完整（impl + golden + test）",
    )


@register("OL14")
def check_ol14(ctx: CheckContext) -> Finding:
    """Stage 6（结构验证）进入前需要 Stage 5（含 cleanup）已完成。"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL14", "FAIL", "状态文件不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("stage_status", {})
        # 新 Stage 1-7 模型：Stage 5 = Construction（per-Phase + cleanup）必须在
        # 进入 Stage 6 = 结构验证前完成。
        if status.get("5") == "completed":
            return ctx.make_finding(
                "OL14", "PASS", "Stage 5 已完成，可进入 Stage 6（结构验证）"
            )
    except ValueError:
        pass
    return ctx.make_finding(
        "OL14",
        "FAIL",
        "Stage 6 入口被阻止：Stage 5 尚未完成",
    )


@register("OL24")
def check_ol24(ctx: CheckContext) -> Finding:
    """.orchestrator_state.json 结构合法（schema v2.0）。"""
    state_path = ctx.file_path(".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return ctx.make_finding("OL24", "FAIL", ".orchestrator_state.json 不存在")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ctx.make_finding("OL24", "FAIL", f"JSON 解析失败: {e}")
    # schema v2.0 必需字段（同时兼容 v1 旧格式 — max_stage 缺失时视为旧版放行）。
    required = ["operator_name", "current_stage", "stage_status"]
    missing = [k for k in required if k not in data]
    if missing:
        return ctx.make_finding(
            "OL24",
            "FAIL",
            f"缺少必需字段: {', '.join(missing)}",
        )
    # 软检查：max_stage 缺失时警告（旧版 schema）。
    if "max_stage" not in data:
        return ctx.make_finding(
            "OL24",
            "WARN",
            "状态文件未声明 max_stage（schema v1 旧格式）；建议通过 state_transition 重新初始化以升级到 v2.0",
        )
    # v2.0 字段为可选，但若存在则必须格式正确。
    if "stage5_phases" in data:
        s5 = data["stage5_phases"]
        if not isinstance(s5, dict) or "phase_status" not in s5:
            return ctx.make_finding(
                "OL24",
                "FAIL",
                "stage5_phases 必须是包含 phase_status 的字典",
            )
    if "rollback_history" in data and not isinstance(data["rollback_history"], list):
        return ctx.make_finding(
            "OL24",
            "FAIL",
            "rollback_history 必须是列表",
        )
    if "artifact_hashes" in data and not isinstance(data["artifact_hashes"], dict):
        return ctx.make_finding(
            "OL24",
            "FAIL",
            "artifact_hashes 必须是字典",
        )
    return ctx.make_finding("OL24", "PASS", "状态文件结构合法 (schema v2.0)")
