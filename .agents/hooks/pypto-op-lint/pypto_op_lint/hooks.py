#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess

from . import checks  # noqa: F401
from .core import (
    GIT_BIN,
    MODE_ENV,
    OP_WORKSPACE_DIR,
    POST_EDIT_BLOCK_ENV,
    SPEC_FILE,
    TEST_COMMAND_PATTERN,
    _run_checks,
)
from .infer import (
    _build_context,
    _find_nearest_op_dir,
    _get_current_stage,
    _infer_op_dir,
    _infer_stage_from_artifacts,
    _infer_stage_from_filename,
    _load_hook_input,
    _rule_ids_for_filename,
)
from .observability import _emit_gate_event, _output_hook_json


def hook_post_edit() -> int:
    """PostToolUse[Write|Edit] — 按文件类型 lint。

    默认开启“产物生成即阻断”模式：当检测到 S0/S1 FAIL 时返回 decision=block，
    由插件侧拦截本次工具调用。
    """
    os.environ[MODE_ENV] = "post-edit"
    data = _load_hook_input()
    file_path = data.get("tool_input", {}).get("file_path", "")
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return 0

    basename = os.path.basename(file_path)
    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_filename(basename)
        if stage == 0:
            return 0
    ctx = _build_context(op_dir, stage)

    # ── SPEC.md 冻结：Stage 2 完成后禁止修改 ──
    if basename == SPEC_FILE and ctx.stage is not None and ctx.stage >= 3:
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=(
                f"[pypto-op-lint] {SPEC_FILE} 在 Stage 2 完成后已冻结，不允许修改。"
                "如需变更需求规格，应通过 state_transition 回退到 Stage 2 重新审核。"
            ),
        )
        return 0

    rule_ids = _rule_ids_for_filename(basename)
    if not rule_ids:
        return 0

    findings = _run_checks(ctx, rule_ids)
    fails = [f for f in findings if f.status == "FAIL"]
    error_fails = [f for f in fails if f.severity in ("S0", "S1")]
    warns = [f for f in findings if f.status == "WARN"]
    infos = [f for f in findings if f.status == "INFO"]
    if not fails and not warns and not infos:
        return 0

    sections = []
    if fails:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in fails]
        sections.append(
            "[pypto-op-lint] 以下规则违规，请立即修正后重新写入文件：\n"
            + "\n".join(lines)
            + "\n\n参考 .agents/skills/pypto-op-develop/references/execution-constraints.md"
        )
    if warns:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in warns]
        sections.append(
            "[pypto-op-lint] 以下提醒建议确认：\n"
            + "\n".join(lines)
            + "\n\n请确认以上提醒项是否需要处理。"
        )
    if infos:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in infos]
        sections.append(
            "[pypto-op-lint] 以下信息提示（不影响门禁）：\n"
            + "\n".join(lines)
        )
    context_msg = "\n\n".join(sections)

    strict_block = os.environ.get(POST_EDIT_BLOCK_ENV, "1") == "1"
    if strict_block and error_fails:
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in error_fails]
        reason = (
            "[pypto-op-lint] 产物写入后即时门禁未通过（S0/S1）：\n"
            + "\n".join(lines)
            + "\n\n请先修复后再继续。"
        )
        _output_hook_json(
            "PostToolUse",
            decision="block",
            reason=reason,
            additionalContext=context_msg,
        )
        # 不返回非零，避免插件侧拿不到结构化结果；实际阻断由插件根据 decision 实施。
        return 0

    _output_hook_json("PostToolUse", decision="allow", reason="", additionalContext=context_msg)
    return 0


def hook_post_bash() -> int:
    """PostToolUse[Bash] — 三态解析测试输出"""
    os.environ[MODE_ENV] = "post-bash"
    data = _load_hook_input()
    command = data.get("tool_input", {}).get("command", "")
    if not TEST_COMMAND_PATTERN.search(command):
        return 0

    stdout = data.get("tool_result", {}).get("stdout", "")
    stderr = data.get("tool_result", {}).get("stderr", "")
    exit_code = data.get("tool_result", {}).get("exit_code", 0)

    verdict = _parse_verdict(stdout, stderr, exit_code)
    detail = _verdict_detail(verdict)
    context_msg = (
        f"[pypto-op-lint parse-result] 确定性判定: {verdict}。{detail}。"
        "请以此结果为准，不要自行解读测试输出。"
    )
    _output_hook_json("PostToolUse", additionalContext=context_msg)
    return 0


def hook_pre_edit_backup() -> int:
    """PreToolUse[Write|Edit] — 状态文件保护 + Stage 6 编辑 impl 前 git auto-commit"""
    data = _load_hook_input()
    file_path = data.get("tool_input", {}).get("file_path", "")

    # 拦截对 .orchestrator_state.json 的直接写入
    if os.path.basename(file_path) == ".orchestrator_state.json":
        _output_hook_json(
            "PreToolUse",
            decision="block",
            reason="禁止直接修改 .orchestrator_state.json，"
                   "请通过 state_transition 工具操作阶段状态。",
        )
        return 0

    if not file_path.endswith("_impl.py"):
        return 0
    op_dir = _infer_op_dir(file_path)
    if not op_dir:
        return 0
    if _get_current_stage(op_dir) != 6:
        return 0

    impl_file = os.path.basename(file_path)
    try:
        _ensure_git_init(op_dir)
        subprocess.run(
            [_git_executable(), "add", impl_file],
            cwd=op_dir, check=True, capture_output=True,
        )
        attempt = _get_stage6_attempt(op_dir)
        subprocess.run(
            [_git_executable(), "commit", "-m",
             f"backup: {impl_file} before stage6 attempt {attempt}"],
            cwd=op_dir, check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # git commit 失败（可能无变更），不拦截
    return 0


def _git_executable() -> str:
    if GIT_BIN:
        return GIT_BIN
    return "/usr/bin/git"


def _ensure_git_init(op_dir: str):
    """确保算子工作区根目录有 git 仓库"""
    current = op_dir
    while current and os.path.basename(current) != OP_WORKSPACE_DIR:
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    git_dir = current if os.path.basename(current) == OP_WORKSPACE_DIR else op_dir
    if not os.path.isdir(os.path.join(git_dir, ".git")):
        subprocess.run([_git_executable(), "init"], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run([_git_executable(), "add", "."], cwd=git_dir,
                       check=True, capture_output=True)
        subprocess.run([_git_executable(), "commit", "-m", "init: operator workspace"],
                       cwd=git_dir, check=True, capture_output=True)


def _get_stage6_attempt(op_dir: str) -> int:
    """从 .orchestrator_state.json 获取 Stage 6 重试次数"""
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("stage_retry_count", {}).get("6", 0)) + 1
    except (ValueError, FileNotFoundError):
        return 1


def hook_stop() -> int:
    """Stop — agent 结束前交付门禁"""
    os.environ[MODE_ENV] = "stop"
    data = _load_hook_input()
    cwd = data.get("cwd", os.getcwd())

    op_dir = _find_nearest_op_dir(cwd)
    if not op_dir:
        return 0

    stage = None
    if not os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        stage = _infer_stage_from_artifacts(op_dir)
        if stage == 0:
            return 0
    ctx = _build_context(op_dir, stage)
    applicable = [r["id"] for r in ctx.rules if ctx.stage in r.get("stages", [])]
    findings = _run_checks(ctx, applicable)
    error_fails = []
    for finding in findings:
        if finding.status == "FAIL" and finding.severity in ("S0", "S1"):
            error_fails.append(finding)

    if error_fails:
        blocking_rules = [f.rule_id for f in error_fails]
        lines = [f"  [{f.rule_id}][{f.severity}] {f.message}" for f in error_fails]
        hint_lines = [f"  - {f.rule_id}: {_rule_fix_hint(f.rule_id)}" for f in error_fails]
        _emit_gate_event(ctx, blocked=True, blocking_rules=blocking_rules)
        _output_hook_json("Stop",
            decision="block",
            reason="[pypto-op-lint] 交付门禁未通过，存在 ERROR（S0/S1）级违规：\n"
                   + "\n".join(lines)
                   + "\n\nblocking_rules: " + ", ".join(blocking_rules)
                   + "\nfix_hints:\n" + "\n".join(hint_lines)
                   + "\n\ndocs_ref: .agents/hooks/pypto-op-lint/rules.json"
                   + "\n\n**⛔ 门禁已阻断：请先修复上述 ERROR 级违规，再继续后续操作。"
                     "修复后重新运行即可。**")
        return 2
    _emit_gate_event(ctx, blocked=False, blocking_rules=[])
    return 0


def _rule_fix_hint(rule_id: str) -> str:
    hints = {
        "OL30": "在 SPEC.md front matter 中填写 supported_dtypes，并在 test 中覆盖对应 dtype",
        "OL31": "在 DESIGN.md front matter 设置 dynamic_axes，并在 impl Tensor 注解使用 pypto.DYNAMIC",
        "OL32": "在 SPEC.md front matter 的 tolerance 中填写 atol/rtol，并校准 test 断言阈值",
        "OL34": "在 SPEC.md front matter 的 p0_shapes 填写 P0 形状，并在 test 用例覆盖",
        "OL39": "为 SPEC.md/DESIGN.md/API_REPORT.md 添加 front matter 块（--- 包裹）",
        "OL40": "补齐 front matter 必填字段并修正字段类型（list/dict）",
        "OL41": "删除代码文件中的 lint 门禁输出文本，确保仅保留可执行源码/文档内容",
    }
    return hints.get(rule_id, "参考 rules.json 中该规则说明修复")


def _parse_verdict(stdout: str, stderr: str, exit_code: int) -> str:
    combined = f"{stdout}\n{stderr}"
    # FAIL 优先于 PASS：多 case 测试中部分失败应判定为整体失败
    if "[PRECISION_FAIL]" in combined:
        return "precision_fail"
    if "[PRECISION_PASS]" in combined:
        return "precision_pass"
    if exit_code != 0:
        return "runtime_error"
    return "no_marker"


def _verdict_detail(verdict: str) -> str:
    details = {
        "precision_pass": "输出中包含 [PRECISION_PASS] 标记，精度通过",
        "precision_fail": "输出中包含 [PRECISION_FAIL] 标记，精度失败",
        "runtime_error": "测试进程返回非零退出码，运行失败",
        "no_marker": "测试运行完毕但未检测到精度标记（[PRECISION_PASS]/[PRECISION_FAIL]）",
    }
    return details.get(verdict, f"未知判定结果: {verdict}")
