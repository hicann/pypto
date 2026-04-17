#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import argparse

from . import checks  # noqa: F401
from .core import (
    CONSISTENCY_RULE_IDS,
    GOLDEN_RULE_IDS,
    IMPL_RULE_IDS,
    TEST_RULE_IDS,
    Finding,
    _has_error_fail,
    _run_checks,
)
from .hooks import hook_post_bash, hook_post_edit, hook_pre_edit_backup, hook_stop
from .infer import _build_context
from .observability import _print_findings


def _cmd_run(findings: list[Finding]) -> int:
    _print_findings(findings)
    return 2 if _has_error_fail(findings) else 0


def cmd_lint_impl(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, IMPL_RULE_IDS))


def cmd_lint_golden(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, GOLDEN_RULE_IDS))


def cmd_lint_test(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, TEST_RULE_IDS))


def cmd_lint_consistency(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    return _cmd_run(_run_checks(ctx, CONSISTENCY_RULE_IDS))


def cmd_check_gate(op_dir: str, stage: int) -> int:
    ctx = _build_context(op_dir, stage)
    gate_rules: list[str] = []
    for rule in ctx.rules:
        # 门禁检查应覆盖当前 stage 的全部交付规则，而不仅是 gate/flow。
        # 否则会出现 Stage 5/6/7 对 impl/test 关键 S1 规则不阻断的问题。
        if rule.get("target") not in ("gate", "flow", "impl", "test", "golden"):
            continue
        if stage not in rule.get("stages", []):
            continue
        gate_rules.append(rule["id"])
    return _cmd_run(_run_checks(ctx, gate_rules))


def main() -> int:
    parser = argparse.ArgumentParser(description="PyPTO 算子开发流程确定性检查工具")
    parser.add_argument("--hook",
                        choices=["post-edit", "post-bash",
                                 "pre-edit-backup", "stop"],
                        help="Hook 模式（从 stdin 读 JSON）")
    parser.add_argument("--lint-impl", action="store_true",
                        help="检查 impl 文件")
    parser.add_argument("--lint-golden", action="store_true",
                        help="检查 golden 文件")
    parser.add_argument("--lint-test", action="store_true",
                        help="检查 test 文件")
    parser.add_argument("--lint-consistency", action="store_true",
                        help="检查跨文件一致性（D5 规则）")
    parser.add_argument("--check-gate", action="store_true",
                        help="检查阶段门禁")
    parser.add_argument("--op-dir", help="算子工作目录")
    parser.add_argument("--stage", type=int, default=5,
                        help="当前阶段 (1-7)")
    args = parser.parse_args()

    if args.hook:
        return {"post-edit": hook_post_edit,
                "post-bash": hook_post_bash,
                "pre-edit-backup": hook_pre_edit_backup,
                "stop": hook_stop,
                }[args.hook]()
    elif args.lint_impl:
        if not args.op_dir:
            parser.error("--lint-impl 需要 --op-dir")
        return cmd_lint_impl(args.op_dir, args.stage)
    elif args.lint_golden:
        if not args.op_dir:
            parser.error("--lint-golden 需要 --op-dir")
        return cmd_lint_golden(args.op_dir, args.stage)
    elif args.lint_test:
        if not args.op_dir:
            parser.error("--lint-test 需要 --op-dir")
        return cmd_lint_test(args.op_dir, args.stage)
    elif args.lint_consistency:
        if not args.op_dir:
            parser.error("--lint-consistency 需要 --op-dir")
        return cmd_lint_consistency(args.op_dir, args.stage)
    elif args.check_gate:
        if not args.op_dir:
            parser.error("--check-gate 需要 --op-dir")
        return cmd_check_gate(args.op_dir, args.stage)
    else:
        parser.print_help()
        return 0
