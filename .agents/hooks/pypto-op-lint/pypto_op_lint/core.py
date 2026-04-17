#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_COMMAND_PATTERN = re.compile(r"python3?\s+.*test_\w+\.py")
PYTHON_BIN = os.path.abspath(sys.executable)
GIT_BIN = shutil.which("git")

SPEC_FILE = "SPEC.md"

API_REPORT_FILE = "API_REPORT.md"

DESIGN_FILE = "DESIGN.md"

OP_WORKSPACE_DIR = "custom"

IMPL_RULE_IDS = [
    "OL01", "OL02", "OL03", "OL04", "OL05", "OL06", "OL07", "OL08",
    "OL16", "OL23", "OL25", "OL26", "OL28", "OL29", "OL37",
]

GOLDEN_RULE_IDS = ["OL15"]

TEST_RULE_IDS = ["OL17", "OL18", "OL19", "OL20", "OL21", "OL22", "OL42"]

CONSISTENCY_RULE_IDS = ["OL30", "OL31", "OL32", "OL33", "OL34", "OL39", "OL40", "OL41", "OL43"]

# Subset of CONSISTENCY_RULE_IDS safe for post-edit: only rules that check
# the edited file itself, not cross-file dependencies.  Cross-file rules
# (OL30-OL34, OL39, OL40, OL43) must wait for the gate / stop check.
POST_EDIT_CONSISTENCY_RULE_IDS = ["OL41"]

STRICT_ENV = "PYPTO_OP_LINT_STRICT"

MODE_ENV = "PYPTO_OP_LINT_MODE"

HOOK_INPUT_ENV = "PYPTO_OP_LINT_HOOK_INPUT"

POST_EDIT_BLOCK_ENV = "PYPTO_OP_LINT_POST_EDIT_BLOCK"


@dataclass
class Finding:
    rule_id: str = ""
    severity: str = ""
    dimension: str = ""
    status: str = "SKIP"
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass
class CheckContext:
    op_dir: str
    op_name: str
    stage: int
    rules: list[dict[str, Any]]
    _ast_cache: dict[str, ast.Module] = field(default_factory=dict, repr=False)
    _parse_errors: dict[str, str] = field(default_factory=dict, repr=False)

    def file_path(self, filename: str) -> str:
        return os.path.join(self.op_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(self.file_path(filename))

    def read_file(self, filename: str) -> str:
        path = self.file_path(filename)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_file(self, filename: str) -> Optional[ast.Module]:
        if filename in self._ast_cache:
            return self._ast_cache[filename]
        source = self.read_file(filename)
        if not source:
            return None
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            self._parse_errors[filename] = str(e)
            return None
        self._ast_cache[filename] = tree
        return tree

    def parse_error(self, filename: str) -> str:
        return self._parse_errors.get(filename, "")

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        return {}

    def make_finding(self, rule_id: str, status: str, message: str,
                     file: str = "", line: int = 0) -> Finding:
        rule = self.get_rule(rule_id)
        return Finding(
            rule_id=rule_id,
            severity=rule.get("severity", "S2"),
            dimension=rule.get("dimension", ""),
            status=status,
            message=message,
            file=file,
            line=line,
        )


CHECKERS: dict[str, Callable[[CheckContext], Finding]] = {}


def register(rule_id: str):
    """装饰器：将检查函数注册到规则 ID"""
    def decorator(fn: Callable[[CheckContext], Finding]):
        CHECKERS[rule_id] = fn
        return fn
    return decorator


def _load_rules() -> list[dict[str, Any]]:
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def _run_checks(ctx: CheckContext, rule_ids: list[str]) -> list[Finding]:
    """执行指定规则的检查"""
    from .observability import _emit_metric_event  # noqa: PLC0415
    findings = []
    mode = os.environ.get(MODE_ENV, "cli")
    strict = os.environ.get(STRICT_ENV, "1") == "1"
    for rid in rule_ids:
        start = time.perf_counter()
        rule = ctx.get_rule(rid)
        if ctx.stage not in rule.get("stages", []):
            finding = ctx.make_finding(rid, "SKIP", "当前阶段不适用")
            findings.append(finding)
            _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
            continue
        checker = CHECKERS.get(rid)
        if not checker:
            finding = ctx.make_finding(rid, "SKIP", "检查函数未注册")
            findings.append(finding)
            _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
            continue
        finding = checker(ctx)
        findings.append(finding)
        _emit_metric_event(ctx, finding, mode, strict, (time.perf_counter() - start) * 1000)
    return findings


def _has_error_fail(findings: list[Finding]) -> bool:
    for finding in findings:
        if finding.status == "FAIL" and finding.severity in ("S0", "S1"):
            return True
    return False
