#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Any, Optional

from .core import SCRIPT_DIR, STRICT_ENV, CheckContext, Finding, _has_error_fail

LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
LOGS_EVENTS_FILE = os.path.join(LOGS_DIR, "lint_events.jsonl")


def _output_hook_json(event: str, **kwargs):
    """输出 hookSpecificOutput JSON 到 stdout"""
    output = {"hookSpecificOutput": {"hookEventName": event, **kwargs}}
    _write_stdout_json(output)


def _print_findings(findings: list[Finding]):
    has_error_fail = _has_error_fail(findings)
    result = {
        "passed": not any(f.status == "FAIL" for f in findings),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "pass": sum(1 for f in findings if f.status == "PASS"),
            "warn": sum(1 for f in findings if f.status == "WARN"),
            "info": sum(1 for f in findings if f.status == "INFO"),
            "fail": sum(1 for f in findings if f.status == "FAIL"),
            "skip": sum(1 for f in findings if f.status == "SKIP"),
            "has_error_fail": has_error_fail,
        },
    }
    _write_stdout_json(result, indent=2)


def _write_stdout_json(payload: dict[str, Any], indent: Optional[int] = None) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=indent)
    os.write(1, f"{content}\n".encode("utf-8"))


def _ensure_logs_dir() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    _ensure_logs_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emit_metric_event(ctx: CheckContext, finding: Finding, mode: str,
                       strict: bool, duration_ms: float) -> None:
    try:
        message_hash = hashlib.sha1(finding.message.encode("utf-8")).hexdigest()[:12]
        event = {
            "ts": int(time.time()),
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "rule_id": finding.rule_id,
            "severity": finding.severity,
            "status": finding.status,
            "mode": mode,
            "strict": strict,
            "duration_ms": round(float(duration_ms), 3),
            "file": finding.file,
            "message_hash": message_hash,
            "event_type": "rule_check",
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass


def _emit_gate_event(ctx: CheckContext, blocked: bool, blocking_rules: list[str]) -> None:
    try:
        event = {
            "ts": int(time.time()),
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "mode": "stop",
            "strict": os.environ.get(STRICT_ENV, "1") == "1",
            "event_type": "gate_decision",
            "blocked": blocked,
            "blocking_rules": blocking_rules,
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass

