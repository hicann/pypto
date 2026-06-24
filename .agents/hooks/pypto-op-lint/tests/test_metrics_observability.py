# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import importlib.util
import json
import os
from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule, run_stop


def test_rule_check_emits_metric_event(tmp_path: Path):
    mod = load_lint_module()
    mod.LOGS_DIR = str(tmp_path / "logs")
    mod.LOGS_EVENTS_FILE = str(tmp_path / "logs" / "lint_events.jsonl")

    op_dir = build_stateless_op_dir(tmp_path, "demo")
    finding = run_rule(mod, op_dir, "OL30")
    assert finding.status == "PASS"

    events_path = Path(mod.LOGS_EVENTS_FILE)
    assert events_path.exists()
    lines = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(e.get("event_type") == "rule_check" and e.get("rule_id") == "OL30" for e in lines)


def test_stop_block_output_contains_blocking_rules_and_fix_hints(tmp_path: Path, capfd):
    mod = load_lint_module()
    mod.LOGS_DIR = str(tmp_path / "logs")
    mod.LOGS_EVENTS_FILE = str(tmp_path / "logs" / "lint_events.jsonl")

    op_dir = build_stateless_op_dir(tmp_path, "demo")
    (op_dir / "SPEC.md").write_text("# SPEC only\n", encoding="utf-8")
    os.environ["PYPTO_OP_LINT_STRICT"] = "1"
    code = run_stop(mod, str(op_dir))
    assert code == 2

    captured = capfd.readouterr().out
    assert "blocking_rules" in captured
    assert "fix_hints" in captured


def test_metrics_report_script_generates_summary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "logs" / "report_metrics.py"
    events = tmp_path / "lint_events.jsonl"
    events.write_text(
        "\n".join([
            json.dumps({"event_type": "rule_check", "rule_id": "OL30", "status": "PASS", "duration_ms": 1.2}),
            json.dumps({"event_type": "rule_check", "rule_id": "OL34", "status": "FAIL", "duration_ms": 2.5}),
            json.dumps({"event_type": "gate_decision", "blocked": True}),
        ]) + "\n",
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("report_metrics", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.BASE = tmp_path
    mod.EVENTS = events
    mod.OUT_JSON = tmp_path / "summary_latest.json"
    mod.OUT_MD = tmp_path / "summary_latest.md"
    summary = mod.build_summary(mod.load_events())
    mod.write_outputs(summary)

    assert mod.OUT_JSON.exists()
    data = json.loads(mod.OUT_JSON.read_text(encoding="utf-8"))
    assert data["total_rule_checks"] == 2
    assert data["total_gate_blocks"] == 1
