#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import json
import os
import select
import sys
from typing import Any, Optional

from .core import (
    API_REPORT_FILE,
    DESIGN_FILE,
    GOLDEN_RULE_IDS,
    HOOK_INPUT_ENV,
    IMPL_RULE_IDS,
    POST_EDIT_CONSISTENCY_RULE_IDS,
    SPEC_FILE,
    TEST_RULE_IDS,
    CheckContext,
    _load_rules,
)


def _infer_op_dir(file_path: str) -> Optional[str]:
    if not file_path:
        return None
    op_dir = os.path.dirname(os.path.abspath(file_path))
    if os.path.isfile(os.path.join(op_dir, ".orchestrator_state.json")):
        return op_dir
    basename = os.path.basename(file_path)
    inferred_op_name = _infer_op_name_from_filename(basename)
    if inferred_op_name and _looks_like_stateless_op_dir(op_dir, inferred_op_name):
        return op_dir
    return None


def _infer_op_name_from_filename(filename: str) -> str:
    if filename.startswith("test_") and filename.endswith(".py"):
        return filename[len("test_"):-len(".py")]
    if filename.endswith("_impl.py"):
        return filename[:-len("_impl.py")]
    if filename.endswith("_golden.py"):
        return filename[:-len("_golden.py")]
    return ""


def _looks_like_stateless_op_dir(op_dir: str, op_name: str) -> bool:
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return False
    expected = {
        f"{op_name}_impl.py",
        f"{op_name}_golden.py",
        f"test_{op_name}.py",
        SPEC_FILE,
        API_REPORT_FILE,
        DESIGN_FILE,
        "README.md",
    }
    return len(files & expected) >= 2


def _infer_stage_from_filename(filename: str) -> int:
    if filename == SPEC_FILE:
        return 1
    if filename == API_REPORT_FILE:
        return 2
    if filename == DESIGN_FILE:
        return 4
    if filename.endswith("_golden.py"):
        return 3
    if filename.endswith("_impl.py") or (filename.startswith("test_") and filename.endswith(".py")):
        return 5
    return 0


def _infer_stage_from_artifacts(op_dir: str) -> int:
    op_name = os.path.basename(op_dir)
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return 0

    if len({f"{op_name}_impl.py", f"test_{op_name}.py", "README.md"} & files) >= 2:
        return 5
    if DESIGN_FILE in files or f"{op_name}_golden.py" in files:
        return 4
    if API_REPORT_FILE in files:
        return 3
    if SPEC_FILE in files:
        return 2
    return 0


def _get_current_stage(op_dir: str) -> int:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if not os.path.isfile(state_path):
        return 0
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("current_stage", 0))
    except ValueError:
        return 0


def _get_op_name(op_dir: str) -> str:
    state_path = os.path.join(op_dir, ".orchestrator_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                name = json.load(f).get("operator_name", "")
                if name:
                    return name
        except ValueError:
            pass
    return os.path.basename(op_dir)


def _build_context(op_dir: str, stage: Optional[int] = None) -> CheckContext:
    rules = _load_rules()
    op_dir = os.path.abspath(op_dir)
    if stage is None:
        stage = _get_current_stage(op_dir)
    op_name = _get_op_name(op_dir)
    return CheckContext(op_dir=op_dir, op_name=op_name, stage=stage, rules=rules)


def _load_hook_input() -> dict[str, Any]:
    raw = ""
    # Try stdin first — non-blocking probe. Works for any caller that pipes data.
    # select(0) returns immediately: data ready → read; no data → skip to env var.
    # On a TTY stdin, select(0) also returns immediately (TTY has no pending data),
    # so this is safe for interactive terminals too — no isatty() check needed.
    if not sys.stdin.closed:
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.read()
        except (ValueError, OSError):
            pass
    # Fall back to env var (OpenCode plugin path)
    if not raw.strip():
        raw = os.environ.get(HOOK_INPUT_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


SPEC_RULE_IDS = ["OL09"]


def _rule_ids_for_filename(filename: str) -> list[str]:
    """Return rule IDs for post-edit checks on a single file.

    Only includes rules that validate the edited file itself.  Cross-file
    consistency rules (target=gate, D5 dimension: OL30-OL34, OL39, OL40,
    OL43) are deferred to the gate/stop check so that half-written
    artefacts don't cause spurious blocks during implementation.
    """
    if filename.endswith("_impl.py"):
        return IMPL_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    if filename.endswith("_golden.py"):
        return GOLDEN_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    is_test_file = filename.startswith("test_") and filename.endswith(".py")
    if is_test_file:
        return TEST_RULE_IDS + POST_EDIT_CONSISTENCY_RULE_IDS
    if filename == SPEC_FILE:
        return SPEC_RULE_IDS
    return []


_MAX_OP_DIR_SEARCH_DEPTH = 8


def _find_nearest_op_dir(cwd: str) -> Optional[str]:
    """从 cwd 向上查找所属算子目录，避免跨算子误判。"""
    abs_cwd = os.path.abspath(cwd)
    current = abs_cwd
    for _ in range(_MAX_OP_DIR_SEARCH_DEPTH):
        if os.path.isfile(os.path.join(current, ".orchestrator_state.json")):
            return current
        if _looks_like_stateless_op_dir(current, os.path.basename(current)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None
