# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

"""Hook 入口集成测试 — 以子进程方式调用 pypto_op_lint.py，验证完整调用链。"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .helpers import build_stateless_op_dir, write_file

SCRIPT = str(Path(__file__).resolve().parents[1] / "pypto_op_lint.py")
PYTHON = sys.executable


def _run_hook(hook: str, payload: dict, env_extra: dict | None = None) -> tuple[int, str]:
    """以子进程执行 hook，返回 (exit_code, stdout)"""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [PYTHON, SCRIPT, "--hook", hook],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=15, env=env,
    )
    return result.returncode, result.stdout.strip()


# ── post-edit: 非算子文件 → 无输出 ──

def test_post_edit_non_operator_file_silent():
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": "/tmp/readme.md"}})
    assert rc == 0
    assert not out


# ── post-edit: 算子 impl 文件 → 返回结构化 JSON ──

def test_post_edit_impl_returns_hook_json(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    impl_path = str(op_dir / "demo_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0
    if out:
        data = json.loads(out)
        assert "hookSpecificOutput" in data
        hook_out = data["hookSpecificOutput"]
        assert hook_out.get("decision") in ("allow", "block")


# ── post-edit: S1 违规 → decision=block ──

def test_post_edit_blocks_on_s1_violation(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 写一个缺少 @jit 装饰器的 impl（违反 OL01）
    bad_impl = """import pypto
def demo_kernel(x, y):
    y[:] = x

def demo_wrapper(x, y):
    return None
"""
    write_file(op_dir / "demo_impl.py", bad_impl)
    impl_path = str(op_dir / "demo_impl.py")
    rc, out = _run_hook("post-edit", {"tool_input": {"file_path": impl_path}})
    assert rc == 0  # exit 0 even on block (structured output)
    data = json.loads(out)
    assert data["hookSpecificOutput"]["decision"] == "block"


# ── post-bash: 非测试命令 → 无输出 ──

def test_post_bash_non_test_command_silent():
    rc, out = _run_hook("post-bash", {"tool_input": {"command": "ls -la"}})
    assert rc == 0
    assert not out


# ── post-bash: 测试命令 → 返回判定结果 ──

def test_post_bash_test_command_returns_verdict():
    payload = {
        "tool_input": {"command": "python3 test_demo.py"},
        "tool_result": {
            "stdout": "some output [PRECISION_PASS] done",
            "stderr": "",
            "exit_code": 0,
        },
    }
    rc, out = _run_hook("post-bash", payload)
    assert rc == 0
    data = json.loads(out)
    ctx = data["hookSpecificOutput"].get("additionalContext", "")
    assert "precision_pass" in ctx


# ── pre-edit-backup: 非 impl 文件 → 无输出 ──

def test_pre_edit_backup_non_impl_silent():
    rc, out = _run_hook("pre-edit-backup", {"tool_input": {"file_path": "/tmp/readme.md"}})
    assert rc == 0
    assert not out


# ── stop: 非算子目录 → 无输出 ──

def test_stop_non_operator_dir_silent():
    rc, out = _run_hook("stop", {"cwd": "/tmp"})
    assert rc == 0
    assert not out


# ── stop: 算子目录有 S1 违规 → exit 2 ──

def test_stop_blocks_on_s1_violation(tmp_path: Path):
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    # 删除 impl 使 OL13 失败
    (op_dir / "demo_impl.py").unlink()
    state = {
        "operator_name": "demo",
        "current_stage": 5,
        "stage_status": {"5": "in_progress"},
    }
    write_file(op_dir / ".orchestrator_state.json", json.dumps(state))
    rc, out = _run_hook("stop", {"cwd": str(op_dir)})
    assert rc == 2
    data = json.loads(out)
    assert data["hookSpecificOutput"]["decision"] == "block"


# ── 边界场景: 空 stdin → 安全退出 ──

def test_hook_empty_stdin():
    """所有 hook 在空输入时应安全退出"""
    for hook in ("post-edit", "post-bash", "pre-edit-backup", "stop"):
        rc, out = _run_hook(hook, {})
        assert rc == 0, f"{hook} failed with rc={rc}"
