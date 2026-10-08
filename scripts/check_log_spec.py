#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the pypto-log-check skill on staged files through a local agent CLI.

The skill is agent-driven, so this hook is a no-op when no agent CLI is
installed (the CI runners).  Only findings on lines added by the current
commit are reported; 致命/严重 findings block the commit.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SKILL_FILE = ".agents/skills/pypto-log-check/SKILL.md"
RULES_FILE = ".agents/skills/pypto-log-check/references/log-rules.md"

# 只有这些目录下的源码参与日志规范检查，其余内容一律不扫。
SCAN_DIRS = (
    "framework/include",
    "framework/src",
    "python/pypto",
    "python/pypto_pro",
    "python/src",
)

BEGIN_MARKER = "LOG_CHECK_JSON_BEGIN"
END_MARKER = "LOG_CHECK_JSON_END"

# Probed in this order; cannbot is the CANN-specific agent.  The
# skip-permissions flags let the agent read the reviewed files without an
# interactive prompt; the prompt itself forbids any write.
AGENT_COMMANDS = {
    "cannbot": lambda prompt: ["cannbot", "run", "--dangerously-skip-permissions", prompt],
    "cursor-agent": lambda prompt: ["cursor-agent", "-p", prompt, "--output-format", "text"],
    "claude": lambda prompt: ["claude", "-p", prompt, "--permission-mode", "plan"],
    "codex": lambda prompt: ["codex", "exec", "--sandbox", "read-only", prompt],
    "opencode": lambda prompt: ["opencode", "run", "--dangerously-skip-permissions", prompt],
}
AGENT_ORDER = ["cannbot", "cursor-agent", "claude", "codex", "opencode"]

# Severity levels come from references/log-rules.md; 致命/严重 block the commit.
BLOCKING_SEVERITIES = ("致命", "严重")
SEVERITY_ALIASES = {
    "fatal": "致命",
    "critical": "致命",
    "blocker": "致命",
    "major": "严重",
    "serious": "严重",
    "medium": "中等",
    "moderate": "中等",
    "minor": "提示",
    "info": "提示",
    "hint": "提示",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

DEFAULT_TIMEOUT = 300


def log(message):
    print(f"[log-spec] {message}", file=sys.stderr)


def select_agent():
    """Return the agent name to use, or None (with a reason logged) when
    no agent CLI is available."""
    requested = os.environ.get("PYPTO_LOG_CHECK_AGENT", "").strip()
    if requested:
        if requested not in AGENT_COMMANDS:
            log(f"PYPTO_LOG_CHECK_AGENT={requested} 不是支持的 agent，"
                f"可选：{', '.join(AGENT_ORDER)}；跳过日志规范检查。")
        elif shutil.which(requested) is None:
            log(f"PYPTO_LOG_CHECK_AGENT={requested} 未安装，跳过日志规范检查。")
        else:
            return requested
        return None
    for name in AGENT_ORDER:
        if shutil.which(name) is not None:
            return name
    log(f"未检测到本地 agent CLI（{', '.join(AGENT_ORDER)}），跳过日志规范检查。")
    return None


def in_scan_scope(path):
    """Return True when *path* lives under one of the scanned directories."""
    normalized = path.replace(os.sep, "/")
    if os.path.isabs(normalized):
        normalized = os.path.relpath(normalized).replace(os.sep, "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return any(
        normalized == directory or normalized.startswith(directory + "/")
        for directory in SCAN_DIRS
    )


def added_lines(path):
    """Return the line numbers added to *path* by the staged diff."""
    result = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = set()
    for line in result.stdout.splitlines():
        match = HUNK_RE.match(line)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            lines.update(range(start, start + count))
    return lines


FINDING_SCHEMA = (
    '{"findings": [{"file": "相对仓库根目录的路径", "line": 行号, '
    '"severity": "致命|严重|中等|提示", "rule": "规则表中的规则名", '
    '"message": "问题描述", "suggestion": "修复建议"}]}'
)


def build_prompt(files):
    file_list = "\n".join(f"- {path}" for path in files)
    return f"""你是 CANN 日志规范评测器，正在 git pre-commit 钩子中运行，工作目录是仓库根目录。

执行步骤：
1. 读取 `{SKILL_FILE}` 和唯一规则来源 `{RULES_FILE}`，规则、严重级和判定尺度全部以 log-rules.md 为准，
不要凭记忆补充或删减规则。
2. 只评测本次提交涉及的下列文件：
{file_list}
   读取每个文件的完整内容作为上下文，识别其中全部运行时日志/打印语句，再按规则表逐条审查。
3. 只发现问题，不要修改任何文件，不要执行仓库中的脚本或命令，不要评测清单之外的文件。
4. 遵守 log-rules.md 的"全局判定尺度"，宁缺毋滥，不报风格类建议。

最后只输出一段 JSON，包裹在下面两行标记之间，标记之外不要有其它内容：
{BEGIN_MARKER}
{FINDING_SCHEMA}
{END_MARKER}

没有问题时输出 {{"findings": []}}。
"""


def run_agent(agent, prompt, timeout):
    """Return the agent stdout, or None when the call failed."""
    argv = AGENT_COMMANDS[agent](prompt)
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log(f"{agent} 超过 {timeout}s 未返回，跳过日志规范检查。")
        return None
    except OSError as exc:
        log(f"{agent} 调用失败（{exc}），跳过日志规范检查。")
        return None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "无输出"
        log(f"{agent} 返回码 {result.returncode}（{tail}），跳过日志规范检查。")
        return None
    return result.stdout


def parse_findings(output):
    """Return the findings list from the agent output, or None if unparsable."""
    text = ANSI_RE.sub("", output or "")
    start = text.rfind(BEGIN_MARKER)
    end = text.rfind(END_MARKER)
    if start != -1 and end > start:
        payload = text[start + len(BEGIN_MARKER):end]
    else:
        brace = text.find('{"findings"')
        if brace == -1:
            brace = text.find('{ "findings"')
        if brace == -1:
            return None
        payload = text[brace:text.rfind("}") + 1]
    try:
        data = json.loads(payload.strip())
    except json.JSONDecodeError:
        return None
    findings = data.get("findings") if isinstance(data, dict) else None
    return findings if isinstance(findings, list) else None


def normalize(finding, allowed_lines):
    """Return a printable finding, or None when it is out of scope."""
    if not isinstance(finding, dict):
        return None
    path = str(finding.get("file", "")).strip()
    if path not in allowed_lines:
        return None
    try:
        line = int(finding.get("line"))
    except (TypeError, ValueError):
        return None
    if line not in allowed_lines[path]:
        return None
    severity = str(finding.get("severity", "")).strip()
    severity = SEVERITY_ALIASES.get(severity.lower(), severity)
    return {
        "file": path,
        "line": line,
        "severity": severity,
        "rule": str(finding.get("rule", "")).strip(),
        "message": str(finding.get("message", "")).strip(),
        "suggestion": str(finding.get("suggestion", "")).strip(),
    }


def report(findings):
    """Print the findings and return the number of blocking ones."""
    blocking = 0
    for item in sorted(findings, key=lambda f: (f["file"], f["line"])):
        log(f"{item['severity']} {item['file']}:{item['line']} "
            f"[{item['rule']}] {item['message']}")
        if item["suggestion"]:
            log(f"         建议：{item['suggestion']}")
        if item["severity"] in BLOCKING_SEVERITIES:
            blocking += 1
    return blocking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if os.environ.get("PYPTO_SKIP_LOG_CHECK"):
        log("PYPTO_SKIP_LOG_CHECK 已设置，跳过日志规范检查。")
        return 0
    if not args.files:
        return 0
    files = [path for path in args.files if in_scan_scope(path)]
    if not files:
        return 0
    if not os.path.exists(RULES_FILE):
        log(f"未找到规则文件 {RULES_FILE}，跳过日志规范检查。")
        return 0

    agent = select_agent()
    if agent is None:
        return 0

    allowed_lines = {path: added_lines(path) for path in files}
    allowed_lines = {path: lines for path, lines in allowed_lines.items() if lines}
    if not allowed_lines:
        return 0

    files = sorted(allowed_lines)
    timeout = int(os.environ.get("PYPTO_LOG_CHECK_TIMEOUT", DEFAULT_TIMEOUT))
    log(f"使用 {agent} 评测 {len(files)} 个文件的日志规范，最长等待 {timeout}s ...")

    output = run_agent(agent, build_prompt(files), timeout)
    if output is None:
        return 0

    findings = parse_findings(output)
    if findings is None:
        log(f"{agent} 输出无法解析为评测结果，跳过日志规范检查。")
        return 0

    in_scope = [f for f in (normalize(f, allowed_lines) for f in findings) if f]
    if not in_scope:
        log("本次改动的日志语句未发现规范问题。")
        return 0

    blocking = report(in_scope)
    if blocking == 0:
        log(f"发现 {len(in_scope)} 个非阻断级问题（中等/提示），不影响提交。")
        return 0
    log(f"发现 {blocking} 个阻断级问题（{'/'.join(BLOCKING_SEVERITIES)}），提交已拦截。")
    log("确认为误报时可用 SKIP=log-spec-check git commit ... 跳过本检查。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
