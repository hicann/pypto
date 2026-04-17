#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from typing import Any, Optional

from .core import SCRIPT_DIR, CheckContext, Finding

NPU_SMI_BIN = shutil.which("npu-smi") or "npu-smi"


def _check_npu_available() -> bool:
    """通过 npu-smi info 检测 NPU 环境是否可用"""
    try:
        result = subprocess.run(
            [NPU_SMI_BIN, "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


DTYPE_ALIASES: dict[str, str] = {
    "float32": "fp32", "fp32": "fp32", "dt_fp32": "fp32", "torch.float32": "fp32",
    "float16": "fp16", "fp16": "fp16", "dt_fp16": "fp16", "torch.float16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16", "dt_bf16": "bf16", "torch.bfloat16": "bf16",
}


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in ("", "null", "None"):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            items = [x.strip() for x in inner.split(",")]
            parsed: list[Any] = []
            for item in items:
                if item.startswith("[") and item.endswith("]"):
                    parsed.append(_parse_scalar(item))
                elif item.startswith(("'", '"')) and item.endswith(("'", '"')):
                    parsed.append(item[1:-1])
                elif re.fullmatch(r"[+-]?\d+", item):
                    parsed.append(int(item))
                elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", item):
                    parsed.append(float(item))
                else:
                    parsed.append(item)
            return parsed
    if value.startswith("{") and value.endswith("}"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        # 回退：解析 YAML 风格的无引号 key 字典，如 {key1: val1, key2: val2}
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for pair in inner.split(","):
            pair = pair.strip()
            if ":" not in pair:
                return value  # 无法解析，返回原始字符串
            k, v = pair.split(":", 1)
            k = k.strip().strip("'\"")
            result[k] = _parse_scalar(v)
        return result
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        return float(value)
    # Bare comma-separated values (no brackets) => parse as list
    if "," in value and not value.startswith(("[", "{")):
        return [_parse_scalar(v.strip()) for v in value.split(",") if v.strip()]
    return value


def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """解析 markdown front matter。

    优先使用 yaml.safe_load 解析（支持多行 YAML），若 YAML 解析失败或
    结果非 dict 则回退到逐行解析。对 YAML 无法识别的 Python 字面量值
    （如 ``{'key': 'val'}``）做后处理兼容。
    """
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end < 0:
        return {}, content

    header = content[4:end]
    body = content[end + 5:]

    # 优先尝试 yaml.safe_load
    try:
        import yaml  # noqa: PLC0415
        parsed = yaml.safe_load(header)
        if isinstance(parsed, dict):
            # 后处理：yaml 会把 Python dict 字面量（如 {'k': v}）解析为字符串，
            # 用 _parse_scalar 重新解析这些字符串值
            for key, val in parsed.items():
                if isinstance(val, str) and val.strip().startswith(("{", "[")):
                    parsed[key] = _parse_scalar(val)
            return parsed, body
    except Exception:
        pass

    # 回退：逐行解析（兼容旧行为）
    meta: dict[str, Any] = {}
    for raw in header.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _parse_scalar(value)
    return meta, body


def _load_doc_meta(ctx: CheckContext, filename: str) -> dict[str, Any]:
    content = ctx.read_file(filename)
    if not content:
        return {}
    meta, _ = _parse_front_matter(content)
    return meta


_tolerance_schema_cache: list[dict[str, Any]] | None = None


def _load_tolerance_schema() -> list[dict[str, Any]]:
    """从 rules.json 加载 tolerance_schema.oneOf 定义（带模块级缓存）。"""
    global _tolerance_schema_cache
    if _tolerance_schema_cache is not None:
        return _tolerance_schema_cache
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        schema: list[dict[str, Any]] = []
    else:
        schema = data.get("tolerance_schema", {}).get("oneOf", [])
    _tolerance_schema_cache = schema
    return schema


def _validate_tolerance(tol: dict[str, Any]) -> list[str]:
    """基于 rules.json 中 tolerance_schema.oneOf 校验 tolerance dict。

    只要匹配任一模式即通过。
    """
    schemas = _load_tolerance_schema()
    if not schemas:
        # 无 schema 定义时回退到基础校验：只检查是否为非空 dict
        return []

    for schema in schemas:
        required = schema.get("required", [])
        if all(k in tol for k in required):
            return []  # 匹配成功

    # 所有模式都不匹配，列出可接受的模式
    modes = [f"{s.get('mode', '?')}({', '.join(s.get('required', []))})" for s in schemas]
    return [f"tolerance must match one of: {' | '.join(modes)}"]


def _validate_doc_schema(doc_type: str, meta: dict[str, Any]) -> list[str]:
    required: dict[str, list[str]] = {
        "SPEC": ["schema_version", "op_name", "supported_dtypes", "p0_shapes", "tolerance"],
        "DESIGN": ["schema_version", "op_name", "dynamic_axes"],
        "API_REPORT": ["schema_version", "op_name"],
    }
    errors: list[str] = []
    for key in required.get(doc_type, []):
        if key not in meta or meta[key] in (None, "", []):
            errors.append(f"missing field: {key}")
    if doc_type == "SPEC":
        if "supported_dtypes" in meta and not isinstance(meta.get("supported_dtypes"), list):
            errors.append("invalid type: supported_dtypes must be list")
        if "p0_shapes" in meta and not isinstance(meta.get("p0_shapes"), list):
            errors.append("invalid type: p0_shapes must be list")
        if "tolerance" in meta and not isinstance(meta.get("tolerance"), dict):
            errors.append("invalid type: tolerance must be dict")
        if isinstance(meta.get("tolerance"), dict):
            errors.extend(_validate_tolerance(meta["tolerance"]))
    if doc_type == "DESIGN":
        if "dynamic_axes" in meta and not isinstance(meta.get("dynamic_axes"), list):
            errors.append("invalid type: dynamic_axes must be list")
    return errors


def _extract_spec_dtypes_from_meta(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("supported_dtypes", [])
    if not isinstance(raw, list):
        return set()
    result: set[str] = set()
    for item in raw:
        key = str(item).strip().lower()
        canonical = DTYPE_ALIASES.get(key)
        if canonical:
            result.add(canonical)
    return result


def _extract_test_dtypes(source: str) -> set[str]:
    """从 test 文件源码中提取使用的 dtype"""
    dtypes: set[str] = set()
    lower = source.lower()
    for alias, canonical in DTYPE_ALIASES.items():
        if alias in lower:
            dtypes.add(canonical)
    return dtypes


def _extract_tolerance(content: str, key: str) -> list[float]:
    """从文本中提取所有 atol 或 rtol 数值"""
    values: list[float] = []
    for pat in [
        rf"{key}\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
        rf"\*\*{key}\*\*\s*[:=]?\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
    ]:
        for m in re.finditer(pat, content, re.IGNORECASE):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                continue
    return values


def _extract_shapes_from_text(content: str) -> set[tuple[int, ...]]:
    """从文本中提取 shape 元组，如 [1, 2048, 4096] 或 (1, 2048, 4096)"""
    shapes: set[tuple[int, ...]] = set()
    for m in re.finditer(r'[\[\(](\d+(?:\s*,\s*\d+)+)[\]\)]', content):
        try:
            dims = tuple(int(x.strip()) for x in m.group(1).split(","))
            if len(dims) >= 2:
                shapes.add(dims)
        except ValueError:
            continue
    return shapes


def _extract_markdown_headings(content: str) -> set[str]:
    _, body = _parse_front_matter(content)
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return {h.strip().lower() for h in headings}


def _has_heading_like(headings: set[str], *keywords: str) -> bool:
    keys = [k.strip().lower() for k in keywords if k.strip()]
    for h in headings:
        if any(k in h for k in keys):
            return True
    return False


def _extract_section_text(content: str, heading_keyword: str) -> str:
    _, body = _parse_front_matter(content)
    lines = body.splitlines()
    key = heading_keyword.strip().lower()
    start = -1
    start_level = 0
    for idx, line in enumerate(lines):
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip().lower()
        if key in title:
            start = idx + 1
            start_level = level
            break
    if start < 0:
        return ""

    buff: list[str] = []
    for line in lines[start:]:
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if m and len(m.group(1)) <= start_level:
            break
        buff.append(line)
    return "\n".join(buff).strip()


def _syntax_error_finding(ctx: CheckContext, rule_id: str, filename: str) -> Optional[Finding]:
    tree = ctx.parse_file(filename)
    if tree is not None:
        return None
    error = ctx.parse_error(filename)
    if not error:
        return None
    return ctx.make_finding(
        rule_id,
        "FAIL",
        f"{filename} 存在语法错误，无法解析: {error}",
        file=filename,
    )


def _extract_design_identifiers(content: str) -> set[str]:
    # 仅从“代码相关上下文”提取变量名，减少自然语言文本噪声：
    # 1) markdown fenced code block
    # 2) inline code (`...`)
    code_blocks = re.findall(r"```[\w-]*\n(.*?)```", content, re.S)
    inline_codes = re.findall(r"`([^`\n]+)`", content)
    scoped_text = "\n".join(code_blocks + inline_codes)

    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", scoped_text))
    blacklist = {
        "input", "output", "tensor", "shape", "dtype", "dynamic", "algorithm",
        "default", "stage", "spec", "design", "report", "loop", "tiling",
        "step", "max", "min", "round", "clamp", "cast", "mul", "add", "sub", "div",
    }
    # 只保留更像“中间变量”的命名（snake_case 优先），减少误报
    return {t for t in tokens if t not in blacklist and ("_" in t or len(t) >= 6)}
