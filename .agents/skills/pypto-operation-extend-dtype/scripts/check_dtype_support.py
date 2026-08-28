#!/usr/bin/env python3
"""Check if a dtype is already supported in pypto operation source code.

Usage:
    python3 check_dtype_support.py \
        --pypto-root /mnt/workspace/gitCode/cann/pypto \
        --operation add \
        --dtypes int64,uint64

Output: JSON with support status per dtype per architecture.
"""

import argparse
import json
from pathlib import Path
import re
import sys

DT_TO_CPP = {
    "int4": "DT_INT4", "int8": "DT_INT8", "int16": "DT_INT16",
    "int32": "DT_INT32", "int64": "DT_INT64", "uint8": "DT_UINT8",
    "uint16": "DT_UINT16", "uint32": "DT_UINT32", "uint64": "DT_UINT64",
    "fp8": "DT_FP8", "fp16": "DT_FP16", "fp32": "DT_FP32",
    "bf16": "DT_BF16", "bool": "DT_BOOL", "double": "DT_DOUBLE",
    "fp8e4m3": "DT_FP8E4M3", "fp8e5m2": "DT_FP8E5M2", "fp8e8m0": "DT_FP8E8M0",
    "hf4": "DT_HF4", "hf8": "DT_HF8",
}


OP_TO_PREFIX = {
    "add": ["ADD"], "sub": ["SUB"], "mul": ["MUL"], "div": ["DIV"],
    "max": ["MAX"], "min": ["MIN"],
    "compare": ["CMP"], "eq": ["CMP"], "ne": ["CMP"], "gt": ["CMP"],
    "ge": ["CMP"], "lt": ["CMP"], "le": ["CMP"],
    "where": ["WHERE"],
    "gather": ["GATHER"], "scatter": ["SCATTER"],
    "concat": ["CONCAT"], "transpose": ["TRANSPOSE"],
    "cast": ["CAST"], "relu": ["RELU"],
    "abs": ["ABS"], "exp": ["EXP"], "log": ["LOG"],
    "sqrt": ["SQRT"], "rsqrt": ["RSQRT"],
    "bitwise_and": ["BITWISE", "AND"], "bitwise_or": ["BITWISE", "OR"],
    "bitwise_xor": ["BITWISE", "XOR"], "bitwise_not": ["BITWISE", "NOT"],
    "bitwise_left_shift": ["BITWISESHIFT", "SHL"],
    "bitwise_right_shift": ["BITWISESHIFT", "SHR"],
}


def get_prefixes(operation: str) -> list:
    """Get possible dtype set prefixes for an operation."""
    op_upper = operation.upper()
    prefixes = OP_TO_PREFIX.get(operation.lower(), [])
    if op_upper not in prefixes:
        prefixes = [op_upper] + prefixes
    return prefixes


OP_TO_CHECK_KEYWORDS = {
    "concat": ["CheckCat", '"CAT"', "InnerConcat"],
    "transpose": ["CheckTranspose", '"TRANSPOSE"'],
    "cast": ["CheckCast", '"CAST"'],
    "expand": ["CheckExpand", '"EXPAND"'],
    "gather": ["CheckGather", '"GATHER"'],
    "scatter": ["CheckScatter", '"SCATTER"'],
    "where": ["CheckWhere", '"WHERE"'],
}


def find_source_files(pypto_root: Path, operation: str) -> list:
    """Find .cpp files that contain the operation's dtype definitions.

    Strategy:
    1. Search for {PREFIX}_A2A3_TYPES or {PREFIX}_A5_TYPES for all possible prefixes.
    2. If not found, search for the operation function name or check function name.
    """
    vector_dir = pypto_root / "framework" / "src" / "interface" / "operation" / "vector"
    if not vector_dir.exists():
        return []

    op_pascal = operation.capitalize()
    op_upper = operation.upper()
    prefixes = get_prefixes(operation)
    check_keywords = OP_TO_CHECK_KEYWORDS.get(operation.lower(), [])
    candidates = []

    for cpp_file in vector_dir.glob("*.cpp"):
        try:
            content = cpp_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        found = False
        for prefix in prefixes:
            if re.search(rf'\b{prefix}_A2A3_TYPES\b', content) or \
               re.search(rf'\b{prefix}_A5_TYPES\b', content):
                candidates.append(str(cpp_file.relative_to(pypto_root)))
                found = True
                break
        if found:
            continue

        for kw in check_keywords:
            if kw in content:
                candidates.append(str(cpp_file.relative_to(pypto_root)))
                found = True
                break
        if found:
            continue

    if not candidates:
        for cpp_file in vector_dir.glob("*.cpp"):
            try:
                content = cpp_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            patterns = [
                rf'\bvoid\s+{op_pascal}\w*\s*\(',
                rf'\bLogicalTensorPtr\s+{op_pascal}\w*\s*\(',
                rf'GetBinaryOpName.*{op_upper}',
                rf'"{op_pascal}"',
                rf'"{op_upper}"',
            ]
            for pat in patterns:
                if re.search(pat, content):
                    candidates.append(str(cpp_file.relative_to(pypto_root)))
                    break

    return candidates


def extract_dtype_sets(file_path: Path, operation: str) -> dict:
    """Extract dtype sets from a source file for the given operation.

    Strategy:
    1. Search for {PREFIX}_A2A3_TYPES and {PREFIX}_A5_TYPES for all possible prefixes.
    2. If not found, search for local a2a3Types/a5Types variables.
    3. If not found, search for supportedTypes within the operation function scope.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    prefixes = get_prefixes(operation)
    op_pascal = operation.capitalize()
    op_upper = operation.upper()
    result = {"a2a3": set(), "a5": set(), "raw_matches": []}

    for prefix in prefixes:
        patterns = [
            (rf'{prefix}_A2A3_TYPES\s*=\s*\{{([^}}]+)\}}', "a2a3"),
            (rf'{prefix}_A5_TYPES\s*=\s*\{{([^}}]+)\}}', "a5"),
        ]

        for pat, arch in patterns:
            for match in re.finditer(pat, content):
                types_str = match.group(1)
                dtypes = re.findall(r'DT_\w+', types_str)
                result[arch].update(dtypes)
                result["raw_matches"].append({
                    "arch": arch,
                    "line": content[:match.start()].count('\n') + 1,
                    "types": dtypes,
                    "prefix": prefix,
                })

    if result["a2a3"] or result["a5"]:
        return result

    local_patterns = [
        (r'(?:a2a3Types|a2a3InputTypes)\s*=\s*\{([^}]+)\}', "a2a3"),
        (r'(?:a5Types|a5InputTypes)\s*=\s*\{([^}]+)\}', "a5"),
    ]
    for pat, arch in local_patterns:
        for match in re.finditer(pat, content):
            types_str = match.group(1)
            dtypes = re.findall(r'DT_\w+', types_str)
            result[arch].update(dtypes)
            result["raw_matches"].append({
                "arch": f"{arch} (local variable)",
                "line": content[:match.start()].count('\n') + 1,
                "types": dtypes,
            })

    if result["a2a3"] or result["a5"]:
        return result

    op_pascal = operation.capitalize()
    check_keywords = OP_TO_CHECK_KEYWORDS.get(operation.lower(), [])
    search_keywords = [op_pascal, op_upper] + check_keywords

    search_regions = []
    for kw in search_keywords:
        kw_escaped = re.escape(kw)
        func_pat = rf'(?:void|LogicalTensorPtr)\s+\w*{kw_escaped}\w*\s*\([^)]*\)\s*\{{'
        for func_match in re.finditer(func_pat, content):
            brace_start = func_match.end() - 1
            depth = 0
            end_pos = brace_start
            for i in range(brace_start, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break
            search_regions.append((func_match.start(), end_pos, content[brace_start + 1:end_pos]))

    for region_start, region_end, func_body in search_regions:
        supported_pat = r'supportedTypes\s*=\s*\{([^}]+)\}'
        for match in re.finditer(supported_pat, func_body):
            types_str = match.group(1)
            dtypes = re.findall(r'DT_\w+', types_str)
            result["a2a3"].update(dtypes)
            result["a5"].update(dtypes)
            line_offset = content[:region_start].count('\n') + \
                          func_body[:match.start()].count('\n') + 1
            result["raw_matches"].append({
                "arch": "both (supportedTypes — no arch distinction)",
                "line": line_offset,
                "types": dtypes,
            })

    if result["a2a3"] or result["a5"]:
        return result

    op_string_names = {
        "concat": '"CAT"', "transpose": '"TRANSPOSE"', "cast": '"CAST"',
        "expand": '"EXPAND"', "gather": '"GATHER"', "scatter": '"SCATTER"',
        "where": '"WHERE"', "full": '"FULL"',
    }
    op_str = op_string_names.get(operation.lower())
    if op_str:
        str_escaped = re.escape(op_str)
        str_pat = rf'{str_escaped}'
        for str_match in re.finditer(str_pat, content):
            region_start = max(0, str_match.start() - 500)
            region_end = min(len(content), str_match.end() + 500)
            region = content[region_start:region_end]
            supported_pat = r'supportedTypes\s*=\s*\{([^}]+)\}'
            for match in re.finditer(supported_pat, region):
                types_str = match.group(1)
                dtypes = re.findall(r'DT_\w+', types_str)
                result["a2a3"].update(dtypes)
                result["a5"].update(dtypes)
                line_offset = content[:region_start].count('\n') + \
                              region[:match.start()].count('\n') + 1
                result["raw_matches"].append({
                    "arch": "both (supportedTypes near operation string)",
                    "line": line_offset,
                    "types": dtypes,
                })

    return result


def check_dtype(pypto_root: Path, operation: str, dtypes: list) -> dict:
    """Main check function."""
    source_files = find_source_files(pypto_root, operation)

    all_sets = {"a2a3": set(), "a5": set(), "raw_matches": []}
    for rel_path in source_files:
        full_path = pypto_root / rel_path
        sets = extract_dtype_sets(full_path, operation)
        all_sets["a2a3"].update(sets["a2a3"])
        all_sets["a5"].update(sets["a5"])
        all_sets["raw_matches"].extend(sets["raw_matches"])

    dtype_results = {}
    for dt in dtypes:
        dt_norm = dt.lower().strip()
        dt_enum = DT_TO_CPP.get(dt_norm, dt_norm.upper() if not dt_norm.startswith("DT_") else dt_norm)

        in_a2a3 = dt_enum in all_sets["a2a3"]
        in_a5 = dt_enum in all_sets["a5"]

        if in_a2a3 and in_a5:
            status = "already_supported"
        elif in_a2a3 or in_a5:
            status = "partially_supported"
        else:
            status = "not_supported"

        dtype_results[dt] = {
            "dt_enum": dt_enum,
            "status": status,
            "in_a2a3": in_a2a3,
            "in_a5": in_a5,
        }

    return {
        "operation": operation,
        "source_files": source_files,
        "current_a2a3_types": sorted(list(all_sets["a2a3"])),
        "current_a5_types": sorted(list(all_sets["a5"])),
        "raw_matches": all_sets["raw_matches"],
        "dtype_results": dtype_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check dtype support in pypto operation source")
    parser.add_argument("--pypto-root", required=True, help="Path to pypto root directory")
    parser.add_argument("--operation", required=True, help="Operation name (e.g., add, sub, compare)")
    parser.add_argument("--dtypes", required=True, help="Comma-separated dtype names (e.g., int64,uint64)")
    args = parser.parse_args()

    pypto_root = Path(args.pypto_root)
    if not pypto_root.exists():
        print(json.dumps({"error": f"pypto root not found: {pypto_root}"}), file=sys.stderr)
        sys.exit(1)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    result = check_dtype(pypto_root, args.operation, dtypes)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
