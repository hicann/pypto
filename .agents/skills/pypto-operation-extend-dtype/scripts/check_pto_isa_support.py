#!/usr/bin/env python3
"""Check if a dtype is supported by pto-isa for a given operation.

Searches pto-isa NPU header files for static_assert type lists and
if constexpr branches that indicate dtype support.

Usage:
    python3 check_pto_isa_support.py \
        --pto-isa-root /mnt/workspace/gitCode/cann/pto-isa \
        --operation add \
        --dtypes int64,uint64

Output: JSON with pto-isa support status per dtype per architecture.
"""

import argparse
import json
from pathlib import Path
import re
import sys

DTYPE_TO_CPP_TYPE = {
    "int4": "int4", "int8": "int8_t", "int16": "int16_t",
    "int32": "int32_t", "int64": "int64_t", "uint8": "uint8_t",
    "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
    "fp8": "float8_t", "fp16": "half", "fp32": "float",
    "bf16": "bfloat16_t", "bool": "bool", "double": "double",
    "fp8e4m3": "float8_e4m3_t", "fp8e5m2": "float8_e5m2_t",
    "fp8e8m0": "float8_e8m0_t", "hf4": "hfloat4", "hf8": "hifloat8_t",
}


OP_TO_HEADER_PATTERNS = {
    "add": ["TAdd.hpp", "TAdds.hpp"],
    "sub": ["TSub.hpp", "TSubS.hpp"],
    "mul": ["TMul.hpp", "TMulS.hpp"],
    "div": ["TDiv.hpp", "TDivs.hpp"],
    "max": ["TMax.hpp", "TMins.hpp", "TColMax.hpp"],
    "min": ["TMin.hpp", "TMins.hpp", "TPartMin.hpp"],
    "abs": ["TAbs.hpp"],
    "exp": ["TExp.hpp"],
    "log": ["TLog.hpp"],
    "sqrt": ["TSqrt.hpp"],
    "rsqrt": ["TRsqrt.hpp"],
    "compare": ["TCmp.h", "TCompare.hpp", "TCmp.hpp"],
    "eq": ["TCmp.h", "TCompare.hpp"],
    "ne": ["TCmp.h", "TCompare.hpp"],
    "gt": ["TCmp.h", "TCompare.hpp"],
    "ge": ["TCmp.h", "TCompare.hpp"],
    "lt": ["TCmp.h", "TCompare.hpp"],
    "le": ["TCmp.h", "TCompare.hpp"],
    "where": ["TSelect.hpp", "TWhere.hpp"],
    "gather": ["TGather.hpp"],
    "scatter": ["TScatter.hpp", "MScatter.hpp"],
    "concat": ["TConcat.hpp"],
    "transpose": ["TTranspose.hpp", "TPermute.hpp"],
    "cast": ["TCvt.hpp"],
    "bitwise_and": ["TAnd.hpp"],
    "bitwise_or": ["TOr.hpp"],
    "bitwise_xor": ["TXor.hpp"],
    "bitwise_not": ["TNot.hpp"],
    "bitwise_left_shift": ["TShl.hpp", "TBitwiseSOp.hpp"],
    "bitwise_right_shift": ["TShr.hpp", "TBitwiseSOp.hpp"],
    "relu": ["TRelu.hpp"],
    "ceil": ["TCeil.hpp"],
    "floor": ["TFloor.hpp"],
    "neg": ["TNeg.hpp"],
    "sigmoid": ["TSigmoid.hpp"],
    "tanh": ["TTanh.hpp"],
}


def find_pto_isa_headers(pto_isa_root: Path, operation: str) -> dict:
    """Find pto-isa header files for the operation in each architecture directory."""
    op_lower = operation.lower()
    npu_include = pto_isa_root / "include" / "pto" / "npu"

    arch_dirs = {
        "a2a3": npu_include / "a2a3",
        "a5": npu_include / "a5",
        "a6": npu_include / "a6",
    }

    candidate_names = OP_TO_HEADER_PATTERNS.get(op_lower, [])
    if not candidate_names:
        pascal = op_lower.title().replace("_", "")
        candidate_names = [f"T{pascal}.hpp", f"T{pascal}.h"]

    result = {}
    for arch, arch_dir in arch_dirs.items():
        if not arch_dir.exists():
            continue

        found_files = []
        for name in candidate_names:
            header_path = arch_dir / name
            if header_path.exists():
                found_files.append(str(header_path.relative_to(pto_isa_root)))

        if not found_files:
            for hpp_file in arch_dir.glob("T*.hpp"):
                try:
                    content = hpp_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                op_pascal = op_lower.title().replace("_", "")
                patterns = [
                    rf'\bOP_NAME\(T{op_pascal.upper()}\b',
                    rf'\b{op_pascal}Op\b',
                    rf'//.*\b{op_lower}\b.*operation',
                    rf'__tf__.*void\s+T{op_pascal}\b',
                ]
                for pat in patterns:
                    if re.search(pat, content, re.IGNORECASE):
                        found_files.append(str(hpp_file.relative_to(pto_isa_root)))
                        break

        result[arch] = found_files

    return result


DTYPE_TO_SIZE = {
    "int4": 1, "int8": 1, "uint8": 1, "bool": 1, "fp8": 1,
    "fp8e4m3": 1, "fp8e5m2": 1, "fp8e8m0": 1, "hf4": 1, "hf8": 1,
    "int16": 2, "uint16": 2, "fp16": 2, "bf16": 2,
    "int32": 4, "uint32": 4, "fp32": 4,
    "int64": 8, "uint64": 8, "double": 8,
}


def extract_supported_types(header_path: Path) -> dict:
    """Extract supported C++ types from a pto-isa header file's static_assert."""
    try:
        content = header_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"supported_types": [], "has_int64_branch": False, "static_assert_lines": [], "sizeof_checks": []}

    supported_types = set()
    static_assert_lines = []

    assert_blocks = re.findall(
        r'static_assert\s*\(\s*([^;]+?)\s*,\s*"[^"]*invalid[^"]*"\s*\)',
        content, re.DOTALL
    )

    for block in assert_blocks:
        type_matches = re.findall(r'std::is_same(?:_v)?<\s*T\s*,\s*(\w+)\s*>', block)
        if type_matches:
            supported_types.update(type_matches)
            static_assert_lines.append(block.strip()[:200])

    if not supported_types:
        type_matches = re.findall(r'std::is_same(?:_v)?<\s*T\s*,\s*(\w+)\s*>', content)
        supported_types.update(type_matches)

    has_int64_branch = bool(re.search(
        r'if\s+constexpr\s*\(\s*std::is_same_v<T,\s*int64_t>\s*\|\|\s*std::is_same_v<T,\s*uint64_t>',
        content
    ))

    has_int64_include = bool(re.search(r'#include.*Int64', content))

    sizeof_checks = []
    sizeof_matches = re.findall(
        r'static_assert\s*\(\s*([^;]+?sizeof\s*\(T\)[^;]+?)\s*,',
        content, re.DOTALL
    )
    for block in sizeof_matches:
        sizes = re.findall(r'sizeof\s*\(T\)\s*==\s*(\d+)', block)
        if sizes:
            sizeof_checks.append({
                "sizes": [int(s) for s in sizes],
                "block": block.strip()[:200],
            })

    return {
        "supported_types": sorted(list(supported_types)),
        "has_int64_branch": has_int64_branch,
        "has_int64_include": has_int64_include,
        "static_assert_lines": static_assert_lines,
        "sizeof_checks": sizeof_checks,
    }


def check_pto_isa(pto_isa_root: Path, operation: str, dtypes: list) -> dict:
    """Main check function."""
    headers_by_arch = find_pto_isa_headers(pto_isa_root, operation)

    arch_results = {}
    for arch, header_files in headers_by_arch.items():
        if not header_files:
            arch_results[arch] = {
                "found": False,
                "headers": [],
                "supported_types": [],
                "dtype_support": {},
            }
            continue

        all_types = set()
        has_int64_branch = False
        all_sizeof_checks = []
        for hf in header_files:
            full_path = pto_isa_root / hf
            info = extract_supported_types(full_path)
            all_types.update(info["supported_types"])
            if info["has_int64_branch"]:
                has_int64_branch = True
            all_sizeof_checks.extend(info.get("sizeof_checks", []))

        all_supported_sizes = set()
        for check in all_sizeof_checks:
            all_supported_sizes.update(check["sizes"])

        dtype_support = {}
        for dt in dtypes:
            dt_lower = dt.lower().strip()
            cpp_type = DTYPE_TO_CPP_TYPE.get(dt_lower, dt_lower)
            is_supported = cpp_type in all_types

            dt_size = DTYPE_TO_SIZE.get(dt_lower)
            if not is_supported and dt_size and dt_size in all_supported_sizes:
                is_supported = True

            dtype_support[dt] = {
                "cpp_type": cpp_type,
                "supported": is_supported,
            }
            if dt_lower in ("int64", "uint64") and has_int64_branch:
                dtype_support[dt]["has_special_branch"] = True
            if dt_size and dt_size in all_supported_sizes:
                dtype_support[dt]["supported_via_sizeof"] = True

        arch_results[arch] = {
            "found": True,
            "headers": header_files,
            "supported_types": sorted(list(all_types)),
            "has_int64_branch": has_int64_branch,
            "sizeof_checks": all_sizeof_checks,
            "dtype_support": dtype_support,
        }

    overall_dtype_status = {}
    for dt in dtypes:
        a2a3_ok = arch_results.get("a2a3", {}).get("dtype_support", {}).get(dt, {}).get("supported", False)
        a5_ok = arch_results.get("a5", {}).get("dtype_support", {}).get(dt, {}).get("supported", False)
        if a2a3_ok and a5_ok:
            status = "supported_all_arch"
        elif a2a3_ok or a5_ok:
            status = "supported_partial_arch"
        else:
            status = "not_supported"
        overall_dtype_status[dt] = {
            "a2a3_supported": a2a3_ok,
            "a5_supported": a5_ok,
            "status": status,
        }

    return {
        "operation": operation,
        "headers_by_arch": headers_by_arch,
        "arch_details": arch_results,
        "dtype_status": overall_dtype_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Check pto-isa dtype support for an operation")
    parser.add_argument("--pto-isa-root", required=True, help="Path to pto-isa root directory")
    parser.add_argument("--operation", required=True, help="Operation name (e.g., add, sub, compare)")
    parser.add_argument("--dtypes", required=True, help="Comma-separated dtype names (e.g., int64,uint64)")
    args = parser.parse_args()

    pto_isa_root = Path(args.pto_isa_root)
    if not pto_isa_root.exists():
        print(json.dumps({"error": f"pto-isa root not found: {pto_isa_root}"}), file=sys.stderr)
        sys.exit(1)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    result = check_pto_isa(pto_isa_root, args.operation, dtypes)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    all_supported = all(
        s["status"] != "not_supported" for s in result["dtype_status"].values()
    )
    sys.exit(0 if all_supported else 1)


if __name__ == "__main__":
    main()
