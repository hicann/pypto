#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Automated Python Script Validator with Multi-Mode Execution Support

This script provides a robust framework for batch validation of Python test or
inference scripts under controlled execution environments. It supports both
standalone script execution (via `python`) and test discovery (via `pytest`),
with conditional logic based on script structure and execution mode.

Key Features:
- Input Flexibility: Accepts either a single `.py` file or a directory path.
- Execution Mode:
    - `npu` (default): Executes scripts as `python script.py [args]`.
    - `sim`: Executes only scripts that explicitly support `--run_mode sim`,
      skipping others with a diagnostic message.
- Smart Dispatch:
    - Scripts containing `if __name__ == "__main__":` are run via `python`.
    - Others are executed via `pytest` for test-case discovery.
- Test Selection: Optional allows targeting specific
  test cases (e.g., `module::function`), passed verbatim to the underlying runner.
- Environment Control: Propagates `DEVICE_ID` to child processes for hardware
  context awareness (e.g., GPU/NPU selection).
- Timeout Protection: Each script is subject to a configurable execution
  timeout (default: 300 seconds); exceeding it results in a failure.
- Result Validation:
    - Success if:
        a) Output contains *none* of: "error", "mismatch", "traceback" (case-insensitive), OR  
        b) Any line matches the pattern: `\bAll\b.*\b(pass|success)\b` (e.g., "All tests passed").
    - Failure otherwise, including timeout or execution exceptions.
- Self-Protection: Automatically excludes itself from execution to prevent recursion.

Exit Behavior:
- Returns 0 on completion (regardless of individual script outcomes).
- Exits early with code 1 if input target is invalid (non-existent, non-.py file, etc.).

Dependencies:
- `pytest` must be installed if any target script lacks a `__main__` guard.

Usage Examples:
    # 1、Run in batches for the specified directory
    python3 examples/validate_examples.py -t examples/02_intermediate --device-id 0

    # 2、Run specified script
    python3 examples/validate_examples.py -t examples/01_beginner/compute/elementwise_ops.py --device-id 0

    # 3、Run the specified case of the specified script
    python3 examples/validate_examples.py -t examples/01_beginner/compute/elementwise_ops.py add::test_add_basic --device-id 0

    # 4、Set the timeout duration for each use case execution (Unit: seconds)
    python3 examples/validate_examples.py -t examples/02_intermediate --device-id 0 --timeout 120

    # 5、Run all scripts in the specified directory in batches in simulation mode (NPU mode by default)
    python3 examples/validate_examples.py -t examples/02_intermediate --run-mode sim --device-id 0
"""

import os
import subprocess
import sys
import argparse
import re
from pathlib import Path


def _has_success_exemption(output: str) -> bool:
    pattern = re.compile(
        r'\bAll\b.*\b(?:pass(?:ed|ing)?|success(?:ful(?:ly)?)?)',
        re.IGNORECASE
    )
    return any(pattern.search(line) for line in output.splitlines())


def _has_main_guard(file_path: Path) -> bool:
    """Check if the file contains 'if __name__ == "__main__":'."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Use regex to match the exact pattern (allow whitespace variations)
        return bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', content))
    except Exception:
        return False


def _supports_run_mode_sim(file_path: Path) -> bool:
    """Check if the script's argparse includes 'run_mode' argument."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Heuristic: look for 'run_mode' in add_argument call
        # Matches: .add_argument(..., "--run_mode", ...) or ...dest="run_mode"...
        pattern = re.compile(r'add_argument\s*\(.*?["\']--run_mode["\']|dest\s*=\s*["\']run_mode["\']', re.DOTALL)
        return bool(pattern.search(content))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute and validate Python scripts or directories."
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="0",
        help="Value for the DEVICE_ID environment variable (default: '0')."
    )
    parser.add_argument(
        "--run-mode",
        choices=["npu", "sim"],
        default="npu",
        help="Execution mode: 'npu' (default) or 'sim'. In 'sim' mode, only scripts supporting --run_mode sim are executed."
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        required=True,
        help="Target: either a .py file path or a directory path."
    )
    parser.add_argument(
        "example_id",
        type=str,
        default=None,
        nargs='?',
        help="Optional test identifier (e.g., 'tensor_creation::test_tensor_creation') to pass to script."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-script execution timeout in seconds (default: 300)."
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["DEVICE_ID"] = args.device_id

    target = Path(args.target).resolve()
    self_path = Path(__file__).resolve()

    # Determine list of .py files to process
    if target.is_file():
        if target.suffix != ".py":
            print(f"Error: Target file '{target}' is not a .py file.", file=sys.stderr)
            sys.exit(1)
        if target.resolve() == self_path:
            print("Error: Cannot execute this validator script itself.", file=sys.stderr)
            sys.exit(1)
        py_files = [target]
        target_dir = target.parent
    elif target.is_dir():
        py_files = sorted(target.rglob("*.py"))
        py_files = [f for f in py_files if f.resolve() != self_path]
        target_dir = target
    else:
        print(f"Error: Target '{target}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not py_files:
        print(f"No valid .py files found in target.")
        return

    relative_paths = [str(f.relative_to(target_dir)) for f in py_files]
    relative_paths.sort()

    success_count = 0
    fail_count = 0
    skipped_sim_count = 0
    failure_details = []
    skipped_sim_scripts = []  # Track scripts skipped due to lack of sim support

    print(f"DEVICE_ID set to: {args.device_id}")
    print(f"Run mode       : {args.run_mode}")
    if args.example_id:
        print(f"Test selector  : {args.example_id}")
    print(f"Target         : {target}")
    print(f"Found {len(relative_paths)} .py file(s). Starting execution...\n")

    for rel_path in relative_paths:
        full_path = target_dir / rel_path
        print(f"Running: {rel_path}")

        # Step 1: Check if script has __main__ guard
        has_main = _has_main_guard(full_path)

        # Step 2: For sim mode, check support
        if args.run_mode == "sim":
            if not _supports_run_mode_sim(full_path):
                msg = "Skipped: script does not support --run_mode sim"
                print(f"⏭️  Skipped: {rel_path} ({msg})")
                skipped_sim_count += 1
                skipped_sim_scripts.append(rel_path)
                print("-" * 50)
                continue

        # Step 3: Build command
        if has_main:
            cmd = [sys.executable, str(full_path)]
            if args.example_id:
                cmd.append(args.example_id)
            if args.run_mode == "sim":
                cmd.append("--run_mode")
                cmd.append("sim")
            
        else:
            # Fallback to pytest
            cmd = ["pytest", str(full_path), "-v"]

        print(f"  → Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=args.timeout
            )
            output = result.stdout + result.stderr

            has_exemption = _has_success_exemption(output)
            lower_output = output.lower()
            detected_keywords = []
            if "error" in lower_output:
                detected_keywords.append("Error")
            if "mismatch" in lower_output:
                detected_keywords.append("mismatch")
            if "traceback" in lower_output:
                detected_keywords.append("Traceback")

            if has_exemption:
                print(f"✅ Success: {rel_path} (exempted via 'All ... pass/success' pattern)")
                success_count += 1
            elif detected_keywords:
                reason = ", ".join(detected_keywords)
                print(f"❌ Failure: {rel_path} (triggered by: {reason})")
                fail_count += 1
                failure_details.append((rel_path, reason))
            else:
                print(f"✅ Success: {rel_path}")
                success_count += 1

        except subprocess.TimeoutExpired:
            msg = "Timeout (exceeded 300s)"
            print(f"❌ Failure: {rel_path} ({msg})")
            fail_count += 1
            failure_details.append((rel_path, msg))
        except Exception as e:
            msg = f"Exception: {e}"
            print(f"❌ Failure: {rel_path} ({msg})")
            fail_count += 1
            failure_details.append((rel_path, msg))

        print("-" * 50)

    # Final summary
    total_executed = success_count + fail_count
    total_original = len(relative_paths)
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    print(f"Target directory/file : {target}")
    print(f"Run mode              : {args.run_mode}")
    print(f"DEVICE_ID             : {args.device_id}")
    print(f"Total scripts found   : {total_original}")
    print(f"Scripts executed      : {total_executed}")
    print(f"✅ Successful          : {success_count}")
    print(f"❌ Failed              : {fail_count}")
    if skipped_sim_count > 0:
        print(f"⏭️  Skipped (no sim support): {skipped_sim_count}")

    if failure_details:
        print("\nFailed Scripts (with diagnostic reasons):")
        for script, reason in failure_details:
            print(f"  • {script} → {reason}")

    if skipped_sim_scripts:
        print("\nScripts Skipped Due to Lack of Sim Support:")
        for script in skipped_sim_scripts:
            print(f"  • {script}")

    print("=" * 60)


if __name__ == "__main__":
    main()