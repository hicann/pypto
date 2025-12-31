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
Parallel Python Script Executor with Retry Logic and Device-Aware Validation

This utility orchestrates the robust, concurrent execution of Python scripts or test suites
across multiple hardware devices (e.g., NPUs/GPUs), featuring intelligent dispatch, adaptive
retry mechanisms, and comprehensive result analysis. Designed for validation pipelines in
hardware-accelerated development environments, it ensures reliable script evaluation under
resource-constrained and failure-prone conditions.

Key Capabilities:
- Target Flexibility: Accepts a single `.py` file or an entire directory for recursive processing.
- Execution Intelligence:
    - Scripts with `if __name__ == "__main__":` are run directly via `python`.
    - Test-only scripts (containing `test_*` functions or `Test*` classes) are dispatched to `pytest`.
    - Automatically skips files lacking executable content.
- Run Mode Control:
    - `npu` (default): Executes all eligible scripts.
    - `sim`: Filters and runs only scripts that explicitly declare support for `--run_mode sim`.
- Device-Aware Parallelism:
    - Leverages a pool of device IDs (e.g., `DEVICE_ID=0,1,2`) to enable true hardware-level concurrency.
    - Each script leases a device during execution and releases it upon completion or failure.
- Resilient Execution Strategy:
    - Initial Parallel Round: All eligible scripts run concurrently (bounded by available devices).
    - Configurable Parallel Retries: Failed scripts are retried up to `N` additional times in parallel.
    - Final Serial Fallback: Remaining failures undergo a last-chance serial retry on a single device
      to rule out resource contention or race conditions.
- Timeout & Isolation:
    - Each script enforces a configurable timeout (default: 300s).
    - Child processes inherit only necessary environment variables (`TILE_FWK_DEVICE_ID`).
- Smart Filtering & Self-Protection:
    - Excludes itself from execution to prevent recursion.
    - Skips non-Python files and invalid targets early with clear diagnostics.
- Granular Test Selection: Supports passing a specific test identifier (e.g., `test_add`) to either
  the script (as CLI arg) or pytest (as `file::test`), enabling focused validation.
- Comprehensive Reporting:
    - Real-time status indicators (✅/❌/⏭️) with device assignment.
    - Final summary categorizing successes, failures, and skip reasons.
    - Optional output snippet preview for failed scripts (last 5 lines).
- Dependency Validation: Checks for `pytest` presence if any target requires test discovery.

Exit Behavior:
- Returns `0` only if all executed scripts succeed (skipped scripts do not affect exit code).
- Returns `1` if any script fails after all retry attempts.
- Exits early with `1` on invalid input (e.g., missing file, empty device list).

Usage Examples:
    # 1. Execute all Python scripts in a directory using single device
    python3 examples/validate_examples.py -t examples/02_intermediate --device_ids 0
    
    # 2. Single-device mode with up to 3 serial retry rounds for unstable scripts
    python3 examples/validate_examples.py -t examples --device_ids 0 --serial-retries 3

    # 3. Run a single script in default NPU mode on device 0
    python3 examples/validate_examples.py -t examples/01_beginner/basic/basic_ops.py --device_ids 0

    # 4. Execute all Python scripts in a directory using multiple devices (parallel execution)
    python3 examples/validate_examples.py -t examples --device_ids 0,1,2,3

    # 5. Execute in simulation mode (only scripts supporting --run_mode sim are run)
    python3 examples/validate_examples.py -t examples --run_mode sim --device_ids 0

    # 6. Concurrent execution in simulation mode
    python3 examples/validate_examples.py -t examples --run_mode sim --device_ids 0,1,2,3,4,5,6,7

    # 7. Set custom timeout (in seconds) for each script execution
    python3 examples/validate_examples.py -t examples/02_intermediate --device_ids 0 --timeout 120

    # 8. Enable parallel retries (initial run + 1 retry rounds) for flaky scripts
    python3 examples/validate_examples.py -t examples --device_ids 0,1 --parallel-retries 1

    # 9. Show last 5 lines of output for failed scripts in final summary (for debugging)
    python3 examples/validate_examples.py -t examples --device_ids 0 --show-fail-details

    # 10. Full production-grade validation: multi-device, retries, timeout, and failure details
    python3 examples/validate_examples.py -t examples --device_ids 0,1,2,3,4,5,6,7 
        --parallel-retries 2 --serial-retries 5 --timeout 300 --show-fail-details

Note: This script is intended for use within CANN-based development workflows and assumes
hardware context awareness via the `TILE_FWK_DEVICE_ID`/
    `TILE_FWK_STEST_DEVICE_ID`/`ASCEND_VISIBLE_DEVICES` environment variable.
"""
import os
import subprocess
import sys
import argparse
import re
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time


def _has_main_guard(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', content))
    except Exception:
        return False


def _supports_run_mode_sim(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        pattern = re.compile(r'add_argument\s*\(.*?["\']--run_mode["\']| \
                             dest\s*=\s*["\']run_mode["\']', re.DOTALL)
        return bool(pattern.search(content))
    except Exception:
        return False


def _has_pytest_tests(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        has_test_func = re.search(r'^\s*def\s+test[a-zA-Z0-9_]*\s*\(', content, re.MULTILINE)
        has_test_class = re.search(r'^\s*class\s+Test[a-zA-Z0-9_]*\s*[:\(]', content, re.MULTILINE)
        return bool(has_test_func or has_test_class)
    except Exception:
        return False


def run_script(args, full_path: Path, rel_path: str, device_queue: queue.Queue,
               timeout: int, safe_print, print_cmd_on_serial=False):
    """
    Execute a single script by leasing a device from device_queue.
    Returns a dict with result info.
    """
    has_main = _has_main_guard(full_path)
    has_tests = _has_pytest_tests(full_path)

    # Early skip logic (no device needed)
    if not has_main and not has_tests:
        safe_print(f"⏭️  Skipped: {rel_path} (no '__main__' guard and no pytest-style tests)")
        return {
            "rel_path": rel_path,
            "status": "skipped_no_tests",
            "message": "no '__main__' guard and no pytest-style tests"
        }

    if args.run_mode == "sim" and not _supports_run_mode_sim(full_path):
        safe_print(f"⏭️  Skipped: {rel_path} (script does not support --run_mode sim)")
        return {
            "rel_path": rel_path,
            "status": "skipped_sim",
            "message": "script does not support --run_mode sim"
        }

    # Lease a device
    try:
        device_id = device_queue.get(timeout=timeout)
    except queue.Empty:
        safe_print(f"❌ Failure: {rel_path} (device acquisition timeout)")
        return {
            "rel_path": rel_path,
            "status": "failure",
            "reason": f"Could not acquire a device within {timeout}s",
            "device_id": None,
            "output_snippet": ""
        }

    # Print start message — omit device info in sim mode
    if args.run_mode == "sim":
        safe_print(f"▶️  Starting: {rel_path}")
    else:
        safe_print(f"▶️  Starting: {rel_path} (device={device_id})")

    try:
        env = os.environ.copy()
        env["TILE_FWK_DEVICE_ID"] = device_id
        env["ASCEND_VISIBLE_DEVICES"] = device_id
        env["TILE_FWK_STEST_DEVICE_ID"] = device_id

        if has_main:
            cmd = [sys.executable, str(full_path)]
            if args.example_id:
                cmd.append(args.example_id)
            if args.run_mode == "sim":
                cmd.extend(["--run_mode", "sim"])
        else:
            if args.example_id:
                cmd = ["pytest", f"{full_path}::{args.example_id}", "-v", "--capture=no"]
            else:
                cmd = ["pytest", str(full_path), "-v", "--capture=no"]

        if print_cmd_on_serial:
            cmd_str = " ".join(str(part) for part in cmd)
            safe_print(f"→  Executing: {cmd_str}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            snippet = "\n".join(output.strip().splitlines()[-5:]) if output.strip() else ""

            if result.returncode == 0:
                safe_print(f"✅ Success: {rel_path}")
                return {
                    "rel_path": rel_path,
                    "status": "success",
                    "device_id": device_id,
                    "output_snippet": None
                }
            else:
                safe_print(f"❌ Failure: {rel_path}")
                return {
                    "rel_path": rel_path,
                    "status": "failure",
                    "reason": f"Non-zero exit code ({result.returncode})",
                    "device_id": device_id,
                    "output_snippet": snippet
                }
        except subprocess.TimeoutExpired:
            safe_print(f"❌ Failure: {rel_path}")
            return {
                "rel_path": rel_path,
                "status": "failure",
                "reason": f"Timeout (exceeded {timeout}s)",
                "device_id": device_id,
                "output_snippet": ""
            }
        except Exception as e:
            safe_print(f"❌ Failure: {rel_path}")
            return {
                "rel_path": rel_path,
                "status": "failure",
                "reason": f"Exception during execution: {e}",
                "device_id": device_id,
                "output_snippet": ""
            }

    finally:
        device_queue.put(device_id)
        safe_print("-" * 50)


def _execute_scripts(args, rel_paths, target_dir, device_ids,
                     max_workers, timeout, safe_print, print_cmd_on_serial=False):
    """Helper to execute a list of scripts with given device pool."""
    if not rel_paths:
        return []

    device_queue = queue.Queue()
    for dev in device_ids:
        device_queue.put(dev)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_rel = {}
        for rel_path in rel_paths:
            full_path = target_dir / rel_path
            future = executor.submit(run_script, args, full_path, rel_path,
                                     device_queue, timeout, safe_print, print_cmd_on_serial)
            future_to_rel[future] = rel_path

        results = []
        for future in as_completed(future_to_rel):
            results.append(future.result())

    # Sort to preserve input order (for readability)
    results.sort(key=lambda x: rel_paths.index(x["rel_path"]))
    return results


def _print_final_summary(success_list, failure_list, skipped_sim_list, skipped_no_tests_list,
                         args, target, device_ids, total_time_sec, safe_print):
    total_original = len(success_list) + len(failure_list) + \
        len(skipped_sim_list) + len(skipped_no_tests_list)

    safe_print("\n" + "=" * 60)
    safe_print("📊 FINAL EXECUTION SUMMARY")
    safe_print("=" * 60)
    safe_print(f"Target directory/file : {target}")
    safe_print(f"Run mode              : {args.run_mode}")
    safe_print(f"DEVICE_IDs            : {', '.join(device_ids)}")
    safe_print(f"Total scripts found   : {total_original}")
    safe_print(f"Total execution time  : {total_time_sec:.2f} seconds")
    safe_print(f"Scripts executed      : {len(success_list) + len(failure_list)}")
    safe_print(f"✅ Successful          : {len(success_list)}")
    safe_print(f"❌ Failed              : {len(failure_list)}")
    if skipped_sim_list:
        safe_print(f"⏭️  Skipped (no sim support): {len(skipped_sim_list)}")
    if skipped_no_tests_list:
        safe_print(f"⏭️  Skipped (no main/test): {len(skipped_no_tests_list)}")

    if failure_list:
        safe_print("\nFailed Scripts:")
        for r in failure_list:
            reason = r.get("reason", "Unknown")
            snippet = r.get("output_snippet", "")
            safe_print(f"  • {r['rel_path']} → {reason}")
            if args.show_fail_details and snippet:
                safe_print("    Output preview:")
                for line in snippet.splitlines():
                    safe_print(f"      {line}")
                safe_print()

    if skipped_sim_list:
        safe_print("\nSkipped Due to Lack of Sim Support:")
        for r in skipped_sim_list:
            safe_print(f"  • {r['rel_path']}")

    if skipped_no_tests_list:
        safe_print("\nSkipped Due to No Executable Content:")
        for r in skipped_no_tests_list:
            safe_print(f"  • {r['rel_path']}")

    safe_print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute and validate Python scripts with configurable \
            parallel retries and final serial fallback."
    )
    parser.add_argument(
        "-d", "--device_ids",
        type=str,
        default="0",
        help="Comma-separated list of DEVICE_IDs (e.g., '0,1,2,3'). Default: '0'."
    )
    parser.add_argument(
        "-r", "--run_mode",
        choices=["npu", "sim"],
        default="npu",
        help="Execution mode: 'npu' (default) or 'sim'. \
            In 'sim' mode, only scripts supporting --run_mode sim are executed."
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
        help="Optional test identifier (e.g., 'test_add' or \
            'test_file.py::test_add') to pass to script or pytest."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-script execution timeout in seconds (default: 300)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers (default: number of device_ids). "
             "Note: actual parallelism is limited by available devices."
    )
    parser.add_argument(
        "--parallel-retries",
        type=int,
        default=1,
        help="Number of additional parallel retry rounds after the initial run (default: 1). "
             "Total parallel rounds = 1 (initial) + N (retries)."
    )
    parser.add_argument(
        "--serial-retries",
        type=int,
        default=5,
        help="Maximum number of serial retry rounds in single-device mode (default: 5). "
             "Total serial runs = 1 (initial) + N (retries). Set to 0 to disable retries."
    )
    parser.add_argument(
        "--show-fail-details",
        action="store_true",
        help="Show last 5 lines of output for each failed script in the final summary."
    )
    args = parser.parse_args()

    # Parse device IDs
    device_ids = [d.strip() for d in args.device_ids.split(",") if d.strip()]
    if not device_ids:
        print("Error: --device_ids cannot be empty.", file=sys.stderr)
        sys.exit(1)

    target = Path(args.target).resolve()
    self_path = Path(__file__).resolve()

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

    # Check pytest requirement
    need_pytest = any(not _has_main_guard(f) and _has_pytest_tests(f) for f in py_files)
    pytest_available = shutil.which("pytest") is not None
    if need_pytest and not pytest_available:
        print("Error: Some scripts require 'pytest'...", file=sys.stderr)
        print("    pip install pytest", file=sys.stderr)
        sys.exit(1)

    # Determine execution strategy based on number of devices
    is_single_device = len(device_ids) == 1

    if is_single_device:
        max_workers = 1
        parallel_retries = 0
    else:
        max_workers = args.max_workers or len(device_ids)
        parallel_retries = max(0, args.parallel_retries)

    print(f"DEVICE_IDs        : {', '.join(device_ids)}")
    print(f"Run mode          : {args.run_mode}")
    if is_single_device:
        print("Execution mode    : Serial (single device)")
    else:
        print(f"Parallel retries  : {parallel_retries} \
              (total parallel rounds = {parallel_retries + 1})")
    if args.example_id:
        print(f"Test selector     : {args.example_id}")
    print(f"Target            : {target}")
    print(f"Found {len(relative_paths)} .py file(s).")
    print("=" * 60)

    # Thread-safe print
    print_lock = threading.Lock()
    
    def safe_print(*a, **kw):
        with print_lock:
            print(*a, **kw)
    
    start_time = time.perf_counter()
    # Initial candidate list: all scripts that are not skipped
    initial_results = []
    candidates_to_run = []
    skipped_sim_scripts = []
    skipped_no_tests_scripts = []

    for rel_path in relative_paths:
        full_path = target_dir / rel_path
        has_main = _has_main_guard(full_path)
        has_tests = _has_pytest_tests(full_path)

        if not has_main and not has_tests:
            skipped_no_tests_scripts.append(rel_path)
            continue

        if args.run_mode == "sim" and not _supports_run_mode_sim(full_path):
            skipped_sim_scripts.append(rel_path)
            continue

        candidates_to_run.append(rel_path)

    # Record skipped results for final summary
    skipped_sim_results = [{"rel_path": p, "status": "skipped_sim"} for p in skipped_sim_scripts]
    skipped_no_tests_results = [{"rel_path": p, "status": "skipped_no_tests"} 
                                for p in skipped_no_tests_scripts]

    current_candidates = candidates_to_run[:]
    all_results_map = {}

    # Add skipped results to final map immediately
    for r in skipped_sim_results + skipped_no_tests_results:
        all_results_map[r["rel_path"]] = r

    # If no executable scripts, just print summary and exit
    if not current_candidates:
        safe_print("ℹ️  No executable scripts found. All were skipped.")
        total_time_sec = time.perf_counter() - start_time
        _print_final_summary([], [], skipped_sim_results, skipped_no_tests_results,
                             args, target, device_ids, total_time_sec, safe_print)
        sys.exit(0)

        # ----------------------------
    # SINGLE DEVICE: SERIAL EXECUTION WITH RETRY LOOP
    # ----------------------------
    if is_single_device:
        safe_print("🚀 Starting Serial Execution — single device mode\n")
        current_candidates = candidates_to_run[:]
        prev_failure_count = len(current_candidates)
        max_serial_retries = max(0, args.serial_retries)  # ensure non-negative
        retry_round = 0

        # Always run at least once (initial run)
        while current_candidates and retry_round <= max_serial_retries:
            if retry_round == 0:
                safe_print(f"▶️  Initial Serial Run — {len(current_candidates)} script(s)\n")
            else:
                safe_print(f"🔁 Serial Retry Round {retry_round}/{max_serial_retries} \
                           — {len(current_candidates)} script(s)\n")

            serial_results = _execute_scripts(
                args, current_candidates, target_dir, device_ids,
                max_workers=1, timeout=args.timeout, 
                safe_print=safe_print, print_cmd_on_serial=True
            )

            # Update global result map
            for r in serial_results:
                all_results_map[r["rel_path"]] = r

            new_failures = [r for r in serial_results if r["status"] == "failure"]
            current_candidates = [r["rel_path"] for r in new_failures]
            current_failure_count = len(current_candidates)

            # Success: break early
            if current_failure_count == 0:
                safe_print("✅ All scripts passed.\n")
                break

            # Check if failure count stopped decreasing
            if current_failure_count >= prev_failure_count:
                safe_print(f"⚠️  Failure count did not decrease (was {prev_failure_count}, \
                           now {current_failure_count}). Stopping retries.\n")
                break

            prev_failure_count = current_failure_count
            retry_round += 1

        # If we exited because of retry limit (and still have failures)
        if current_candidates and retry_round > max_serial_retries:
            safe_print(f"🛑 Reached maximum serial retries ({max_serial_retries}). Stopping.\n")

        success_list = [r for r in all_results_map.values() if r["status"] == "success"]
        failure_list = [r for r in all_results_map.values() if r["status"] == "failure"]

        total_time_sec = time.perf_counter() - start_time
        _print_final_summary(success_list, failure_list, skipped_sim_results, 
                             skipped_no_tests_results, args, target, 
                             device_ids, total_time_sec, safe_print)
        sys.exit(1 if len(failure_list) > 0 else 0)

    # ----------------------------
    # MULTI DEVICE: PARALLEL + RETRIES + FINAL SERIAL FALLBACK
    # ----------------------------
    else:
        for round_idx in range(parallel_retries + 1):
            round_name = "Initial" if round_idx == 0 else f"Retry {round_idx}"
            safe_print(f"🚀 Starting Parallel Round {round_idx + 1}/{parallel_retries + 1} \
                       ({round_name}) — {len(current_candidates)} script(s)\n")

            round_results = _execute_scripts(
                args, current_candidates, target_dir, device_ids, max_workers, 
                args.timeout, safe_print, print_cmd_on_serial=False
            )

            # Update final results map
            for r in round_results:
                all_results_map[r["rel_path"]] = r

            # Check for failures
            round_failures = [r for r in round_results if r["status"] == "failure"]
            if not round_failures:
                safe_print(f"✅ All scripts passed in Parallel Round {round_idx + 1}. \
                           No further retries needed.")
                success_list = [r for r in all_results_map.values() if r["status"] == "success"]
                failure_list = []
                
                total_time_sec = time.perf_counter() - start_time
                _print_final_summary(success_list, failure_list, skipped_sim_results, 
                                     skipped_no_tests_results, args, target, 
                                     device_ids, total_time_sec, safe_print)
                sys.exit(0)

            # Prepare next round
            current_candidates = [r["rel_path"] for r in round_failures]
            safe_print(f"🔁 {len(current_candidates)} script(s) failed and will be retried.")

        # Final serial retry loop for remaining failures (with convergence and max retries)
        if current_candidates:
            safe_print(f"🔂 Starting Final Serial Retry Loop — {len(current_candidates)} \
                       remaining failed script(s)\n")
            serial_candidates = current_candidates[:]
            prev_failure_count = len(serial_candidates)
            max_serial_retries = max(0, args.serial_retries)
            serial_retry_round = 0

            while serial_candidates and serial_retry_round <= max_serial_retries:
                if serial_retry_round == 0:
                    safe_print(f"▶️  Final Serial Run — {len(serial_candidates)} script(s)\n")
                else:
                    safe_print(f"🔁 Final Serial Retry {serial_retry_round}/{max_serial_retries} \
                               — {len(serial_candidates)} script(s)\n")

                serial_device_ids = [device_ids[0]]  # use first device
                serial_results = _execute_scripts(
                    args, serial_candidates, target_dir, serial_device_ids,
                    max_workers=1, timeout=args.timeout, 
                    safe_print=safe_print, print_cmd_on_serial=True
                )

                # Update global results map
                for r in serial_results:
                    all_results_map[r["rel_path"]] = r

                new_failures = [r for r in serial_results if r["status"] == "failure"]
                serial_candidates = [r["rel_path"] for r in new_failures]
                current_failure_count = len(serial_candidates)

                if current_failure_count == 0:
                    safe_print("✅ All scripts passed after final serial retry loop.\n")
                    break

                if current_failure_count >= prev_failure_count:
                    safe_print(f"⚠️  Final serial retry: failure count did not \
                               decrease (was {prev_failure_count}, \
                               now {current_failure_count}). Stopping.\n")
                    break

                prev_failure_count = current_failure_count
                serial_retry_round += 1

            if serial_candidates and serial_retry_round > max_serial_retries:
                safe_print(f"🛑 Reached maximum final serial retries \
                           ({max_serial_retries}). Stopping.\n")

        success_list = [r for r in all_results_map.values() if r["status"] == "success"]
        failure_list = [r for r in all_results_map.values() if r["status"] == "failure"]

        safe_print("\n🏁 All execution rounds completed.")
        total_time_sec = time.perf_counter() - start_time
        _print_final_summary(success_list, failure_list, skipped_sim_results, 
                             skipped_no_tests_results, args, target, 
                             device_ids, total_time_sec, safe_print)

        sys.exit(1 if len(failure_list) > 0 else 0)


if __name__ == "__main__":
    main()