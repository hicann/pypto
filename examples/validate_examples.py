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
Parallel Python Script Executor with Adaptive Retry Logic

This utility orchestrates the concurrent execution of Python scripts across multiple hardware devices
(NPUs), featuring intelligent script analysis, adaptive execution strategies,
and comprehensive reporting. Designed for validation workflows in CANN-based hardware-accelerated 
environments, it ensures reliable script evaluation.

Key Capabilities:
- Target Flexibility: Processes single .py files or recursively scans directories, excluding itself.
- Intelligent Script Analysis (via AST parsing):
    - Detects if name == 'main' guards for direct execution
    - Identifies pytest-style tests (test* functions or Test* classes) for pytest dispatch
    - Verifies '--run_mode' argument support for simulation mode filtering
    - Automatically skips files with no executable content
- Dual Execution Modes:
    - npu (default): Executes eligible scripts on physical hardware devices with device isolation
    - sim: Filters and runs only scripts that explicitly support '--run_mode' argument 
        using virtual workers
- Resource Management:
    - Thread-safe device leasing system for hardware resource allocation
    - Hierarchical process termination (parent + children) on timeout or failure
    - Device-specific environment isolation (TILE_FWK_DEVICE_ID, ASCEND_VISIBLE_DEVICES, 
        TILE_FWK_STEST_DEVICE_ID)
- Adaptive Execution Strategies:
    - Single-Device Mode: Serial execution with progressive retry rounds (default: 3)
    - Multi-Device Mode:
        * Initial parallel execution across available physical/virtual devices
        * Configurable parallel retry rounds (default: 1)
        * Final serial fallback for persistent failures to eliminate resource contention
- Granular Test Selection: Passes specific test identifiers to scripts or pytest as needed
- Comprehensive Reporting:
    - Real-time emoji-enhanced status indicators (✅/❌/⏭️/⚠️) with device assignment
    - Final categorized summary with success/failure/skip counts
    - Optional failure diagnostics showing last 5 lines of output
    - Structured retry progression tracking
- Safety Features:
    - Per-script timeout enforcement (default: 300s) with cleanup guarantees
    - Dependency validation (pytest availability check)
    - Process group isolation for reliable cleanup

Exit Behavior:
- Returns 0 only if all executed scripts succeed (skipped scripts don't affect exit code)
- Returns 1 if any script fails after all retry attempts
- Early exits with descriptive errors for invalid inputs or missing dependencies

Usage Examples:
    # 1. Execute directory on single NPU device
    python3 examples/validate_examples.py -t examples/02_intermediate -d 0

    # 2. Multi-device parallel execution
    python3 examples/validate_examples.py -t examples -d 0,1,2,3

    # 3. Execute specific script on device 0
    python3 examples/validate_examples.py -t examples/01_beginner/basic/basic_ops.py -d 0

    # 4. Simulation mode (single virtual worker)
    python3 examples/validate_examples.py -t examples --run_mode sim -w 1

    # 5. Concurrent execution in simulation mode (16 virtual workers)
    python3 examples/validate_examples.py -t examples --run_mode sim -w 16

    # 6. Custom timeout per script
    python3 examples/validate_examples.py -t examples/02_intermediate -d 0 --timeout 120

    # 7. Show failure diagnostics in summary
    python3 examples/validate_examples.py -t examples -d 0 --show-fail-details

    # 8. Full configuration
    python3 examples/validate_examples.py -t examples -d 0,1,2,3 
        --parallel-retries 2 --serial-retries 5 
        --timeout 300 --show-fail-details --allow-pytest-auto-detect

Note: This tool is designed specifically for CANN-based development workflows. In npu mode, device
parallelism is determined by provided device IDs. In sim mode, parallelism is controlled by the
--workers parameter which creates virtual device slots.
"""
import os
import subprocess
import sys
import argparse
import ast
import re
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
import signal
from typing import List, Dict, Any, Optional, Tuple, Union
import psutil


def _has_main_guard(file_path: Path) -> bool:
    """Check if the file contains an 'if __name__ == '__main__'' guard using AST parsing"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Avoid exceptions for empty files
        if not content.strip():
            return False
            
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if condition is __name__ == '__main__'
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == '__name__' and
                    len(node.test.ops) == 1 and
                    isinstance(node.test.ops[0], ast.Eq) and
                    len(node.test.comparators) == 1 and
                    isinstance(node.test.comparators[0], ast.Constant) and
                    node.test.comparators[0].value == '__main__'):
                    return True
        return False
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        # The file may not be Python code or may have encoding issues
        return False
    except Exception as e:
        print(f"Warning: Error parsing {file_path} with AST: {e}", file=sys.stderr)
        return False


def _supports_run_mode_sim(file_path: Path) -> bool:
    """Check if the file supports the '--run_mode' argument using AST parsing"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Avoid exceptions for empty files
        if not content.strip():
            return False
            
        tree = ast.parse(content, filename=str(file_path))
        
        # Check argparse-related code
        for node in ast.walk(tree):
            # Check add_argument calls
            if (isinstance(node, ast.Call) and 
                hasattr(node.func, 'attr') and node.func.attr == 'add_argument'):
                for arg in node.args:
                    if (isinstance(arg, ast.Constant) and 
                        isinstance(arg.value, str) and 
                        '--run_mode' in arg.value):
                        return True
                
                # Check keyword arguments
                for keyword in node.keywords:
                    if keyword.arg == 'dest' and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value == 'run_mode':
                            return True
        
        return False
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return False
    except Exception as e:
        print(f"Warning: Error parsing {file_path} with AST for run_mode support: {e}", 
              file=sys.stderr)
        return False


def _has_pytest_tests(file_path: Path) -> bool:
    """Check if the file contains pytest-style tests using AST parsing"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Avoid exceptions for empty files
        if not content.strip():
            return False
            
        tree = ast.parse(content, filename=str(file_path))
        
        has_test_function = False
        has_test_class = False
        
        for node in ast.walk(tree):
            # Check test functions
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test'):
                    has_test_function = True
            
            # Check test classes
            if isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    has_test_class = True
        
        return has_test_function or has_test_class
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return False
    except Exception as e:
        print(f"Warning: Error parsing {file_path} with AST for pytest tests: {e}", file=sys.stderr)
        return False


def _terminate_process_and_children(proc: subprocess.Popen) -> None:
    """Terminate a process and all its child processes"""
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        
        # Terminate child processes first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Wait for child processes to terminate
        gone, still_alive = psutil.wait_procs(children, timeout=3)
        
        # Force-kill remaining processes
        for child in still_alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Terminate parent process
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # The process may no longer exist
        try:
            proc.kill()
        except Exception as e:
            print(f"Unexpected error while killing process: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error terminating process tree: {e}", file=sys.stderr)


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
        safe_print(f"⏭️  Skipped: {rel_path} (script does not support --run_mode)")
        return {
            "rel_path": rel_path,
            "status": "skipped_sim",
            "message": "script does not support --run_mode"
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

    proc = None
    try:
        env = os.environ.copy()
        # Only set device-related environment variables in NPU mode
        if args.run_mode != "sim":
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
            if not args.allow_pytest_auto_detect:
                safe_print(f"⚠️  Warning: {rel_path} relies on pytest but auto-detection \
                            is not explicitly allowed.")
                safe_print("    The script will still be executed with pytest. \
                           Use the flag to suppress this warning.")
            
            if args.example_id:
                cmd = ["pytest", f"{full_path}::{args.example_id}", "-v", "--capture=no"]
            else:
                cmd = ["pytest", str(full_path), "-v", "--capture=no"]

        if print_cmd_on_serial:
            cmd_str = " ".join(str(part) for part in cmd)
            safe_print(f"→  Executing: {cmd_str}")

        # Create a new process group for easier cleanup later
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        try:
            # Wait for process completion or timeout
            stdout, stderr = proc.communicate(timeout=timeout)
            output = stdout + stderr
            snippet = "\n".join(output.strip().splitlines()[-5:]) if output.strip() else ""

            if proc.returncode == 0:
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
                    "reason": f"Non-zero exit code ({proc.returncode})",
                    "device_id": device_id,
                    "output_snippet": snippet
                }
        except subprocess.TimeoutExpired:
            # Timeout handling: terminate the process and its child processes
            _terminate_process_and_children(proc)
            stdout, stderr = proc.communicate()
            output = stdout + stderr
            snippet = "\n".join(output.strip().splitlines()[-5:]) if output.strip() else ""
            
            safe_print(f"❌ Failure: {rel_path}")
            return {
                "rel_path": rel_path,
                "status": "failure",
                "reason": f"Timeout (exceeded {timeout}s)",
                "device_id": device_id,
                "output_snippet": snippet
            }
        except Exception as e:
            # Exception handling: terminate the process and its child processes
            if proc:
                _terminate_process_and_children(proc)
            safe_print(f"❌ Failure: {rel_path}")
            return {
                "rel_path": rel_path,
                "status": "failure",
                "reason": f"Exception during execution: {e}",
                "device_id": device_id,
                "output_snippet": ""
            }

    finally:
        if proc and proc.poll() is None:
            # Ensure the process has been terminated
            _terminate_process_and_children(proc)
            
        device_queue.put(device_id)
        safe_print("-" * 50)


class ExecutionStrategy:
    """Base class for execution strategies"""
    
    def __init__(self, args, target_dir, device_ids, timeout, safe_print):
        self.args = args
        self.target_dir = target_dir
        self.device_ids = device_ids
        self.timeout = timeout
        self.safe_print = safe_print
        
    def execute(self, scripts: List[str], 
                all_results_map: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Execute scripts and return success and failure lists"""
        raise NotImplementedError()


class SingleDeviceStrategy(ExecutionStrategy):
    """Execution strategy for a single device"""
    
    def execute(self, scripts: List[str], 
                all_results_map: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
        current_candidates = scripts[:]
        prev_failure_count = len(current_candidates)
        max_serial_retries = max(0, self.args.serial_retries)
        retry_round = 0

        # Always run at least once (initial run)
        while current_candidates and retry_round <= max_serial_retries:
            if retry_round == 0:
                self.safe_print(f"▶️  Initial Serial Run — {len(current_candidates)} script(s)\n")
            else:
                self.safe_print(f"🔁 Serial Retry Round {retry_round}/{max_serial_retries} "
                               f"— {len(current_candidates)} script(s)\n")

            serial_results = _execute_scripts(
                self.args, current_candidates, self.target_dir, self.device_ids,
                workers=1, timeout=self.timeout, 
                safe_print=self.safe_print, print_cmd_on_serial=True
            )

            # Update global result map
            for r in serial_results:
                all_results_map[r["rel_path"]] = r

            new_failures = [r for r in serial_results if r["status"] == "failure"]
            current_candidates = [r["rel_path"] for r in new_failures]
            current_failure_count = len(current_candidates)

            # Success: break early
            if current_failure_count == 0:
                self.safe_print("✅ All scripts passed.\n")
                break

            # Check if failure count stopped decreasing
            if current_failure_count >= prev_failure_count:
                self.safe_print(f"⚠️  Failure count did not decrease (was {prev_failure_count}, "
                               f"now {current_failure_count}). Stopping retries.\n")
                break

            prev_failure_count = current_failure_count
            retry_round += 1

        # If we exited because of retry limit (and still have failures)
        if current_candidates and retry_round > max_serial_retries:
            self.safe_print(f"🛑 Reached maximum serial retries ({max_serial_retries}). Stopping.\n")

        success_list = [r for r in all_results_map.values() if r["status"] == "success"]
        failure_list = [r for r in all_results_map.values() if r["status"] == "failure"]
        
        return success_list, failure_list


class MultiDeviceStrategy(ExecutionStrategy):
    """Execution strategy for multiple devices"""
    
    def execute(self, scripts: List[str], 
                all_results_map: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
        current_candidates = scripts[:]
        parallel_retries = max(0, self.args.parallel_retries)
        
        # Determine actual workers to use based on mode
        if self.args.run_mode == "sim":
            actual_workers = len(self.device_ids)  # Use virtual devices count
        else:
            actual_workers = len(self.device_ids)  # Use physical devices count
        
        # Parallel execution rounds
        for round_idx in range(parallel_retries + 1):
            round_name = "Initial" if round_idx == 0 else f"Retry {round_idx}"
            self.safe_print(f"🚀 Starting Parallel Round {round_idx + 1}/{parallel_retries + 1} "
                           f"({round_name}) — {len(current_candidates)} script(s)\n")

            round_results = _execute_scripts(
                self.args, current_candidates, self.target_dir, self.device_ids, 
                workers=actual_workers, timeout=self.timeout, 
                safe_print=self.safe_print, print_cmd_on_serial=False
            )

            # Update final results map
            for r in round_results:
                all_results_map[r["rel_path"]] = r

            # Check for failures
            round_failures = [r for r in round_results if r["status"] == "failure"]
            if not round_failures:
                self.safe_print(f"✅ All scripts passed in Parallel Round {round_idx + 1}. "
                               f"No further retries needed.")
                success_list = [r for r in all_results_map.values() if r["status"] == "success"]
                failure_list = []
                return success_list, failure_list

            # Prepare next round
            current_candidates = [r["rel_path"] for r in round_failures]
            self.safe_print(f"🔁 {len(current_candidates)} script(s) failed and will be retried.")

        # Final serial retry loop for remaining failures
        if current_candidates:
            self.safe_print(f"🔂 Starting Final Serial Retry Loop — {len(current_candidates)} "
                           f"remaining failed script(s)\n")
            serial_candidates = current_candidates[:]
            prev_failure_count = len(serial_candidates)
            max_serial_retries = max(0, self.args.serial_retries)
            serial_retry_round = 0

            while serial_candidates and serial_retry_round <= max_serial_retries:
                if serial_retry_round == 0:
                    self.safe_print(f"▶️  Final Serial Run — {len(serial_candidates)} script(s)\n")
                else:
                    self.safe_print(f"🔁 Final Serial Retry {serial_retry_round}/{max_serial_retries} "
                                   f"— {len(serial_candidates)} script(s)\n")

                # Use only the first device for serial retries
                serial_device_ids = [self.device_ids[0]]
                serial_results = _execute_scripts(
                    self.args, serial_candidates, self.target_dir, serial_device_ids,
                    workers=1, timeout=self.timeout, 
                    safe_print=self.safe_print, print_cmd_on_serial=True
                )

                # Update global results map
                for r in serial_results:
                    all_results_map[r["rel_path"]] = r

                new_failures = [r for r in serial_results if r["status"] == "failure"]
                serial_candidates = [r["rel_path"] for r in new_failures]
                current_failure_count = len(serial_candidates)

                if current_failure_count == 0:
                    self.safe_print("✅ All scripts passed after final serial retry loop.\n")
                    break

                if current_failure_count >= prev_failure_count:
                    self.safe_print(f"⚠️  Final serial retry: failure count did not "
                                   f"decrease (was {prev_failure_count}, "
                                   f"now {current_failure_count}). Stopping.\n")
                    break

                prev_failure_count = current_failure_count
                serial_retry_round += 1

            if serial_candidates and serial_retry_round > max_serial_retries:
                self.safe_print(f"🛑 Reached maximum final serial retries "
                               f"({max_serial_retries}). Stopping.\n")

        success_list = [r for r in all_results_map.values() if r["status"] == "success"]
        failure_list = [r for r in all_results_map.values() if r["status"] == "failure"]
        
        self.safe_print("\n🏁 All execution rounds completed.")
        return success_list, failure_list


def _execute_scripts(args, rel_paths, target_dir, device_ids,
                     workers, timeout, safe_print, print_cmd_on_serial=False):
    """Helper to execute a list of scripts with given device pool."""
    if not rel_paths:
        return []

    device_queue = queue.Queue()
    for dev in device_ids:
        device_queue.put(dev)

    with ThreadPoolExecutor(max_workers=workers) as executor:
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
    if args.run_mode == "sim":
        safe_print(f"Workers (sim mode)    : {len(device_ids)}")
    else:
        safe_print(f"DEVICE_IDs (npu mode) : {', '.join(device_ids)}")
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
        description="Execute and validate Python scripts with configurable "
                    "parallel retries and final serial fallback."
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        required=True,
        help="Target: either a .py file path or a directory path."
    )
    parser.add_argument(
        "-r", "--run_mode",
        choices=["npu", "sim"],
        default="npu",
        help="Execution mode: 'npu' (default) or 'sim'. "
             "In 'sim' mode, only scripts supporting --run_mode are executed."
    )
    parser.add_argument(
        "-d", "--device_ids",
        type=str,
        default="0",
        help="Comma-separated list of DEVICE_IDs (e.g., '0,1,2,3'). Default: '0'. "
             "Only effective in 'npu' mode. In 'sim' mode, use --workers instead."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (only effective in 'sim' mode). "
             "In 'npu' mode, parallelism is determined by device count. "
             "Default: auto-detected CPU count (up to 16) in sim mode."
    )
    parser.add_argument(
        "example_id",
        type=str,
        default=None,
        nargs='?',
        help="Optional test identifier (e.g., 'test_add' or "
             "'test_file.py::test_add') to pass to script or pytest."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-script execution timeout in seconds (default: 300)."
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
        default=3,
        help="Maximum number of serial retry rounds in single-device mode (default: 3). "
             "Total serial runs = 1 (initial) + N (retries). Set to 0 to disable retries."
    )
    parser.add_argument(
        "--show-fail-details",
        action="store_true",
        help="Show last 5 lines of output for each failed script in the final summary."
    )
    parser.add_argument(
        "--allow-pytest-auto-detect",
        action="store_true",
        help="Allow automatic detection and execution of pytest-style tests. "
             "Without this flag, a warning will be shown when pytest tests are detected."
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
        print("Error: Some scripts require 'pytest' but it is not installed.", file=sys.stderr)
        print("    Please install pytest with: pip install pytest", file=sys.stderr)
        sys.exit(1)
    
    if need_pytest and not args.allow_pytest_auto_detect:
        print("\n⚠️  WARNING: Some scripts appear to contain pytest-style tests, \
                but --allow-pytest-auto-detect was not set.")
        print("    Add the --allow-pytest-auto-detect flag to explicitly allow pytest execution.")
        print("    Use --allow-pytest-auto-detect to suppress this warning.\n")

    # Handle workers and device configuration based on run mode
    if args.run_mode == "sim":
        # SIM mode: workers parameter determines parallelism
        if args.workers is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            args.workers = min(16, cpu_count)
            print(f"💡 Sim mode: auto-setting workers to {args.workers} (CPU cores: {cpu_count})")
        
        # Create virtual devices for SIM mode (just placeholders for the queue)
        virtual_devices = [str(i) for i in range(args.workers)]
        actual_device_ids = virtual_devices
        print(f"💡 Sim mode: using {args.workers} virtual workers")
        
        # Determine if we should use single or multi-device strategy based on workers
        is_single_device = (args.workers == 1)
    else:
        # NPU mode: device_ids determines parallelism, workers parameter is ignored
        actual_device_ids = device_ids
        print(f"💡 NPU mode: using {len(device_ids)} physical devices")
        
        # Determine if we should use single or multi-device strategy based on physical devices
        is_single_device = (len(device_ids) == 1)

    print(f"Run mode          : {args.run_mode}")
    if args.run_mode == "sim":
        print(f"Workers (sim mode): {args.workers}")
    else:
        print(f"DEVICE_IDs (npu mode): {', '.join(device_ids)}")
    
    if is_single_device:
        print("Execution mode    : Serial (single device)")
    else:
        print(f"Parallel retries  : {args.parallel_retries} "
              f"(total parallel rounds = {args.parallel_retries + 1})")
    if args.example_id:
        print(f"Test selector     : {args.example_id}")
    if not args.allow_pytest_auto_detect:
        print("Pytest auto-detect: Disabled (warning only for pytest-style tests)")
    else:
        print("Pytest auto-detect: Enabled")
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
                             args, target, actual_device_ids, total_time_sec, safe_print)
        sys.exit(0)

    # Create execution strategy
    strategy = SingleDeviceStrategy(args, target_dir, actual_device_ids, args.timeout, safe_print) \
        if is_single_device else MultiDeviceStrategy(args, target_dir, actual_device_ids, 
                                                     args.timeout, safe_print)
    
    # Execute scripts using the selected strategy
    success_list, failure_list = strategy.execute(current_candidates, all_results_map)

    # Generate final summary
    total_time_sec = time.perf_counter() - start_time
    _print_final_summary(success_list, failure_list, skipped_sim_results, 
                         skipped_no_tests_results, args, target, 
                         actual_device_ids, total_time_sec, safe_print)

    sys.exit(1 if len(failure_list) > 0 else 0)


if __name__ == "__main__":
    main()