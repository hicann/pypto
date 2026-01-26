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
    - Skips the entire script by default when @pytest.mark.skip decorators are detected and 
        the script is run via if __name__ == '__main__':. (Can be disabled via configuration.)
    - Identifies pytest-style tests (test* functions or Test* classes) for pytest dispatch
    - Verifies '--run_mode' argument support for simulation mode filtering
    - Automatically skips files with no executable content
- Dual Execution Modes:
    - npu (default): Executes eligible scripts on physical hardware devices with device isolation
    - sim: Filters and runs only scripts that explicitly support '--run_mode' argument 
        using virtual workers
- Script-level skip control:
    - Automatically excludes scripts containing @pytest.mark.skip decorators (enabled by default),
        mimicking pytest's skip behavior at the file level.
        Can be disabled via --no-skip-pytest-mark-skip.
    - Scripts without a '__main__' guard but with pytest-style tests are executed using pytest
        by default. This behavior can be disabled via --no-pytest-auto-detect, which will skip
        such scripts instead.
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
        * Optional skip of serial fallback via --no-serial-fallback flag
- Granular Test Selection: Passes specific test identifiers to scripts or pytest as needed
- Comprehensive Reporting:
    - Real-time emoji-enhanced status indicators (✅/❌/⏭️/⚠️) with device assignment
    - Final categorized summary with success/failure/skip counts
    - Optional failure diagnostics showing last OUTPUT_SNIPPET_LINES lines of output
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

    # 8. Include scripts marked with @pytest.mark.skip (override default behavior)
    python3 examples/validate_examples.py -t examples -d 0 --no-skip-pytest-mark-skip

    # 9. Disable pytest auto-detection (skip scripts without __main__ guard)
    python3 examples/validate_examples.py -t examples -d 0 --no-pytest-auto-detect
    
    # 10. Skip serial fallback in multi-device mode (only parallel retries)
    python3 examples/validate_examples.py -t examples -d 0,1,2,3 --no-serial-fallback

    # 11. Full configuration
    python3 examples/validate_examples.py -t examples -d 0,1,2,3
        --parallel_retries 2 --serial_retries 5 --timeout 300
        --show-fail-details

Note: This tool is designed specifically for CANN-based development workflows. In npu mode, device
parallelism is determined by provided device IDs. In sim mode, parallelism is controlled by the
--workers parameter which creates virtual device slots.
"""
import argparse
import ast
import functools
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Tuple
import psutil


# =============================================================================
# Module-level Constants (Eliminate Magic Numbers)
# =============================================================================
SHUTDOWN_CHECK_INTERVAL: float = 0.5  # Interval to check for shutdown requests (seconds)
PROCESS_TERMINATE_TIMEOUT: int = 3    # Time to wait for process to terminate gracefully (seconds)
CHILD_PROCESS_WAIT_TIMEOUT: int = 3   # Time to wait for child processes to terminate (seconds)
FUTURE_RESULT_BUFFER: int = 30        # Extra buffer time for future results beyond script timeout (seconds)
DEFAULT_SCRIPT_TIMEOUT: int = 300     # Default per-script execution timeout (seconds)
DEFAULT_PARALLEL_RETRIES: int = 1     # Default number of parallel retry rounds
DEFAULT_SERIAL_RETRIES: int = 3       # Default number of serial retry rounds
DEFAULT_SIM_WORKERS: int = 16         # Maximum default number of workers in simulation mode
OUTPUT_SNIPPET_LINES: int = 5         # Number of lines to show in output snippets

# Environment variable names for device configuration (NPU mode)
ENV_TILE_FWK_DEVICE_ID: str = "TILE_FWK_DEVICE_ID"
ENV_ASCEND_VISIBLE_DEVICES: str = "ASCEND_VISIBLE_DEVICES"
ENV_TILE_FWK_STEST_DEVICE_ID: str = "TILE_FWK_STEST_DEVICE_ID"


class ExecutionStatus(Enum):
    """Enumeration of possible script execution statuses."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED_NO_TESTS = "skipped_no_tests"
    SKIPPED_SIM = "skipped_sim"
    SKIPPED_PYTEST_MARK = "skipped_pytest_mark"
    SKIPPED_PYTEST_DISABLED = "skipped_pytest_disabled"


@dataclass
class ExecutionResult:
    """Unified result type for script execution.

    This dataclass provides a consistent structure for all execution results,
    regardless of the outcome (success, failure, skip, or cancellation).
    """
    rel_path: str
    status: ExecutionStatus
    device_id: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    output_snippet: Optional[str] = None

    @classmethod
    def success(cls, rel_path: str, device_id: Optional[str] = None) -> "ExecutionResult":
        """Create a success result."""
        return cls(rel_path=rel_path, status=ExecutionStatus.SUCCESS, device_id=device_id)

    @classmethod
    def failure(cls, rel_path: str, reason: str, device_id: Optional[str] = None,
                output_snippet: Optional[str] = None) -> "ExecutionResult":
        """Create a failure result."""
        return cls(rel_path=rel_path, status=ExecutionStatus.FAILURE,
                   reason=reason, device_id=device_id, output_snippet=output_snippet)

    @classmethod
    def cancelled(cls, rel_path: str, message: str) -> "ExecutionResult":
        """Create a cancelled result."""
        return cls(rel_path=rel_path, status=ExecutionStatus.CANCELLED, message=message)

    @classmethod
    def skipped(cls, rel_path: str, status: ExecutionStatus, message: str) -> "ExecutionResult":
        """Create a skipped result."""
        return cls(rel_path=rel_path, status=status, message=message)


@dataclass
class ExecutionContext:
    """Encapsulates execution parameters to reduce function argument count.

    This context object bundles together the parameters needed for script execution,
    providing a cleaner API for run_script and related functions.
    """
    args: argparse.Namespace
    device_queue: "queue.Queue[str]"
    timeout: int
    process_manager: "ProcessManager"
    safe_print: Callable[..., None]
    print_cmd_on_serial: bool = False
    estimated_queue_depth: int = 1


@dataclass
class SummaryData:
    """Encapsulates data for the final execution summary.

    This dataclass bundles together all the information needed to print
    the final execution summary, reducing the parameter count of _print_final_summary.
    """
    success_list: List["ExecutionResult"]
    failure_list: List["ExecutionResult"]
    skipped_sim_list: List["ExecutionResult"]
    skipped_no_tests_list: List["ExecutionResult"]
    skipped_pytest_mark_list: List["ExecutionResult"]
    skipped_pytest_disabled_list: List["ExecutionResult"]
    args: argparse.Namespace
    target: Path
    device_ids: List[str]
    total_time_sec: float
    safe_print: Callable[..., None]

    @property
    def total_original(self) -> int:
        """Calculate total number of scripts found."""
        return (len(self.success_list) + len(self.failure_list) +
                len(self.skipped_sim_list) + len(self.skipped_no_tests_list) +
                len(self.skipped_pytest_disabled_list))


@dataclass
class CollectionParams:
    """Encapsulates parameters for result collection to reduce function argument count.

    This dataclass bundles together all parameters needed for collecting and formatting
    execution results, adhering to the guideline of limiting function parameters.
    """
    proc: subprocess.Popen
    stdout: Optional[str]
    stderr: Optional[str]
    rel_path: str
    device_id: str
    safe_print: Callable[..., None]


# =============================================================================
# Process Manager - Encapsulated Process Tracking and Graceful Shutdown
# =============================================================================
class ProcessManager:
    """Manages process tracking and graceful shutdown for concurrent script execution.

    This class encapsulates all process-related state and operations, providing:
    - Thread-safe process registration and unregistration
    - Cooperative shutdown via signal handling
    - Process tree termination for reliable cleanup

    Design principles:
    1. Signal handlers only set flags (async-signal-safe)
    2. Actual cleanup happens in regular code paths
    3. Atomic process registration to avoid race conditions
    4. Cooperative shutdown via periodic flag checking

    Usage:
        manager = ProcessManager()
        manager.setup_signal_handlers()
        # ... use manager throughout execution ...
        manager.cleanup_all()
    """

    def __init__(self) -> None:
        """Initialize the process manager with empty state."""
        self._active_processes: List[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._print_lock = threading.Lock()
        self._safe_print: Optional[Callable[..., None]] = None

    def setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        Should be called early, before any child processes are created.
        """
        if os.name != 'nt':  # Unix-like systems
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            signal.signal(signal.SIGINT, self._signal_handler)

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def create_safe_print(self) -> Callable[..., None]:
        """Create and return a thread-safe print function."""
        def safe_print(*args, **kwargs):
            with self._print_lock:
                print(*args, **kwargs)
        self._safe_print = safe_print
        return safe_print

    def create_and_register_process(
        self, cmd: List[str], popen_kwargs: Dict[str, Any]
    ) -> Tuple[Optional[subprocess.Popen], bool]:
        """Atomically create a process and register it for tracking.

        This method holds the lock during process creation, eliminating the race
        condition window between process creation and registration.

        Args:
            cmd: Command to execute
            popen_kwargs: Keyword arguments for subprocess.Popen

        Returns:
            Tuple of (process, success):
            - (proc, True) if process was created and registered successfully
            - (None, False) if shutdown was requested before creation
            - (proc, False) if process was created but shutdown was requested during creation
              (caller should terminate the process)
        """
        with self._lock:
            if self._shutdown_event.is_set():
                return None, False
            proc = subprocess.Popen(cmd, **popen_kwargs)
            # Check again after creation - if shutdown was requested during Popen,
            # we still need to track the process but signal failure
            if self._shutdown_event.is_set():
                # Process created but shutdown requested - return process for cleanup
                # but don't register (caller will terminate it)
                return proc, False
            self._active_processes.append(proc)
            return proc, True

    def unregister_process(self, proc: subprocess.Popen) -> None:
        """Unregister a process from tracking."""
        with self._lock:
            if proc in self._active_processes:
                self._active_processes.remove(proc)

    def cleanup_all(self) -> None:
        """Clean up all tracked active processes.

        Should be called from regular code paths (not from signal handlers).
        """
        with self._lock:
            procs = self._active_processes.copy()

        if not procs:
            return

        if self._safe_print:
            self._safe_print(f"\n⚠️  Shutdown requested. Cleaning up {len(procs)} active process(es)...")

        for proc in procs:
            try:
                if proc.poll() is None:
                    terminate_process_tree(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
                if self._safe_print:
                    self._safe_print(f"Warning: Error cleaning up process {proc.pid}: {e}")

    def wait_for_process(self, proc: subprocess.Popen, timeout: int,
                         check_interval: float = SHUTDOWN_CHECK_INTERVAL
                         ) -> Tuple[Optional[str], Optional[str], bool]:
        """Wait for process completion while periodically checking for shutdown.

        Args:
            proc: The subprocess to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check for shutdown (seconds)

        Returns:
            Tuple of (stdout, stderr, was_shutdown_requested)
        """
        start_time = time.perf_counter()
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        while True:
            if self._shutdown_event.is_set():
                return None, None, True

            if proc.poll() is not None:
                try:
                    remaining_stdout, remaining_stderr = proc.communicate(timeout=1)
                    stdout_parts.append(remaining_stdout or "")
                    stderr_parts.append(remaining_stderr or "")
                except (subprocess.TimeoutExpired, OSError):
                    pass
                return "".join(stdout_parts), "".join(stderr_parts), False

            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                raise subprocess.TimeoutExpired(proc.args, timeout)

            remaining_time = min(check_interval, timeout - elapsed)
            if remaining_time > 0:
                time.sleep(remaining_time)

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        """Signal handler that only sets the shutdown flag (async-signal-safe)."""
        self._shutdown_event.set()


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and all its child processes.

    Args:
        proc: The subprocess.Popen object to terminate
    """
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
        _, still_alive = psutil.wait_procs(children, timeout=CHILD_PROCESS_WAIT_TIMEOUT)

        # Force-kill remaining processes
        for child in still_alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent process
        proc.terminate()
        try:
            proc.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()

    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
        # Process may have already exited or we lack permissions
        # Try direct kill as a fallback
        try:
            proc.kill()
        except OSError:
            # Process already gone or inaccessible, nothing more to do
            pass


# =============================================================================
# AST Analysis Functions
# =============================================================================
@dataclass
class ScriptAnalysis:
    """Result of analyzing a Python script's AST.

    This dataclass consolidates all information extracted from a single AST parse,
    eliminating the need for multiple parsing passes over the same file.
    """
    has_main_guard: bool = False
    has_pytest_tests: bool = False
    has_pytest_skip_mark: bool = False
    supports_run_mode: bool = False


def _parse_ast(file_path: Path, context: str = "") -> Optional[ast.Module]:
    """Parse a Python file and return its AST, or None if parsing fails.

    This helper function centralizes file reading and AST parsing logic,
    eliminating code duplication across AST analysis functions.

    Args:
        file_path: Path to the Python file to parse
        context: Context string for error messages (e.g., "main guard check")

    Returns:
        Parsed AST module, or None if parsing fails due to:
        - Empty file
        - Syntax errors
        - Encoding issues
        - File not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            return None

        return ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        # Log syntax errors since they likely indicate a real issue
        context_msg = f" for {context}" if context else ""
        print(f"Warning: Syntax error parsing {file_path}{context_msg}: {e}", file=sys.stderr)
        return None
    except (UnicodeDecodeError, OSError) as e:
        # Log file access errors for debugging (user may not be aware of permission issues)
        context_msg = f" for {context}" if context else ""
        print(f"Warning: Cannot read {file_path}{context_msg}: {e}", file=sys.stderr)
        return None
    except (ValueError, TypeError) as e:
        # Unexpected but non-fatal errors - log and return None
        context_msg = f" for {context}" if context else ""
        print(f"Warning: Error parsing {file_path} with AST{context_msg}: {e}", file=sys.stderr)
        return None


@functools.lru_cache(maxsize=1024)
def _analyze_script_cached(file_path_str: str) -> ScriptAnalysis:
    """Internal cached implementation of script analysis.
    """
    file_path = Path(file_path_str)
    result = ScriptAnalysis()
    tree = _parse_ast(file_path, "script analysis")
    if tree is None:
        return result

    for node in ast.walk(tree):
        # Check for __name__ == '__main__' guard
        if isinstance(node, ast.If):
            if (isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__' and
                len(node.test.ops) == 1 and
                isinstance(node.test.ops[0], ast.Eq) and
                len(node.test.comparators) == 1 and
                isinstance(node.test.comparators[0], ast.Constant) and
                node.test.comparators[0].value == '__main__'):
                result.has_main_guard = True

        # Check for pytest-style test functions and classes
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('test'):
                result.has_pytest_tests = True

            # Check for @pytest.mark.skip decorator
            for decorator in node.decorator_list:
                # Check for @pytest.mark.skip(...)
                if (isinstance(decorator, ast.Call) and
                    isinstance(decorator.func, ast.Attribute) and
                    isinstance(decorator.func.value, ast.Attribute) and
                    decorator.func.value.attr == 'mark' and
                    isinstance(decorator.func.value.value, ast.Name) and
                    decorator.func.value.value.id == 'pytest' and
                    decorator.func.attr == 'skip'):
                    result.has_pytest_skip_mark = True
                # Check for @pytest.mark.skip (without call)
                if (isinstance(decorator, ast.Attribute) and
                    isinstance(decorator.value, ast.Attribute) and
                    decorator.value.attr == 'mark' and
                    isinstance(decorator.value.value, ast.Name) and
                    decorator.value.value.id == 'pytest' and
                    decorator.attr == 'skip'):
                    result.has_pytest_skip_mark = True

        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            result.has_pytest_tests = True

        # Check for --run_mode argument support
        if (isinstance(node, ast.Call) and
            hasattr(node.func, 'attr') and node.func.attr == 'add_argument'):
            for arg in node.args:
                if (isinstance(arg, ast.Constant) and
                    isinstance(arg.value, str) and
                    '--run_mode' in arg.value):
                    result.supports_run_mode = True

            for keyword in node.keywords:
                if keyword.arg == 'dest' and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value == 'run_mode':
                        result.supports_run_mode = True

    return result


def analyze_script(file_path: Path) -> ScriptAnalysis:
    """Analyze a Python script and extract all relevant information in a single pass.

    This function parses the AST once and extracts all needed information,
    avoiding the overhead of multiple parse operations for the same file.
    Results are cached using functools.lru_cache for thread-safe memoization.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        ScriptAnalysis with all extracted information
    """
    return _analyze_script_cached(str(file_path))


def get_script_analysis(file_path: Path) -> ScriptAnalysis:
    """Get cached script analysis.

    This function resolves the path and delegates to the lru_cache-decorated
    analyze_script function, which handles thread-safe caching automatically.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        ScriptAnalysis with all extracted information
    """
    return _analyze_script_cached(str(file_path.resolve()))


# =============================================================================
# Script Execution Helper Functions (Modular Design)
# =============================================================================
def _check_skip_conditions(
    ctx: ExecutionContext, analysis: ScriptAnalysis, rel_path: str
) -> Optional[ExecutionResult]:
    """Check if script should be skipped based on analysis results.

    Note: Skip conditions are checked in two places by design:
    1. In main() during initial categorization - for upfront filtering and summary stats
    2. Here in run_script() - as a safety check for scripts that may have been
       added to the candidate list incorrectly or for defensive programming

    The checks in main() prevent unnecessary device queue contention by filtering
    early. This function provides a defensive second check and generates proper
    skip results with logging.

    Args:
        ctx: Execution context containing args and safe_print
        analysis: ScriptAnalysis result
        rel_path: Relative path for display

    Returns:
        ExecutionResult if script should be skipped, None otherwise
    """
    if not analysis.has_main_guard and not analysis.has_pytest_tests:
        ctx.safe_print(f"⏭️  Skipped: {rel_path} (no '__main__' guard and no pytest-style tests)")
        return ExecutionResult.skipped(
            rel_path, ExecutionStatus.SKIPPED_NO_TESTS,
            "no '__main__' guard and no pytest-style tests"
        )

    if not analysis.has_main_guard and analysis.has_pytest_tests and not ctx.args.pytest_auto_detect:
        ctx.safe_print(f"⏭️  Skipped: {rel_path} (pytest auto-detect disabled, no '__main__' guard)")
        return ExecutionResult.skipped(
            rel_path, ExecutionStatus.SKIPPED_PYTEST_DISABLED,
            "pytest auto-detect disabled, script has no '__main__' guard"
        )

    if ctx.args.skip_pytest_mark_skip and analysis.has_pytest_skip_mark:
        ctx.safe_print(f"⏭️  Skipped: {rel_path} (contains @pytest.mark.skip)")
        return ExecutionResult.skipped(
            rel_path, ExecutionStatus.SKIPPED_PYTEST_MARK,
            "contains @pytest.mark.skip decorator"
        )

    if ctx.args.run_mode == "sim" and not analysis.supports_run_mode:
        ctx.safe_print(f"⏭️  Skipped: {rel_path} (script does not support --run_mode)")
        return ExecutionResult.skipped(
            rel_path, ExecutionStatus.SKIPPED_SIM,
            "script does not support --run_mode"
        )

    return None


def _acquire_device(
    ctx: ExecutionContext, rel_path: str
) -> Tuple[Optional[str], Optional[ExecutionResult]]:
    """Acquire a device from the queue with timeout and shutdown checks.

    Args:
        ctx: Execution context containing device_queue, timeout, etc.
        rel_path: Relative path for display

    Returns:
        Tuple of (device_id, error_result):
        - (device_id, None) on success
        - (None, ExecutionResult) on failure or cancellation
    """
    device_acquisition_timeout = (ctx.estimated_queue_depth * ctx.timeout) + FUTURE_RESULT_BUFFER
    device_wait_start = time.perf_counter()

    while True:
        if ctx.process_manager.is_shutdown_requested():
            return None, ExecutionResult.cancelled(rel_path, "Shutdown requested during device acquisition")
        try:
            device_id = ctx.device_queue.get(timeout=SHUTDOWN_CHECK_INTERVAL)
            return device_id, None
        except queue.Empty:
            elapsed = time.perf_counter() - device_wait_start
            if elapsed >= device_acquisition_timeout:
                ctx.safe_print(f"❌ Failure: {rel_path} (device acquisition timeout)")
                return None, ExecutionResult.failure(
                    rel_path, f"Could not acquire a device within {device_acquisition_timeout}s",
                    device_id=None, output_snippet=""
                )


def _build_command(
    args, analysis: ScriptAnalysis, full_path: Path, device_id: str
) -> Tuple[List[str], Dict[str, str]]:
    """Build the command and environment for script execution.

    This function centralizes all command and environment configuration,
    including device-specific environment variables for NPU mode.

    Args:
        args: Parsed command-line arguments
        analysis: ScriptAnalysis result
        full_path: Absolute path to the script
        device_id: Device ID for environment setup (used in NPU mode)

    Returns:
        Tuple of (command_list, environment_dict)
    """
    env = os.environ.copy()

    # Set device-specific environment variables in NPU mode
    if args.run_mode != "sim":
        env[ENV_TILE_FWK_DEVICE_ID] = device_id
        env[ENV_ASCEND_VISIBLE_DEVICES] = device_id
        env[ENV_TILE_FWK_STEST_DEVICE_ID] = device_id

    if analysis.has_main_guard:
        cmd = [sys.executable, str(full_path)]
        if args.example_id:
            cmd.append(args.example_id)
        if args.run_mode == "sim":
            cmd.extend(["--run_mode", "sim"])
    else:
        # Execute pytest-style tests
        if args.example_id:
            cmd = ["pytest", f"{full_path}::{args.example_id}", "-v", "--capture=no"]
        else:
            cmd = ["pytest", str(full_path), "-v", "--capture=no"]

    return cmd, env


def _execute_process(
    cmd: List[str],
    env: Dict[str, str],
    device_id: str,
    ctx: ExecutionContext,
    rel_path: str
) -> Tuple[Optional[subprocess.Popen], Optional[str], Optional[str], Optional[ExecutionResult]]:
    """Execute the process and wait for completion.

    Args:
        cmd: Command to execute
        env: Environment variables (already configured with device settings)
        device_id: Device ID (for error reporting)
        ctx: Execution context
        rel_path: Relative path for display

    Returns:
        Tuple of (proc, stdout, stderr, error_result):
        - (proc, stdout, stderr, None) on successful execution (may have non-zero exit)
        - (proc, None, None, ExecutionResult) on error or cancellation
        - (None, None, None, ExecutionResult) if process couldn't be created
    """
    if ctx.print_cmd_on_serial:
        cmd_str = " ".join(str(part) for part in cmd)
        ctx.safe_print(f"→  Executing: {cmd_str}")

    # Final shutdown check before creating process
    if ctx.process_manager.is_shutdown_requested():
        return None, None, None, ExecutionResult.cancelled(
            rel_path, "Shutdown requested before process creation"
        )

    # Create popen kwargs with process group settings
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
        "text": True,
    }
    if os.name != 'nt':
        if sys.version_info >= (3, 11):
            popen_kwargs["process_group"] = 0
        else:
            popen_kwargs["start_new_session"] = True

    # Atomically create and register process
    proc, registered = ctx.process_manager.create_and_register_process(cmd, popen_kwargs)
    if proc is None:
        return None, None, None, ExecutionResult.cancelled(
            rel_path, "Shutdown requested before process creation"
        )
    if not registered:
        terminate_process_tree(proc)
        return None, None, None, ExecutionResult.cancelled(
            rel_path, "Shutdown requested during process creation"
        )

    try:
        stdout, stderr, was_shutdown = ctx.process_manager.wait_for_process(
            proc, ctx.timeout, check_interval=SHUTDOWN_CHECK_INTERVAL
        )

        if was_shutdown:
            terminate_process_tree(proc)
            ctx.safe_print(f"🛑 Cancelled: {rel_path} (shutdown requested)")
            return proc, None, None, ExecutionResult.cancelled(
                rel_path, "Shutdown requested during execution"
            )

        return proc, stdout, stderr, None

    except subprocess.TimeoutExpired:
        # Try to capture any buffered output before terminating
        timeout_output = ""
        try:
            # Read available output without blocking (using poll)
            if proc.stdout:
                # Set a short timeout to avoid hanging
                remaining_stdout, remaining_stderr = proc.communicate(timeout=1)
                timeout_output = (remaining_stdout or "") + (remaining_stderr or "")
        except (subprocess.TimeoutExpired, OSError):
            # Process didn't respond, continue with termination
            pass
        terminate_process_tree(proc)
        ctx.safe_print(f"❌ Failure: {rel_path}")
        snippet = _extract_output_snippet(timeout_output) if timeout_output else ""
        return proc, None, None, ExecutionResult.failure(
            rel_path, f"Timeout (exceeded {ctx.timeout}s)",
            device_id=device_id, output_snippet=snippet
        )

    except OSError as e:
        if proc:
            terminate_process_tree(proc)
        ctx.safe_print(f"❌ Failure: {rel_path}")
        return proc, None, None, ExecutionResult.failure(
            rel_path, f"Exception during execution: {e}",
            device_id=device_id, output_snippet=""
        )


def _collect_result(params: CollectionParams) -> ExecutionResult:
    """Collect and format the execution result.

    Args:
        params: CollectionParams containing process, output, and context information

    Returns:
        ExecutionResult with success or failure status
    """
    output = (params.stdout or "") + (params.stderr or "")
    snippet = _extract_output_snippet(output)

    if params.proc.returncode == 0:
        params.safe_print(f"✅ Success: {params.rel_path}")
        return ExecutionResult.success(params.rel_path, params.device_id)
    else:
        params.safe_print(f"❌ Failure: {params.rel_path}")
        return ExecutionResult.failure(
            params.rel_path, f"Non-zero exit code ({params.proc.returncode})",
            device_id=params.device_id, output_snippet=snippet
        )


def run_script(ctx: ExecutionContext, full_path: Path, rel_path: str) -> ExecutionResult:
    """Execute a single script by leasing a device from the device queue.

    This function orchestrates the script execution pipeline:
    1. Analyze: Check script properties and skip conditions
    2. Acquire Device: Lease a device from the queue
    3. Execute: Run the script process
    4. Collect: Gather and format execution results

    Args:
        ctx: Execution context containing all execution parameters
        full_path: Absolute path to the script
        rel_path: Relative path for display purposes

    Returns:
        ExecutionResult with the outcome of the script execution
    """
    # Phase 1: Check for early shutdown
    if ctx.process_manager.is_shutdown_requested():
        return ExecutionResult.cancelled(rel_path, "Shutdown requested before execution")

    # Phase 2: Analyze - Check skip conditions
    analysis = get_script_analysis(full_path)
    skip_result = _check_skip_conditions(ctx, analysis, rel_path)
    if skip_result is not None:
        return skip_result

    # Phase 3: Acquire Device
    if ctx.process_manager.is_shutdown_requested():
        return ExecutionResult.cancelled(rel_path, "Shutdown requested before device acquisition")

    device_id, error_result = _acquire_device(ctx, rel_path)
    if error_result is not None:
        return error_result

    # Print start message
    if ctx.args.run_mode == "sim":
        ctx.safe_print(f"▶️  Starting: {rel_path}")
    else:
        ctx.safe_print(f"▶️  Starting: {rel_path} (device={device_id})")

    proc = None
    try:
        # Phase 4: Build Command (includes environment configuration)
        cmd, env = _build_command(ctx.args, analysis, full_path, device_id)

        # Phase 5: Execute Process
        proc, stdout, stderr, exec_error = _execute_process(
            cmd, env, device_id, ctx, rel_path
        )
        if exec_error is not None:
            return exec_error

        # Phase 6: Collect Result
        collection_params = CollectionParams(
            proc=proc,
            stdout=stdout,
            stderr=stderr,
            rel_path=rel_path,
            device_id=device_id,
            safe_print=ctx.safe_print
        )
        return _collect_result(collection_params)

    finally:
        # Cleanup: Unregister process and return device to queue
        if proc:
            ctx.process_manager.unregister_process(proc)
            if proc.poll() is None:
                terminate_process_tree(proc)
        # Only return device to queue if we successfully acquired one
        # (prevents None from polluting the device pool)
        if device_id is not None:
            ctx.device_queue.put(device_id)
        ctx.safe_print("-" * 50)


def _extract_output_snippet(output: str) -> str:
    """Extract the last N lines from output for error reporting."""
    if not output.strip():
        return ""
    return "\n".join(output.strip().splitlines()[-OUTPUT_SNIPPET_LINES:])


# =============================================================================
# Execution Strategy Classes (Strategy Pattern with ABC)
# =============================================================================
class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies.

    This class defines the interface for script execution strategies and provides
    common functionality for retry logic to avoid code duplication.
    """

    def __init__(self, args, target_dir: Path, device_ids: List[str],
                 timeout: int, safe_print: Callable[..., None],
                 process_manager: ProcessManager) -> None:
        self.args = args
        self.target_dir = target_dir
        self.device_ids = device_ids
        self.timeout = timeout
        self.safe_print = safe_print
        self.process_manager = process_manager

    @abstractmethod
    def execute(self, scripts: List[str],
                all_results_map: Dict[str, ExecutionResult]) -> Tuple[List[ExecutionResult], List[ExecutionResult]]:
        """Execute scripts and return success and failure lists."""
        pass

    def _run_serial_retry_loop(
        self,
        candidates: List[str],
        all_results_map: Dict[str, ExecutionResult],
        max_retries: int,
        device_ids: List[str],
        log_prefix: str = ""
    ) -> List[str]:
        """Common serial retry loop logic used by both strategies.

        Args:
            candidates: List of script paths to execute
            all_results_map: Dictionary to store results
            max_retries: Maximum number of retry rounds
            device_ids: Device IDs to use for execution
            log_prefix: Prefix for log messages (e.g., "Final " for multi-device)

        Returns:
            List of remaining failed script paths
        """
        current_candidates = candidates[:]
        prev_failure_count = len(current_candidates)
        retry_round = 0

        while current_candidates and retry_round <= max_retries:
            if self.process_manager.is_shutdown_requested():
                self.safe_print(f"🛑 Shutdown requested. Stopping {log_prefix.lower()}execution.\n")
                break

            if retry_round == 0:
                self.safe_print(f"▶️  {log_prefix}Serial Run — {len(current_candidates)} script(s)\n")
            else:
                self.safe_print(f"🔁 {log_prefix}Serial Retry {retry_round}/{max_retries} "
                               f"— {len(current_candidates)} script(s)\n")

            results = execute_scripts(
                self.args, current_candidates, self.target_dir, device_ids,
                workers=1, timeout=self.timeout, safe_print=self.safe_print,
                process_manager=self.process_manager, print_cmd_on_serial=True
            )

            for r in results:
                all_results_map[r.rel_path] = r

            if self.process_manager.is_shutdown_requested():
                self.safe_print(f"🛑 Shutdown requested. Stopping {log_prefix.lower()}retries.\n")
                break

            new_failures = [r for r in results if r.status == ExecutionStatus.FAILURE]
            current_candidates = [r.rel_path for r in new_failures]
            current_failure_count = len(current_candidates)

            if current_failure_count == 0:
                suffix = ' after ' + log_prefix.lower() + 'retry loop' if log_prefix else ''
                self.safe_print(f"✅ All scripts passed{suffix}.\n")
                break

            if current_failure_count >= prev_failure_count:
                self.safe_print(f"⚠️  {log_prefix}Failure count did not decrease "
                               f"(was {prev_failure_count}, now {current_failure_count}). Stopping retries.\n")
                break

            prev_failure_count = current_failure_count
            retry_round += 1

        if current_candidates and retry_round > max_retries and not self.process_manager.is_shutdown_requested():
            self.safe_print(f"🛑 Reached maximum {log_prefix.lower()}serial retries ({max_retries}). Stopping.\n")

        return current_candidates

    def _collect_final_results(
        self,
        all_results_map: Dict[str, ExecutionResult]
    ) -> Tuple[List[ExecutionResult], List[ExecutionResult]]:
        """Collect success and failure lists from results map."""
        success_list = [r for r in all_results_map.values()
                        if r.status == ExecutionStatus.SUCCESS]
        failure_list = [r for r in all_results_map.values()
                        if r.status == ExecutionStatus.FAILURE]
        return success_list, failure_list


class SingleDeviceStrategy(ExecutionStrategy):
    """Execution strategy for a single device (serial execution)."""

    def execute(self, scripts: List[str],
                all_results_map: Dict[str, ExecutionResult]) -> Tuple[List[ExecutionResult], List[ExecutionResult]]:
        max_serial_retries = max(0, self.args.serial_retries)
        self._run_serial_retry_loop(
            scripts, all_results_map, max_serial_retries, self.device_ids
        )
        return self._collect_final_results(all_results_map)


class MultiDeviceStrategy(ExecutionStrategy):
    """Execution strategy for multiple devices (parallel execution with serial fallback)."""

    def execute(self, scripts: List[str],
                all_results_map: Dict[str, ExecutionResult]) -> Tuple[List[ExecutionResult], List[ExecutionResult]]:
        current_candidates = scripts[:]
        parallel_retries = max(0, self.args.parallel_retries)
        actual_workers = len(self.device_ids)

        # Parallel execution rounds
        for round_idx in range(parallel_retries + 1):
            if self.process_manager.is_shutdown_requested():
                self.safe_print("🛑 Shutdown requested. Stopping parallel execution.\n")
                break

            round_name = "Initial" if round_idx == 0 else f"Retry {round_idx}"
            self.safe_print(f"🚀 Starting Parallel Round {round_idx + 1}/{parallel_retries + 1} "
                           f"({round_name}) — {len(current_candidates)} script(s)\n")

            round_results = execute_scripts(
                self.args, current_candidates, self.target_dir, self.device_ids,
                workers=actual_workers, timeout=self.timeout, safe_print=self.safe_print,
                process_manager=self.process_manager, print_cmd_on_serial=False
            )

            for r in round_results:
                all_results_map[r.rel_path] = r

            if self.process_manager.is_shutdown_requested():
                self.safe_print("🛑 Shutdown requested. Stopping retries.\n")
                break

            round_failures = [r for r in round_results if r.status == ExecutionStatus.FAILURE]
            if not round_failures:
                self.safe_print(f"✅ All scripts passed in Parallel Round {round_idx + 1}. "
                               f"No further retries needed.")
                return self._collect_final_results(all_results_map)

            current_candidates = [r.rel_path for r in round_failures]
            self.safe_print(f"🔁 {len(current_candidates)} script(s) failed and will be retried.")

        # Final serial retry loop for remaining failures
        if current_candidates and not self.process_manager.is_shutdown_requested():
            if self.args.no_serial_fallback:
                self.safe_print(f"⏭️  Skipping serial fallback (--no-serial-fallback enabled). "
                               f"{len(current_candidates)} script(s) remain failed.\n")
            else:
                self.safe_print(f"🔂 Starting Final Serial Retry Loop — {len(current_candidates)} "
                               f"remaining failed script(s)\n")
                max_serial_retries = max(0, self.args.serial_retries)
                serial_device_ids = [self.device_ids[0]]
                self._run_serial_retry_loop(
                    current_candidates, all_results_map, max_serial_retries,
                    serial_device_ids, log_prefix="Final "
                )

        if not self.process_manager.is_shutdown_requested():
            self.safe_print("\n🏁 All execution rounds completed.")

        return self._collect_final_results(all_results_map)


def execute_scripts(
    args, rel_paths: List[str], target_dir: Path, device_ids: List[str],
    workers: int, timeout: int, safe_print: Callable[..., None],
    process_manager: ProcessManager, print_cmd_on_serial: bool = False
) -> List[ExecutionResult]:
    """Execute a list of scripts with given device pool.

    This function supports graceful shutdown and properly waits for all futures.

    Args:
        args: Parsed command-line arguments
        rel_paths: List of relative script paths
        target_dir: Base directory for scripts
        device_ids: List of device IDs to use
        workers: Number of parallel workers
        timeout: Per-script timeout
        safe_print: Thread-safe print function
        process_manager: ProcessManager for process tracking
        print_cmd_on_serial: Whether to print commands in serial mode

    Returns:
        List of ExecutionResult objects
    """
    if not rel_paths:
        return []

    if process_manager.is_shutdown_requested():
        return [ExecutionResult.cancelled(p, "Shutdown requested")
                for p in rel_paths]

    device_queue: queue.Queue = queue.Queue()
    for dev in device_ids:
        device_queue.put(dev)

    # Calculate estimated queue depth for device acquisition timeout
    # This represents how many scripts each device is expected to handle.
    # NOTE: This is an average-based estimate that may be inaccurate when script
    # execution times vary significantly. Fast scripts may wait longer than needed,
    # while slow scripts may timeout prematurely. Consider adjusting --timeout
    # if experiencing unexpected timeouts with mixed script durations.
    estimated_queue_depth = math.ceil(len(rel_paths) / len(device_ids)) if device_ids else 1

    # Create execution context (shared across all script executions)
    ctx = ExecutionContext(
        args=args,
        device_queue=device_queue,
        timeout=timeout,
        process_manager=process_manager,
        safe_print=safe_print,
        print_cmd_on_serial=print_cmd_on_serial,
        estimated_queue_depth=estimated_queue_depth
    )

    # Pre-build path order lookup for O(1) sorting later
    path_order = {p: i for i, p in enumerate(rel_paths)}

    results: List[ExecutionResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_rel = {}
        for rel_path in rel_paths:
            if process_manager.is_shutdown_requested():
                results.append(
                    ExecutionResult.cancelled(rel_path, "Shutdown requested before submission")
                )
                continue

            full_path = target_dir / rel_path
            future = executor.submit(run_script, ctx, full_path, rel_path)
            future_to_rel[future] = rel_path

        # Collect results - wait for each future properly
        # Use queue depth-aware timeout for Future.result()
        future_timeout = (estimated_queue_depth * timeout) + FUTURE_RESULT_BUFFER
        for future in as_completed(future_to_rel):
            rel_path = future_to_rel[future]
            try:
                result = future.result(timeout=future_timeout)
                results.append(result)
            except TimeoutError:
                # This should rarely happen as run_script has its own timeout
                results.append(
                    ExecutionResult.failure(rel_path, "Future result timeout")
                )
            except Exception as e:
                results.append(
                    ExecutionResult.failure(rel_path, f"Unexpected error: {e}")
                )

        if process_manager.is_shutdown_requested():
            process_manager.cleanup_all()

    # Sort results using O(1) dictionary lookup instead of O(n) list.index()
    results.sort(key=lambda x: path_order.get(x.rel_path, float('inf')))
    return results


def _print_final_summary(summary: SummaryData) -> None:
    """Print the final execution summary.

    Args:
        summary: SummaryData containing all information for the summary
    """
    safe_print = summary.safe_print
    args = summary.args

    safe_print("\n" + "=" * 60)
    safe_print("📊 FINAL EXECUTION SUMMARY")
    safe_print("=" * 60)
    safe_print(f"Target directory/file : {summary.target}")
    safe_print(f"Run mode              : {args.run_mode}")
    if args.run_mode == "sim":
        safe_print(f"Workers (sim mode)    : {len(summary.device_ids)}")
    else:
        safe_print(f"DEVICE_IDs (npu mode) : {', '.join(summary.device_ids)}")
    safe_print(f"Total scripts found   : {summary.total_original}")
    safe_print(f"Total execution time  : {summary.total_time_sec:.2f} seconds")
    safe_print(f"Scripts executed      : {len(summary.success_list) + len(summary.failure_list)}")
    safe_print(f"✅ Successful          : {len(summary.success_list)}")
    safe_print(f"❌ Failed              : {len(summary.failure_list)}")
    if summary.skipped_sim_list:
        safe_print(f"⏭️  Skipped (no sim support): {len(summary.skipped_sim_list)}")
    if summary.skipped_no_tests_list:
        safe_print(f"⏭️  Skipped (no main/test): {len(summary.skipped_no_tests_list)}")
    if summary.skipped_pytest_mark_list:
        safe_print(f"⏭️  Skipped (@pytest.mark.skip): {len(summary.skipped_pytest_mark_list)}")
    if summary.skipped_pytest_disabled_list:
        safe_print(f"⏭️  Skipped (pytest disabled): {len(summary.skipped_pytest_disabled_list)}")

    if summary.failure_list:
        safe_print("\nFailed Scripts:")
        for r in summary.failure_list:
            reason = r.reason or "Unknown"
            snippet = r.output_snippet or ""
            safe_print(f"  • {r.rel_path} → {reason}")
            if args.show_fail_details and snippet:
                safe_print("    Output preview:")
                for line in snippet.splitlines():
                    safe_print(f"      {line}")
                safe_print()

    if summary.skipped_sim_list:
        safe_print("\nSkipped Due to Lack of Sim Support:")
        for r in summary.skipped_sim_list:
            safe_print(f"  • {r.rel_path}")

    if summary.skipped_no_tests_list:
        safe_print("\nSkipped Due to No Executable Content:")
        for r in summary.skipped_no_tests_list:
            safe_print(f"  • {r.rel_path}")

    if summary.skipped_pytest_mark_list:
        safe_print("\nSkipped Due to @pytest.mark.skip:")
        for r in summary.skipped_pytest_mark_list:
            safe_print(f"  • {r.rel_path}")

    if summary.skipped_pytest_disabled_list:
        safe_print("\nSkipped Due to Pytest Auto-Detect Disabled:")
        for r in summary.skipped_pytest_disabled_list:
            safe_print(f"  • {r.rel_path}")

    safe_print("=" * 60)


def main() -> None:
    # Initialize ProcessManager early for signal handling
    process_manager = ProcessManager()
    process_manager.setup_signal_handlers()

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
        default=DEFAULT_SIM_WORKERS,
        help=f"Number of parallel workers (only effective in 'sim' mode). "
             f"In 'npu' mode, parallelism is determined by device count. "
             f"Default: {DEFAULT_SIM_WORKERS}."
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
        default=DEFAULT_SCRIPT_TIMEOUT,
        help=f"Per-script execution timeout in seconds (default: {DEFAULT_SCRIPT_TIMEOUT})."
    )
    parser.add_argument(
        "--parallel_retries",
        type=int,
        default=DEFAULT_PARALLEL_RETRIES,
        help=f"Number of additional parallel retry rounds after the initial run "
             f"(default: {DEFAULT_PARALLEL_RETRIES}). "
             f"Total parallel rounds = 1 (initial) + N (retries)."
    )
    parser.add_argument(
        "--serial_retries",
        type=int,
        default=DEFAULT_SERIAL_RETRIES,
        help=f"Maximum number of serial retry rounds in single-device mode "
             f"(default: {DEFAULT_SERIAL_RETRIES}). "
             f"Total serial runs = 1 (initial) + N (retries). Set to 0 to disable retries."
    )
    parser.add_argument(
        "--show-fail-details",
        action="store_true",
        help=f"Show last {OUTPUT_SNIPPET_LINES} lines of output for each failed script in the final summary."
    )
    parser.add_argument(
        "--no-pytest-auto-detect",
        action="store_false",
        dest="pytest_auto_detect",
        default=True,
        help="Disable automatic detection and execution of pytest-style tests. "
             "By default, scripts without a '__main__' guard but with pytest-style tests "
             "will be executed using pytest. With this flag, such scripts will be skipped."
    )
    parser.add_argument(
        "--no-skip-pytest-mark-skip",
        action="store_false",
        dest="skip_pytest_mark_skip",
        default=True,
        help="Disable the skipping of scripts based on @pytest.mark.skip decorator. "
             "By default, scripts with @pytest.mark.skip are skipped."
    )
    parser.add_argument(
        "--no-serial-fallback",
        action="store_true",
        dest="no_serial_fallback",
        default=False,
        help="Skip the final serial retry loop in multi-device mode. "
             "By default, when parallel execution has failures, a serial retry loop is executed. "
             "With this flag, serial fallback is skipped and only parallel retries are performed."
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

    # Pre-analyze all scripts to cache results and check pytest requirement
    script_analyses: Dict[Path, ScriptAnalysis] = {}
    for f in py_files:
        script_analyses[f] = get_script_analysis(f)

    need_pytest = args.pytest_auto_detect and any(
        not analysis.has_main_guard and analysis.has_pytest_tests
        for analysis in script_analyses.values()
    )
    pytest_available = shutil.which("pytest") is not None

    if need_pytest and not pytest_available:
        print("Error: Some scripts require 'pytest' but it is not installed.", file=sys.stderr)
        print("    Please install pytest with: pip install pytest", file=sys.stderr)
        sys.exit(1)

    # Handle workers and device configuration based on run mode
    if args.run_mode == "sim":
        # SIM mode: workers parameter determines parallelism
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
        if args.no_serial_fallback:
            print("Serial fallback   : Disabled (--no-serial-fallback)")
        else:
            print(f"Serial fallback   : Enabled (max retries: {args.serial_retries})")
    if args.example_id:
        print(f"Test selector     : {args.example_id}")
    if not args.pytest_auto_detect:
        print("Pytest auto-detect: Disabled (pytest-only scripts will be skipped)")
    else:
        print("Pytest auto-detect: Enabled")
    print(f"Target            : {target}")
    print(f"Found {len(relative_paths)} .py file(s).")
    print("=" * 60)

    # Create thread-safe print function via ProcessManager
    safe_print = process_manager.create_safe_print()

    start_time = time.perf_counter()
    exit_code = 0
    cancelled_by_signal = False

    try:
        # Initial candidate list: all scripts that are not skipped
        candidates_to_run = []
        skipped_sim_scripts = []
        skipped_no_tests_scripts = []
        skipped_pytest_mark_scripts = []
        skipped_pytest_disabled_scripts = []

        for rel_path in relative_paths:
            # Check for early shutdown
            if process_manager.is_shutdown_requested():
                safe_print("\n⚠️  Shutdown requested during initialization. Aborting.")
                cancelled_by_signal = True
                break

            full_path = target_dir / rel_path
            analysis = script_analyses.get(full_path) or get_script_analysis(full_path)

            if not analysis.has_main_guard and not analysis.has_pytest_tests:
                skipped_no_tests_scripts.append(rel_path)
                continue

            # Skip pytest-only scripts when pytest auto-detection is disabled
            if not analysis.has_main_guard and analysis.has_pytest_tests and not args.pytest_auto_detect:
                skipped_pytest_disabled_scripts.append(rel_path)
                continue

            if args.run_mode == "sim" and not analysis.supports_run_mode:
                skipped_sim_scripts.append(rel_path)
                continue

            if args.skip_pytest_mark_skip and analysis.has_pytest_skip_mark:
                skipped_pytest_mark_scripts.append(rel_path)
                continue

            candidates_to_run.append(rel_path)

        # Record skipped results for final summary
        skipped_sim_results = [
            ExecutionResult.skipped(
                p, ExecutionStatus.SKIPPED_SIM, "script does not support --run_mode"
            )
            for p in skipped_sim_scripts
        ]
        skipped_no_tests_results = [
            ExecutionResult.skipped(
                p, ExecutionStatus.SKIPPED_NO_TESTS,
                "no '__main__' guard and no pytest-style tests"
            )
            for p in skipped_no_tests_scripts
        ]
        skipped_pytest_mark_results = [
            ExecutionResult.skipped(
                p, ExecutionStatus.SKIPPED_PYTEST_MARK,
                "contains @pytest.mark.skip decorator"
            )
            for p in skipped_pytest_mark_scripts
        ]
        skipped_pytest_disabled_results = [
            ExecutionResult.skipped(
                p, ExecutionStatus.SKIPPED_PYTEST_DISABLED, "pytest auto-detect disabled"
            )
            for p in skipped_pytest_disabled_scripts
        ]

        current_candidates = candidates_to_run[:]
        all_results_map: Dict[str, ExecutionResult] = {}

        # Add skipped results to final map immediately
        all_skipped_results = (
            skipped_sim_results + skipped_no_tests_results +
            skipped_pytest_mark_results + skipped_pytest_disabled_results
        )
        for r in all_skipped_results:
            all_results_map[r.rel_path] = r

        success_list: List[ExecutionResult] = []
        failure_list: List[ExecutionResult] = []

        if cancelled_by_signal:
            pass  # Early exit due to signal
        elif not current_candidates:
            safe_print("ℹ️  No executable scripts found. All were skipped.")
        else:
            # Create execution strategy with ProcessManager
            if is_single_device:
                strategy: ExecutionStrategy = SingleDeviceStrategy(
                    args, target_dir, actual_device_ids, args.timeout,
                    safe_print, process_manager
                )
            else:
                strategy = MultiDeviceStrategy(
                    args, target_dir, actual_device_ids, args.timeout,
                    safe_print, process_manager
                )

            # Execute scripts using the selected strategy
            success_list, failure_list = strategy.execute(current_candidates, all_results_map)

            # Check if we were cancelled during execution
            if process_manager.is_shutdown_requested():
                cancelled_by_signal = True

        # Generate final summary
        total_time_sec = time.perf_counter() - start_time

        if cancelled_by_signal:
            safe_print("\n" + "=" * 60)
            safe_print("⚠️  EXECUTION CANCELLED BY SIGNAL")
            safe_print("=" * 60)
            safe_print(f"Total execution time before cancellation: {total_time_sec:.2f} seconds")

            # Count results by status
            cancelled_count = sum(1 for r in all_results_map.values()
                                  if r.status == ExecutionStatus.CANCELLED)
            completed_success = sum(1 for r in all_results_map.values()
                                    if r.status == ExecutionStatus.SUCCESS)
            completed_failure = sum(1 for r in all_results_map.values()
                                    if r.status == ExecutionStatus.FAILURE)

            safe_print(f"Scripts completed successfully: {completed_success}")
            safe_print(f"Scripts failed: {completed_failure}")
            safe_print(f"Scripts cancelled: {cancelled_count}")
            safe_print("=" * 60)
            exit_code = 130  # Standard exit code for SIGINT
        else:
            summary = SummaryData(
                success_list=success_list,
                failure_list=failure_list,
                skipped_sim_list=skipped_sim_results,
                skipped_no_tests_list=skipped_no_tests_results,
                skipped_pytest_mark_list=skipped_pytest_mark_results,
                skipped_pytest_disabled_list=skipped_pytest_disabled_results,
                args=args,
                target=target,
                device_ids=actual_device_ids,
                total_time_sec=total_time_sec,
                safe_print=safe_print
            )
            _print_final_summary(summary)
            exit_code = 1 if len(failure_list) > 0 else 0

    finally:
        # Final cleanup: ensure all processes are terminated
        if process_manager.is_shutdown_requested():
            process_manager.cleanup_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()