#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests Execution Acceleration

This module provides test case parallel execution acceleration, supporting multi-process
concurrent test case execution. Main features include:

- Multi-container/process parallel test case execution to improve testing efficiency
- Intelligent case sorting with load balancing based on historical duration estimates
- Case duration caching mechanism to optimize repeated execution scenarios
- Real-time execution status monitoring and exception handling
- Detailed execution reports and statistics (including container execution summary,
  case duration statistics, exception information, etc.)
- CPU affinity configuration to optimize performance in multi-CPU environments

Main classes:
- TestsAccelerate: Main class for test acceleration, providing a complete parallel execution framework
- CaseDesc: Test case description, including name and estimated duration
- ExecParam: Execution parameter configuration
- ExecResult: Execution result statistics and report generation
- CntrContext: Container/process execution context
- CaseContext: Case execution context

Usage example:
1. Inherit from TestsAccelerate and implement the _prepare_get_params method
2. Call prepare() for preparation
3. Call process() to execute tests
4. Call post() to get execution results
"""

from abc import ABC
import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
import json
import logging
import multiprocessing
from multiprocessing import Event, JoinableQueue, Process, Value, cpu_count
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.args_action import ArgsEnvDictAction
from utils.executable import Exec
from utils.table import Table


class ArgsCaseListAction(argparse.Action):
    """Parse the cases field from command-line arguments (adapted for custom metadata parameters)"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: List[str],
        option_string: Optional[str] = None,
    ) -> None:
        # Parse each string, split by colon and flatten
        case_list = []
        for value in values:
            cases = [cs.strip() for cs in value.split(':') if cs.strip()]  # Split each string and filter empty strings
            case_list.extend(cases)
        # Set the result to the namespace
        setattr(namespace, self.dest, case_list)


class TestsAccelerate(ABC):
    """Tests Acceleration"""

    @dataclasses.dataclass
    class ExecParam:
        """Execution parameters"""

        cntr_id: Optional[int] = None
        envs_func: Optional[Callable] = None
        custom: Optional[Any] = None

        def __init__(self, cntr_id: int, envs_func: Optional[Callable] = None, custom: Optional[Any] = None):
            self.cntr_id = cntr_id
            self.envs_func = envs_func
            self.custom = custom

        def get_envs(self) -> Optional[Dict[str, str]]:
            """Get additional environment variable configuration"""
            if self.envs_func:
                return self.envs_func(self)
            return None

    @dataclasses.dataclass
    class ExecResult:
        """Execution result"""

        cntr_name: str = "Cntr"
        act_duration: Optional[timedelta] = None  # Actual total duration
        ori_duration: Optional[timedelta] = None  # Original total duration (estimated)
        cntr_max_duration: Optional[timedelta] = None  # Longest duration among all Cntrs
        cntr_min_duration: Optional[timedelta] = None  # Shortest duration among all Cntrs
        cntr_execution_details: JoinableQueue = JoinableQueue()
        cntr_duration_dict: Dict[int, timedelta] = dataclasses.field(default_factory=dict)
        case_execution_details: JoinableQueue = JoinableQueue()
        case_exception_details: JoinableQueue = JoinableQueue()
        case_terminate_details: JoinableQueue = JoinableQueue()

        @property
        def revenue_desc(self) -> str:
            diff = self.ori_duration - self.act_duration
            rate = float(diff / self.act_duration) * 100
            desc = f"Revenue(Act/Ori, {self.act_duration.total_seconds():.2f}/"
            desc += f"{self.ori_duration.total_seconds():.2f}) {rate:.2f}%"
            return desc

        @property
        def cntr_latency_desc(self) -> str:
            diff = self.cntr_max_duration - self.cntr_min_duration
            rate = float(diff / self.cntr_min_duration) * 100
            desc = f"Latency(Max/Min/Diff, {self.cntr_max_duration.total_seconds():.2f}/"
            desc += f"{self.cntr_min_duration.total_seconds():.2f}/{diff.total_seconds():.2f}) {rate:.2f}%"
            return desc

        @staticmethod
        def save_case_duration_to_json(
            sorted_datas: List[List[Any]],
            dump_item_num: int = 100,
            dump_min_duration: float = 5,
            path: Optional[Path] = None,
        ):
            # Path handling
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            # Data processing
            item_num = 0
            case_name_idx = 1
            duration_idx = 2
            duration_dict = {}
            for item in sorted_datas:
                case_name = item[case_name_idx]
                duration = float(item[duration_idx])
                duration_dict[case_name] = duration
                item_num += 1
                if item_num >= dump_item_num:
                    break
                if duration <= dump_min_duration:
                    break
            # Persist data to disk
            with path.open("w", encoding="utf-8") as f:
                json.dump(duration_dict, f, indent=4)

        def get_cntr_exec_info(self) -> Tuple[str, str]:
            """Get Container execution information statistics.

            :returns:
                Tuple[str, str]:
                    - Container execution statistics table (str)
                    - Container parallel execution revenue description (str)
            """
            heads = [self.cntr_name, "Total", "Success", "Failed", "Duration"]
            datas = []
            self.ori_duration = timedelta()
            while not self.cntr_execution_details.empty():
                _brief = self.cntr_execution_details.get()
                devs_id = int(_brief[0])
                case_total = int(_brief[1])
                case_pass = int(_brief[2])
                case_fail = int(_brief[3])
                devs_duration = _brief[-1]
                # Duration statistics
                if self.cntr_max_duration is None:
                    self.cntr_max_duration = devs_duration
                self.cntr_max_duration = max(self.cntr_max_duration, devs_duration)
                if self.cntr_min_duration is None:
                    self.cntr_min_duration = devs_duration
                self.cntr_min_duration = min(self.cntr_min_duration, devs_duration)
                # Save results
                self.cntr_duration_dict[devs_id] = devs_duration
                self.ori_duration += devs_duration
                datas.append([devs_id, case_total, case_pass, case_fail, f"{devs_duration.total_seconds():.2f}"])
                self.cntr_execution_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                brief = Table.table(datas=datas, headers=heads)
            # Calculate parallel execution revenue
            desc = f"Duration {self.act_duration.total_seconds():.2f} secs, {self.revenue_desc}"
            return f"\n\n{self.cntr_name} Execution Brief:{brief}", desc

        def get_case_exec_terminate_info(self) -> Tuple[str, int]:
            """Get Case execution termination information.

            :returns:
                Tuple[str, int]:
                    - Case termination execution information
                    - Case termination count
            """
            heads = ["Idx", self.cntr_name, "CaseName", "Duration"]
            datas = []
            while not self.case_terminate_details.empty():
                _brief = self.case_terminate_details.get()
                cntr_id = int(_brief[0])
                case_name = str(_brief[1])
                case_duration = _brief[2]
                datas.append([cntr_id, case_name, f"{case_duration.total_seconds():.2f}"])
                self.case_terminate_details.task_done()
            brief = "\nNone"
            if len(datas) != 0:
                datas = [[f"{idx}/{len(datas)}"] + ele for idx, ele in enumerate(datas, start=1)]
                brief = Table.table(datas=datas, headers=heads)
            return f"\n\nCase Terminate Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_exception_info(self) -> Tuple[str, int]:
            """Get Case execution exception information.

            :returns:
                Tuple[str, int]:
                    - Case exception execution information
                    - Case exception count
            """
            datas = []
            brief = ""
            while not self.case_exception_details.empty():
                chunk = self.case_exception_details.get()
                if len(chunk) != 0:
                    brief += chunk
                else:
                    datas.append(str(brief))
                    brief = ""
                self.case_exception_details.task_done()
            brief = "\nNone" if len(datas) == 0 else ""
            for idx, data in enumerate(datas, start=1):
                brief += f"\nIdx:{idx}/{len(datas)}\n{data}"
            return f"\n\nCase Exception Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_duration_info(
            self,
            case_dict: Dict[str, Exec.CaseDesc],
            min_print_cnt: Optional[int] = None,
            dump_json_path: Optional[Path] = None,
            dump_item_num: int = 100,
            dump_min_duration: float = 5,
        ) -> str:
            """Get Case execution duration statistics.

            :return: Case execution duration statistics.
            """
            heads = [self.cntr_name, "CaseName", "Duration", "Estimate", f"Ratio({self.cntr_name})", "Ratio(Total)"]
            datas = []
            while not self.case_execution_details.empty():
                _brief = self.case_execution_details.get()
                cntr_id = _brief[0]
                case_name = str(_brief[1])
                case_duration = _brief[2]
                case_desc = case_dict.get(case_name, None)
                case_estimate = ""
                if case_desc and case_desc.duration:
                    case_estimate = timedelta(seconds=case_desc.duration).total_seconds()
                cntr_duration = self.cntr_duration_dict[cntr_id]
                ratio_cntr = float(case_duration / cntr_duration) * 100
                ratio_process = float(case_duration / self.act_duration) * 100
                datas.append(
                    [
                        cntr_id,
                        case_name,
                        case_duration.total_seconds(),
                        case_estimate,
                        f"{case_duration.total_seconds():.2f}/{cntr_duration.total_seconds():.2f} {ratio_cntr:.2f}%",
                        f"{case_duration.total_seconds():.2f}/{self.act_duration.total_seconds():.2f} "
                        f"{ratio_process:.2f}%",
                    ]
                )
                self.case_execution_details.task_done()
            brief = "\nNone"
            add_desc = ""
            if len(datas) != 0:
                # Sort data by duration in descending order, then convert format
                duration_idx = 2  # 2 is idx of duration
                datas = sorted(datas, key=lambda x: x[duration_idx], reverse=True)
                for item in datas:
                    item[duration_idx] = f"{item[duration_idx]:.2f}"
                # Persist results, commonly used to accelerate local repeated execution
                self.save_case_duration_to_json(
                    sorted_datas=datas,
                    path=dump_json_path,
                    dump_item_num=dump_item_num,
                    dump_min_duration=dump_min_duration,
                )
                # Abbreviation feature
                if min_print_cnt:
                    # Print 50 additional cases beyond those with configured estimated duration
                    print_cnt = min_print_cnt + 50
                    ori_len = len(datas)
                    datas = datas[:print_cnt]
                    cur_len = len(datas)
                    if ori_len > cur_len:
                        hidden_cnt = ori_len - cur_len
                        hidden_first_data = datas[-1]  # Get the last case after slicing
                        add_desc = f"\n({hidden_cnt} durations <= {hidden_first_data[2]}s hidden.)"  # 2 Duration
                # Result summary
                brief = Table.table(datas=datas, headers=heads, auto_sort=False)
            return f"\n\nCase Duration Brief:{brief}" + add_desc

    @dataclasses.dataclass
    class CntrContext:
        """Cntr processing context"""

        cntr_id: int = 0
        exec_param: Optional[Any] = None
        success: int = 0
        failed: int = 0
        ts: Optional[datetime] = None
        exit_code: int = 0

        def __init__(self, cntr_id: int, exec_param):
            self.cntr_id = cntr_id
            self.exec_param = exec_param
            self.ts = datetime.now(tz=timezone.utc)

        @property
        def total(self) -> int:
            return self.success + self.failed

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.total, self.success, self.failed, (datetime.now(tz=timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class CaseContext:
        """Case processing context"""

        cntr_id: int = 0
        exec_param: Optional[Any] = None
        ts: Optional[datetime] = None
        case_name: str = ""

        def __init__(self, cntr_id: int, exec_param, case_name):
            self.cntr_id = cntr_id
            self.exec_param = exec_param
            self.case_name = case_name
            self.ts = datetime.now(tz=timezone.utc)

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.case_name, (datetime.now(tz=timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class MoveContext:
        """Move process context"""

        ele_count: int
        src_queue: JoinableQueue
        dst_queue: JoinableQueue

        def __init__(self, src: JoinableQueue, dst: JoinableQueue):
            self.ele_count = 0
            self.src_queue = src
            self.dst_queue = dst

        def move(self, timeout: int = 1) -> bool:
            try:
                ele = self.src_queue.get(timeout=timeout)
                self.src_queue.task_done()
                if ele is None:
                    return False
                if isinstance(ele, str):
                    if len(ele) == 0:
                        self.ele_count += 1
                else:
                    self.ele_count += 1
                self.dst_queue.put(ele)
            except (queue.Empty, KeyboardInterrupt):
                pass
            return True

    def __init__(self, args, scene_mark: str, cntr_name: str):
        """
        :param args: Command-line arguments
        :param cntr_name: Container name, used for display output
        """
        # Scene identifier
        self.mark: str = scene_mark

        # Case execution parameters, execution behavior control parameters
        self.exe: Exec = Exec(file=args.target[0], envs=args.envs, timeout=args.timeout_case)
        self.exe_params: List[TestsAccelerate.ExecParam] = []
        self.exe_result: TestsAccelerate.ExecResult = TestsAccelerate.ExecResult(cntr_name=cntr_name)
        self.exe_timeout: Optional[int] = args.timeout
        self.exe_halt_on_error: bool = args.halt_on_error  # Terminate subsequent Case execution on failure

        # Case management
        self.case_duration_json: Path = self._init_get_case_duration_json(args=args)
        self.case_duration_max_num: int = self._init_get_case_duration_max_num(args=args)
        self.case_duration_min_sec: float = self._init_get_case_duration_min_sec(args=args)
        self.case_list: List[Exec.CaseDesc] = []
        self.case_dict: Dict[str, Exec.CaseDesc] = {}
        self.case_ordered_cnt: int = 0
        # GCOV concurrent write isolation: Set GCOV_PREFIX before getting case list
        # to avoid concurrent writes from main process --gtest_list_tests
        self._init_gcov_prefix()
        self.case_ordered_cnt, self.case_list, self.case_dict = self.exe.get_case_name_info(
            case_name_list=args.cases, duration_json=self.case_duration_json
        )

        self.case_queue: JoinableQueue = JoinableQueue()
        self.case_execution_queue: JoinableQueue = JoinableQueue()  # Collect info when Case completes normally
        self.case_exception_queue: JoinableQueue = JoinableQueue()  # Collect error info when Case execution fails
        self.case_terminate_queue: JoinableQueue = JoinableQueue()  # Collect info when Case execution is terminated
        self.case_exec_count = Value('i', 0)  # DFX, track Case completion progress

        # Container management
        self.cntr_name: str = cntr_name
        self.cntr_execution_queue: JoinableQueue = JoinableQueue()  # Container execution result statistics reporting
        self.cntr_terminate_event = Event()  # Used to notify other Container processes to stop
        self.cntr_exit_count = Value('i', 0)  # DFX, track Container exit progress

        # CPU affinity management
        self.cpu_rank_size: Optional[int] = self._init_get_cpu_rank_size(args=args)
        self.cpu_affinity_policy: Optional[int] = None

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        lst = [
            ["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
            ["Timeout", self.exe_timeout],
            ["HaltOnError", self.exe_halt_on_error],
            [f"{self.cntr_name}Num", self.cntr_num],
            [f"{self.cntr_name}List", [p.cntr_id for p in self.exe_params]],
            ["CaseNum", self.case_num],
            ["CaseTimeout", self.exe.timeout],
            ["CaseDurationFile", self.case_duration_json],
            ["CaseDurationMaxNum", self.case_duration_max_num],
            ["CaseDurationMinSecs", self.case_duration_min_sec],
            ["Executable", self.exe.file],
        ]
        if self.cpu_rank_size:
            lst.append(["CpuRankSize", self.cpu_rank_size])
            lst.append(["CpuAffinityPolicy", f"{self.cpu_affinity_policy_str}({self.cpu_affinity_policy})"])
        return lst

    @property
    def cntr_num(self) -> int:
        return len(self.exe_params)

    @property
    def case_num(self) -> int:
        return len(self.case_list)

    @property
    def cpu_affinity_policy_str(self) -> str:
        if not self.cpu_affinity_policy:
            return "Disable"
        elif self.cpu_affinity_policy == 1:
            return "Even Allocation"  # Even allocation
        elif self.cpu_affinity_policy == 2:
            return "Cyclic Reuse Allocation"  # Cyclic reuse allocation
        else:
            return "Unknown"

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser):
        """Register command-line arguments

        Notes:
            1. This function should be used in conjunction with the get_container_manager function;
            2. This function registers the 'cases' field, but get_container_manager does not parse it;
               the field should be parsed by the caller;

        :param parser: ArgumentParser created externally
        """
        # Execution parameters
        parser.add_argument(
            "-t", "--target", nargs=1, type=str, required=True, help="Specific target executable file path."
        )
        parser.add_argument(
            "-e",
            "--env",
            nargs="+",
            action=ArgsEnvDictAction,
            default={},
            dest="envs",
            help="Specify additional environment variables to set when executing the target.",
        )
        parser.add_argument("--timeout", nargs="?", type=int, default=None, help="Timeout for executing all cases.")
        parser.add_argument(
            "--timeout_case", nargs="?", type=int, default=None, help="Timeout for executing single case."
        )
        parser.add_argument(
            "--halt_on_error",
            action="store_true",
            default=False,
            help="If any case failed, subsequent cases are not executed.",
        )
        # Case parameters
        parser.add_argument(
            "-c",
            "--cases",
            nargs='*',
            action=ArgsCaseListAction,
            default=[],
            required=False,
            help="Cases, multiple cases are separated by ':'",
        )
        # Others
        parser.add_argument(
            "--cpu_rank_size",
            nargs="?",
            type=int,
            default=None,
            help="Specify the rank size for CPU affinity grouping.",
        )
        # Case duration cache related parameters
        parser.add_argument(
            "--dump_case_duration_json",
            nargs="?",
            type=Path,
            default=None,
            help="Specify the path to the case duration json cache file.",
        )
        parser.add_argument(
            "--dump_case_duration_max_num",
            nargs="?",
            type=int,
            default=None,
            help="Maximum number of cases to dump to duration json cache.",
        )
        parser.add_argument(
            "--dump_case_duration_min_secends",
            nargs="?",
            type=int,
            default=None,
            help="Minimum duration (in seconds) for cases to dump to duration json cache.",
        )

    @staticmethod
    def _init_get_cpu_rank_size(args) -> Optional[int]:
        cpu_rank_size = None
        if args.cpu_rank_size:
            cpu_rank_size = args.cpu_rank_size
        else:
            cpu_rank_size_str = os.environ.get("PYPTO_TESTS_CASE_EXECUTE_CPU_RANK_SIZE", None)
            if cpu_rank_size_str:
                cpu_rank_size = int(cpu_rank_size_str)
        if cpu_rank_size and cpu_rank_size > 0:
            return cpu_rank_size
        return None

    @staticmethod
    def _init_get_case_duration_json(args) -> Path:
        """Initialize case_duration_json

        Command-line argument takes priority, then environment variable, then default value
        """
        if args.dump_case_duration_json:
            return args.dump_case_duration_json.resolve()

        # Get from environment variable
        env_json_path = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_JSON", None)
        if env_json_path:
            return Path(env_json_path).resolve()

        # Default value
        tagert = Path(args.target[0])
        return tagert.parent / f"{tagert.stem}_duration.json"

    @staticmethod
    def _init_get_case_duration_max_num(args) -> int:
        """Initialize case_duration_max_num

        Command-line argument takes priority, then environment variable, then default value
        """
        if args.dump_case_duration_max_num is not None:
            return args.dump_case_duration_max_num

        # Get from environment variable
        env_max_num = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_MAX_NUM", None)
        if env_max_num:
            return int(env_max_num)

        # Default value
        return 500

    @staticmethod
    def _init_get_case_duration_min_sec(args) -> float:
        """Initialize case_duration_min_sec

        Command-line argument takes priority, then environment variable, then default value
        """
        if args.dump_case_duration_min_secends is not None:
            return float(args.dump_case_duration_min_secends)

        # Get from environment variable
        env_min_sec = os.environ.get("PYPTO_TESTS_DUMP_CASE_DURATION_MIN_SECONDS", None)
        if env_min_sec:
            return float(env_min_sec)

        # Default value
        return 5.0

    def _init_gcov_prefix(self):
        """Initialize GCOV_PREFIX isolation configuration

        Set GCOV_PREFIX_STRIP, inherited by Cntr processes via fork.
        The main process does not set GCOV_PREFIX (.gcda is written to the build directory,
        no concurrent write issue), reducing unnecessary .gcda copies.
        """
        gcov_data_dir = self.exe.envs.get("PYPTO_GCOV_DATA_DIR")
        if not gcov_data_dir:
            return
        gcov_root = Path(gcov_data_dir)
        if gcov_root.exists():
            shutil.rmtree(gcov_root)
        gcov_root.mkdir(parents=True, exist_ok=True)
        build_dir = str(gcov_root.parent.resolve())
        strip_count = len(Path(build_dir).parts) - 1
        os.environ["GCOV_PREFIX_STRIP"] = str(strip_count)

    def _cntr_set_gcov_prefix(self, cntr_id: int):
        """Set independent GCOV_PREFIX for Cntr process

        Case subprocess inherits Cntr's os.environ; when GTest exits, .gcda is written to
        the isolated directory, avoiding concurrent write conflicts.
        GCOV_PREFIX_STRIP is set by the main process _init_gcov_prefix and inherited via fork,
        no need to set it again.
        """
        gcov_data_dir = self.exe.envs.get("PYPTO_GCOV_DATA_DIR")
        if not gcov_data_dir:
            return
        cntr_dir = Path(gcov_data_dir) / f"cntr_{cntr_id}"
        cntr_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GCOV_PREFIX"] = str(cntr_dir)

    @staticmethod
    def _move(src: JoinableQueue, dst: JoinableQueue):
        TestsAccelerate._set_process_desc()
        ctx = TestsAccelerate.MoveContext(src=src, dst=dst)
        while True:
            if not ctx.move():
                break
        logging.info("%s Exit, Move %s elements.", TestsAccelerate._get_process_desc(), ctx.ele_count)

    @staticmethod
    def _get_process_desc() -> str:
        cur_process = multiprocessing.current_process()
        return f"{cur_process.name}"

    @staticmethod
    def _set_process_desc():
        try:
            import setproctitle

            setproctitle.setproctitle(TestsAccelerate._get_process_desc())
        except ModuleNotFoundError:
            pass

    def prepare(self):
        """Execution preparation"""
        self.exe_params = self._prepare_get_params()
        if self.cntr_num == 0:
            raise ValueError("ExecParams is empty, won't run any task.")
        if self.cntr_num > self.case_num:
            logging.info(
                "CaseNum(%s) less than len(ExecParams)=%s, will only start the first %s %s.",
                self.case_num,
                self.cntr_num,
                self.case_num,
                self.cntr_name,
            )
            self.exe_params = self.exe_params[:self.case_num]
        # CPU affinity settings
        self._prepare_determine_cpu_affinity_policy()

    def process(self):
        """Execute tasks"""
        logging.info("\n\n%s Accelerate Args:%s", self.mark, Table.table(datas=self.brief))
        # Execution flow
        ts = datetime.now(tz=timezone.utc)
        self._main()
        self.exe_result.act_duration = datetime.now(tz=timezone.utc) - ts

    def post(self) -> bool:
        """Post-processing, get execution result summary"""
        # Cntr execution information collection and summary
        cntr_exec_brief, cntr_revenue_desc = self.exe_result.get_cntr_exec_info()

        # Case execution information collection and summary
        case_exec_brief, case_exec_result = self._post_case_exec_info()

        out = f"{self.mark}, HaltOnError({self.exe_halt_on_error}), {cntr_revenue_desc}"
        out += cntr_exec_brief
        out += case_exec_brief

        if case_exec_result:
            logging.info(out)
            logging.info(
                "Use %s %s | Exec %s case | %s | %s",
                self.cntr_num,
                self.cntr_name,
                self.case_num,
                self.exe_result.revenue_desc,
                self.exe_result.cntr_latency_desc,
            )
        else:
            logging.error(out)
        return case_exec_result

    def _prepare_determine_cpu_affinity_policy(self):
        """Initialize CPU affinity policy

        The parameters required for policy determination (e.g. CntrNum) are not available
        at class construction time, so this process is deferred to the prepare stage.
        """
        self.cpu_affinity_policy = None
        if self.cpu_rank_size and self.cpu_rank_size > 0:
            if self.cntr_num * self.cpu_rank_size <= cpu_count():
                self.cpu_affinity_policy = 1  # Policy 1: Even allocation (each CPU group maps to 1 cntr)
            else:
                # Policy 2: Cyclic reuse of core groups (expected CPU count exceeds total)
                self.cpu_affinity_policy = 2
        logging.info(
            "Determine CpuAffinity, Policy=%s(%s), CntrNum=%s, CpuNum=%s, CpuRankSize=%s",
            self.cpu_affinity_policy_str,
            self.cpu_affinity_policy,
            self.cntr_num,
            cpu_count(),
            self.cpu_rank_size,
        )

    def _prepare_get_params(self) -> List[ExecParam]:
        return []

    def _post_case_exec_info(self) -> Tuple[str, bool]:
        """Get Case execution information.

        :returns:
            Tuple[str, bool]:
                - Case execution information
                - Case execution success/failure determination
        """
        terminate_brief, terminate_count = self.exe_result.get_case_exec_terminate_info()
        exception_brief, exception_count = self.exe_result.get_case_exec_exception_info()
        duration_brief = self.exe_result.get_case_exec_duration_info(
            case_dict=self.case_dict,
            min_print_cnt=self.case_ordered_cnt,
            dump_json_path=self.case_duration_json,
            dump_item_num=self.case_duration_max_num,
            dump_min_duration=self.case_duration_min_sec,
        )

        # Case execution overall summary
        remaining_count = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                remaining_count += 1
            self.case_queue.task_done()
        success_count = self.case_num - remaining_count - terminate_count - exception_count
        execution_heads = ["Total", "Success", "Failed", "Terminate", "Remaining"]
        execution_datas = [[self.case_num, success_count, exception_count, terminate_count, remaining_count]]
        execution_brief = Table.table(datas=execution_datas, headers=execution_heads)
        execution_brief = f"\n\nCase Execution Brief:{execution_brief}"

        rst = (terminate_count + exception_count + remaining_count) == 0
        out = execution_brief + duration_brief + terminate_brief + exception_brief
        return out, rst

    def _main(self):
        """Case execution, manage execution state (main process)

        :return: Whether execution succeeded
        """
        # Create and start subprocesses for task processing
        cntr_step = 1
        cntr_process_group = []
        try:
            # Task preparation
            self._push_all_case_sync()
            # Start reporting monitor processes
            self._start_move_process_grp()
            # Create and start task subprocesses for task processing
            cntr_process_group = self._start_cntr_process_grp()
            # Wait for task processing to complete
            self._join_cntr_process_grp(cntr_process_grp=cntr_process_group, step=cntr_step)
        except KeyboardInterrupt:
            logging.info("MainProcess Recv download terminate event.")
        finally:
            self._stop_cntr_process_grp(cntr_process_grp=cntr_process_group, timeout=cntr_step)
            self._stop_move_process_grp()

    def _push_all_case_sync(self):
        """Insert cases into the queue synchronously, insert termination signals based on Container count"""
        for cs in self.case_list:
            self.case_queue.put(cs.name)
        for _ in range(self.cntr_num):
            self.case_queue.put(None)

    def _start_move_process_grp(self) -> List[Process]:
        """Start Move process group

        :return: List of Move processes
        """
        move_grp = []
        desc_list = self._get_move_process_grp_desc_list()
        for name, src_queue, dst_queue in desc_list:
            process = Process(
                name=f"MoveProcess({name})",
                target=self._move,
                args=(
                    src_queue,
                    dst_queue,
                ),
            )
            process.start()
            move_grp.append(process)
        return move_grp

    def _stop_move_process_grp(self):
        """Stop Move process group"""
        desc_list = self._get_move_process_grp_desc_list()
        for _, src_queue, _ in desc_list:
            src_queue.put(None)
            src_queue.join()

    def _get_move_process_grp_desc_list(self) -> List[Tuple[str, JoinableQueue, JoinableQueue]]:
        pairs = [
            ("CaseExecution", self.case_execution_queue, self.exe_result.case_execution_details),
            ("CaseException", self.case_exception_queue, self.exe_result.case_exception_details),
            ("CaseTerminate", self.case_terminate_queue, self.exe_result.case_terminate_details),
            (f"{self.cntr_name}Execution", self.cntr_execution_queue, self.exe_result.cntr_execution_details),
        ]
        return pairs

    def _start_cntr_process_grp(self, delay: int = 2) -> List[Process]:
        """Start Cntr process group

        :param delay: Delay duration after each Cntr starts and before processing specific Cases;
            in multi-consumer mode, add delay to each consumer startup to wait for all consumers
            to be ready
        :return: Cntr process group
        """
        process_group: List[Process] = []
        for exec_param in self.exe_params:
            process = Process(
                name=f"{self.cntr_name}Process({self.cntr_name}[{exec_param.cntr_id}])",
                target=self._cntr,
                args=(
                    exec_param.cntr_id,
                    exec_param,
                    delay,
                ),
            )
            process_group.append(process)
            process.start()
        return process_group

    def _join_cntr_process_grp(self, cntr_process_grp: List[Process], step: int = 1):
        """Synchronously wait for Cntr process group to complete

        :param cntr_process_grp: Cntr process group
        :param step: Internal check interval in seconds
        """
        s_time = time.time()
        while True:
            if not self._wait_cntr_one_step(cntr_process_grp=cntr_process_grp, s_time=s_time, step=step):
                break

    def _wait_cntr_one_step(self, cntr_process_grp: List[Process], s_time, step: int = 1) -> bool:
        """Block current process and check Cntr process group completion status

        :param cntr_process_grp: Cntr process group
        :param s_time: Process group start time
        :param step: Check interval
        :return: Whether to continue checking
        """
        time.sleep(step)
        need_next_step = True
        timeout = int(time.time() - s_time) > self.exe_timeout if self.exe_timeout else False
        if timeout:
            # Stop all subprocesses from processing new tasks
            self.cntr_terminate_event.set()
            need_next_step = False
            time.sleep(step)
        alive_process_count = 0
        for process in cntr_process_grp:
            if process.is_alive():
                if timeout:
                    logging.info("%s timeout, terminate it.", process.name)
                    os.kill(process.pid, signal.SIGINT)  # Stop the current task being processed by the subprocess
                alive_process_count += 1
                continue
            if process.exitcode != 0 and self.exe_halt_on_error:
                need_next_step = False
                logging.info("MainProcess Recv %s upload terminate event", process.name)
                break
        need_next_step = False if alive_process_count == 0 else need_next_step
        if not need_next_step:
            self._stop_cntr_process_grp(cntr_process_grp=cntr_process_grp, timeout=step)
        return need_next_step

    def _stop_cntr_process_grp(self, cntr_process_grp: List[Process], timeout: int = 1):
        """Stop Cntr process group

        :param cntr_process_grp: Cntr process group
        :param timeout: Wait timeout duration for exit
        """
        self.cntr_terminate_event.set()  # Stop all subprocesses from processing new tasks
        for process in cntr_process_grp:
            # When this script is called via build_ci.py through CMake, build_ci.py sends SIGINT
            # to the entire process group (including Cntr/Case subprocesses).
            # In this case, prefer waiting for subprocesses to exit on their own.
            if process.is_alive():
                process.join(timeout=timeout)
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)  # Stop the current task being processed by the subprocess
                logging.info("MainProcess Send download terminate event to %s.", process.name)
                process.join(timeout=timeout)

    def _cntr(self, cntr_id: int, exec_param, delay: int):
        """Container process

        Notes:
            1. During Container process execution, no Exception will be raised; case execution
               exception information is reported to the exception queue;
            2. The Container process exits when the task queue is empty or the termination event is set;

        :param cntr_id: ContainerId
        :param exec_param: ContainerParam
        """
        self._set_process_desc()
        self._cntr_set_cpu_affinity(cntr_id=cntr_id)
        self._cntr_set_gcov_prefix(cntr_id=cntr_id)
        ctx = TestsAccelerate.CntrContext(cntr_id=cntr_id, exec_param=exec_param)
        try:
            time.sleep(delay)
            while not self.cntr_terminate_event.is_set():
                # Case acquisition
                case_name = self._cntr_get_case()
                if case_name is None:
                    break
                # Case processing
                need_next = self._cntr_deal_case(case_name=case_name, ctx=ctx)
                if not need_next:
                    break  # No need to process next Case, exit
        except KeyboardInterrupt:
            pass
        # Container execution result statistics and reporting
        self._put_cntr_execution_info(info=ctx.brief)
        if not ctx.exit_code:
            logging.info("%s Send terminate event upload.", self._get_process_desc())
        logging.info(
            "%s Exit[%s] %s %s",
            self._get_process_desc(),
            ctx.exit_code,
            self._cntr_progress(update=True),
            self._case_progress(update=False),
        )
        exit(ctx.exit_code)  # Pass Container execution result via exit_code to trigger upstream awareness

    def _cntr_get_case(self) -> Optional[str]:
        """Get case to be executed

        :return: Name of the case to be executed, None means no pending cases
        """
        try:
            case_name = self.case_queue.get()
            self.case_queue.task_done()
        except queue.Empty:
            case_name = None  # Queue is empty, normal exit
        except KeyboardInterrupt:
            case_name = None  # Forced termination while waiting for pending cases, normal exit
        return case_name

    def _cntr_deal_case(self, case_name: str, ctx: CntrContext) -> Optional[bool]:
        """Process a single Case

        :param case_name: Case name
        :param ctx: Cntr processing context
        :return: Whether to continue processing the next Case
        """
        process = None
        try:
            # Case process startup
            process = Process(
                name=f"CaseProcess({self.cntr_name}[{ctx.cntr_id}] Case[{case_name}])",
                target=self._case,
                args=(
                    ctx.cntr_id,
                    ctx.exec_param,
                    case_name,
                ),
            )
            process.start()
            process.join()
        except KeyboardInterrupt:
            if process and process.is_alive():
                # Kill subprocess when forced termination occurs during case execution
                logging.info(
                    "%s Recv terminate event download, stop running Case[%s]", self._get_process_desc(), case_name
                )
                os.kill(process.pid, signal.SIGINT)
                process.join()  # Wait for Case process to finish
        finally:
            need_next = self._cntr_deal_case_finally(process=process, case_name=case_name, ctx=ctx)
        return need_next

    def _cntr_deal_case_finally(self, process: Process, case_name: str, ctx: CntrContext) -> bool:
        """Handle single Case completion

        :param process: CaseProcess
        :param case_name: Case name
        :param ctx: Cntr processing context
        :return: Whether to continue processing the next Case
        """
        if process is None:
            return False
        if process.exitcode == 0:
            ctx.success += 1
            return True
        ctx.failed += 1
        if not self.exe_halt_on_error:
            return True
        self.cntr_terminate_event.set()
        ctx.exit_code = process.exitcode
        logging.info("%s Recv Case[%s] upload terminate event.", self._get_process_desc(), case_name)
        return False

    def _execute_case(
        self, ctx: CaseContext, param: ExecParam, case_name: str
    ) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        """Unified case execution entry point - subclasses override this method to implement different modes"""
        return self.exe.run(params=[f"--gtest_filter={case_name}"], envs=param.get_envs())

    def _cntr_set_cpu_affinity(self, cntr_id: int):
        """Set CPU affinity at Cntr startup

        CPU affinity configured at the Cntr process level will be inherited by all Cases
        executed by that Cntr.
        """
        if not self.cpu_affinity_policy:
            return
        # Determine CPU group index
        if self.cpu_affinity_policy == 1:
            group_idx = cntr_id
        else:
            cpu_rank_num = cpu_count() // self.cpu_rank_size
            group_idx = cntr_id % cpu_rank_num
        # Calculate CPU group contents
        start_core = group_idx * self.cpu_rank_size
        end_core = min(start_core + self.cpu_rank_size, cpu_count())  # Prevent exceeding total CPU count
        cpu_core_list = [int(i) for i in range(start_core, end_core)]
        try:
            os.sched_setaffinity(0, cpu_core_list)  # 0 represents the current process PID
            # Verify the setting (optional)
        except OSError as e:
            # CPU affinity setting failure does not affect case execution
            logging.error("%s[%s] Failed to set CPU affinity: %s", self.cntr_name, cntr_id, e)
        current_affinity = os.sched_getaffinity(0)  # 0 represents the current process PID
        logging.debug("%s[%s] cpu affinity cores: %s", self.cntr_name, cntr_id, current_affinity)

    def _case(self, cntr_id: int, param: ExecParam, case_name: str):
        """Specific case execution process

        Implements execution context isolation for each Case via subprocess,
        preventing Cases from affecting each other.

        :param cntr_id: Container ID
        :param case_name: Case name
        """
        self._set_process_desc()
        ctx = TestsAccelerate.CaseContext(cntr_id=cntr_id, exec_param=param, case_name=case_name)
        run_desc = f"Run {self.mark}{self.exe.brief} Case({case_name})"
        try:
            logging.info("%s[%s] [BGN] %s", self.cntr_name, cntr_id, run_desc)
            ret, cmd, _ = self._execute_case(ctx, param, case_name)
            if ret.returncode:
                self._case_exception_exit(
                    cntr_id=cntr_id, cmd=cmd, ret_code=ret.returncode, out=ret.stdout, err=ret.stderr
                )
            else:
                msg = f"{ret.stdout}\n{ret.stderr}"
                logging.info(
                    "%s[%s] [END] %s %s Output Below:\n%s",
                    self.cntr_name,
                    cntr_id,
                    run_desc,
                    self._case_progress(update=True),
                    msg,
                )
                self._put_case_execution_info(info=ctx.brief)
        except subprocess.TimeoutExpired as e:
            self._put_case_terminate_info(info=ctx.brief)  # On timeout, proactively exit and report elapsed duration
            self._case_exception_exit(cntr_id=cntr_id, cmd=str(e), ret_code=1, out=None, err=str(e.output))
        except KeyboardInterrupt:
            # On forced termination, proactively exit and report elapsed duration
            self._put_case_terminate_info(info=ctx.brief)
            logging.info("%s Recv terminate event download, stop running.", self._get_process_desc())

    def _case_exception_exit(
        self, cntr_id: int, cmd: str, ret_code: int, out: Optional[str] = None, err: Optional[str] = None
    ):
        """Handle case execution process abnormal exit

        :param cntr_id: CntrId
        :param cmd: Failed command line
        :param ret_code: Process exit code
        :param out: Output information
        :param err: Exception information
        """
        # Collect error scene information and report
        msg = f"{self.cntr_name} : {cntr_id}\nCmd : {cmd}\nRetCode : {ret_code}\nstdout :\n{out}\nstderr :\n{err}"
        self._put_case_exception_info(info=msg)
        # Post-exception handling
        if self.exe_halt_on_error:
            self.cntr_terminate_event.set()
            logging.info("%s Send terminate event upload.", self._get_process_desc())
            exit(ret_code)  # Trigger Container execution process to detect Case execution exception

    def _cntr_progress(self, update=True) -> str:
        """Get Container processing progress; caller should acquire lock (dfx_output_lock) before calling"""
        if update:
            with self.cntr_exit_count.get_lock():
                self.cntr_exit_count.value += 1
        cnt = int(self.cntr_exit_count.value)
        pgs = cnt / self.cntr_num * 100
        return f"{self.cntr_name}Progress[{cnt}/{self.cntr_num} {pgs:.2f}%]"

    def _case_progress(self, update=True) -> str:
        """Get Case processing progress; caller should acquire lock (dfx_output_lock) before calling"""
        if update:
            with self.case_exec_count.get_lock():
                self.case_exec_count.value += 1
        cnt = int(self.case_exec_count.value)
        pgs = cnt / self.case_num * 100
        return f"CaseProgress[{cnt}/{self.case_num} {pgs:.2f}%]"

    def _put_case_execution_info(self, info: List[Any]):
        self.case_execution_queue.put(info)

    def _put_case_exception_info(self, info: str, chunk_size: int = 4096):
        for i in range(0, len(info), chunk_size):
            self.case_exception_queue.put(info[i:i + chunk_size])
        self.case_exception_queue.put("")  # Insert separator

    def _put_case_terminate_info(self, info: List[Any]):
        self.case_terminate_queue.put(info)

    def _put_cntr_execution_info(self, info: List[Any]):
        self.cntr_execution_queue.put(info)
