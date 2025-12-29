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
"""GTest 执行加速
"""
import argparse
import dataclasses
import logging
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from multiprocessing import JoinableQueue, Event, Process, Value
from typing import List, Any, Optional, Tuple, Dict, Callable

from utils.args_action import ArgsEnvDictAction, ArgsGTestFilterListAction
from utils.executable import Executable
from utils.table import Table


class GTestAccelerate(ABC):
    """GTest 加速
    """

    @dataclasses.dataclass
    class ExecParam:
        """执行参数
        """
        cntr_id: Optional[int] = None
        envs_func: Optional[Callable] = None
        custom: Optional[Any] = None

        def __init__(self, cntr_id: int, envs_func: Optional[Callable] = None, custom: Optional[Any] = None):
            self.cntr_id = cntr_id
            self.envs_func = envs_func
            self.custom = custom

        def get_envs(self) -> Optional[Dict[str, str]]:
            """获取额外的环境变量配置
            """
            if self.envs_func:
                return self.envs_func(self)
            return None

    @dataclasses.dataclass
    class ExecResult:
        """执行结果
        """
        cntr_name: str = "Cntr"
        duration: Optional[timedelta] = None
        cntr_execution_details: JoinableQueue = JoinableQueue()
        cntr_duration_dict: Dict[int, timedelta] = dataclasses.field(default_factory=dict)
        case_execution_details: JoinableQueue = JoinableQueue()
        case_exception_details: JoinableQueue = JoinableQueue()
        case_terminate_details: JoinableQueue = JoinableQueue()

        def get_cntr_exec_info(self) -> Tuple[str, str]:
            """获取 Container 执行信息统计.

            :returns:
                Tuple[str, str]:
                    - Container 执行信息统计表(str)
                    - Container 并行执行收益描述(str)
            """
            heads: List[str] = [self.cntr_name, "Total", "Success", "Failed", "Duration"]
            datas: List[List[Any]] = []
            duration_sum: timedelta = timedelta()
            while not self.cntr_execution_details.empty():
                _brief = self.cntr_execution_details.get()
                devs_id: int = int(_brief[0])
                case_total: int = int(_brief[1])
                case_pass: int = int(_brief[2])
                case_fail: int = int(_brief[3])
                devs_duration: timedelta = _brief[-1]
                self.cntr_duration_dict[devs_id] = devs_duration
                duration_sum += devs_duration
                datas.append([devs_id, case_total, case_pass, case_fail, f"{devs_duration.total_seconds():.2f}"])
                self.cntr_execution_details.task_done()
            brief: str = "\nNone"
            if len(datas) != 0:
                brief = Table.table(datas=datas, headers=heads)
            # 并行执行收益计算
            rate: float = float((duration_sum - self.duration) / self.duration) * 100
            desc: str = f"Duration {self.duration.total_seconds():.2f} secs, Revenue(Act/Ori, "
            desc += f"{self.duration.total_seconds():.2f}/{duration_sum.total_seconds():.2f}) {rate:.2f}%"
            return f"\n\n{self.cntr_name} Execution Brief:{brief}", desc

        def get_case_exec_terminate_info(self) -> Tuple[str, int]:
            """获取 Case 执行终止信息.

            :returns:
                Tuple[str, int]:
                    - Case 终止执行情况信息
                    - Case 终止执行数量
            """
            heads: List[str] = ["Idx", self.cntr_name, "CaseName", "Duration"]
            datas: List[Any] = []
            while not self.case_terminate_details.empty():
                _brief = self.case_terminate_details.get()
                job_id: int = int(_brief[0])
                case_name: str = str(_brief[1])
                case_duration: timedelta = _brief[2]
                datas.append([job_id, case_name, f"{case_duration.total_seconds():.2f}"])
                self.case_terminate_details.task_done()
            brief: str = "\nNone"
            if len(datas) != 0:
                datas = [[f"{idx}/{len(datas)}"] + ele for idx, ele in enumerate(datas, start=1)]
                brief = Table.table(datas=datas, headers=heads)
            return f"\n\nCase Terminate Brief({len(datas)}):{brief}", len(datas)

        def get_case_exec_exception_info(self) -> Tuple[str, int]:
            """获取 Case 执行异常信息.

            :returns:
                Tuple[str, int]:
                    - Case 异常执行情况信息
                    - Case 异常执行数量
            """
            datas: List[str] = []
            brief: str = ""
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

        def get_case_exec_duration_info(self) -> str:
            """获取 Case 执行耗时统计信息.

            :return: Case 执行耗时统计信息.
            """
            heads: List[str] = [self.cntr_name, "CaseName", "Duration",
                                f"Ratio({self.cntr_name})", "Ratio(Total)"]
            datas: List[List[Any]] = []
            while not self.case_execution_details.empty():
                _brief = self.case_execution_details.get()
                job_idx: int = _brief[0]
                case_name: str = str(_brief[1])
                case_duration: timedelta = _brief[2]
                job_duration: timedelta = self.cntr_duration_dict[job_idx]
                ratio_job: float = float(case_duration / job_duration) * 100
                ratio_process: float = float(case_duration / self.duration) * 100
                datas.append(
                    [job_idx, case_name, case_duration.total_seconds(),
                     f"{case_duration.total_seconds():.2f}/{job_duration.total_seconds():.2f} {ratio_job:.2f}%",
                     f"{case_duration.total_seconds():.2f}/{self.duration.total_seconds():.2f} "
                     f"{ratio_process:.2f}%"])
                self.case_execution_details.task_done()
            brief: str = "\nNone"
            if len(datas) != 0:
                # 把 data 按耗时降序重排, 重排后转换格式
                duration_idx: int = 2  # 2 is idx of duration
                datas = sorted(datas, key=lambda x: x[duration_idx], reverse=True)
                for item in datas:
                    item[duration_idx] = f"{item[duration_idx]:.2f}"
                brief = Table.table(datas=datas, headers=heads, auto_sort=False)
            return f"\n\nCase Duration Brief:{brief}"

    @dataclasses.dataclass
    class CntrContext:
        """Cntr处理上下文
        """
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
        """Case处理上下文
        """
        cntr_id: int = 0
        exec_param: Optional[Any] = None
        ts: Optional[datetime] = None
        gtest_filter: str = ""

        def __init__(self, cntr_id: int, exec_param, gtest_filter):
            self.cntr_id = cntr_id
            self.exec_param = exec_param
            self.gtest_filter = gtest_filter
            self.ts = datetime.now(tz=timezone.utc)

        @property
        def brief(self) -> List[Any]:
            return [self.cntr_id, self.gtest_filter, (datetime.now(tz=timezone.utc) - self.ts)]

    @dataclasses.dataclass
    class MoveContext:
        """Move进程处理上下文
        """
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

    def __init__(self, args, params: List[ExecParam], cntr_name: str = "Cntr"):
        """
        :param args: 命令行参数
        :param params: 执行参数
        :param cntr_name: 容器名称, 用于回显内容
        """
        # 用例执行参数, 执行行为控制参数
        self.exe: Executable = Executable(file=args.target[0], envs=args.envs, timeout=args.timeout_case)
        self.exe_params: List[GTestAccelerate.ExecParam] = params
        self.exe_result: GTestAccelerate.ExecResult = GTestAccelerate.ExecResult(cntr_name=cntr_name)
        self.exe_timeout: Optional[int] = args.timeout
        self.exe_halt_on_error: bool = args.halt_on_error  # 失败时终止后续 Case 执行

        # 用例管理
        self.case_list: List[str] = args.cases
        self.case_queue: JoinableQueue = JoinableQueue()
        self.case_execution_queue: JoinableQueue = JoinableQueue()  # Case 正常执行结束时，收集相关信息
        self.case_exception_queue: JoinableQueue = JoinableQueue()  # Case 执行失败时, 用于收集错误信息
        self.case_terminate_queue: JoinableQueue = JoinableQueue()  # Case 被终止执行时, 收集相关信息
        self.case_exec_count: Value = Value('i', 0)  # DFX, 统计 Case 完成进度

        # 容器管理
        self.cntr_name: str = cntr_name
        self.cntr_execution_queue: JoinableQueue = JoinableQueue()  # Container 执行结果统计上报
        self.cntr_terminate_event: Event = Event()  # 用于通知其他 Container 进程结束运行
        self.cntr_exit_count: Value = Value('i', 0)  # DFX, 统计 Container 退出进度

        # 其他
        if len(self.exe_params) == 0:
            raise ValueError("ExecParams is empty, won't run any task.")
        if len(params) > len(self.case_list):
            logging.info("CaseNum(%s) less than len(ExecParams)=%s, will only start the first %s %s.",
                         len(self.case_list), len(self.exe_params), len(self.case_list), self.cntr_name)
            self.exe_params = self.exe_params[:len(self.case_list)]
        logging.info("\n\n%s Accelerate Args:%s", self.mark, Table.table(datas=self.brief))

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        lst = [
            ["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
            ["Timeout", self.exe_timeout],
            ["HaltOnError", self.exe_halt_on_error],
            [f"{self.cntr_name}Num", len(self.exe_params)],
            [f"{self.cntr_name}List", [p.cntr_id for p in self.exe_params]],
            ["CaseNum", len(self.case_list)],
            ["CaseTimeout", self.exe.timeout],
            ["Executable", self.exe.file],
        ]
        for k, v in self.exe.envs.items():
            lst.append([k, v])
        return lst

    @property
    @abstractmethod
    def mark(self) -> str:
        pass

    @staticmethod
    def reg_args(parser: argparse.ArgumentParser):
        """注册命令行参数

        注意事项:
            1. 本函数应与 get_container_manager 函数协同使用;
            2. 本函数注册了 'gtest_filter' 字段, 但 get_container_manager 内不会解析处理, 该字段应由使用者解析处理;

        :param parser: ArgumentParser 外部创建
        """
        # 执行所需参数
        parser.add_argument("-t", "--target", nargs=1, type=str, required=True,
                            help="Specific target executable file path.")
        parser.add_argument("-e", "--env",
                            nargs="+", action=ArgsEnvDictAction, default={}, dest="envs",
                            help="Specify additional environment variables to set when executing the target.")
        parser.add_argument("--timeout", nargs="?", type=int, default=None,
                            help="Timeout for executing all cases.")
        parser.add_argument("--timeout_case", nargs="?", type=int, default=None,
                            help="Timeout for executing single case.")
        parser.add_argument("--halt_on_error", action="store_true", default=False,
                            help="If any case failed, subsequent cases are not executed.")
        # 用例参数
        parser.add_argument("--gtest_filter",
                            nargs="+", action=ArgsGTestFilterListAction, default=[], required=True, dest="cases",
                            help="GTestFilter, multiple cases are separated by ':'")

    @staticmethod
    def _move(src: JoinableQueue, dst: JoinableQueue):
        GTestAccelerate._set_process_desc()
        ctx: GTestAccelerate.MoveContext = GTestAccelerate.MoveContext(src=src, dst=dst)
        while True:
            if not ctx.move():
                break
        logging.info("%s Exist, Move %s elements.", GTestAccelerate._get_process_desc(), ctx.ele_count)

    @staticmethod
    def _get_process_desc() -> str:
        cur_process = multiprocessing.current_process()
        return f"{cur_process.name}"

    @staticmethod
    def _set_process_desc():
        try:
            import setproctitle
            setproctitle.setproctitle(GTestAccelerate._get_process_desc())
        except ModuleNotFoundError:
            pass

    def process(self):
        """执行任务
        """
        # 执行流程
        ts: datetime = datetime.now(tz=timezone.utc)
        self._main()
        self.exe_result.duration = datetime.now(tz=timezone.utc) - ts

    def post(self) -> bool:
        """后处理, 获得执行结果汇总
        """
        # Cntr 执行信息收集汇总
        cntr_exec_brief, cntr_revenue_desc = self.exe_result.get_cntr_exec_info()

        # Case 执行信息收集汇总
        case_exec_brief, case_exec_result = self._post_case_exec_info()

        out: str = f"{self.mark}, HaltOnError({self.exe_halt_on_error}), {cntr_revenue_desc}"
        out += cntr_exec_brief
        out += case_exec_brief

        if case_exec_result:
            logging.info(out)
        else:
            logging.error(out)
        return case_exec_result

    def _post_case_exec_info(self) -> Tuple[str, bool]:
        """获取 Case 执行信息.

        :returns:
            Tuple[str, bool]:
                - Case 执行情况信息
                - Case 执行成功与否判定结果
        """
        terminate_brief, terminate_count = self.exe_result.get_case_exec_terminate_info()
        exception_brief, exception_count = self.exe_result.get_case_exec_exception_info()
        duration_brief = self.exe_result.get_case_exec_duration_info()

        # Case 执行总体情况汇总
        remaining_count: int = 0
        while not self.case_queue.empty():
            cs = self.case_queue.get()
            if cs is not None:
                remaining_count += 1
            self.case_queue.task_done()
        success_count: int = len(self.case_list) - remaining_count - terminate_count - exception_count
        execution_heads = ["Total", "Success", "Failed", "Terminate", "Remaining"]
        execution_datas = [[len(self.case_list), success_count, exception_count, terminate_count, remaining_count]]
        execution_brief = Table.table(datas=execution_datas, headers=execution_heads)
        execution_brief = f"\n\nCase Execution Brief:{execution_brief}"

        rst: bool = (terminate_count + exception_count + remaining_count) == 0
        out: str = execution_brief + duration_brief + terminate_brief + exception_brief
        return out, rst

    def _main(self):
        """用例执行, 管理执行状态(主进程)

        :return: 执行成功与否
        """
        # 创建并启动子进程, 进行任务处理
        cntr_step: int = 1
        cntr_process_group: List[Process] = []
        try:
            # 任务准备
            self._push_all_case_sync()
            # 启动上报监控进程
            self._start_move_process_grp()
            # 创建并启动任务子进程, 进行任务处理
            cntr_process_group = self._start_cntr_process_grp()
            # 等待任务处理结束
            self._join_cntr_process_grp(cntr_process_grp=cntr_process_group, step=cntr_step)
        except KeyboardInterrupt:
            logging.info("MainProcess Recv download terminate event.")
        finally:
            self._stop_cntr_process_grp(cntr_process_grp=cntr_process_group, timeout=cntr_step)
            self._stop_move_process_grp()

    def _push_all_case_sync(self):
        """以同步方式将待执行用例插入待执行队列, 按 Container 数量插入终止信号
        """
        for cs in self.case_list:
            self.case_queue.put(cs)
        for _ in range(len(self.exe_params)):
            self.case_queue.put(None)

    def _start_move_process_grp(self) -> List[Process]:
        """启动 Move 进程组

        :return: Move 进程列表
        """
        move_grp: List[Process] = []
        desc_list = self._get_move_process_grp_desc_list()
        for name, src_queue, dst_queue in desc_list:
            process: Process = Process(name=f"MoveProcess({name})", target=self._move, args=(src_queue, dst_queue,))
            process.start()
            move_grp.append(process)
        return move_grp

    def _stop_move_process_grp(self):
        """停止 Move 进程组
        """
        desc_list = self._get_move_process_grp_desc_list()
        for _, src_queue, _ in desc_list:
            src_queue.put(None)
            src_queue.join()

    def _get_move_process_grp_desc_list(self) -> List[Tuple[str, JoinableQueue, JoinableQueue]]:
        pairs: List[Tuple[str, JoinableQueue, JoinableQueue]] = [
            ("CaseExecution", self.case_execution_queue, self.exe_result.case_execution_details),
            ("CaseException", self.case_exception_queue, self.exe_result.case_exception_details),
            ("CaseTerminate", self.case_terminate_queue, self.exe_result.case_terminate_details),
            (f"{self.cntr_name}Execution", self.cntr_execution_queue, self.exe_result.cntr_execution_details),
        ]
        return pairs

    def _start_cntr_process_grp(self, delay: int = 5) -> List[Process]:
        """启动 Cntr 进程组

        :param delay: 各 Cntr 启动后, 处理具体 Case 前延迟时长, 在多消费者模式下, 各消费者启动时增加一定延迟, 等待所有消费者启动完成
        :return: Cntr 进程组
        """
        process_group: List[Process] = []
        for exec_param in self.exe_params:
            process = Process(name=f"{self.cntr_name}Process({self.cntr_name}[{exec_param.cntr_id}])",
                              target=self._cntr, args=(exec_param.cntr_id, exec_param, delay,))
            process_group.append(process)
            process.start()
        return process_group

    def _join_cntr_process_grp(self, cntr_process_grp: List[Process], step: int = 1):
        """以同步方式等待 Cntr 进程组完成

        :param cntr_process_grp: Cntr 进程组
        :param step: 内部检测步长, 单位为秒
        """
        s_time = time.time()
        while True:
            if not self._wait_cntr_one_step(cntr_process_grp=cntr_process_grp, s_time=s_time, step=step):
                break

    def _wait_cntr_one_step(self, cntr_process_grp: List[Process], s_time: time, step: int = 1) -> bool:
        """阻塞当前进程, 检测 Cntr 进程组完成情况

        :param cntr_process_grp: Cntr 进程组
        :param s_time: 进程组启动时间
        :param step: 检测步长
        :return: 是否要继续检测
        """
        time.sleep(step)
        need_next_step: bool = True
        timeout: bool = int(time.time() - s_time) > self.exe_timeout if self.exe_timeout else False
        if timeout:
            # 停止所有子进程对新任务的处理
            self.cntr_terminate_event.set()
            need_next_step = False
            time.sleep(step)
        alive_process_count: int = 0
        for process in cntr_process_grp:
            if process.is_alive():
                if timeout:
                    logging.info("%s timeout, terminate it.", process.name)
                    os.kill(process.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
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
        """停止 Cntr 进程组

        :param cntr_process_grp: Cntr 进程组
        :param timeout: 等待退出超时时长
        """
        self.cntr_terminate_event.set()  # 停止所有子进程对新任务的处理
        for process in cntr_process_grp:
            # 当通过 build_ci.py 经 CMake 调用本脚本时, build_ci.py 会向整个进程组(包括 Cntr/Case 子进程)发送 SIGINT 信号.
            # 此时优先等待子进程自主退出.
            if process.is_alive():
                process.join(timeout=timeout)
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)  # 停止对应子进程当前处理的任务
                logging.info("MainProcess Send download terminate event to %s.", process.name)
                process.join(timeout=timeout)

    def _cntr(self, cntr_id: int, exec_param, delay: int):
        """Container 进程

        说明:
            1. Container 进程执行时, 不会产生 Exception, 用例执行异常信息会上报至异常信息队列;
            2. Container 进程在任务队列为空, 或异常终止事件被设置时退出;

        :param cntr_id: ContainerId
        :param exec_param: ContainerParam
        """
        self._set_process_desc()
        ctx: GTestAccelerate.CntrContext = GTestAccelerate.CntrContext(cntr_id=cntr_id, exec_param=exec_param)
        try:
            time.sleep(delay)
            while not self.cntr_terminate_event.is_set():
                # 用例获取
                gtest_filter = self._cntr_get_case()
                if gtest_filter is None:
                    break
                # 用例处理
                need_next = self._cntr_deal_case(gtest_filter=gtest_filter, ctx=ctx)
                if not need_next:
                    break  # 不需处理下一个 Case, 退出处理
        except KeyboardInterrupt:
            pass
        # Container 执行结果统计与上报
        self._put_cntr_execution_info(info=ctx.brief)
        if not ctx.exit_code:
            logging.info("%s Send terminate event upload.", self._get_process_desc())
        logging.info("%s Exist[%s] %s %s",
                     self._get_process_desc(), ctx.exit_code,
                     self._cntr_progress(update=True), self._case_progress(update=False))
        exit(ctx.exit_code)

    def _cntr_get_case(self) -> Optional[str]:
        """获取待执行用例

        :return: 待执行用例名, None 表示无待执行用例
        """
        try:
            gtest_filter = self.case_queue.get()
            self.case_queue.task_done()
        except queue.Empty:
            gtest_filter = None  # 队列为空, 正常退出
        except KeyboardInterrupt:
            gtest_filter = None  # 等待获取待执行用例过程中, 强制终止时, 正常退出
        return gtest_filter

    def _cntr_deal_case(self, gtest_filter: str, ctx: CntrContext) -> Optional[bool]:
        """处理单个 Case

        :param gtest_filter: GTestFilter
        :param ctx: Cntr 处理上下文
        :return: 需要继续处理下个 Case
        """
        process: Optional[Process] = None
        try:
            # 用例进程启动
            process = Process(name=f"CaseProcess({self.cntr_name}[{ctx.cntr_id}] Case[{gtest_filter}])",
                              target=self._case, args=(ctx.cntr_id, ctx.exec_param, gtest_filter,))
            process.start()
            process.join()
        except KeyboardInterrupt:
            if process and process.is_alive():
                # 用例执行过程中, 强制终止时, 杀停子进程
                logging.info("%s Recv terminate event download, stop running Case[%s]",
                             self._get_process_desc(), gtest_filter)
                os.kill(process.pid, signal.SIGINT)
                process.join()  # 等待 Case 进程结束
        finally:
            need_next = self._cntr_deal_case_finally(process=process, gtest_filter=gtest_filter, ctx=ctx)
        return need_next

    def _cntr_deal_case_finally(self, process: Process, gtest_filter: str, ctx: CntrContext) -> bool:
        """处理单个 Case 结束

        :param process: CaseProcess
        :param gtest_filter: GTestFilter
        :param ctx: Cntr 处理上下文
        :return: 需要继续处理下个 Case
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
        logging.info("%s Recv Case[%s] upload terminate event.", self._get_process_desc(), gtest_filter)
        return False

    def _case(self, cntr_id: int, param: ExecParam, gtest_filter: str):
        """具体用例执行进程

        通过子进程实现各 Case 执行上下文隔离, 避免 Case 间相互影响

        :param cntr_id: Container ID
        :param gtest_filter: GTestFilter
        """
        self._set_process_desc()
        ctx: GTestAccelerate.CaseContext = GTestAccelerate.CaseContext(cntr_id=cntr_id, exec_param=param,
                                                                       gtest_filter=gtest_filter)
        run_desc: str = f"Run {self.mark}{self.exe.brief} GTestFilter({gtest_filter})"
        try:
            logging.info("%s[%s] [BGN] %s", self.cntr_name, cntr_id, run_desc)
            ret, cmd, _ = self.exe.run(gtest_filter=ctx.gtest_filter, envs=ctx.exec_param.get_envs())
            if ret.returncode:
                self._case_exception_exit(cntr_id=cntr_id, cmd=cmd,
                                          ret_code=ret.returncode, out=ret.stdout, err=ret.stderr)
            else:
                msg = f"{ret.stdout}\n{ret.stderr}"
                logging.info("%s[%s] [END] %s %s Output Below:\n%s",
                             self.cntr_name, cntr_id, run_desc, self._case_progress(update=True), msg)
                self._put_case_execution_info(info=ctx.brief)
        except subprocess.TimeoutExpired as e:
            self._put_case_terminate_info(info=ctx.brief)  # 执行超时时, 主动退出执行, 上报已运行时长
            self._case_exception_exit(cntr_id=cntr_id, cmd=str(e),
                                      ret_code=1, out=None, err=str(e.output))
        except KeyboardInterrupt:
            self._put_case_terminate_info(info=ctx.brief)  # 强制终止时, 主动退出执行, 上报已运行时长
            logging.info("%s Recv terminate event download, stop running.", self._get_process_desc())

    def _case_exception_exit(self, cntr_id: int, cmd: str, ret_code: int,
                             out: Optional[str] = None, err: Optional[str] = None):
        """用例执行进程异常退出处理

        :param cntr_id: CntrId
        :param cmd: 失败命令行
        :param ret_code: 进程退出码
        :param out: 输出信息
        :param err: 异常信息
        """
        # 收集错误现场信息并上报
        msg: str = (f"{self.cntr_name} : {cntr_id}\n"
                    f"Cmd : {cmd}\n"
                    f"RetCode : {ret_code}\n"
                    f"stdout :\n{out}\n"
                    f"stderr :\n{err}")
        self._put_case_exception_info(info=msg)
        # 异常后处理
        if self.exe_halt_on_error:
            self.cntr_terminate_event.set()
            logging.info("%s Send terminate event upload.", self._get_process_desc())
            exit(ret_code)  # 触发 Container 执行进程感知 Case 执行异常

    def _cntr_progress(self, update=True) -> str:
        """获取 Container 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            with self.cntr_exit_count.get_lock():
                self.cntr_exit_count.value += 1
        cnt: int = int(self.cntr_exit_count.value)
        pgs: float = cnt / len(self.exe_params) * 100
        return f"{self.cntr_name}Progress[{cnt}/{len(self.exe_params)} {pgs:.2f}%]"

    def _case_progress(self, update=True) -> str:
        """获取 Case 处理进展, 调用本函数前, 由调用方加锁(dfx_output_lock)
        """
        if update:
            with self.case_exec_count.get_lock():
                self.case_exec_count.value += 1
        cnt: int = int(self.case_exec_count.value)
        pgs: float = cnt / len(self.case_list) * 100
        return f"CaseProgress[{cnt}/{len(self.case_list)} {pgs:.2f}%]"

    def _put_case_execution_info(self, info: List[Any]):
        self.case_execution_queue.put(info)

    def _put_case_exception_info(self, info: str, chunk_size: int = 4096):
        for i in range(0, len(info), chunk_size):
            self.case_exception_queue.put(info[i:i + chunk_size])
        self.case_exception_queue.put("")  # 插入分隔符

    def _put_case_terminate_info(self, info: List[Any]):
        self.case_terminate_queue.put(info)

    def _put_cntr_execution_info(self, info: List[Any]):
        self.cntr_execution_queue.put(info)
