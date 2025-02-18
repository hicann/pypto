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
"""需要执行 target 工具基类定义.
"""
import csv
import shlex
import shutil
import subprocess
import sys
from abc import abstractmethod, ABC
from datetime import timezone, datetime, timedelta
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional

from utils.executable import Executable

from .case_abc import CaseAbc
from .tools_abc import ToolsAbc


class ToolsRunAbc(ToolsAbc, ABC):

    def __init__(self, args):
        super().__init__(args=args)

        # 路径管理
        self.timestamp: str = str(datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"))
        self.output_root: Path = Path(Path.cwd(), "../../../../../output_tools", self.timestamp).resolve()

        # 用例执行参数
        self.exe: Executable = Executable(file=Path(args.target[0]).resolve(), envs=args.envs)

        # 用例管理
        self.case_lines: List[Dict[str, Any]] = []  # 原始 enable 的 case line
        self.case_list: List[CaseAbc] = []  # 具体 case list, 子类负责初始化
        self.cases_golden_impl_path: List[Path] = args.cases_golden_impl_path
        self.cases_golden_output_path: Path = Path(args.cases_golden_output_path[0]).resolve()  # 指定 Golden 路径
        self.cases_golden_output_clean: bool = args.cases_golden_output_clean  # 清理 Golden 标记
        self.init_case_lines(args=args)
        self.init_case_list(args=args)

        # 执行控制
        self.halt_on_error: bool = args.halt_on_error  # 失败时终止后续 Case 执行
        self.device_list: List[int] = []
        self.init_device_list(args=args)

    @property
    def brief(self) -> List[Any]:
        ver = sys.version_info
        datas = [["Python3", f"{sys.executable} ({ver.major}.{ver.minor}.{ver.micro})"],
                 ["Executable", self.exe.file]]
        for _k, _v in self.exe.envs.items():
            datas.append([_k, _v])
        datas += [["CaseNum", len(self.case_list)],
                  ["CaseGoldenImplPathNum", len(self.cases_golden_impl_path)],
                  ["CaseGoldenOutputPath", self.cases_golden_output_path],
                  ["CaseGoldenOutputClean", self.cases_golden_output_clean],
                  ["TimeStamp", self.timestamp],
                  ["OutputRoot", self.output_root],
                  ["HaltOnError", self.halt_on_error],
                  ["DeviceIdList", self.device_list]]
        return super().brief + datas

    def init_case_lines(self, args):
        cases_names: str = args.cases[0] if args.cases else ""
        cases_csv: Path = Path(args.cases_csv_file[0]).resolve() if args.cases_csv_file else None
        # cases lines
        if cases_names == "":  # 未指定 --cases, 使用 .csv
            if cases_csv is None:
                raise ValueError("Neither specify cases nor cases_csv_file.")
            with open(cases_csv, mode='r', newline='') as fp:
                csv_reader = csv.DictReader(fp)
                csv_lines = list(csv_reader)
            for line in csv_lines:
                if line.get(CaseAbc.FieldType.Enable.value, "False").lower() == "true":
                    self.case_lines.append(dict(line))
        else:  # 指定 --cases, 使用指定值
            cases_name_list = str(cases_names).split(":")
            for name in cases_name_list:
                self.case_lines.append({CaseAbc.FieldType.Network.value: "ManualSpecify",
                                        CaseAbc.FieldType.Name.value: name,
                                        CaseAbc.FieldType.Enable.value: "True"})

    @abstractmethod
    def init_case_list(self, args):
        pass

    def init_device_list(self, args):
        self.device_list = [0]
        if args.device is not None:
            self.device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]

    def clean(self) -> bool:
        if self.clean_flg:
            output_tools_name: str = Path(self.output_root).parent.name
            output_tools_dir = Path(Path(self.output_root, "../../"), output_tools_name).resolve()
            if output_tools_dir.exists():
                shutil.rmtree(output_tools_dir)
        return True

    def prepare(self) -> bool:
        # Golden 生成
        cases_name_str: str = ":".join([c.name for c in self.case_list])
        golden_ctrl_py: Path = Path(self.source_root, "framework/tests/cmake/scripts/golden_ctrl.py")
        cmd = f"{sys.executable} {golden_ctrl_py} -o={self.cases_golden_output_path} -c={cases_name_str}"
        for imp_d in self.cases_golden_impl_path:
            cmd += f" --path={imp_d}"
        cmd += " --clean" if self.cases_golden_output_clean else ""
        ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
        ret.check_returncode()
        return True

    def run_case(self, cs: CaseAbc, device_id: int,
                 envs: Optional[Dict[str, str]] = None) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        act_env = envs if envs else {}
        act_env.update({"TILE_FWK_DEVICE_ID": f"{device_id}"})
        return self.exe.run(gtest_filter=cs.name, envs=act_env)

    @abstractmethod
    def process_case_prepare(self, cs: CaseAbc, device_id: int) -> bool:
        pass

    @abstractmethod
    def process_case_process(self, cs: CaseAbc, device_id: int) -> bool:
        pass

    @abstractmethod
    def process_case_post(self, cs: CaseAbc, device_id: int) -> bool:
        pass

    @abstractmethod
    def post(self) -> bool:
        pass
