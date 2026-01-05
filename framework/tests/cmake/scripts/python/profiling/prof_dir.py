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
""" 性能分析工具, PROF结果目录.
"""
import copy
import csv
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from profiling.prof_case import ProfCase


class ProfDir:
    """Profiling 结果目录
    """

    def __init__(self, src_root: Path, path: Path, prof_case: ProfCase, device_id: int = 0, level: str = "l1"):
        self._src_root: Path = src_root
        self._device_id: int = device_id
        self._level: str = level
        self._work_dir: Path = path.resolve()
        self._data_dir: Path = Path(self._work_dir, f"device_{self._device_id}/data")
        self._rest_dir: Path = Path(self._work_dir, f"device_{self._device_id}/result")
        # 解析/统计 结果
        self._rest: ProfCase = copy.deepcopy(prof_case)
        self._statistic_rst_file: Path = Path(self._rest_dir, ProfCase.STATISTIC_RST_FILE_NAME)

    @property
    def working_dir(self) -> Path:
        return self._work_dir

    @property
    def timestamp(self) -> str:
        return self._work_dir.name

    @property
    def result_case(self) -> ProfCase:
        return self._rest

    @property
    def result_dump_dir(self) -> Path:
        d = Path(self._rest_dir, "dump")
        return d

    @property
    def statistic_rst_file(self) -> Path:
        return self._statistic_rst_file

    def check(self) -> bool:
        return self._data_dir.exists() and any(self._data_dir.iterdir())

    def delete(self):
        shutil.rmtree(self._work_dir)

    def parse(self):
        self.parse_work_flow()
        self.parse_perfetto()

    def parse_work_flow(self):
        task_info_csv_file: Path = Path(self._rest_dir, "work_flow/tilefwk_task_info.csv")
        if not task_info_csv_file.exists():
            self._rest_dir.mkdir(parents=True, exist_ok=True)
            wf_py = Path(self._src_root, "tools/tilefwk_prof_data_parser.py")
            cmd = f"{sys.executable} {wf_py} -p {self._data_dir} --output=work_flow"
            ret = subprocess.run(shlex.split(cmd), cwd=self._rest_dir,
                                 capture_output=True, check=False, text=True, encoding='utf-8')
            if ret.returncode:
                logging.error("Cmd[%s] failed, RetCode[%s]\nstdout:\n%s\nstderr:\n%s",
                              cmd, ret.returncode, ret.stdout, ret.stderr)
                raise subprocess.CalledProcessError(ret.returncode, ret.args, ret.stdout, ret.stderr)
        t_min = int(sys.maxsize)
        t_max = -1
        with open(task_info_csv_file, mode='r', newline='', encoding='utf-8') as fp:
            csv_reader = csv.DictReader(fp)
            lines = list(csv_reader)
        for line in lines:
            v1 = int(line.get("startCycle"))
            v2 = int(line.get("endCycle"))
            t_min = min(min(v1, v2), t_min)
            t_max = max(max(v1, v2), t_max)
        self._rest.update(k=ProfCase.FieldType.Cycle.value, v=(t_max - t_min))

    def parse_perfetto(self):
        prof_log_json = Path(self._rest_dir, "work_flow/tilefwk_prof_data.json")
        topo_json = Path(self.result_dump_dir, "topo.json")
        if prof_log_json.exists() and topo_json.exists():
            # 多卡预热场景下, 暂无法获取准确 json, 暂不支持 perfetto 输出
            py = Path(self._src_root, "tools/draw_swim_lane.py")
            cmd = f"{sys.executable} {py} {prof_log_json} {topo_json}"
            ret = subprocess.run(shlex.split(cmd), cwd=self._src_root,
                                 capture_output=True, check=False, text=True, encoding='utf-8')
            if ret.returncode:
                logging.error("Cmd[%s] failed, RetCode[%s]\nstdout:\n%s\nstderr:\n%s",
                              cmd, ret.returncode, ret.stdout, ret.stderr)
                raise subprocess.CalledProcessError(ret.returncode, ret.args, ret.stdout, ret.stderr)
