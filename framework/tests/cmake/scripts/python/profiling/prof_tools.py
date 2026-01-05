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
""" 性能分析工具.
"""
import logging
import math
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from profiling.prof_case import ProfCase
from profiling.prof_dir import ProfDir
from utils.table import Table
from utils.tools_run_abc_sp import ToolsRunAbcSp


class ProfTools(ToolsRunAbcSp):

    def __init__(self, args):
        # 用例管理
        self.level: str = args.prof_level
        self.warn_up_cnt: Optional[int] = args.prof_warn_up_cnt and args.prof_warn_up_cnt[0]
        self.try_cnt: Optional[int] = args.prof_try_cnt and args.prof_try_cnt[0]
        self.max_cnt: Optional[int] = args.prof_max_cnt and args.prof_max_cnt[0]
        super().__init__(args=args)

        # 与当前用例关联
        self.case_prof_ori_dir_root: Optional[Path] = None  # 表示当前用例产生原始 PROF 的根目录
        self.case_prof_dirs: Dict[str, ProfDir] = {}  # 存储当前用例的所有 ProfDir, {TimeStamp: ProfDir}
        self.case_rst_dir_root: Optional[Path] = None  # 表示当前用例产生结果根目录
        self.case_statistic_all_file: Optional[Path] = None
        self.case_statistic_rst_file: Optional[Path] = None
        self.case_output_dir_bak: Optional[Path] = None  # 适配二进制复用场景

        # 与本次运行的所有用例关联
        self.statistic_rst_file: Path = Path(self.output_root, f"profiling/{ProfCase.STATISTIC_RST_FILE_NAME}")
        self.statistic_all_file: Path = Path(self.output_root, f"profiling/{ProfCase.STATISTIC_ALL_FILE_NAME}")

        logging.info("\n\nTools Profiling Args:\n%s", Table.table(datas=self.brief))
        if len(self.device_list) > 1:
            logging.info("Specify Multi Device(%s), use first device(%s) to profiling.",
                         self.device_list, self.device_id)

    @property
    def brief(self) -> List[Any]:
        datas = [["Level", self.level],
                 ["WarnUpCnt", self.warn_up_cnt if self.warn_up_cnt is not None else "None"],
                 ["TryCnt", self.try_cnt if self.try_cnt is not None else "None"],
                 ["MaxCnt", self.max_cnt if self.max_cnt is not None else "None"]]
        return super().brief + datas

    def init_case_list(self, args):
        for line in self.case_lines:
            c = ProfCase(desc=line, warn_up_cnt=self.warn_up_cnt, try_cnt=self.try_cnt, max_cnt=self.max_cnt)
            self.case_list.append(c)

    def prepare(self) -> bool:
        if not super().prepare():
            return False
        # 此处清理汇总表, 清理动作不受 clean 标记控制
        if self.statistic_rst_file.exists():
            self.statistic_rst_file.unlink()
        if self.statistic_all_file.exists():
            self.statistic_all_file.unlink()
        # 产生复用二进制, 在多卡情况下, 复用多卡并行执行所有用例, 预先产生所复用二进制
        if len(self.device_list) > 1:
            # 复用 STest 多卡执行脚本实现多卡并行产生 Kernel 二进制
            pys = Path(self.source_root, "framework/tests/cmake/scripts/stest_accelerate.py")
            cases_name_str = ":".join([c.name for c in self.case_list])
            cmd = f"{sys.executable} {pys} -t={self.exe.file} -c={cases_name_str}"
            for _k, _v in self.exe.envs.items():
                cmd += f" --env {_k}={_v}"
            cmd += " --halt_on_error" if self.halt_on_error else ""
            for dev_id in self.device_list:
                cmd += f" -d={dev_id}"
            ts = datetime.now(tz=timezone.utc)
            logging.info("[BGN] Multi Device(%s) Warn-Up, Cmd=%s", self.device_list, cmd)
            ret = subprocess.run(shlex.split(cmd), capture_output=False, check=True, text=True, encoding='utf-8')
            ret.check_returncode()
            logging.info("[END] Multi Device(%s) Warn-Up, Duration %s secs",
                         self.device_list, (datetime.now(tz=timezone.utc) - ts).seconds)
        return True

    def process_case_prepare(self, cs: ProfCase, device_id: int) -> bool:
        # 清理
        self.case_prof_dirs.clear()
        self.case_prof_ori_dir_root: Path = Path(self.output_root, f"profiling/{cs.name}/{self.level}/origin")
        self.case_rst_dir_root: Path = Path(self.output_root, f"profiling/{cs.name}/{self.level}/result")
        if self.case_rst_dir_root.exists():
            shutil.rmtree(self.case_rst_dir_root)
        self.case_rst_dir_root.mkdir(parents=True)
        self.case_statistic_rst_file = Path(self.case_rst_dir_root, ProfCase.STATISTIC_RST_FILE_NAME)
        self.case_statistic_all_file = Path(self.case_rst_dir_root, ProfCase.STATISTIC_ALL_FILE_NAME)
        # 预热 及 输出路径预处理
        # 在二进制复用场景下, 相关输出仅会在首次执行时产生
        output_dirs = [d for d in Path(self.exe.file.parent, "output").glob(pattern="output_*") if d.is_dir()]
        output_dirs.sort(reverse=True)  # 取最近一次
        self.case_output_dir_bak = output_dirs[0] if len(output_dirs) != 0 else None
        for i in range(1, cs.prof_warn_up_cnt + 1):
            _, _, tc = self.run_case(cs=cs, device_id=device_id)
            logging.info("%s Profiling Prepare, Warn-up(%s/%s), Duration %s secs.",
                         cs.full_name, i, cs.prof_warn_up_cnt, tc.seconds)
        output_dirs = [d for d in Path(self.exe.file.parent, "output").glob(pattern="output_*") if d.is_dir()]
        output_dirs.sort(reverse=False)
        idx = output_dirs.index(self.case_output_dir_bak) + 1 if self.case_output_dir_bak else 0
        self.case_output_dir_bak = output_dirs[idx] if idx < len(output_dirs) else None
        case_output_dir_bak_name = self.case_output_dir_bak.name if self.case_output_dir_bak else "None"
        logging.info("%s output back dir(%s)", cs.full_name, case_output_dir_bak_name)
        return True

    def process_case_process(self, cs: ProfCase, device_id: int) -> bool:
        env = {
            "PROFILER_SAMPLECONFIG": "{"
                                     + f"\"stars_acsq_task\":\"off\","
                                     + f"\"app\":\"{self.exe.file.name}\","
                                     + f"\"prof_level\":\"l2\","
                                     + f"\"taskTime\":\"l2\","
                                     + f"\"result_dir\":\"{self.case_prof_ori_dir_root}\","
                                     + f"\"app_dir\":\"{self.exe.file.parent}\","
                                     + f"\"ai_core_profiling\":\"off\","
                                     + f"\"aicpuTrace\":\"on\""
                                     + "}"
        }
        cur_cnt = 0
        for i in range(1, cs.prof_try_cnt + 1):
            # 性能采集
            self.run_case(cs=cs, device_id=device_id, envs=env)
            # 采集结果检查
            rest_sub_dirs = [d for d in self.case_prof_ori_dir_root.glob(pattern="PROF_*") if d.is_dir()]
            rest_sub_dirs.sort(reverse=True)
            if len(rest_sub_dirs) == 0:
                logging.warning("%s Profiling Collection failed(Try:%s/%s), can't get any PROF_* dir in %s",
                                cs.full_name, i, cs.prof_try_cnt, self.case_prof_ori_dir_root)
                continue
            prof_dir = ProfDir(src_root=self.source_root, path=rest_sub_dirs[0],
                               prof_case=cs, device_id=device_id, level=self.level)
            if not prof_dir.check():
                logging.warning("%s Profiling Collection failed(Try:%s/%s), prof dir(%s) check failed, "
                                "delete current prof result",
                                cs.full_name, i, cs.prof_try_cnt, prof_dir.working_dir)
                prof_dir.delete()
                continue
            # 采集成功处理
            if not self.process_case_process_post(prof_dir=prof_dir):
                continue
            cur_cnt += 1
            logging.info("%s Profiling Collection update(Try:%s/%s, SuccCnt:%s/%s), TimeStamp(%s)",
                         cs.full_name, i, cs.prof_try_cnt, cur_cnt, cs.prof_max_cnt, prof_dir.timestamp)
            if cur_cnt >= cs.prof_max_cnt:
                break
        # 采集结果汇总检查
        if cur_cnt < 1:
            logging.error("%s Profiling Collection failed, can't get any legal PROF_* dir.", cs.full_name)
            return False
        return True

    def process_case_process_post(self, prof_dir: ProfDir) -> bool:
        """性能采集成功后处理

        :param prof_dir: 当前 ProfDir
        """
        if not self.process_case_process_post_dump_output(prof_dir=prof_dir):
            return False
        return True

    def process_case_process_post_dump_output(self, prof_dir: ProfDir) -> bool:
        output_dirs = [d for d in Path(self.exe.file.parent, "output").glob(pattern="output_*") if d.is_dir()]
        output_dirs.sort(reverse=True)
        if len(output_dirs) == 0:
            logging.error("%s can't get any output dir after execute.", prof_dir.result_case.full_name)
            return False
        output_dir = output_dirs[0]
        if not Path(output_dir, "topo.json").exists():
            if self.case_output_dir_bak is None:
                raise RuntimeError("Case Output Back Dir is None")
        logging.info("%s dump %s output dir", prof_dir.result_case.full_name, output_dir.name)
        shutil.copytree(src=output_dir, dst=Path(prof_dir.result_dump_dir))
        return True

    def process_case_post(self, cs: ProfCase, device_id: int) -> bool:
        self.case_post_rebuild(cs=cs, device_id=device_id)
        rest_prof_dir = self.case_post_parse(cs=cs)
        self.case_post_statistic(cs=cs, rest_prof_dir=rest_prof_dir)
        return True

    def case_post_rebuild(self, cs: ProfCase, device_id: int):
        """对采集结果进行重建
        """
        rest_sub_dirs = [d for d in self.case_prof_ori_dir_root.glob(pattern="PROF_*") if d.is_dir()]
        for sub_dir in rest_sub_dirs:
            prof_dir = ProfDir(src_root=self.source_root, path=sub_dir,
                               prof_case=cs, device_id=device_id, level=self.level)
            prof_dir.result_case.update(k=ProfCase.FieldType.TimeStamp.value, v=prof_dir.timestamp)
            self.case_prof_dirs.update({prof_dir.timestamp: prof_dir})

    def case_post_parse(self, cs: ProfCase) -> ProfDir:
        """对采集结果进行解析, 并做初步结果筛选
        """
        # 遍历当前 Case 所采集的各个 ProfDir, 产生 Cycle
        cycle_lst = []
        for idx, (timestamp, prof_dir) in enumerate(self.case_prof_dirs.items(), start=1):
            prof_dir.parse()
            cycle_lst.append(prof_dir.result_case.cycle)
            logging.info("%s Profiling Statistics update(SuccCnt:%s/%s), TimeStamp(%s)",
                         cs.full_name, idx, len(self.case_prof_dirs), timestamp)
        # 遍历当前 Case 所采集的各个 ProfDir, 查找 Cycle 最接近平均值的结果, 作为当前 Case 采集的最终结果
        rest_prof_dir = None
        cycle_avg = math.ceil(sum(cycle_lst) / len(cycle_lst))
        cycle_sub_abs_min = sys.maxsize
        for _, prof_dir in self.case_prof_dirs.items():
            cycle_sub_abs = abs(prof_dir.result_case.cycle - cycle_avg)
            if cycle_sub_abs < cycle_sub_abs_min:
                cycle_sub_abs_min = cycle_sub_abs
                rest_prof_dir = prof_dir
        if rest_prof_dir is None:
            raise RuntimeError("%s can't get result", cs.full_name)
        return rest_prof_dir

    def case_post_statistic(self, cs: ProfCase, rest_prof_dir: ProfDir):
        """对解析结果做统计, 统计结果反标 ProfDir
        """
        cs.update(k=ProfCase.FieldType.TimeStamp.value, v=rest_prof_dir.timestamp)
        cs.update(k=ProfCase.FieldType.Cycle.value, v=rest_prof_dir.result_case.cycle)
        cs.update(k=ProfCase.FieldType.JitterRate.value, v=float(0))

        heads = []
        datas = []
        for idx, (timestamp, prof_dir) in enumerate(self.case_prof_dirs.items(), start=1):
            # JitterRate, 计算本 Case 的抖动率时, 仅统计劣化情况, 并取 ProfDir 最小值作为结果
            jitter_rate = (cs.cycle - prof_dir.result_case.cycle) / prof_dir.result_case.cycle
            prof_dir.result_case.update(k=ProfCase.FieldType.JitterRate.value, v=jitter_rate)
            if cs.jitter_rate >= jitter_rate:
                cs.update(k=ProfCase.FieldType.JitterRate.value, v=jitter_rate)
            # ProfDir 统计结果落盘
            logging.debug("%s Prof(%s) dump, Output %s", cs.full_name, prof_dir.timestamp, prof_dir.statistic_rst_file)
            logging.debug("%s Prof(%s) dump, Output %s", cs.full_name, prof_dir.timestamp, self.case_statistic_all_file)
            logging.debug("%s Prof(%s) dump, Output %s", cs.full_name, prof_dir.timestamp, self.statistic_all_file)
            prof_dir.result_case.dump_csv(file=prof_dir.statistic_rst_file)  # 落盘至 ProfDir
            prof_dir.result_case.dump_csv(file=self.case_statistic_all_file, append=True)  # 落盘至 当前 Case 总表
            prof_dir.result_case.dump_csv(file=self.statistic_all_file, append=True)  # 落盘至 所有 Case 总表
            # ProfDir 打屏结果输出
            heads, brief_ds = prof_dir.result_case.brief
            datas.append(brief_ds)
            logging.info("%s Profiling Statistics dump(SuccCnt:%s/%s), TimeStamp(%s)",
                         cs.full_name, idx, len(self.case_prof_dirs), timestamp)
        # 当前 Case 结果落盘
        shutil.copytree(rest_prof_dir.working_dir, Path(self.case_rst_dir_root, rest_prof_dir.timestamp))
        logging.debug("%s dump, Output %s", cs.full_name, self.case_statistic_rst_file)
        logging.debug("%s dump, Output %s", cs.full_name, self.statistic_rst_file)
        cs.dump_csv(file=self.case_statistic_rst_file)  # 落盘至 当前 Case 结果表
        cs.dump_csv(file=self.statistic_rst_file, append=True)  # 落盘至 所有 Case 结果表
        # Case 打屏结果输出
        cs_heads, cs_datas = cs.brief
        logging.info("\nCase(Profiling) Result\n%s\nCase(Profiling) Details\n%s",
                     Table.table(datas=[cs_datas], headers=cs_heads), Table.table(datas=datas, headers=heads))
