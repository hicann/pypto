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
"""生成覆盖率
"""
import argparse
import logging
import math
import os
import re
import subprocess
import dataclasses
import zipfile
import shutil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timezone

import yaml

from utils.table import Table


class GenCoverage:

    class FilterPathAction(argparse.Action):
        """自定义 Action: 解析 filter 参数时校验路径并格式化"""
        def __call__(self, parser, namespace, values, option_string=None):
            # 获取当前已收集的列表(初始为 None)
            cur_values = getattr(namespace, self.dest, None) or []
            path = Path(values)
            # 仅保留存在的路径, 并格式化(目录加 /*)
            if path.exists():
                if path.is_dir():
                    cur_values.append(f" {path}/*")
                else:
                    cur_values.append(f" {path}")
            # 更新命名空间的值
            setattr(namespace, self.dest, cur_values)

    @dataclasses.dataclass
    class LCovAblity:
        """lcov 能力"""

        lcov_version: str = ""
        lcov_supported_exclude: bool = False
        lcov_supported_parallel: bool = False

        genhtml_version: str = ""
        genhtml_supported_hierarchical: bool = False
        genhtml_supported_parallel: bool = False

        def __init__(self):
            self.init_lcov_version()
            self.lcov_supported_exclude = self._check_param_support(exe="lcov", param="--exclude")
            self.lcov_supported_parallel = self._check_param_support(exe="lcov", param="--parallel")
            self.init_genhtml_version()
            self.genhtml_supported_hierarchical = self._check_param_support(exe="genhtml", param="--hierarchical")
            self.genhtml_supported_parallel = self._check_param_support(exe="genhtml", param="--parallel")

        def __str__(self) -> str:
            desc = f"\nlcov"
            desc += f"\n    Version            : {self.lcov_version}"
            desc += f"\n    Ablities"
            desc += f"\n        --exclude      : {self.lcov_supported_exclude}"
            desc += f"\n        --parallel     : {self.lcov_supported_parallel}"
            desc += f"\ngenhtml"
            desc += f"\n    Version            : {self.genhtml_version}"
            desc += f"\n    Ablities"
            desc += f"\n        --hierarchical : {self.genhtml_supported_hierarchical}"
            desc += f"\n        --parallel     : {self.genhtml_supported_parallel}"
            return desc

        @classmethod
        def parse_version(cls, version: str) -> str:
            # 提取版本号(兼容 2.3.2, 2.3.2-1, 2.3.2+rc1 等格式)
            version = version.strip()
            # 正则匹配:提取 主版本.次版本.补丁版本(忽略后缀)
            version_match = re.search(r'version (\d+\.\d+\.\d+)', version)
            if not version_match:
                # 兼容只有两位版本号的情况(如 2.3)
                version_match = re.search(r'version (\d+\.\d+)', version)
                if not version_match:
                    raise RuntimeError(f"Can't get version from {version}")
                # 补全三位版本号(如 2.3 → 2.3.0)
                base_version = version_match.group(1)
                return f"{base_version}.0"
            else:
                return version_match.group(1)

        @classmethod
        def _check_param_support(cls, exe: str, param: str) -> bool:
            """检查指定可执行文件是否支持指定参数

            Args:
                exe: 可执行文件名(如 "lcov", "genhtml")
                param: 要检查的参数(如 "--exclude")

            Returns:
                bool: 是否支持该参数
            """
            try:
                ret = subprocess.run(f"{exe} --help".split(), capture_output=True, check=True, encoding='utf-8')
                ret.check_returncode()
                help_output = ret.stdout
                return param in help_output
            except (FileNotFoundError, subprocess.CalledProcessError):
                return False

        def init_lcov_version(self):
            """检查 lcov 环境
            """
            try:
                ret = subprocess.run('lcov --version'.split(), capture_output=True, check=True, encoding='utf-8')
                ret.check_returncode()
                self.lcov_version = self.parse_version(ret.stdout)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"lcov is required to generate coverage data, please install.") from e

        def init_genhtml_version(self):
            try:
                ret = subprocess.run('genhtml --version'.split(), capture_output=True, check=True, encoding='utf-8')
                ret.check_returncode()
                self.genhtml_version = self.parse_version(ret.stdout)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"genhtml is required to generate coverage html report, please install.") from e

    def __init__(self, args):
        self.src_root: Optional[Path] = Path(args.source[0]).resolve() if args.source else None
        self.data_dir: Path = Path(args.data[0]).resolve()
        self.result_dir: Path = Path(args.result[0]).resolve() if args.result else Path(self.data_dir, 'cov_result')
        self.job_num: int = self.get_job_num(args=args)

        # 全量覆盖率
        self.full_cov_info_file: Path = Path(self.result_dir, 'coverage.info')
        self.full_html_report_path: Path = Path(self.result_dir, "html")
        self.filter_lst: List[str] = args.filter

        # 增量覆盖率
        self.incr_flag: bool = self.get_increment_flag(args)
        incr_root = Path(self.result_dir, "increment")
        self.incr_cov_info_file: Path = Path(incr_root, f"{self.full_cov_info_file.name}")
        self.incr_html_report_path: Path = Path(incr_root, self.full_html_report_path.name)
        self.incr_text_report_file: Path = Path(incr_root, "coverage_report.txt")

        # 合法性检查
        self.lcov_ability = self.LCovAblity()
        self.chk_env()

        # 输出路径准备
        if not self.data_dir.exists():
            raise ValueError(f"The dir({self.data_dir}) required to find the .da files not exist.")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.full_html_report_path.mkdir(parents=True, exist_ok=True)
        if self.incr_flag:
            incr_root.mkdir(parents=True, exist_ok=True)
            self.incr_html_report_path.mkdir(parents=True, exist_ok=True)

        # 增量覆盖率计算相关数据
        # 变更文件及其行号范围, 格式为 {"file_rel_path": [(start_line, end_line)]}
        self.latest_changes: Dict[str, List[Tuple[int, int]]] = {}
        # 原始覆盖率数据, 格式为 {"file_abs_path": {"lines": {line_number: hit_count}}}
        self.full_cov_data: Dict[str, Dict[str, Dict[int, int]]] = {}
        # 增量覆盖率结果
        self.incr_cov_rst: Dict[str, Any] = {}

    def __str__(self) -> str:
        desc = f"\nGenerateCoverage"
        desc += f"\n    SrcRoot      : {self.src_root}"
        desc += f"\n    DataDir      : {self.data_dir}"
        desc += f"\n    ResultDir    : {self.result_dir}"
        desc += f"\n    JobNum       : {self.job_num}"

        desc += f"\n    Full"
        desc += f"\n      FilterList : {self.filter_lst}"
        desc += f"\n     CovInfoFile : {self.full_cov_info_file}"
        desc += f"\n      HtmlReport : {self.full_html_report_path}"
        if self.incr_flag:
            desc += f"\n    Increment"
            desc += f"\n     CovInfoFile : {self.incr_cov_info_file}"
            desc += f"\n      HtmlReport : {self.incr_html_report_path}"
        desc += f"{self.lcov_ability}"
        desc += f"\n"
        return desc

    @classmethod
    def reg_args(cls, parser):
        """注册命令行参数
        """
        parser.add_argument("-s", "--source",
                            required=True, nargs=1, type=Path,
                            help="Specify the source base directory.")
        parser.add_argument("-d", "--data",
                            required=True, nargs=1, type=Path,
                            help="Specify the *.da's base directory.")
        parser.add_argument("-r", "--result",
                            required=False, nargs=1, type=Path,
                            help="Specify the result output directory.")
        parser.add_argument("-f", "--filter",
                            required=False, action=cls.FilterPathAction, type=str,
                            help="Specify filter file/dir in coverage info.")
        parser.add_argument("-j", "--job_num",
                            nargs="?", type=int, default=None,
                            help="Specify parallel job num.")
        parser.add_argument("-i", "--increment",
                            action="store", type=str, default=None,
                            choices=["true", "false", "TRUE", "FALSE", "True", "False", "1", "0"],
                            help="Enable increment coverage calculation based on latest commit.")

    @classmethod
    def get_job_num(cls, args):
        """获取并行任务数
        """
        if args.job_num:
            job_num = args.job_num
        else:
            if os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", 0):
                job_num = int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"), 0)
            elif os.environ.get("PYPTO_TESTS_PARALLEL_NUM", 0):
                job_num = int(os.environ.get("PYPTO_TESTS_PARALLEL_NUM", 0))
            else:
                job_num = int(math.ceil(float(cpu_count()) * 0.8))    # use 0.8 cpu
        job_num = min(max(int(job_num), 1), cpu_count(), 48)
        return job_num

    @classmethod
    def get_increment_flag(cls, args):
        """获取增量覆盖率标志
        """
        env_val = os.environ.get("PYPTO_BUILD_GCOV_INCREMENT", "")
        true_list = ["true", "1"]
        if args.increment is not None:
            return args.increment.lower() in true_list
        else:
            return env_val.lower() in true_list

    @classmethod
    def get_file_stats(cls, file_cov_info: Dict[str, Dict[int, int]],
                       line_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """处理单个文件的覆盖率数据

        Args:
            file_cov_info: 文件的覆盖率数据, 格式为 {"lines": {line_number: hit_count}}
            line_ranges: 文件的变更行范围

        Returns:
            dict: 文件的覆盖率统计, 格式为 {
                "total_lines": int,
                "covered_lines": int,
                "coverage_rate": float,
                "lines": {line_number: hit_count}
            }
        """
        def process_line_range(start_line: int, end_line: int, lines_coverage: Dict[int, int],
                               file_stats: Dict[str, Any]):
            """处理单个行范围的覆盖率数据
            """
            for line_number in range(start_line, end_line + 1):
                file_stats["total_lines"] += 1

                # 检查行是否被覆盖
                hit_count = lines_coverage.get(line_number, 0)
                file_stats["lines"][line_number] = hit_count

                if hit_count > 0:
                    file_stats["covered_lines"] += 1

        file_stats = {
            "total_lines": 0,
            "covered_lines": 0,
            "lines": {}
        }

        lines_coverage = file_cov_info.get("lines", {})
        for start_line, end_line in line_ranges:
            process_line_range(start_line, end_line, lines_coverage, file_stats)

        return file_stats

    @classmethod
    def main(cls):
        """主函数
        """
        # 参数注册
        parser = argparse.ArgumentParser(description="Generate Coverage", epilog="Best Regards!")
        cls.reg_args(parser=parser)
        # 参数处理
        ctrl = GenCoverage(args=parser.parse_args())
        logging.info("%s", ctrl)
        # 流程处理
        ctrl.process()

    @classmethod
    def _merge_line_ranges(cls, lines: List[int]) -> List[Tuple[int, int]]:
        lines = sorted(lines)
        if not lines:
            return []
        ranges = []
        start = lines[0]
        end = start
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = line
                end = line
        ranges.append((start, end))
        return ranges

    @classmethod
    def _get_line_ranges_str(cls, ranges: List[Tuple[int, int]]) -> str:
        if not ranges:
            return ""
        parts = []
        for start, end in ranges:
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}~{end}")
        return ", ".join(parts)

    def chk_env(self):
        """检查环境依赖
        """
        # 检查 git 是否安装
        if self.incr_flag:
            self.chk_env_git()

    def chk_env_git(self):
        """检查 git 环境
        """
        # 检查 git 是否安装
        try:
            subprocess.run('git --version'.split(), capture_output=True, check=True, encoding='utf-8')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"git is required for increment coverage, please install.") from e
        # 检查是否在 git 仓库中
        try:
            subprocess.run('git rev-parse --is-inside-work-tree'.split(),
                           cwd=self.src_root, capture_output=True, check=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"The source directory {self.src_root} is not a git repository.") from e

    def process(self):
        """使用 lcov 生成覆盖率
        """
        # 处理全量覆盖率
        self.gen_full_cov_info_file()
        self.gen_cov_html_report(cov_file=self.full_cov_info_file, dest=self.full_html_report_path)

        # 处理增量覆盖率
        if not self.incr_flag:
            return

        self.gen_full_cov_data()
        self.detect_changes()
        if not self.latest_changes:
            return
        self.gen_incr_cov_info_file()
        self.gen_cov_html_report(cov_file=self.incr_cov_info_file, dest=self.incr_html_report_path,
                                 hierarchical=False)
        self.detect_incr_cov_rst()
        self.gen_inc_cov_text_report()

        # 压缩结果目录, 仅在增量覆盖率使能时压缩, 避免影响 CI 执行性能
        self.compress_result_root()

    def gen_full_cov_info_file(self):
        """生成过滤后的全量覆盖率文件
        """
        # 生成覆盖率原始统计文件
        cmd = f"lcov -c -d {self.data_dir} -o {self.full_cov_info_file}"
        if self.lcov_ability.lcov_supported_exclude:
            for filter_path in self.filter_lst:
                cmd += f" --exclude {filter_path}"
        if self.lcov_ability.lcov_supported_parallel:
            cmd += f" --rc geninfo_unexecuted_blocks=1"  # 接受未执行块
            cmd += f" --ignore-errors negative"
            cmd += f" -j {self.job_num}"
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated%s coverage file %s, cmd: %s",
                     "" if self.lcov_ability.lcov_supported_exclude else " origin", self.full_cov_info_file, cmd)
        # 滤掉某些文件/路径的覆盖率信息
        filtered_file = Path(self.full_cov_info_file.parent,
                             f"{self.full_cov_info_file.stem}_filtered{self.full_cov_info_file.suffix}")
        if self.lcov_ability.lcov_supported_exclude:
            # CI 兼容处理, 当存在 exclude 选项时, 直接复制原始文件
            shutil.copy(src=self.full_cov_info_file, dst=filtered_file)
        else:
            filter_str = " ".join(self.filter_lst)
            cmd = f"lcov --remove {self.full_cov_info_file} {filter_str} -o {filtered_file}"
            ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
            ret.check_returncode()
            logging.info("Generated filtered coverage file %s, cmd: %s", filtered_file, cmd)
            self.full_cov_info_file = filtered_file

    def gen_cov_html_report(self, cov_file: Path, dest: Path, scene: str = "full", hierarchical: bool = True):
        """生成完整的 html 报告
        """
        prefix = f"-p {self.src_root}" if self.src_root else ""
        cmd = f'genhtml {cov_file} {prefix} -o {dest}'
        if self.lcov_ability.genhtml_supported_hierarchical and hierarchical:
            cmd += f" --hierarchical"
        if self.lcov_ability.genhtml_supported_parallel:
            cmd += f" --rc check_data_consistency=0"  # 关闭数据一致性校验
            cmd += f" -j {self.job_num}"
        ret = subprocess.run(cmd.split(), capture_output=True, check=False, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated %s coverage html report in %s, cmd: %s", scene, dest, cmd)

    def gen_full_cov_data(self):
        """解析 lcov 覆盖率数据文件, 生成原始覆盖率数据
        """
        self.full_cov_data.clear()
        rel_path_str = None

        with open(self.full_cov_info_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()

            # 匹配文件路径
            if line.startswith('SF:'):
                rel_path_str = line[3:]
                self.full_cov_data[rel_path_str] = {"lines": {}}

            # 匹配行覆盖率数据
            elif line.startswith('DA:') and rel_path_str:
                parts = line[3:].split(',')
                if len(parts) >= 2:
                    line_number = int(parts[0])
                    hit_count = int(parts[1])
                    self.full_cov_data[rel_path_str]["lines"][line_number] = hit_count

        if self.full_cov_data:
            logging.info("Detected %d files coverage data from %s", len(self.full_cov_data), self.full_cov_info_file)
        else:
            logging.error("Failed to parse coverage data from %s", self.full_cov_info_file)

    def detect_changes(self):
        """确定最新一次代码变更情况
        """
        # 获取最新一次代码变更情况
        self._detect_changes_from_git_diff()
        if not self.latest_changes:
            logging.error("No code changes detected, skipping increment coverage calculation.")
            return

        # 根据 classify_rule.yaml 过滤变更
        self._filter_changes_from_classify_rule()
        if not self.latest_changes:
            logging.info("No changes after filtering by classify_rule.yaml, skipping increment coverage calculation.")
            return

        # 根据规则过滤变更, 如过滤头文件等
        self._filter_changes_by_rules()
        if not self.latest_changes:
            logging.info("No changes after filtering by rules")
            return

    def gen_incr_cov_info_file(self):
        """生成增量覆盖率文件
        """
        def write_uncovered_lines(f, line_ranges):
            """写入未覆盖的行
            """
            for start_line, end_line in line_ranges:
                for line_number in range(start_line, end_line + 1):
                    f.write(f'DA:{line_number},0\n')

        def write_covered_lines(f, line_ranges, file_coverage):
            """写入覆盖的行
            """
            for start_line, end_line in line_ranges:
                for line_number in range(start_line, end_line + 1):
                    hit_count = file_coverage["lines"].get(line_number, 0)
                    f.write(f'DA:{line_number},{hit_count}\n')

        # 生成增量 lcov 文件
        with open(self.incr_cov_info_file, 'w') as f:
            for rel_path_str, line_ranges in self.latest_changes.items():
                abs_path_str = str(Path(self.src_root, rel_path_str))
                f.write(f'SF:{abs_path_str}\n')

                if abs_path_str not in self.full_cov_data:
                    # 为未在覆盖率数据中的文件写入未覆盖的行
                    write_uncovered_lines(f, line_ranges)
                else:
                    file_coverage = self.full_cov_data[abs_path_str]
                    write_covered_lines(f, line_ranges, file_coverage)

                f.write('end_of_record\n')

    def detect_incr_cov_rst(self):
        """计算增量代码覆盖率结果
        """
        def create_uncovered_stats(line_ranges):
            """为未在覆盖率数据中的文件创建统计信息
            """
            file_stats = {
                "total_lines": 0,
                "covered_lines": 0,
                "lines": {}
            }
            for start_line, end_line in line_ranges:
                for line_number in range(start_line, end_line + 1):
                    file_stats["total_lines"] += 1
                    file_stats["lines"][line_number] = 0
            return file_stats

        cov_rst = {
            "total_lines": 0,
            "covered_lines": 0,
            "coverage_rate": 0.0,
            "files": {}
        }

        for rel_path_str, line_ranges in self.latest_changes.items():
            # 检查文件是否在覆盖率数据中
            file_abs_path = Path(self.src_root, rel_path_str)
            file_cov_info = self.full_cov_data.get(str(file_abs_path), None)

            if file_cov_info is None:
                logging.warning("File %s not found in coverage data, considered as uncovered", rel_path_str)
                # 为未在覆盖率数据中的文件创建统计信息，标记为未覆盖
                file_stats = create_uncovered_stats(line_ranges)
            else:
                file_stats = self.get_file_stats(file_cov_info=file_cov_info, line_ranges=line_ranges)

            # 计算文件的覆盖率
            if file_stats["total_lines"] > 0:
                file_stats["coverage_rate"] = file_stats["covered_lines"] / file_stats["total_lines"] * 100

            cov_rst["files"][rel_path_str] = file_stats
            cov_rst["total_lines"] += file_stats["total_lines"]
            cov_rst["covered_lines"] += file_stats["covered_lines"]

        # 计算总体覆盖率
        if cov_rst["total_lines"] > 0:
            cov_rst["coverage_rate"] = cov_rst["covered_lines"] / cov_rst["total_lines"] * 100

        self.incr_cov_rst = cov_rst

    def gen_inc_cov_text_report(self):
        """生成文本格式的增量覆盖率报告
        """
        lines = []

        # 生成总体覆盖率报告
        lines.append("=" * 80)
        lines.append("Increment Code Coverage Report")
        lines.append("=" * 80)
        lines.append("Comparison Range: Latest commit (HEAD~1..HEAD)")

        # Add latest commit message
        commit_msg = self._get_latest_commit_message()
        if commit_msg:
            lines.append("Commit Message:")
            lines.append("-" * 80)
            for line in commit_msg.split('\n'):
                lines.append(f"  {line}")
            lines.append("-" * 80)

        lines.append(f"Overall Coverage: {self.incr_cov_rst['coverage_rate']:.2f}%")
        lines.append(f"Total Changed Lines: {self.incr_cov_rst['total_lines']}")
        lines.append(f"Covered Lines: {self.incr_cov_rst['covered_lines']}")
        lines.append("=" * 80)

        # 生成文件覆盖率详情
        brief = ""
        if self.incr_cov_rst['files']:
            lines.append("File Coverage Details:")
            heads = ["File", "Coverage", "Lines[Covered/Total]", "Uncovered Lines"]
            datas = []
            for file_path, file_stats in self.incr_cov_rst['files'].items():
                cov_desc = f"{file_stats['covered_lines']}/{file_stats['total_lines']}"
                uncov_lines = [l for l, count in file_stats['lines'].items() if count == 0]
                uncov_ranges = self._merge_line_ranges(lines=uncov_lines)
                uncov_ranges_str = self._get_line_ranges_str(ranges=uncov_ranges)
                datas.append([file_path, f"{file_stats['coverage_rate']:.2f}%", f"{cov_desc}", f"{uncov_ranges_str}"])
            brief = Table.table(datas=datas, headers=heads)

        report = '\n'.join(lines) + "\n" + brief
        with open(self.incr_text_report_file, 'w') as f:
            f.write(report)
        logging.info("\n" + report)

    def compress_result_root(self):
        """压缩结果目录为 zip 格式
        """
        def add_files_to_zip(zipf, result_root):
            """将文件添加到 zip 文件中
            """
            for root, _, files in os.walk(result_root):
                for file in files:
                    file_path = Path(root, file)
                    arcname = f"{result_root.name}/{file_path.relative_to(result_root)}"
                    zipf.write(file_path, arcname)

        if not self.result_dir.exists():
            logging.warning(f"Result directory {self.result_dir} does not exist, skipping compression.")
            return

        zip_file = self.result_dir.with_suffix('.zip')
        try:
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                add_files_to_zip(zipf, self.result_dir)
            logging.info("Compressed result directory to %s", zip_file)
        except Exception as e:
            logging.error("Failed to compress result directory: %s", e)
            raise

    def _detect_changes_from_git_diff(self):
        """获取最近一次提交的代码变更
        """
        # 运行 git diff 命令, 获取最近一次提交的变更
        ret = subprocess.run(['git', 'diff', '--unified=0', 'HEAD~1', 'HEAD'],
                             cwd=self.src_root, capture_output=True, text=True, encoding='utf-8', check=True)
        diff_output = ret.stdout

        file_pattern = re.compile(r'^diff --git a/(.*) b/(.*)$')  # 匹配文件路径的正则表达式
        hunk_pattern = re.compile(r'^@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@')  # 匹配行号的正则表达式
        lines = diff_output.split('\n')

        rel_path_str = None
        current_line = 0
        start_line = None

        def finalize_current_range():
            """完成当前变更范围的处理并添加到变更列表
            """
            nonlocal rel_path_str, start_line, current_line
            if rel_path_str and start_line is not None:
                self.latest_changes[rel_path_str].append((start_line, current_line - 1))

        for line in lines:
            # 处理文件路径匹配
            file_match = file_pattern.match(line)
            if file_match:
                finalize_current_range()
                rel_path_str = file_match.group(2)
                self.latest_changes[rel_path_str] = []
                current_line = 0
                start_line = None
                continue

            # 跳过处理如果当前没有文件
            if rel_path_str not in self.latest_changes:
                continue

            # 处理行号范围匹配
            hunk_match = hunk_pattern.match(line)
            if hunk_match:
                finalize_current_range()
                current_line = int(hunk_match.group(3))
                start_line = None
                continue

            # 处理新增行
            if line.startswith('+') and not line.startswith('+++'):
                if start_line is None:
                    start_line = current_line
                current_line += 1
                continue

            # 处理未变更行
            if not line.startswith('-') and not line.startswith('---') and not line.startswith('@@'):
                finalize_current_range()
                start_line = None
                current_line += 1
                continue

        # 处理最后一个文件的最后一个范围
        finalize_current_range()

    def _filter_changes_from_classify_rule(self):
        """根据 classify_rule.yaml 文件过滤变更
        """
        # 读取 classify_rule.yaml 文件
        classify_rule_path = Path(self.src_root, "classify_rule.yaml")
        if not classify_rule_path.exists():
            logging.warning(f"classify_rule.yaml file not found at {classify_rule_path}, returning original changes")
            return

        with open(classify_rule_path, 'r', encoding='utf-8') as f:
            classify_rule = yaml.safe_load(f)

        # 提取 release 和 unrelease 路径
        pypto_config = classify_rule.get("pypto", {})
        src_config = pypto_config.get("src", {})
        release_paths = src_config.get("release", [])
        unrelease_paths = src_config.get("unrelease", [])

        filtered_changes = {}

        for file_path, line_ranges in self.latest_changes.items():
            # 检查文件是否在 unrelease 路径中
            in_unrelease = False
            for unrelease_path in unrelease_paths:
                if file_path.startswith(unrelease_path):
                    in_unrelease = True
                    break
            if in_unrelease:
                continue

            # 检查文件是否在 release 路径中
            in_release = False
            for release_path in release_paths:
                if file_path.startswith(release_path):
                    in_release = True
                    break
            if in_release:
                filtered_changes[file_path] = line_ranges

        if filtered_changes:
            logging.info("Detected %d files changes:", len(filtered_changes))
            for file, line_ranges in filtered_changes.items():
                logging.info("    %s: %s", file, line_ranges)

        self.latest_changes = filtered_changes

    def _filter_changes_by_rules(self):
        """根据规则过滤变更
        """
        filtered_changes = {}
        for rel_path_str, ori_line_ranges in self.latest_changes.items():
            # 过滤头文件(与 CI 保持一致, 头文件不参与增量覆盖率计算)
            if rel_path_str.endswith(('.h', '.hpp', '.hxx')):
                continue
            # 根据原始覆盖率数据过滤变更行号, 排除无法覆盖的行(如空行/}/已被编译器优化)
            line_ranges = self._filter_changes_line_ranges_by_ori_cov_data(rel_path_str, ori_line_ranges)
            if line_ranges:
                filtered_changes[rel_path_str] = line_ranges
        self.latest_changes = filtered_changes

    def _filter_changes_line_ranges_by_ori_cov_data(self, rel_path_str: str, line_ranges: List[Tuple[int, int]]):
        """根据原始覆盖率数据过滤变更行号
        """
        # 检查对应文件是否存在原始覆盖率数据, 若不存, 说明对应文件在此次变更中属于删除变更, 反馈空列表
        abs_path_str = str(Path(self.src_root, rel_path_str))
        full_cov_lines_data = self.full_cov_data.get(abs_path_str, {}).get("lines", None)
        if not full_cov_lines_data:
            return []

        # 逐行检查变更行, 若变更行不在覆盖率数据中, 说明其属于无法覆盖的行(如空行/}/已被编译器优化), 跳过
        new_lines = []
        for start_line, end_line in line_ranges:
            for line_number in range(start_line, end_line + 1):
                hit_count = full_cov_lines_data.get(line_number, None)
                if hit_count is not None:
                    new_lines.append(line_number)

        # 合并连续的行号范围
        if not new_lines:
            return []
        return self._merge_line_ranges(lines=new_lines)

    def _get_latest_commit_message(self):
        """获取最新提交的 commit message
        """
        try:
            ret = subprocess.run(['git', 'log', '-1', '--pretty=format:%s\n%b'],
                                 cwd=self.src_root, capture_output=True, text=True, encoding='utf-8', check=True)
            return ret.stdout.strip()
        except Exception as e:
            logging.warning(f"Failed to get commit message: {e}")
            return ""


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    ts = datetime.now(tz=timezone.utc)
    GenCoverage.main()
    duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
    logging.info("Generate Coverage use %s secs.", duration)
