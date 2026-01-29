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
import zipfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timezone

import yaml


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

    def __init__(self, args):
        self.src_root: Optional[Path] = Path(args.source[0]).resolve() if args.source else None
        self.data_dir: Path = Path(args.data[0]).resolve()
        self.job_num: int = self.get_job_num(args=args)

        # 生成路径
        info_file = Path(args.info[0]).resolve() if args.info else Path(self.data_dir, 'cov_result/coverage.info')
        report = Path(args.html[0]).resolve() if args.html else Path(info_file.parent, "html")

        # 全量覆盖率
        self.full_cov_info_file: Path = info_file
        self.full_html_report_path: Path = report
        self.filter_lst: List[str] = args.filter

        # 增量覆盖率
        self.incr_flag: bool = self.get_increment_flag(args)
        self.incr_root = Path(info_file.parent, "increment")
        self.incr_cov_info_file: Path = Path(self.incr_root, f"{info_file.name}")
        self.incr_html_report_path: Path = Path(self.incr_root, report.name)
        self.incr_text_report_file: Path = Path(self.incr_root, "coverage_report.txt")

        # 合法性检查
        self.lcov_version: str = ""
        self.lcov_version_new: bool = False  # 用于标识 lcov 版本符合要求, 可以使用使用 -j 及 --exclude 能力
        self.chk_env()
        if not self.data_dir.exists():
            raise ValueError(f"The dir({self.data_dir}) required to find the .da files not exist.")

        # 输出路径准备
        self.full_cov_info_file.parent.mkdir(parents=True, exist_ok=True)
        self.full_html_report_path.mkdir(parents=True, exist_ok=True)
        if self.incr_flag:
            self.incr_cov_info_file.parent.mkdir(parents=True, exist_ok=True)
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
        desc += f"\n    JobNum       : {self.job_num}"
        desc += f"\n    lcov         : {self.lcov_version} ({self.lcov_version_new})"

        desc += f"\n    Full"
        desc += f"\n      FilterList : {self.filter_lst}"
        desc += f"\n     CovInfoFile : {self.full_cov_info_file}"
        desc += f"\n      HtmlReport : {self.full_html_report_path}"
        if self.incr_flag:
            desc += f"\n    Increment"
            desc += f"\n     CovInfoFile : {self.incr_cov_info_file}"
            desc += f"\n      HtmlReport : {self.incr_html_report_path}"
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
        parser.add_argument("-i", "--info_file", dest="info",
                            required=False, nargs=1, type=Path,
                            help="Specify coverage info file path.")
        parser.add_argument("-f", "--filter",
                            required=False, action=cls.FilterPathAction, type=str,
                            help="Specify filter file/dir in coverage info.")
        parser.add_argument("--html_report", dest="html",
                            required=False, nargs=1, type=Path,
                            help="Specify coverage html report dir.")
        parser.add_argument("-j", "--job_num",
                            nargs="?", type=int, default=None,
                            help="Specify parallel job num.")
        parser.add_argument("--increment",
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
            if env_val.lower() in true_list:
                return True
            else:
                return False

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

    def chk_env(self):
        """检查环境依赖
        """
        # 检查 lcov
        self.chk_env_lcov()
        # 检查 git 是否安装
        if self.incr_flag:
            self.chk_env_git()

    def chk_env_lcov(self):
        """检查 lcov 环境
        """
        try:
            ret = subprocess.run('lcov --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
            # 提取版本号(兼容 2.3.2, 2.3.2-1, 2.3.2+rc1 等格式)
            version_output = ret.stdout.strip()
            # 正则匹配:提取 主版本.次版本.补丁版本(忽略后缀)
            version_match = re.search(r'version (\d+\.\d+\.\d+)', version_output)
            if not version_match:
                # 兼容只有两位版本号的情况(如 2.3)
                version_match = re.search(r'version (\d+\.\d+)', version_output)
                if not version_match:
                    raise RuntimeError(f"Can't get version from {version_output}")
                # 补全三位版本号(如 2.3 → 2.3.0)
                base_version = version_match.group(1)
                self.lcov_version = f"{base_version}.0"
            else:
                self.lcov_version = version_match.group(1)
            # 验证版本是否 >=2.3.2
            req = [2, 3, 2]
            parts = list(map(int, self.lcov_version.split('.')))
            # 版本对比逻辑:逐位比较主, 次, 补丁版本
            self.lcov_version_new = (parts[0] > req[0] or
                                     (parts[0] == req[0] and parts[1] > req[1]) or
                                     (parts[0] == req[0] and parts[1] == req[1] and parts[2] >= req[2]))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"lcov is required to generate coverage data, please install.") from e
        try:
            ret = subprocess.run('genhtml --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"genhtml is required to generate coverage html report, please install.") from e

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
        # 生成全量覆盖率报告
        self.gen_full_cov_info_file()
        self.gen_cov_html_report(cov_file=self.full_cov_info_file, dest=self.full_html_report_path)

        # 生成增量覆盖率报告
        if self.incr_flag:
            self.gen_incr_cov_info_file()
            if self.latest_changes:
                self.gen_cov_html_report(cov_file=self.incr_cov_info_file, dest=self.incr_html_report_path)
                # 计算增量覆盖率结果
                self.detect_incr_cov_rst()
                # 生成文本报告
                self.gen_inc_cov_text_report()
                # 压缩增量覆盖率目录
                self.compress_incr_root()

    def gen_full_cov_info_file(self):
        """生成过滤后的全量覆盖率文件
        """
        # 生成覆盖率原始统计文件
        cmd = f"lcov -c -d {self.data_dir} -o {self.full_cov_info_file}"
        if self.lcov_version_new:
            for filter_path in self.filter_lst:
                cmd += f" --exclude {filter_path}"
            cmd += f" --rc geninfo_unexecuted_blocks=1"  # 接受未执行块
            cmd += f" --ignore-errors negative"
            cmd += f" -j {self.job_num}"
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated%s coverage file %s, cmd: %s", "" if self.lcov_version_new else " origin",
                     self.full_cov_info_file, cmd)
        # 滤掉某些文件/路径的覆盖率信息
        if not self.lcov_version_new:
            filtered_file = Path(self.full_cov_info_file.parent,
                                 f"{self.full_cov_info_file.stem}_filtered{self.full_cov_info_file.suffix}")
            filter_str = " ".join(self.filter_lst)
            cmd = f"lcov --remove {self.full_cov_info_file} {filter_str} -o {filtered_file}"
            ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
            ret.check_returncode()
            logging.info("Generated filtered coverage file %s, cmd: %s", filtered_file, cmd)
            self.full_cov_info_file = filtered_file

    def gen_cov_html_report(self, cov_file: Path, dest: Path, scene: str = "full"):
        """生成完整的 html 报告
        """
        prefix = f"-p {self.src_root}" if self.src_root else ""
        cmd = f'genhtml {cov_file} {prefix} -o {dest}'
        if self.lcov_version_new:
            cmd += f" --rc check_data_consistency=0"  # 关闭数据一致性校验
            cmd += f" -j {self.job_num}"
        ret = subprocess.run(cmd.split(), capture_output=True, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("Generated %s coverage html report in %s, cmd: %s", scene, dest, cmd)

    def gen_incr_cov_info_file(self):
        """生成增量率覆盖率文件
        """
        # 获取最新一次代码变更情况
        self.detect_changes_from_git_diff()
        if not self.latest_changes:
            logging.error("No code changes detected, skipping increment coverage calculation.")
            return

        # 根据 classify_rule.yaml 过滤变更
        self.detect_changes_from_classify_rule()
        if not self.latest_changes:
            logging.info("No changes after filtering by classify_rule.yaml, skipping increment coverage calculation.")
            return

        # 解析原始覆盖率数据
        self.detect_full_cov_data()
        if not self.full_cov_data:
            return

        # 生成增量 lcov 文件
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

    def detect_changes_from_git_diff(self):
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

    def detect_changes_from_classify_rule(self):
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

    def detect_full_cov_data(self):
        """解析 lcov 覆盖率数据文件, 确定原始覆盖率数据
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

    def get_latest_commit_message(self):
        """获取最新提交的 commit message
        """
        try:
            ret = subprocess.run(['git', 'log', '-1', '--pretty=format:%s\n%b'],
                                 cwd=self.src_root, capture_output=True, text=True, encoding='utf-8', check=True)
            return ret.stdout.strip()
        except Exception as e:
            logging.warning(f"Failed to get commit message: {e}")
            return ""

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
        commit_msg = self.get_latest_commit_message()
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
        if self.incr_cov_rst['files']:
            lines.append("File Coverage Details:")
            lines.append("-" * 80)

            for file_path, file_stats in self.incr_cov_rst['files'].items():
                lines.append(f"File: {file_path}")
                lines.append(f"  Coverage: {file_stats['coverage_rate']:.2f}%")
                lines.append(f"  Changed Lines: {file_stats['total_lines']}")
                lines.append(f"  Covered Lines: {file_stats['covered_lines']}")

                # 显示未覆盖的行
                uncovered = [l for l, count in file_stats['lines'].items() if count == 0]
                if uncovered:
                    lines.append(f"  Uncovered Lines: {sorted(uncovered)}")

                lines.append("-" * 80)

        report = '\n'.join(lines)
        with open(self.incr_text_report_file, 'w') as f:
            f.write(report)
        logging.info("\n" + report)

    def compress_incr_root(self):
        """压缩增量覆盖率目录为 zip 格式
        """
        def add_files_to_zip(zipf, incr_root):
            """将文件添加到 zip 文件中
            """
            for root, _, files in os.walk(incr_root):
                for file in files:
                    file_path = Path(root, file)
                    arcname = f"{incr_root.name}/{file_path.relative_to(incr_root)}"
                    zipf.write(file_path, arcname)

        if not self.incr_root.exists():
            logging.warning(f"Increment root directory {self.incr_root} does not exist, skipping compression.")
            return

        zip_file = self.incr_root.with_suffix('.zip')
        try:
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                add_files_to_zip(zipf, self.incr_root)
            logging.info("Compressed increment coverage directory to %s", zip_file)
        except Exception as e:
            logging.error("Failed to compress increment coverage directory: %s", e)
            raise


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    ts = datetime.now(tz=timezone.utc)
    GenCoverage.main()
    duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
    logging.info("Generate Coverage use %s secs.", duration)
