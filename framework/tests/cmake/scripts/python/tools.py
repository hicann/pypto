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
"""工具总入口.
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from profiling.prof_tools import ProfTools
from utils.args_action import ArgsEnvDictAction


class ToolsCtrl:

    @classmethod
    def reg_run_args(cls, pre_sub_parsers):
        parser = pre_sub_parsers.add_parser('run', help="Tools need run executable")
        parser.add_argument("-t", "--target",
                            nargs=1, type=Path, required=True,
                            help="Specific target executable binary path")
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Specific parallel accelerate device, "
                                 "If this parameter is not specified, 0 device will be used by default.")
        parser.add_argument("-e", "--env",
                            nargs="+", action=ArgsEnvDictAction, default={}, dest="envs",
                            help="Specify additional environment variables to set when executing the target.")
        parser.add_argument("--cases",
                            nargs=1, type=str, default="",
                            help="Specify case name, multiple name are separated by ':'")
        parser.add_argument("--cases_csv_file",
                            nargs=1, type=Path, default=None,
                            help="Specify cases csv file path")
        parser.add_argument("--golden_impl_path", dest="cases_golden_impl_path",
                            nargs="?", type=str, action="append", required=True,
                            help="Golden impl path, relative path to the source root directory.")
        parser.add_argument("--golden_output_path", dest="cases_golden_output_path",
                            nargs=1, type=str, default="", required=True,
                            help="Specific Cases golden output path.")
        parser.add_argument("--golden_output_clean", dest="cases_golden_output_clean",
                            action="store_true", default=False,
                            help="Clean Cases golden output.")
        parser.add_argument("--halt_on_error", action="store_true", default=False,
                            help="If any case failed, subsequent cases are not executed.")
        sub_parsers = parser.add_subparsers()
        cls.reg_run_prof_args(pre_sub_parsers=sub_parsers)

    @classmethod
    def reg_run_prof_args(cls, pre_sub_parsers):
        parser = pre_sub_parsers.add_parser('profiling', help="Profiling", aliases=['prof'])
        parser.add_argument("--level", dest="prof_level",
                            nargs="?", type=str, default="l1",
                            choices=["l1", "l2"],
                            help="Specify profiling level")
        parser.add_argument("--warn_up_cnt", dest="prof_warn_up_cnt",
                            nargs=1, type=int, default=None,
                            help="Specify profiling warn up cnt")
        parser.add_argument("--try_cnt", dest="prof_try_cnt",
                            nargs=1, type=int, default=None,
                            help="Specify profiling try cnt")
        parser.add_argument("--max_cnt", dest="prof_max_cnt",
                            nargs=1, type=int, default=None,
                            help="Specify profiling max cnt")
        parser.set_defaults(processor=ProfTools)

    @classmethod
    def main(cls):
        """主处理流程
        """
        parser = argparse.ArgumentParser(description=f"Tile Framework Tools Ctrl.", epilog="Best Regards!")

        # 主命令参数注册
        parser.add_argument("-c", "--clean", dest="tools_output_clean",
                            action="store_true", default=False,
                            help="Specify clean flag, clean tools output dir")
        parser.add_argument("--intercept",
                            action="store_true", default=False,
                            help="Intercept if have failed case result")
        parser_sub_parsers = parser.add_subparsers()

        # 子命令参数注册(Run), 组织需要执行 target 的相关工具
        cls.reg_run_args(pre_sub_parsers=parser_sub_parsers)

        # 流程处理
        args = parser.parse_args()
        if 'processor' in args:
            ts = datetime.now(tz=timezone.utc)
            tools = args.processor(args=args)
            ret = tools.clean()
            ret = ret and tools.prepare()
            ret = ret and tools.process()
            ret = ret and tools.post()
            logging.info("Tools(%s), Duration %s sec", tools.__class__.__name__,
                         (datetime.now(tz=timezone.utc) - ts).seconds)
        else:
            raise ValueError("Must Specify a sub-command.")
        return ret


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    exit(0 if ToolsCtrl.main() else 1)
