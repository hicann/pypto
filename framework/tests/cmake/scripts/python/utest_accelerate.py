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
"""UTest 用例并行执行.
"""
import argparse
import logging
import math
import os
from multiprocessing import cpu_count
from typing import List

from accelerate.gtest_accelerate import GTestAccelerate


class UTestAccelerate(GTestAccelerate):
    """UTest 执行加速

    通过多进程并行执行, 以提升 UTest 执行效率.
    """

    @property
    def mark(self) -> str:
        return "UTest"

    @staticmethod
    def main() -> bool:
        """主处理流程
        """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"UTest Execute Accelerate", epilog="Best Regards!")
        UTestAccelerate.reg_args(parser=parser)
        parser.add_argument("-j", "--job_num", nargs="?", type=int, default=None,
                            help="Specific parallel accelerate job num.")
        # 流程处理
        args = parser.parse_args()
        params = []

        # 获取job_num
        if args.job_num:
            job_num = args.job_num
        else:
            if os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", 0):
                job_num = int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"), 0)
            elif os.environ.get("PYPTO_UTEST_PARALLEL_NUM", 0):
                job_num = int(os.environ.get("PYPTO_UTEST_PARALLEL_NUM", 0))
            else:
                job_num = int(math.ceil(float(cpu_count()) * 0.8))    # use 0.8 cpu
        job_num = min(min(min(max(int(job_num), 1), cpu_count()), 16), len(args.cases))

        for job_idx in range(job_num):
            params.append(GTestAccelerate.ExecParam(cntr_id=job_idx))
        ctrl = UTestAccelerate(args=args, params=params)
        ctrl.process()
        return ctrl.post()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    exit(0 if UTestAccelerate.main() else 1)
