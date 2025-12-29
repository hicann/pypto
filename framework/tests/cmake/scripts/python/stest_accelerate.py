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
"""STest 用例并行执行.
"""
import argparse
import logging
from typing import List, Any, Optional, Dict

from accelerate.gtest_accelerate import GTestAccelerate


class STestAccelerate(GTestAccelerate):
    """STest 执行加速

    通过多进程并行执行, 以提升 STest 执行效率.
    """

    @property
    def mark(self) -> str:
        return "STest"

    @staticmethod
    def main() -> bool:
        """主处理流程
        """
        # 参数注册
        parser = argparse.ArgumentParser(description=f"STest Execute Accelerate", epilog="Best Regards!")
        STestAccelerate.reg_args(parser=parser)
        parser.add_argument("-d", "--device", nargs="?", type=int, action="append",
                            help="Specific parallel accelerate device, "
                                 "If this parameter is not specified, 0 device will be used by default.")
        # 流程处理
        args = parser.parse_args()
        params: List[GTestAccelerate.ExecParam] = []
        device_list: List[int] = [0]
        if args.device is not None:
            device_list = [int(d) for d in list(set(args.device)) if d is not None and str(d) != ""]
        for _id in device_list:
            p = GTestAccelerate.ExecParam(cntr_id=_id, envs_func=STestAccelerate.set_device_id_envs)
            params.append(p)
        ctrl = STestAccelerate(args=args, params=params, cntr_name="Device")
        ctrl.process()
        return ctrl.post()

    @staticmethod
    def set_device_id_envs(p: Any) -> Optional[Dict[str, str]]:
        self: GTestAccelerate.ExecParam = p
        return {"TILE_FWK_DEVICE_ID": f"{self.cntr_id}"}


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - PID[%(process)d] - %(levelname)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    exit(0 if STestAccelerate.main() else 1)
