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
"""需要执行 target 工具基类定义, 串行单进程版.
"""
import logging
from abc import ABC
from datetime import datetime, timezone
from typing import List, Any

from utils.table import Table

from .case_abc import CaseAbc
from .tools_run_abc import ToolsRunAbc


class ToolsRunAbcSp(ToolsRunAbc, ABC):

    def __init__(self, args):
        super().__init__(args=args)

        # 执行控制
        self.device_id: int = self.device_list[0]

    @property
    def brief(self) -> List[Any]:
        datas = [["DeviceId", self.device_id]]
        return super().brief + datas

    def process(self) -> bool:
        rst: bool = True
        for cs in self.case_list:
            rst = rst and self.process_case(cs=cs, device_id=self.device_id)
            if not rst and self.halt_on_error:
                return False
        return rst

    def process_case(self, cs: CaseAbc, device_id: int) -> bool:
        if not cs.enable:
            logging.info("%s disable, skip it.", cs.full_name)
            return True
        # 用例准备
        ts = datetime.now(tz=timezone.utc)
        ret = self.process_case_prepare(cs=cs, device_id=device_id)
        te = datetime.now(tz=timezone.utc)
        if ret:
            logging.info("%s prepare success, cost %s secs", cs.full_name, (te - ts).seconds)
        else:
            logging.error("%s prepare failed, cost %s secs", cs.full_name, (te - ts).seconds)
            return False
        # 用例执行
        ts = datetime.now(tz=timezone.utc)
        ret = self.process_case_process(cs=cs, device_id=device_id)
        te = datetime.now(tz=timezone.utc)
        if ret:
            logging.info("%s process success, cost %s secs", cs.full_name, (te - ts).seconds)
        else:
            logging.error("%s process failed, cost %s secs", cs.full_name, (te - ts).seconds)
            return False
        # 用例后处理
        ts = datetime.now(tz=timezone.utc)
        ret = self.process_case_post(cs=cs, device_id=device_id)
        te = datetime.now(tz=timezone.utc)
        if ret:
            logging.info("%s post success, cost %s secs", cs.full_name, (te - ts).seconds)
        else:
            logging.error("%s post failed, cost %s secs", cs.full_name, (te - ts).seconds)
            return False
        return True

    def post(self) -> bool:
        heads = []
        datas = []
        for cs in self.case_list:
            if cs.result:
                continue
            heads, ds = cs.brief
            datas.append(ds)
        if len(datas) != 0:
            out: str = ""
            out += f"{self.__class__.__name__} post, InterceptFlag({self.intercept}), "
            out += f"Total execute {len(self.case_list)} cases, has {len(datas)} failed case brief below:\n"
            out += Table.table(datas=datas, headers=heads)
            if self.intercept:
                logging.error("%s", out)
            else:
                logging.warning("%s", out)
            return not self.intercept
        return True
