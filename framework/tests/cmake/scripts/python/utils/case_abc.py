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
"""用例基类定义.

范围如下:
    1. 承载参数, 结果等数据存储, 数据统计, 数据落盘相关功能;
    2. 承载用例执行功能;
"""
import csv
from abc import ABC, abstractmethod
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class CaseAbc(ABC):
    @unique
    class FieldType(Enum):
        Network: str = "NetworkType"
        Name: str = "TestCaseName"
        Enable: str = "Enable"

    def __init__(self, desc: Optional[Dict[str, Any]]):
        self._fields_dict: Dict[str, Any] = {}
        desc = desc if desc else {}
        for _k, _v in desc.items():
            self._fields_dict.update({_k: _v})

    @property
    def brief(self) -> Tuple[List[Any], List[Any]]:
        """获取缩略描述信息, 主要用于打屏输出
        """
        heads = [CaseAbc.FieldType.Network.value, CaseAbc.FieldType.Name.value, "Result"]
        datas = [self.network, self.name, self._get_field_rst(self.result)]
        return heads, datas

    @property
    def detail(self) -> Tuple[List[Any], List[Any]]:
        """获取详细描述信息, 主要用于落盘输出
        """
        heads = [CaseAbc.FieldType.Network.value, CaseAbc.FieldType.Name.value, CaseAbc.FieldType.Enable.value,
                 "Result"]
        datas = [self.network, self.name, self.enable, self._get_field_rst(self.result)]
        return heads, datas

    @property
    def network(self) -> str:
        return self._get_field_str(field=CaseAbc.FieldType.Network.value)

    @property
    def name(self) -> str:
        return self._get_field_str(field=CaseAbc.FieldType.Name.value)

    @property
    def full_name(self) -> str:
        return f"Case[{self.network}" + ":" + f"{self.name}]"

    @property
    def enable(self) -> bool:
        return self._get_field_bool(field=CaseAbc.FieldType.Enable.value)

    @property
    @abstractmethod
    def result(self) -> bool:
        """子类必须实现该属性, 用于获取执行结果
        """
        pass

    @classmethod
    def _get_field_rst(cls, rst: bool) -> str:
        return "Success" if rst else "Failed"

    def update(self, k: str, v: Any):
        self._fields_dict.update({k: v})

    def dump_csv(self, file: Path, append: bool = False):
        heads, datas = self.detail
        append = append and file.exists() and file.is_file() and file.stat().st_size > 0
        mode = 'a' if append else 'w'
        with open(file, mode=mode, newline='', encoding='utf-8') as fp:
            csv_writer = csv.writer(fp)
            if not append:
                csv_writer.writerow(heads)
            csv_writer.writerow(datas)

    def _get_field_str(self, field: str, default: str = "unknown"):
        return str(self._get_field(field=field, default=default))

    def _get_field_int(self, field: str, default: int = 0):
        return int(self._get_field(field=field, default=default))

    def _get_field_float(self, field: str, default: float = float(0)):
        return float(self._get_field(field=field, default=default))

    def _get_field_bool(self, field: str, default: bool = False):
        return self._get_field_str(field=field, default=str(default)).lower() == "true"

    def _get_field(self, field: str, default):
        val = self._fields_dict.get(field, None)
        return val if val is not None and str(val) != "" else default
