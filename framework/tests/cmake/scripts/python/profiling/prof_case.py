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
"""性能分析工具.
"""
from enum import Enum, unique
from typing import List, Optional, Dict, Any, Tuple

from utils.case_abc import CaseAbc


class ProfCase(CaseAbc):
    """Profiling 用例, 仅承载数据存储, 统计, 落盘相关功能.
    """

    @unique
    class FieldType(Enum):
        # Args
        ProfWarnUpCnt: str = "ProfWarnUpCnt"
        ProfTryCnt: str = "ProfTryCnt"
        ProfMaxCnt: str = "ProfMaxCnt"
        # Result
        TimeStamp: str = "TimeStamp"
        Us: str = "Us"
        Cycle: str = "Cycle"
        CycleThreshold: str = "CycleThreshold"
        JitterRate: str = "JitterRate"
        JitterRateThreshold: str = "JitterRateThreshold"

    PROF_WARN_UP_CNT_DEF: int = 5
    PROF_TRY_CNT_DEF: int = 10
    PROF_MAX_CNT_DEF: int = 5

    PROF_CYCLE_JITTER_RATE_THRESHOLD_DEF: float = float(1.05)  # 默认容许 5% 以内的波动
    PROF_JITTER_RATE_THRESHOLD_DEF: float = float(-0.08)  # 默认容许向下 8% 的抖动

    STATISTIC_RST_FILE_NAME: str = "prof_statistic_result.csv"
    STATISTIC_ALL_FILE_NAME: str = "prof_statistic_all.csv"

    def __init__(self, desc: Optional[Dict[str, Any]],
                 warn_up_cnt: Optional[int] = None,
                 try_cnt: Optional[int] = None,
                 max_cnt: Optional[int] = None):
        super().__init__(desc)
        warn_up = warn_up_cnt if warn_up_cnt is not None and warn_up_cnt >= 0 else self.prof_warn_up_cnt
        try_cnt = max(try_cnt if try_cnt is not None and try_cnt >= 0 else self.prof_try_cnt, 1)
        max_cnt = max(min(try_cnt, max_cnt if max_cnt is not None and max_cnt >= 0 else self.prof_max_cnt), 1)
        self._fields_dict.update({self.FieldType.ProfWarnUpCnt.value: warn_up,
                                  self.FieldType.ProfTryCnt.value: try_cnt,
                                  self.FieldType.ProfMaxCnt.value: max_cnt})

    @property
    def brief(self) -> Tuple[List[Any], List[Any]]:
        """获取缩略描述信息, 主要用于打屏输出
        """
        heads, datas = super().brief
        heads_ext = ["CtrlCnt(W/T/M)", ProfCase.FieldType.TimeStamp.value,
                     ProfCase.FieldType.Us.value, "Cycle(A/T/R)", "JitterRate(A/T/R)"]
        datas_ext = [f"{self.prof_warn_up_cnt} {self.prof_try_cnt} {self.prof_max_cnt}", self.timestamp,
                     f"{self.us:.2f}",
                     f"{self.cycle} {self.cycle_threshold} {self._get_field_rst(self.cycle_result)}",
                     f"{(self.jitter_rate * 100):5.2f}% {(self.jitter_rate_threshold * 100):5.2f}% "
                     f"{self._get_field_rst(self.jitter_rate_result)}"]
        return heads + heads_ext, datas + datas_ext

    @property
    def detail(self) -> Tuple[List[Any], List[Any]]:
        """获取详细描述信息, 主要用于落盘输出
        """
        heads, datas = super().detail
        heads_ext = [
            ProfCase.FieldType.ProfWarnUpCnt.value, ProfCase.FieldType.ProfTryCnt.value,
            ProfCase.FieldType.ProfMaxCnt.value, ProfCase.FieldType.TimeStamp.value,
            ProfCase.FieldType.Us.value,
            ProfCase.FieldType.Cycle.value, ProfCase.FieldType.CycleThreshold.value, "CycleResult",
            ProfCase.FieldType.JitterRate.value, ProfCase.FieldType.JitterRateThreshold.value, "JitterRateResult"]
        datas_ext = [self.prof_warn_up_cnt, self.prof_try_cnt, self.prof_max_cnt, self.timestamp, self.us,
                     self.cycle, self.cycle_threshold, self._get_field_rst(self.cycle_result),
                     self.jitter_rate, self.jitter_rate_threshold, self._get_field_rst(self.jitter_rate_result)]
        return heads + heads_ext, datas + datas_ext

    @property
    def result(self) -> bool:
        return self.cycle_result and self.jitter_rate_result

    @property
    def prof_warn_up_cnt(self) -> int:
        return self._get_field_int(field=ProfCase.FieldType.ProfWarnUpCnt.value, default=self.PROF_WARN_UP_CNT_DEF)

    @property
    def prof_try_cnt(self) -> int:
        return self._get_field_int(field=ProfCase.FieldType.ProfTryCnt.value, default=self.PROF_TRY_CNT_DEF)

    @property
    def prof_max_cnt(self) -> int:
        return self._get_field_int(field=ProfCase.FieldType.ProfMaxCnt.value, default=self.PROF_MAX_CNT_DEF)

    @property
    def timestamp(self) -> str:
        return self._get_field_str(field=ProfCase.FieldType.TimeStamp.value)

    @property
    def us(self) -> float:
        return self.cycle / 50

    @property
    def cycle(self) -> int:
        return self._get_field_int(field=ProfCase.FieldType.Cycle.value)

    @property
    def cycle_threshold(self) -> int:
        return self._get_field_int(field=ProfCase.FieldType.CycleThreshold.value)

    @property
    def cycle_result(self) -> bool:
        return self.cycle <= self.cycle_threshold * self.PROF_CYCLE_JITTER_RATE_THRESHOLD_DEF

    @property
    def jitter_rate(self) -> float:
        return self._get_field_float(field=ProfCase.FieldType.JitterRate.value)

    @property
    def jitter_rate_threshold(self) -> float:
        return self._get_field_float(field=ProfCase.FieldType.JitterRateThreshold.value,
                                     default=self.PROF_JITTER_RATE_THRESHOLD_DEF)

    @property
    def jitter_rate_result(self) -> bool:
        return self.jitter_rate >= self.jitter_rate_threshold
