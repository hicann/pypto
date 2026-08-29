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
"""Helper table."""

import copy
from typing import Any, List, Optional


class Table:
    """Table processing, reducing dependency on other external libraries"""

    @staticmethod
    def table(
        datas: List[List[Any]], headers: Optional[List[Any]] = None, col_width_max: int = 128, auto_sort: bool = True
    ) -> str:
        """Get a formatted table string

        :param datas: Two-dimensional table data, each sublist represents a row
        :param headers: Optional list of headers
        :param col_width_max: Maximum column width per column
        :param auto_sort: Automatically sort in ascending order
        :return: Formatted grid table string
        """
        # Normalize to a table with equal number of columns,
        # and calculate the maximum string length required for each column
        _heads = None if headers is None else copy.deepcopy(headers)
        _datas = copy.deepcopy(datas)
        if _heads is not None:
            _datas.append(_heads)
        col_num = max(len(row) for row in _datas)
        col_widths = [0] * col_num
        for i, row in enumerate(_datas):
            if len(row) < col_num:
                _datas[i] = row + [""] * (col_num - len(row))
                row = _datas[i]
            for j, col in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(col)))
        if _heads is not None:
            _datas = _datas[:-1]
            if auto_sort:
                _datas.sort(reverse=False)

        # Build header (if present)
        separator = Table._make_separator(widths=col_widths, width_max=col_width_max, div='-')
        lines = []
        if headers is not None:
            lines.append(separator)
            lines.append(Table._make_details(widths=col_widths, datas=_heads, width_max=col_width_max))
            lines.append(Table._make_separator(widths=col_widths, width_max=col_width_max, div='='))
        else:
            lines.append(separator)

        # Add data rows
        for row in _datas:
            lines.append(Table._make_details(widths=col_widths, datas=row, width_max=col_width_max, mode="left"))
            lines.append(separator)

        return "\n" + "\n".join(lines)

    @staticmethod
    def _make_separator(widths: List[int], width_max: int = 128, cross: str = '+', div: str = '-') -> str:
        """Build a separator line

        :param widths: Column widths for each column
        :param width_max: Maximum column width per column
        :param cross: Character at intersection points
        :param div: Character for division line elements
        :return: Separator line string
        """
        line = cross
        for width in widths:
            width = min(width, width_max)
            line += div * (width + 2) + cross
        return line

    @staticmethod
    def _make_details(widths: List[int], datas: List[Any], width_max: int = 128, mode: str = "center") -> str:
        """Build data row details

        :param widths: Column widths for each column
        :param width_max: Maximum column width per column
        :param datas: Elements in a single row
        :param mode: Alignment mode for elements
        :return: Data row string
        """
        line = "|"
        mode = mode.lower()
        for idx, ele in enumerate(datas):
            ele = str(ele)
            if mode in ["right", "r"]:
                ele = ele.rjust(widths[idx])
            elif mode in ["left", "l"]:
                ele = ele.ljust(widths[idx])
            else:
                ele = ele.center(widths[idx])
            ele = ele[:width_max]
            line += f" {ele} |"
        return line
