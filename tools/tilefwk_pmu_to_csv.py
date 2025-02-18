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
"""
"""
import os
import argparse
import struct
import json
from tabulate import tabulate
import pandas as pd

# MsprofAdditonalInfo: 256 byte
TILEFWK_STARS_PMU_FMT = "HHIIIQ" + "HHIHBB" + "I" * 7 + "IIQIIIIIIII" * 4
MSPROF_REPORT_AICPU_HCCL_OP_INFO = 10
PROF_DATATYPE_PMU = 3


class TileFwkPmuStructBean():
    def __init__(self: any, *args) -> None: # 利用struct 的unpack的特性传递 解压后数据
        filed = args[0]
        self._magic_number = filed[0]
        self._level = filed[1]
        self._type = filed[2]
        self._thread_id = filed[3]
        self._data_len = filed[4]
        self._time_stamp = filed[5]
        self._head = filed[6]
        self._core_id = filed[7] & 127 # 取后低7位core_id
        self._data_type = (filed[7] >> 10) & 63 # 取高6位data_type
        self._task_id = filed[8]
        self._stream_id = filed[9]
        self._task_count = filed[10]
        self._res0 = filed[11]
        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_PMU:
            return
        self._task_list = []
        self._task_count = min(4, self._task_count)
        for i in range(0, self._task_count):
            self._task_list.append([
                filed[19 + i * 11], # sub graph id
                filed[20 + i * 11], # sub task id
                filed[21 + i * 11], # total cycle
                filed[22 + i * 11], # pmu cnt 0
                filed[23 + i * 11], # pmu cnt 1
                filed[24 + i * 11], # pmu cnt 2
                filed[25 + i * 11], # pmu cnt 3
                filed[26 + i * 11], # pmu cnt 4
                filed[27 + i * 11], # pmu cnt 5
                filed[28 + i * 11], # pmu cnt 6
                filed[29 + i * 11]  # pmu cnt 7
            ])


    @property
    def data(self: any) -> list():
        if self._type != MSPROF_REPORT_AICPU_HCCL_OP_INFO or self._data_type != PROF_DATATYPE_PMU:
            return []
        print("----------------------------------------------------------")
        print("TileFwk pmu get level: ", self._level)
        print("TileFwk pmu get type: ", self._type)
        print("TileFwk pmu get head: ", self._head)
        print("TileFwk pmu get core id: ", self._core_id)
        print("TileFwk pmu get data type: ", self._data_type)
        print("TileFwk pmu get task count: ", self._task_count)
        print("----------------------------------------------------------")
        task_head_list = [
            self._thread_id,
            self._task_id,
            self._stream_id,
            self._core_id
        ]
        print("task_head_list: ", task_head_list)
        print("task_pmu_list: ", self._task_list)
        for item in self._task_list:
            item[0:0] = task_head_list
        return self._task_list


def parse(file_path: str) -> list():
    bean_list = []
    print("start to parse")
    for name in os.listdir(file_path):
        if "aicpu.data" not in name:
            continue
        print("support data: " + name)
        # 构建完整的文件或目录路径
        file_name = os.path.join(file_path, name)
        # 如果是文件，则打印其路径
        print("support file path: ", file_name)
        file_size = os.path.getsize(file_name)
        if not file_size and "complete" not in file_name:
            print(file_name + " file size is empty.")
            return bean_list

        struct_size = struct.calcsize(TILEFWK_STARS_PMU_FMT)
        with open(file_name, "rb") as fd:
            context = fd.read()
            for _index in range(file_size // struct_size):
                data = context[_index * struct_size:(_index + 1) * struct_size]
                append_list = TileFwkPmuStructBean(struct.unpack(TILEFWK_STARS_PMU_FMT, data)).data
                if not append_list:
                    continue
                bean_list += append_list
    print("total_pmu_list: ", bean_list)
    return bean_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="请输入要解析的绝对路径")
    parser.add_argument("-pe", "--pmuEvent", nargs="?", type=int, default=2, choices=[1, 2, 4, 5, 6, 7, 8],
                            help="pmuEvent.")
    parser.add_argument('--output', default='', help="pmu数据存储路径")
    args = parser.parse_args()
    print("start parser pmu data:" + args.path)
    task_pmu_list = parse(args.path)
    if not task_pmu_list:
        print("empty pmu list")
        return
    print("end parser pmu data")
    table_header = ["thread id", "task id", "stream id", "core id", "seqNo", "sub task id", "total cycle"]
    table_pmu_header = []
    if args.pmuEvent == 1:
        table_pmu_header = ["cube_fp16_exec", "cube_int8_exec", "vec_fp32_exec", "vec_fp16_128lane_exec",
                            "vec_fp16_64lane_exec", "vec_int32_exec", "vec_misc_exec"]
    elif args.pmuEvent == 2:
        table_pmu_header = ["vec_busy_cycles", "cube_busy_cycles", "scalar_busy_cycles", "mte1_busy_cycles",
                            "mte2_busy_cycles", "mte3_busy_cycles", "icache_miss", "icache_req"]
    elif args.pmuEvent == 4:
        table_pmu_header = ["ub_read_req", "ub_write_req", "l1_read_req", "l1_write_req", "l2_read_req",
                            "l2_write_req", "main_read_req", "main_write_req"]
    elif args.pmuEvent == 5:
        table_pmu_header = ["l0a_read_req", "l0a_read_req", "l0b_read_req", "l0b_write_req", "l0c_read_req",
                            "l0c_write_req"]
    elif args.pmuEvent == 6:
        table_pmu_header = ["bankgroup_stall_cycles", "bank_stall_cycles", "vec_resc_conflict_cycles"]
    elif args.pmuEvent == 7:
        table_pmu_header = ["ub_read_bw_mte", "l2_write_bw", "main_mem_write_bw", "ub_write_bw_mte",
                            "ub_read_bw_vector", "ub_write_bw_vector", "ub_read_bw_scalar", "ub_write_bw_scalar"]
    elif args.pmuEvent == 8:
        table_pmu_header = ["write_cache_hit", "write_cache_miss_allocate", "r0_read_cache_hit",
                            "r0_read_cache_miss_allocate", "r1_read_cache_hit", "r1_read_cache_miss_allocate"]
    if len(table_pmu_header) < 8:
        task_pmu_list = [item[:len(table_pmu_header) - 8] for item in task_pmu_list]
    table_header += table_pmu_header
    print(tabulate(task_pmu_list, headers=table_header, tablefmt='grid', floatfmt=".1f", showindex="always"))
    df = pd.DataFrame(task_pmu_list, columns=table_header)
    if args.output == "":
        df.to_csv('tilefwk_prof_pmu.csv', index=False)
    else:
        df.to_csv(f'{args.output}/tilefwk_prof_pmu.csv', index=False)


if __name__ == '__main__':
    main()
