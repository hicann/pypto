#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import json
import copy
import struct
import argparse
import logging
import multiprocessing
from itertools import groupby
import ml_dtypes
import numpy as np
import pandas as pd


# ===================== 核心配置（需和C/C++端一致）=====================
DEV_SHAPE_DIM_MAX = 5  # 替换为实际值
BYTE_ORDER = "<"       # 小端：< ；大端：> ；本机字节序：=

# 单个字段的字节数定义（无对齐，纯原始字节）
FIELD_SIZES = {
    "uint32_t": 4,
    "int32_t": 4,
    "int64_t": 8,
    "uint64_t": 8
}


logging.basicConfig(
    level=logging.DEBUG,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式（含时间、级别、内容）
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("app.log", encoding="utf-8")  # 输出到文件（持久化）
    ]
)


def _get_data_type(data_type: int):
    """数据类型数值转可读字符串"""
    _data_type_full_mapping = {
        0: ("DT_INT4", ml_dtypes.int4),
        1: ("DT_INT8", np.int8),
        2: ("DT_INT16", np.int16),
        3: ("DT_INT32", np.int32),
        4: ("DT_INT64", np.int64),
        5: ("DT_FP8", ml_dtypes.float8_e4m3fn),
        6: ("DT_FP16", np.float16),
        7: ("DT_FP32", np.float32),
        8: ("DT_BF16", ml_dtypes.bfloat16),
        9: ("DT_HF4", None),                    # 暂不支持解析
        10: ("DT_HF8", None),                   # 暂不支持解析
        11: ("DT_UINT8", np.uint8),
        12: ("DT_UINT16", np.uint16),
        13: ("DT_UINT32", np.uint32),
        14: ("DT_UINT64", np.uint64),
        15: ("DT_BOOL", np.bool_),
        16: ("DT_DOUBLE", np.float64),
        17: ("DT_BOTTOM", None)
    }
    return _data_type_full_mapping.get(data_type, f"UNKNOWN({data_type})")


class VerifyRes:
    def __init__(self):
        self.verify_codegen_op_info_list = None
        self.verify_tensorgraph_op_info_list = None
        self.verify_path = ""

    @staticmethod
    def _compare_codegen_tensors(tensor_infos, tensor_infos_new):

        for i, tensor_info in enumerate(tensor_infos_new):
            dump_tshape = tensor_info.get("shape")
            verify_tensor_info = tensor_info["verify_dup_tensor"]
            verify_tshape = tensor_info["valid_shape"]
            tensor_infos[i]["verify_tensor_file"] = tensor_info["verify_dup_tensor"]
            
            if os.path.exists(verify_tensor_info) and len(verify_tshape) == len(dump_tshape):
                dtype = _get_data_type(tensor_info["dataType"])[1]
                
                verify_tensor_data = np.fromfile(verify_tensor_info, dtype)
                verify_tensor_data = verify_tensor_data.reshape(verify_tshape)
                
                data = np.fromfile(tensor_info["bin_file"], dtype)
                data = data.reshape(dump_tshape)
                
                slices = []
                for dim in range(data.ndim):
                    stop = min(verify_tshape[dim], dump_tshape[dim])
                    slices.append(slice(0, stop))
                
                tensor_infos[i]["cmp_res"] = np.allclose(
                    data[tuple(slices)], 
                    verify_tensor_data[tuple(slices)], 
                    1e-3, 1e-3
                )
            else:
                tensor_infos[i]["cmp_res"] = "NO_CMP"
    
    def read_verify_result(self, verify_path):
        self.verify_path = verify_path
        verify_res_file = os.path.join(self.verify_path, "verify_result.csv")
        if not os.path.exists(verify_res_file):
            logging.error(f"verify path {verify_path} not exist.")
            return
        
        df = pd.read_csv(verify_res_file, encoding="utf-8")
        df_clean = df.dropna(subset=["rawTensorMagic"]).copy()
        df_clean["rawTensorMagic"] = df_clean["rawTensorMagic"].astype(int)

        codegen_filter = df_clean["verifyType"].str.contains("CodegenPreproc", na=False)
        df_codegen = df_clean[codegen_filter]
        df_codegen = df_codegen.dropna(subset=["callopMagic"]).copy()
        df_codegen["callopMagic"] = df_codegen["callopMagic"].astype(int)
        self.verify_codegen_op_info_list = df_codegen

        tensor_graph_filter = df_clean["verifyType"].str.contains("tensor_graph", na=False)
        self.verify_tensorgraph_op_info_list = df_clean[tensor_graph_filter]


    def get_verify_res_single(self, tensor_info, op_info_list):
        raw_magic = tensor_info.get("rawMagic")
        ioflag = tensor_info.get("ioflag")
        callop_magic = tensor_info.get("callopMagic")
        tensor_info_offset_str = '_'.join(str(item) for item in tensor_info.get("offset"))

        verify_dup_tensor = ""
        valid_shape = []
        loop_info = ""
        
        for op_info in op_info_list:
            if callop_magic != op_info.get("callopMagic"):
                continue 
            if raw_magic != op_info.get("callopRawMagic"):
                continue
            
            if ioflag == "input" and op_info.get("opCode") in ["COPY_IN", "VIEW"]:
                verify_op_offset = json.loads(op_info.get("offset"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("outputTensor")
                    valid_shape = json.loads(op_info.get("outputValidShape"))
                    loop_info = op_info.get("loopInfo")
            elif ioflag == "output" and op_info.get("opCode") in ["COPY_OUT"]:
                verify_op_offset = json.loads(op_info.get("offset"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("inputTensors")   # COPY_OUT的op只会有一个输入
                    valid_shape = json.loads(op_info.get("inputValidShape"))
                    loop_info = op_info.get("loopInfo")

        if verify_dup_tensor:
            verify_dup_tensor = os.path.join(self.verify_path, op_info.get("verifyType"), verify_dup_tensor)
        tensor_info["verify_dup_tensor"] = verify_dup_tensor
        tensor_info["valid_shape"], tensor_info["loop_info"] = valid_shape, loop_info

    def get_verify_codegen_res(self, tensor_infos):
        if self.verify_codegen_op_info_list is None:
            logging.info("verify codegen op info is None.")
            return tensor_infos
        callop_magic = tensor_infos[0].get("callopMagic")   # callop
        tensor_infos_new = copy.deepcopy(tensor_infos)
        op_info_list = self.verify_codegen_op_info_list.copy(deep=True)
        op_info_list = op_info_list[op_info_list["callopMagic"] == callop_magic]
        all_match = False
        while not all_match:
            self.get_verify_res_single(tensor_infos_new[0], op_info_list.to_dict(orient='records'))
            if not tensor_infos_new[0].get("loop_info"):
                break
            op_info_list_with_loop = op_info_list[op_info_list["loopInfo"] == tensor_infos_new[0].get("loop_info")]
            all_match = True
            for i, tensor_info in enumerate(tensor_infos_new):
                if i == 0:
                    continue
                tensor_info["verify_dup_tensor"] = ""   # 先清理上一次的结果
                self.get_verify_res_single(tensor_info, op_info_list_with_loop.to_dict(orient='records'))
                if not tensor_info["verify_dup_tensor"]:
                    all_match = False
                    break
            if all_match:
                break
            op_info_list = op_info_list[op_info_list["loopInfo"] != tensor_infos_new[0].get("loop_info")]
        if not all_match:
            for _, tensor_info in enumerate(tensor_infos):
                tensor_info["verify_tensor_file"] = "" 
                tensor_info["cmp_res"] = "NO_CMP"
            return tensor_infos

        self._compare_codegen_tensors(tensor_infos, tensor_infos_new)
        return tensor_infos

    def get_verify_tensor_graph_res(self, tensor_info):
        raw_magic = tensor_info.get("rawMagic")

        verify_dup_tensor = ""
        valid_shape = []
        
        # verify_tensorgraph_op_info_list
        if self.verify_tensorgraph_op_info_list is None or self.verify_tensorgraph_op_info_list.empty:
            return verify_dup_tensor, valid_shape
        
        # 按rawTensorMagic过滤
        filtered_df = self.verify_tensorgraph_op_info_list[
            self.verify_tensorgraph_op_info_list["rawTensorMagic"] == raw_magic
        ]
        if filtered_df.empty:
            return verify_dup_tensor, valid_shape

        sorted_df = filtered_df.sort_values(by="No.", ascending=True)
        last_op_info = sorted_df.iloc[-1]
        verify_dup_tensor = last_op_info.get("outputTensor")
        valid_shape = json.loads(last_op_info.get("outputValidShape"))
        if verify_dup_tensor:
            verify_dup_tensor = os.path.join(self.verify_path, last_op_info.get("verifyType"), verify_dup_tensor)
        
        return verify_dup_tensor, valid_shape

_verify_res = VerifyRes()


class CompactDumpTensorInfoParser:
    def __init__(self, dump_tensor_path):
        self.dump_tensor_path = dump_tensor_path
        # 计算单个结构体的紧凑总字节数（无对齐）
        self.struct_compact_size = self._calc_compact_size()
        # 定义字段解析顺序和类型（严格匹配C/C++结构体）
        self.field_specs = [
            ("headSize", "uint32_t"),
            ("funcId", "uint32_t"),
            ("taskId", "uint32_t"),
            ("callopMagic", "uint32_t"),
            ("coreId", "int32_t"),
            ("dataType", "int32_t"),
            ("rawMagic", "int32_t"),
            ("dims", "int32_t"),
            ("exeStart", "int64_t"),
            ("exeEnd", "int64_t"),
            ("rootHash", "uint64_t"),
            ("funcHash", "uint64_t"),
            ("timeStamp", "uint64_t"),
            ("shape", "uint64_t", DEV_SHAPE_DIM_MAX),  # 数组：类型 + 长度
            ("offset", "uint64_t", DEV_SHAPE_DIM_MAX),
            ("rawShape", "uint64_t", DEV_SHAPE_DIM_MAX),
            ("tensorAddr", "uint64_t")
        ]
        self.raw_tensor_info = {}
        self.task_tensor_info = {}
    
    @staticmethod
    def _calc_compact_size():
        """计算无对齐的紧凑总字节数"""
        total = 0
        # 基础字段
        total += FIELD_SIZES["uint32_t"] * 4  # headSize ~ taskId
        total += FIELD_SIZES["int32_t"] * 4   # coreId ~ dims
        total += FIELD_SIZES["int64_t"] * 2   # exeStart ~ exeEnd
        total += FIELD_SIZES["uint64_t"] * 3   # rootHash ~ timeStamp
        # 数组字段
        array_size = FIELD_SIZES["uint64_t"] * DEV_SHAPE_DIM_MAX
        total += array_size * 3  # shape + offset + rawShape
        # 最后一个字段
        total += FIELD_SIZES["uint64_t"]      # tensorAddr
        return total

    @staticmethod
    def _parse_field(bin_data: bytes, offset: int, field_type: str, array_len: int = 1) -> tuple:
        """解析单个字段（支持标量/数组）
        Returns:
            (解析后的值, 字段占用的总字节数)
        """
        field_size = FIELD_SIZES[field_type]
        total_bytes = field_size * array_len
        # 校验数据长度
        if offset + total_bytes > len(bin_data):
            raise ValueError(f"字段解析失败：偏移{offset}，需要{total_bytes}字节，剩余{len(bin_data)-offset}字节")
        
        # 构建单个元素的格式符
        fmt_char = {
            "uint32_t": "I",
            "int32_t": "i",
            "int64_t": "q",
            "uint64_t": "Q"
        }[field_type]
        # 拼接格式符（字节序 + 元素格式符*数量）
        fmt = BYTE_ORDER + fmt_char * array_len
        
        # 解析数据
        values = struct.unpack_from(fmt, bin_data, offset)
        # 标量返回单个值，数组返回元组
        if array_len == 1:
            return values[0], total_bytes
        else:
            return values, total_bytes
    
    @staticmethod
    def _verify_merged_tensor(merge_tensor_info, raw_data):
        # 获取验证张量信息
        verify_tensor_info, verify_tshape = _verify_res.get_verify_tensor_graph_res(merge_tensor_info)
        dump_tshape = merge_tensor_info.get("rawShape")
        
        # 验证张量存在且形状完全匹配时才进行比较
        if os.path.exists(verify_tensor_info) and len(verify_tshape) == len(dump_tshape) and \
                all(vdim == ddim for vdim, ddim in zip(verify_tshape, dump_tshape)):
            
            merge_tensor_info["verify_tensor_file"] = verify_tensor_info
            dtype = _get_data_type(merge_tensor_info["dataType"])[1]
            
            # 读取验证张量并进行比较
            verify_tensor_data = np.fromfile(verify_tensor_info, dtype)
            verify_tensor_data = verify_tensor_data.reshape(verify_tshape)
            merge_tensor_info["cmp_res"] = np.allclose(raw_data, verify_tensor_data, 1e-3, 1e-3)
        
        return merge_tensor_info

    def parse_single(self, bin_data: bytes, offset: int = 0) -> dict:
        """解析单个紧凑存储的DumpTensorInfo结构体"""
        result = {}
        current_offset = offset
        
        # 逐个解析字段（严格按顺序）
        for spec in self.field_specs:
            if len(spec) == 2:
                # 标量字段：(name, type)
                name, field_type = spec
                value, bytes_used = self._parse_field(bin_data, current_offset, field_type)
            else:
                # 数组字段：(name, type, array_len)
                name, field_type, array_len = spec
                value, bytes_used = self._parse_field(bin_data, current_offset, field_type, array_len)
            
            result[name] = value
            current_offset += bytes_used
        

        dims = result.get("dims")
        if dims > 0 and dims < DEV_SHAPE_DIM_MAX:
            result["shape"] = result["shape"][:dims]
            result["offset"] = result["offset"][:dims]
            result["rawShape"] = result["rawShape"][:dims]
        
        # 衍生字段（可选）
        result["exeDuration"] = result.get("exeEnd") - result.get("exeStart")
        result["dataTypeStr"] = _get_data_type(result.get("dataType", 17))[0]
        
        return result

    def parse_file(self, file_path: str) -> list[dict]:
        """解析整个紧凑存储的bin文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        with open(file_path, "rb") as f:
            bin_data = f.read()
        
        tensor_info = self.parse_single(bin_data, 0)
        dtype = _get_data_type(tensor_info["dataType"])[1]
        data = np.frombuffer(bin_data, dtype, offset=tensor_info["headSize"])
        bin_file = f"{file_path[:-6]}.data"
        data.tofile(bin_file)
        
        tensor_info["ioflag"] = "output"
        if "input" in bin_file.split("_")[-1]:
            tensor_info["ioflag"] = "input"

        tensor_info["bin_file"] = bin_file

        if tensor_info["ioflag"] == "output":
            if tensor_info["rawMagic"] not in self.raw_tensor_info:
                self.raw_tensor_info[tensor_info["rawMagic"]] = []
            self.raw_tensor_info[tensor_info["rawMagic"]].append(tensor_info)
        
        if tensor_info["taskId"] not in self.task_tensor_info:
            self.task_tensor_info[tensor_info["taskId"]] = []
        self.task_tensor_info[tensor_info["taskId"]].append(tensor_info)
        return tensor_info
    
    def tensor_compare(self):
        merged_result = []
        if not self.task_tensor_info:
            for _, tensor_infos in self.task_tensor_info.items():
                merged_result.extend(tensor_infos)
            return merged_result

        num_tasks = len(self.task_tensor_info)
        num_cpus = os.cpu_count() or 1
        num_processes = min(8, num_cpus, num_tasks)
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = []
            for _, tensor_infos in self.task_tensor_info.items():
                tasks.append(tensor_infos)
            
            try:
                results = pool.map(_verify_res.get_verify_codegen_res, tasks)
            except Exception as e:
                logging.error(f"Tensor comparison failed with error: {e}")
                for tensor_infos in tasks:
                    merged_result.extend(tensor_infos)
                return merged_result

        for result in results:
            merged_result.extend(result)
        
        return merged_result

    def merge_raw_tensor_data(self, raw_magic, tensor_infos):
        # 创建合并张量的基础信息
        merge_tensor_info = {}
        merge_tensor_info["rawMagic"] = raw_magic
        merge_tensor_info["dataTypeStr"] = tensor_infos[0]["dataTypeStr"]
        merge_tensor_info["ioflag"] = tensor_infos[0]["ioflag"]
        merge_tensor_info["rawShape"] = tensor_infos[0]["rawShape"]
        merge_tensor_info["dataType"] = tensor_infos[0]["dataType"]
        merge_tensor_info["rootHash"] = 0
        merge_tensor_info["funcHash"] = 0
        
        # 生成保存路径
        file_path = os.path.join(self.dump_tensor_path, 
                                f"raw_{raw_magic}_{tensor_infos[0]['dataTypeStr']}_{tensor_infos[0]['ioflag']}.data")
        merge_tensor_info["bin_file"] = file_path
        
        # 按offset排序张量
        tensor_infos_sorted = sorted(tensor_infos, key=lambda x: x["offset"])
        grouped_tensors = {}
        for key, group in groupby(tensor_infos_sorted, key=lambda x: x["offset"]):
            grouped_tensors[key] = list(group)
        if len(grouped_tensors) == 1:
            return merge_tensor_info, None
        
        # 执行合并操作
        dtype = _get_data_type(merge_tensor_info["dataType"])[1]
        raw_data = np.zeros(merge_tensor_info["rawShape"], dtype)
        
        for tensor_info in tensor_infos:
            is_tensor_valid = True
            data = np.fromfile(tensor_info["bin_file"], dtype)
            data = data.reshape(tensor_info.get("shape"))
            
            # 计算切片范围
            raw_slices, data_slices = [], []
            for dim in range(data.ndim):
                start = tensor_info["offset"][dim]
                stop = min(merge_tensor_info["rawShape"][dim], start + data.shape[dim])
                if start >= stop:
                    is_tensor_valid = False
                
                raw_slices.append(slice(start, stop))
                data_slices.append(slice(0, min(merge_tensor_info["rawShape"][dim] - start, data.shape[dim])))
            
            # 合并有效张量
            if is_tensor_valid:
                raw_data[tuple(raw_slices)] = data[tuple(data_slices)]
        
        # 保存合并后的张量
        raw_data.tofile(file_path)
        return merge_tensor_info, raw_data

    def merge_raw_tensor(self):
        merge_tensor_infos = []
        for raw_magic, tensor_infos in self.raw_tensor_info.items():
            # 合并张量数据
            merge_tensor_info, raw_data = self.merge_raw_tensor_data(raw_magic, tensor_infos)
            
            # 如果有合并后的数据，进行验证
            if raw_data is not None:
                merge_tensor_info = self._verify_merged_tensor(merge_tensor_info, raw_data)
                merge_tensor_infos.append(merge_tensor_info)
        return merge_tensor_infos


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser dump_tensor.")
    parser.add_argument("--dump_tensor_path", type=str, default="output/dump_tensor/device_0", 
                        help="directory like output/dump_tensor/device_x")
    parser.add_argument("--verify_path", type=str, default="", help="Path to verify_result.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not os.path.exists(args.dump_tensor_path):
        logging.error(f"目录不存在：{args.dump_tensor_path}")
        return
    # 初始化紧凑解析器
    parser = CompactDumpTensorInfoParser(args.dump_tensor_path)
    logging.info(f"单个结构字节数：{parser.struct_compact_size}")

    _verify_res.read_verify_result(args.verify_path)

    for dir_path, _, file_names in os.walk(args.dump_tensor_path):
        for file_name in file_names:
            if not file_name.endswith(".tdump"):
                continue
            bin_file = os.path.join(dir_path, file_name)
            parser.parse_file(bin_file)
    
    tensor_infos = parser.tensor_compare()
    merge_tensor_infos = parser.merge_raw_tensor()
    tensor_infos.extend(merge_tensor_infos)
    df = pd.DataFrame(tensor_infos)
    df["rootHash"] = "'" + df["rootHash"].astype(str)
    df["funcHash"] = "'" + df["funcHash"].astype(str)
    logging.info(df)

    df.to_csv(os.path.join(args.dump_tensor_path, "tensor_info.csv"), index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
