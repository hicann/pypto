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
import struct
import argparse
import logging
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


class VerifyRes:
    def __init__(self):
        self.verify_op_info_list = []
        self.verify_path = ""
    
    def read_verify_result(self, verify_path):
        self.verify_path = verify_path
        verify_res_file = os.path.join(self.verify_path, "verify_result.csv")
        if not os.path.exists(verify_res_file):
            return
        
        df = pd.read_csv(verify_res_file, encoding="utf-8")
        df_clean = df.dropna(subset=["rawTensorMagic", "callopMagic"]).copy()
        df_clean["rawTensorMagic"] = df_clean["rawTensorMagic"].astype(int)
        df_clean["callopMagic"] = df_clean["callopMagic"].astype(int)
        codegen_filter = df_clean["verifyType"].str.contains("CodegenPreproc", na=False)
        df_res = df_clean[codegen_filter]
        self.verify_op_info_list = df_res.to_dict(orient='records')


    def get_verify_res(self, tensor_info) -> dict:
        raw_magic = tensor_info.get("rawMagic")
        ioflag = tensor_info.get("ioflag")
        callop_magic = tensor_info.get("callopMagic")
        tensor_info_offset_str = '_'.join(str(item) for item in tensor_info.get("offset"))

        verify_dup_tensor = ""
        valid_shape = []
        
        for op_info in self.verify_op_info_list:
            if callop_magic != op_info.get("callopMagic"):
                continue 
            
            if ioflag == "input" and f"tensor_Incast_{raw_magic}" in op_info.get("inputTensors") and   \
                op_info.get("opCode") in ["COPY_IN", "VIEW"]:
                verify_op_offset = json.loads(op_info.get("offset"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("outputTensor")
                    valid_shape = json.loads(op_info.get("outputValidShape"))
            elif ioflag == "output" and raw_magic == op_info.get("rawTensorMagic"):
                verify_op_offset = json.loads(op_info.get("offset"))
                verify_op_offset_str = '_'.join(str(item) for item in verify_op_offset)
                if verify_op_offset_str == tensor_info_offset_str:
                    verify_dup_tensor = op_info.get("inputTensors")   # COPY_OUT的op只会有一个输入
                    valid_shape = json.loads(op_info.get("inputValidShape"))

        if verify_dup_tensor:
            verify_dup_tensor = os.path.join(self.verify_path, op_info.get("verifyType"), verify_dup_tensor)
        return verify_dup_tensor, valid_shape
        

_verify_res = VerifyRes()


class CompactDumpTensorInfoParser:
    def __init__(self):
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

    @staticmethod
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
        result["dataTypeStr"] = self._get_data_type(result.get("dataType", 17))[0]
        
        return result

    def parse_file(self, file_path: str) -> list[dict]:
        """解析整个紧凑存储的bin文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        with open(file_path, "rb") as f:
            bin_data = f.read()
        
        tensor_info = self.parse_single(bin_data, 0)

        dtype = self._get_data_type(tensor_info["dataType"])[1]
        data = np.frombuffer(bin_data, dtype, offset=tensor_info["headSize"])
    
        bin_file = f"{file_path[:-4]}.data"
        data.tofile(bin_file)
        
        tensor_info["ioflag"] = "output"
        if "input" in bin_file.split("_")[-1]:
            tensor_info["ioflag"] = "input"

        tensor_info["bin_file"] = file_path

        verify_tensor_info, verify_tshape = _verify_res.get_verify_res(tensor_info)
        dump_tshape = tensor_info.get("shape")
        tensor_info["verify_tensor_file"] = verify_tensor_info
        if os.path.exists(verify_tensor_info) and len(verify_tshape) == len(dump_tshape):
            verify_tensor_data = np.fromfile(verify_tensor_info, dtype)
            verify_tensor_data = verify_tensor_data.reshape(verify_tshape)
            data = data.reshape(dump_tshape)
            # dump tensor可能存在无效数据，只对吧有效部分
            slices = []
            for dim in range(data.ndim):
                stop = min(verify_tshape[dim], dump_tshape[dim])
                slices.append(slice(0, stop))
            tensor_info["cmp_res"] = np.allclose(data[tuple(slices)], verify_tensor_data[tuple(slices)])
        else:
            tensor_info["cmp_res"] = "NO_CMP"

        return tensor_info


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser dump_tensor.")
    parser.add_argument("--dump_tensor_path", type=str, default="output/dump_tensor/device_0", 
                        help="directory like output/dump_tensor/device_x")
    parser.add_argument("--verify_path", type=str, default="", help="Path to verify_result.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()
    # 初始化紧凑解析器
    parser = CompactDumpTensorInfoParser()
    logging.info(f"单个结构字节数：{parser.struct_compact_size}")

    _verify_res.read_verify_result(args.verify_path)

    tensor_infos = []
    for dir_path, _, file_names in os.walk(args.dump_tensor_path):
        for file_name in file_names:
            if not file_name.endswith(".tdump"):
                continue
            bin_file = os.path.join(dir_path, file_name)
            tensor_infos.append(parser.parse_file(bin_file))
    df = pd.DataFrame(tensor_infos)
    df["rootHash"] = "'" + df["rootHash"].astype(str)
    df["funcHash"] = "'" + df["funcHash"].astype(str)
    logging.info(df)

    df.to_csv(os.path.join(args.dump_tensor_path, "tensor_info.csv"), index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
