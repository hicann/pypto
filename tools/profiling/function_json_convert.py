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
import json
from enum import Enum


def get_sematic(func_index, opmagic, func_data):
    if func_index >= len(func_data):
        return ""
    for call_op in func_data[func_index]["operations"]:
        if call_op["opmagic"] == opmagic:
            if "semantic_label" in call_op:
                return call_op["semantic_label"]["label"]
            else:
                return ""
    return ""


def get_hash_order(func_index, func_data):
    """Get hashOrder info from program.json by func_index.

    Traverse all operations in the function to find valid hashOrder values.
    Some ops (e.g., added by l1_reuse/nbuffer pass) have all hashOrder=-1,
    so we search for any op with valid values in the same function.

    Args:
        func_index: Index of the function in func_data (should be leaf function for hashOrder)
        func_data: List of functions from program.json

    Returns:
        tuple: (l1ReuseHashOrder, cubeMergeHashOrder, vecMergeHashOrder)
    """
    if func_index >= len(func_data):
        return -1, -1, -1

    func = func_data[func_index]

    for op in func["operations"]:
        l1_reuse_hash_order = op.get("l1ReuseHashOrder", -1)
        cube_merge_hash_order = op.get("cubeMergeHashOrder", -1)
        vec_merge_hash_order = op.get("vecMergeHashOrder", -1)
        if l1_reuse_hash_order != -1 or cube_merge_hash_order != -1 or vec_merge_hash_order != -1:
            return l1_reuse_hash_order, cube_merge_hash_order, vec_merge_hash_order

    return -1, -1, -1


class DataType(Enum):
    INT4 = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    FP8 = 5
    FP16 = 6
    FP32 = 7
    BF16 = 8
    HF4 = 9
    HF8 = 10
    UINT8 = 11
    UINT16 = 12
    UINT32 = 13
    UINT64 = 14
    BOOL = 15
    DOUBLE = 16
    FP8E5M2 = 17
    FP8E4M3 = 18
    FP8E8M0 = 19
    BOTTOM = 20


class MemType(Enum):
    UB = 0
    L1 = 1
    L0A = 2
    L0B = 3
    L0C = 4
    FIX = 5
    FIX_QUANT_PRE = 6
    FIX_RELU_PRE = 7
    FIX_RELU_POST = 8
    FIX_QUANT_POST = 9
    FIX_ELT_ANTIQ = 10
    FIX_MTE2_ANTIQ = 11
    BT = 12
    L2 = 13
    L3 = 14
    DEVICE_DDR = 15
    HOST1 = 16
    FAR1 = 17
    FAR2 = 18
    WORKSPACE = 19
    VECTOR_REG = 20


data_type_dict = {member.value: member.name for member in DataType}
mem_type_dict = {member.value: member.name for member in MemType}


def get_data_type_str(data_type):
    if data_type in data_type_dict.keys():
        return data_type_dict[data_type]
    return "DataType" + str(data_type)


def get_mem_type_str(mem_type):
    if mem_type in mem_type_dict.keys():
        return mem_type_dict[mem_type]
    return "MemType" + str(mem_type)


def convert_rawtensor_to_str(rawtensor):
    res = ""
    res += "raw@" + str(rawtensor["rawmagic"])
    res += " " + get_data_type_str(rawtensor["datatype"])
    res += " " + str(rawtensor["rawshape"])
    if "symbol" in rawtensor:
        res += " " + rawtensor["symbol"]
    return res


def convert_operand_to_str(operand):
    res = ""
    res += "tensor@" + str(operand["magic"])
    res += " " + get_mem_type_str(operand["mem_type"]["tobe"])
    res += " " + str(operand["shape"])
    res += " " + str(operand["offset"])
    res += " " + convert_rawtensor_to_str(operand["rawtensor"])
    return res


def convert_operands_to_str(operands):
    res = []
    for operand in operands:
        res.append(convert_operand_to_str(operand))
    return ", ".join(set(res))


def get_in_out_operand_str(is_inoperand, func_index, opmagic, func_data):
    if func_index >= len(func_data):
        return ""
    for call_op in func_data[func_index]["operations"]:
        if call_op["opmagic"] == opmagic:
            if is_inoperand:
                return convert_operands_data(call_op['ioperands'])
            else:
                return convert_operands_data(call_op['ooperands'])
    return ""


def get_in_out_operands_data(is_inoperand, func_index, opmagic, func_data):
    if func_index >= len(func_data):
        return []
    for call_op in func_data[func_index]["operations"]:
        if call_op["opmagic"] == opmagic:
            if is_inoperand:
                return convert_operands_data(call_op['ioperands'])
            else:
                return convert_operands_data(call_op['ooperands'])
    return []


def convert_operand_data(operand):
    res = dict()
    tensor_info = dict()
    tensor_info['shape'] = operand['shape']
    tensor_info['dtype'] = operand['rawtensor']['datatype']
    tensor_info['rawmagic'] = operand['rawtensor']['rawmagic']
    res[operand['magic']] = tensor_info
    return res


def convert_operands_data(operands):
    res = []
    for operand in operands:
        res.append(convert_operand_data(operand))
    return res


def get_tensors(func_hash, func_hash_data):
    tensors_dict = dict()
    if func_hash not in func_hash_data:
        return tensors_dict
    tensors = func_hash_data[func_hash]['tensors']
    for tensor in tensors:
        tensors_dict[tensor['magic']] = tensor
    return tensors_dict


def get_rawtensors(func_hash, func_hash_data):
    rawtensors_dict = dict()
    if func_hash not in func_hash_data:
        return rawtensors_dict
    rawtensors = func_hash_data[func_hash]['rawtensors']
    for rawtensor in rawtensors:
        rawtensors_dict[rawtensor['rawmagic']] = rawtensor
    return rawtensors_dict
