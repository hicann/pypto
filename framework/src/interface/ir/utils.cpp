/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file utils.cpp
 * \brief
 */

#include "ir/utils.h"
#include "ir/value.h"

#include <ostream>
#include <unordered_map>

namespace pto {

// Initialize static member
std::map<ObjectType, int> IDGen::counters_;

int IDGen::NextID(ObjectType type) {
    return ++counters_[type];
}

void IDGen::Reset(ObjectType type) {
    counters_[type] = 0;
}

void IDGen::ResetAll() {
    counters_.clear();
}

void PrintIndent(std::ostream& os, int indent) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
}

static std::unordered_map<DataType, std::string> dataTypeNameDict = {
    {DataType::BOOL, "bool"},
    {DataType::INT4, "int4"},
    {DataType::INT8, "int8"},
    {DataType::INT16, "int16"},
    {DataType::INT32, "int32"},
    {DataType::INT64, "int64"},
    {DataType::UINT8, "uint8"},
    {DataType::UINT16, "uint16"},
    {DataType::UINT32, "uint32"},
    {DataType::UINT64, "uint64"},
    {DataType::FP8, "fp8"},
    {DataType::FP16, "fp16"},
    {DataType::BF16, "bf16"},
    {DataType::FP32, "fp32"},
    {DataType::FP64, "fp64"},
    {DataType::HF4, "hf4"},
    {DataType::HF8, "hf8"},
    {DataType::BOTTOM, "bottom"},
    {DataType::UNKNOWN, "unknown"}
};

std::string DataTypeToString(DataType type) {
    if (dataTypeNameDict.count(type)) {
        return dataTypeNameDict[type];
    } else {
        return "unknown";
    }
}

DataType StringToValueType(const std::string& name) {
    if (name == "bool") return DataType::BOOL;
    if (name == "int4") return DataType::INT4;
    if (name == "int8") return DataType::INT8;
    if (name == "i8") return DataType::INT8;
    if (name == "int16") return DataType::INT16;
    if (name == "i16") return DataType::INT16;
    if (name == "int32") return DataType::INT32;
    if (name == "i32") return DataType::INT32;
    if (name == "int64") return DataType::INT64;
    if (name == "i64") return DataType::INT64;
    if (name == "uint8") return DataType::UINT8;
    if (name == "u8") return DataType::UINT8;
    if (name == "uint16") return DataType::UINT16;
    if (name == "u16") return DataType::UINT16;
    if (name == "uint32") return DataType::UINT32;
    if (name == "u32") return DataType::UINT32;
    if (name == "uint64") return DataType::UINT64;
    if (name == "u64") return DataType::UINT64;
    if (name == "fp8") return DataType::FP8;
    if (name == "fp16") return DataType::FP16;
    if (name == "f16") return DataType::FP16;
    if (name == "bf16") return DataType::BF16;
    if (name == "fp32") return DataType::FP32;
    if (name == "f32") return DataType::FP32;
    if (name == "fp64") return DataType::FP64;
    if (name == "f64") return DataType::FP64;
    if (name == "hf4") return DataType::HF4;
    if (name == "hf8") return DataType::HF8;
    if (name == "bottom") return DataType::BOTTOM;
    if (name == "unknown") return DataType::UNKNOWN;
    // Default to INT32 if unknown
    return DataType::UNKNOWN;
}

} // namespace pto

