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
 * \file type.cpp
 * \brief
 */

#include "ir/type.h"
#include "ir/utils.h"

#include <ostream>
#include <cstring>

namespace pto {

// ========== Type System Implementation ==========
static struct DTypeInfo dtypeInfo[] = {
    {      DataType::INT4,       "int4",  4, 1, false,  true, false},
    {      DataType::INT8,       "int8",  8, 1, false,  true, false},
    {     DataType::INT16,      "int16", 16, 2, false,  true, false},
    {     DataType::INT32,      "int32", 32, 4, false,  true, false},
    {     DataType::INT64,      "int64", 64, 8, false,  true, false},
    {DataType::FP8_E4M3FN, "fp8_e4m3fn",  8, 1,  true, false, false},
    {      DataType::FP16,    "float16", 16, 2,  true, false, false},
    {      DataType::FP32,    "float32", 32, 4,  true, false, false},
    {      DataType::BF16,   "bfloat16", 16, 2,  true, false, false},
    {       DataType::HF4,        "hf4",  4, 1,  true, false, false},
    {       DataType::HF8,        "hf8",  8, 1,  true, false, false},
    {     DataType::UINT8,      "uint8",  8, 1, false, false,  true},
    {    DataType::UINT16,     "uint16", 16, 2, false, false,  true},
    {    DataType::UINT32,     "uint32", 32, 4, false, false,  true},
    {    DataType::UINT64,     "uint64", 64, 8, false, false,  true},
    {      DataType::BOOL,       "bool",  8, 1, false,  true, false},
    {      DataType::FP64,    "float64", 64, 8,  true, false, false},
    {  DataType::FP8_E5M2,   "fp8_e5m2",  8, 1,  true, false, false},
    {   DataType::UNKNOWN,    "unknown",  0, 0, false, false, false},
};

DTypeInfo &DTypeInfoOf(DataType dtype) {
    return dtypeInfo[static_cast<int>(dtype)];
}

DTypeInfo &DTypeInfoOf(const char *name) {
    for (auto &spec : dtypeInfo) {
        if (strcmp(spec.name, name) == 0) {
            return spec;
        }
    }
    return DTypeInfoOf(DataType::UNKNOWN);
}

uint64_t Type::GetDataTypeSize(DataType dataType) {
    return DTypeInfoOf(dataType).bytes;
}

void ScalarType::Print(std::ostream& os) const {
    os << DTypeInfoOf(dataType_).name;
}

void TensorType::Print(std::ostream& os) const {
    os << DTypeInfoOf(dataType_).name;
}

uint64_t TileType::GetTypeSize() const {
    uint64_t elementSize = GetDataTypeSize(dataType_);
    uint64_t totalElements = 1;
    for (size_t dim : shape_) {
        totalElements *= dim;
    }
    return elementSize * totalElements;
}

void TileType::Print(std::ostream& os) const {
    os << "tile<[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i];
        if (i + 1 < shape_.size()) {
            os << ", ";
        }
    }
    os << "], " << DTypeInfoOf(dataType_).name << ">";
}

} // namespace pto

