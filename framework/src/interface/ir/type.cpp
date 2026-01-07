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

namespace pto {

// ========== Type System Implementation ==========

uint64_t Type::GetDataTypeSize(DataType dataType) {
    switch (dataType) {
    case DataType::INT4:
    case DataType::HF4:
        return 1;  // 4 bits still need 1 byte
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOL:
    case DataType::FP8:
    case DataType::HF8:
        return 1;
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FP16:
    case DataType::BF16:
        return 2;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FP32:
        return 4;
    case DataType::INT64:
    case DataType::UINT64:
    case DataType::FP64:
        return 8;
    case DataType::BOTTOM:
    case DataType::UNKNOWN:
    default:
        return 0;
    }
}

void ScalarType::Print(std::ostream& os) const {
    os << DataTypeToString(dataType_);
}

void TensorType::Print(std::ostream& os) const {
    os << DataTypeToString(dataType_);
}

uint64_t TileType::GetTypeSize() const {
    uint64_t elementSize = GetDataTypeSize();
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
    os << "], " << DataTypeToString(dataType_) << ">";
}

} // namespace pto

