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
#include "ir/type.h"

#include <ostream>

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
} // namespace pto

