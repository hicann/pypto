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
 * \file opcode.cpp
 * \brief
 */

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "ir/opcode.h"
#include "ir/utils_defop.h"

namespace pto {

static std::unordered_map<Opcode, std::string> opcodeNameDict = {

#define DEFOP DEFOP_OPCODE_DICT
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

};

std::string GetOpcodeName(Opcode opcode) {
    if (opcodeNameDict.count(opcode)) {
        return opcodeNameDict[opcode];
    } else {
        return "";
    }
}

} // namespace pto
