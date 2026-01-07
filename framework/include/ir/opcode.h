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
 * \file opcode.h
 * \brief
 */

#pragma once

#include "utils_defop.h"

namespace pto {

enum class Opcode : int64_t {
    OP_INVALID,

#define DEFOP DEFOP_OPCODE

#include "operation.def"
    OP_END_COMMON,

#include "tile_graph.def"
    OP_END_TILE_GRAPH,

#undef DEFOP

    OP_COMMON_BEGIN = OP_INVALID + 1,
    OP_COMMON_END = OP_END_COMMON,
    OP_TILE_GRAPH_BEGIN = OP_END_COMMON + 1,
    OP_TILE_GRAPH_END = OP_END_TILE_GRAPH - 1,
};

std::string GetOpcodeName(Opcode opcode);

} // namespace pto
