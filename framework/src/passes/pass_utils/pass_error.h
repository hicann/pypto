/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pass_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// 前端传入Tensor错误
enum class TensorErr : uint32_t {
    TENSOR_NULL_POINTER         = 40000U,
    TENSOR_INVALID_MEMORY_TYPE  = 40001U,
    TENSOR_SUBGRAPH_BOUNDARY    = 40002U,
    TENSOR_SHAPE_MISMATCH       = 40003U,
    TENSOR_UNSUPPORTED_DATATYPE = 40004U,
    TENSOR_MEMORY_ALLOCATION    = 40005U,
    TENSOR_DYNAMIC_ATTR         = 40006U,
    TENSOR_MEMORY_CORRUPTION    = 40007U
};

// 前端传入Operation错误
enum class OperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 41000U,
    OP_NULL_POINTER          = 41001U,
    OP_INVALID_OPCODE        = 41002U,
    OP_PRODUCER_CONSUMER     = 41003U,
    OP_SPECIAL_CONSTRAINT    = 41004U,
    OP_NESTING_DEPTH         = 41005U,
    OP_SEQUENCE_ERROR        = 41006U
};

// 前端传入Function错误
enum class FunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE       = 42000U,
    FUNCTION_BOUNDARY_COMPLETENESS = 42001U,
    FUNCTION_GRAPH_CONNECTION      = 42002U,
    FUNCTION_EXPAND_FEATURE        = 42003U,
    FUNCTION_MEMORY_REACHABILITY   = 42004U,
    FUNCTION_UNIQUENESS            = 42005U,
    FUNCTION_SPECIAL_STRUCTURE     = 42006U
};

// 前端传入Graph错误
enum class GraphErr : uint32_t {
    GRAPH_LOOP_DETECTION       = 43000U,
    GRAPH_TOPOLOGY_STRUCTURE   = 43001U,
    GRAPH_SUBGRAPH_EMPTY       = 43002U,
    GRAPH_SUBGRAPH_ID_INVALID  = 43003U,
    GRAPH_EDGE_CONSISTENCY     = 43004U,
    GRAPH_COLOR_CONSISTENCY    = 43005U,
    GRAPH_READY_STATE          = 43006U,
    GRAPH_AIV_AIC_MIX          = 43007U
};

// 前端传入Config错误
enum class ConfigErr : uint32_t {
    CONFIG_MEMORY_TYPE_REACHABLE  = 44000U,
    CONFIG_SUBGRAPH_BOUNDARY      = 44001U,
    CONFIG_TENSOR_MEMORY_TYPE     = 44002U
};

// 前端传入Manager错误
enum class ManagerErr : uint32_t {
};

}  // namespace npu::tile_fwk

