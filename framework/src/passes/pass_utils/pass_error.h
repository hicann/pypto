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
    TENSOR_NULL_POINTER = 0x40000U,
    TENSOR_INVALID_MEMORY_TYPE = 0x40001U,
    TENSOR_SUBGRAPH_BOUNDARY = 0x40002U,
    TENSOR_SHAPE_MISMATCH = 0x40003U,
    TENSOR_UNSUPPORTED_DATATYPE = 0x40004U,
    TENSOR_MEMORY_ALLOCATION = 0x40005U,
    TENSOR_DYNAMIC_ATTR = 0x40006U,
    TENSOR_MEMORY_CORRUPTION = 0x40007U
};

// 前端传入Operation错误
enum class OperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 0x41000U,
    OP_NULL_POINTER = 0x41001U,
    OP_INVALID_OPCODE = 0x41002U,
    OP_PRODUCER_CONSUMER = 0x41003U,
    OP_SPECIAL_CONSTRAINT = 0x41004U,
    OP_NESTING_DEPTH = 0x41005U,
    OP_SEQUENCE_ERROR = 0x41006U
};

// 前端传入Function错误
enum class FunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE = 0x42000U,
    FUNCTION_BOUNDARY_COMPLETENESS = 0x42001U,
    FUNCTION_GRAPH_CONNECTION = 0x42002U,
    FUNCTION_EXPAND_FEATURE = 0x42003U,
    FUNCTION_MEMORY_REACHABILITY = 0x42004U,
    FUNCTION_UNIQUENESS = 0x42005U,
    FUNCTION_SPECIAL_STRUCTURE = 0x42006U
};

// 前端传入Graph错误
enum class GraphErr : uint32_t {
    GRAPH_LOOP_DETECTION = 0x43000U,
    GRAPH_TOPOLOGY_STRUCTURE = 0x43001U,
    GRAPH_SUBGRAPH_EMPTY = 0x43002U,
    GRAPH_SUBGRAPH_ID_INVALID = 0x43003U,
    GRAPH_EDGE_CONSISTENCY = 0x43004U,
    GRAPH_COLOR_CONSISTENCY = 0x43005U,
    GRAPH_READY_STATE = 0x43006U,
    GRAPH_AIV_AIC_MIX = 0x43007U
};

// 前端传入Config错误
enum class ConfigErr : uint32_t {
    CONFIG_MEMORY_TYPE_REACHABLE = 0x44000U,
    CONFIG_SUBGRAPH_BOUNDARY = 0x44001U,
    CONFIG_TENSOR_MEMORY_TYPE = 0x44002U
};

// 前端传入Manager错误
enum class ManagerErr : uint32_t {};

} // namespace npu::tile_fwk
