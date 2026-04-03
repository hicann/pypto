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
 * \file distributed_error.h
 * \brief 通信相关的错误码
 */

#pragma once
#include <cstdint>

namespace npu::tile_fwk {

enum class DistributedErrorCode : uint32_t {
    // FA0xxx: 参数错误
    INVALID_GROUP_NAME = 0xA0000,
    INVALID_WORLD_SIZE = 0xA0001,
    INVALID_TENSOR_DIM = 0xA0002,
    INVALID_TENSOR_SHAPE = 0xA0003,
    INVALID_TENSOR_DTYPE = 0xA0004,
    INVALID_TENSOR_FORMAT = 0xA0005,
    INVALID_OPERAND_NUM = 0xA0006,
    INVALID_SHMEM_TENSOR = 0xA0007,
    INVALID_SHMEM_VIEW_PARAM = 0xA0008,
    INVALID_OP_TYPE = 0xA0009,

    // FA1xxx: 配置错误
    INVALID_TILE_DIM = 0xA1000,
    INVALID_TILE_SHAPE = 0xA1001,
    INVALID_ALIGNMENT = 0xA1002,

    // FA2xxx: 运行时错误
    WIN_SIZE_EXCEED_LIMIT = 0xA2000,
    TILE_NUM_EXCEED_LIMIT = 0xA2001,
    DIVISION_BY_ZERO = 0xA2002,

    // FA3xxx: machine 相关的错误
    AICPU_TASK_TIMEOUT = 0xA3000,
    AICPU_TASK_NUM_EXCEED_LIMIT = 0xA3001,
    AICPU_TASK_QUEUE_EMPTY = 0xA3002,
    AICPU_TASKID_NOT_IN_MAP = 0xA3003,
    INVALID_GROUP_INDEX = 0xA3004,
    NULLPTR = 0xA3005,
};
} // namespace npu::tile_fwk
