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
 * \file connection_matrix.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "connection_matrix_impl.h"
namespace npu::tile_fwk {
using ConnectionMatrixImplPtr = std::shared_ptr<ConnectionMatrixImpl>;

class ConnectionMatrix {
public:
    explicit ConnectionMatrix(Function *func);
    ~ConnectionMatrix() = default;

    bool IsConnected(const Operation &a, const Operation &b) const;

    bool IsConnected(uint64_t indexA, uint64_t indexB) const;

    void SetConnectivity(const std::unordered_set<Operation *> &producers, Operation &op);

    void Generate(Function *func);
    
    uint64_t GetIndex(const Operation &op) const;

    const LargeBitmap &GetBitMap(const Operation &op) const;

    const LargeBitmap &GetBitMap(uint64_t index) const;
    
    static constexpr uint64_t INVALID_INDEX = std::numeric_limits<uint64_t>::max();

private:
    ConnectionMatrixImplPtr impl_{nullptr};
};
}
