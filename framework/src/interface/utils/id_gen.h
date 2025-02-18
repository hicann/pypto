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
 * \file id_gen.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk {
enum class IdType {
    RAW_TENSOR,
    FUNCTION,
    LOGICAL_TENSOR,
    TENSOR_INDEX,
    CG_USING_NAME, // gen using name for codegen
    CG_VAR_NAME, // gen variable for codegen
};

template <IdType T>
class IdGen {
public:
    static auto &Inst() {
        static IdGen<T> inst;
        return inst;
    }

    auto NewId() { return id_++; }

    auto CurId() const { return id_; }

    void Reset() { id_ = 0; }

    void SetId(int id) { id_ = id;}

private:
    IdGen() = default;

    int id_{0};
};
} // namespace npu::tile_fwk