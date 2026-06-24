/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class InferWriteConflict : public Pass {
public:
    InferWriteConflict() : Pass("InferWriteConflict") {}
    ~InferWriteConflict() override = default;

private:
    bool MayOverlap(const Operation* op0, const Operation* op1);
    bool MayOverlap(const std::vector<Operation*>& prods);
    Status RunOnFunction(Function& function) override;
};
} // namespace npu::tile_fwk
