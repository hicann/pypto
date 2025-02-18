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
 * \file auto_cast.h
 * \brief
 */

#ifndef PASS_AUTO_CAST_H_
#define PASS_AUTO_CAST_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
class AutoCast : public Pass {
public:
    AutoCast() : Pass("AutoCast") {}
    ~AutoCast() override = default;
    Status RunOnFunction(Function &function) override;
    bool SupportBF16(Operation *op);
    Status InsertCast(Function &function);
    bool IsLegalCast(DataType ds, DataType dt);
    std::vector<Operation *> GetCastChain(Operation *tailOp);
    Status ShortenChain(Function &function, const std::vector<Operation *> &castChain, Operation *tailOp);
    Status RemoveRedundantCastChain(Function &function);
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    void InsertCastOp(Function &function, LogicalTensorPtr src, LogicalTensorPtr tgt, const TileShape &tileShape);
    Status GetInOutConnectedTensor(Function &function);
    std::unordered_set<int> inCastConnectedTensors_;
    std::unordered_set<int> outCastConnectedTensors_;
    std::unordered_set<Operation *> addedCast_;
};
}
}
#endif // PASS_AUTO_CAST_H_