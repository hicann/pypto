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
 * \file infer_memory_conflict.h
 * \brief
 */

#ifndef PASS_INFER_DISCONTINUOUS_INPUT_H_
#define PASS_INFER_DISCONTINUOUS_INPUT_H_

#include <vector>
#include <unordered_map>

#include "passes/pass_interface/pass.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class InferDiscontinuousInput : public Pass {
public:
    InferDiscontinuousInput() : Pass("InferDiscontinuousInput") {}
    ~InferDiscontinuousInput() override = default;

private:
    Status RunOnFunction(Function &function) override;
    Status InferFromIncast();
    void InsertViewOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    void InsertAssembleOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    void InsertCopyOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Status InsertTensorCopy(Function &function);
    void Init(Function& function);
    std::vector<std::pair<LogicalTensorPtr, Operation *>> FilterCopyScenes(
        const std::vector<std::pair<LogicalTensorPtr, Operation *>> &);
    std::map<LogicalTensorPtr, std::vector<std::pair<LogicalTensorPtr, Operation *>>> insertCopys_;
    std::map<Operation *, size_t> opInputDegree_;
    std::map<LogicalTensorPtr, size_t> tensorProducers_;
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_INFER_DISCONTINUOUS_INPUT_H_