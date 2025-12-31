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
 * \file remove_redundant_op.h
 * \brief
 */

#ifndef REMOVE_REDUNDANT_OP_H
#define REMOVE_REDUNDANT_OP_H
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
const std::unordered_set<Opcode> matchOpcodeWithDynshape = {Opcode::OP_VIEW, Opcode::OP_EXPAND};
const std::unordered_set<Opcode> matchOpcodeWithoutDynshape = {Opcode::OP_REGISTER_COPY, Opcode::OP_ASSEMBLE};
class RemoveRedundantOp : public Pass {
public:
    RemoveRedundantOp() : Pass("RemoveRedundantOp") {}
    ~RemoveRedundantOp() override = default;
private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status ProcessRedundantOpWithDynShape(Operation &op, Function &function) const;
    Status ProcessRedundantOpWithoutDynShape(Operation &op, Function &function) const;
    Status RemoveDummyOp(Function &function) const;
    Status RemoveViewAssemble(Function &function) const;
    void ProcessPerfectMatch(Function &function, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const;
    void RemoveViewAssembleForOutcast(Function &function, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const;
    void GenerateNewView(Function &function, Operation &op, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const;
    bool IsNotSameViewInput(LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const;
    bool IsDataReplace(LogicalTensorPtr &endTensor) const;
};
} // namespace tile_fwk
} // namespace npu
#endif  // REMOVE_REDUNDANT_OP_H