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

namespace npu::tile_fwk {

/*
    RemoveRedundantOp: 如果op的类型为VIEW且op的输入tensor和输出tensor相同，则认为该VIEW op为冗余op，将其删除，
    并改变图中的连接关系.
*/
class RemoveRedundantOp : public Pass {
public:
    RemoveRedundantOp() : Pass("RemoveRedundantOp") {}
    ~RemoveRedundantOp() override = default;
private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status NeedToDelete(const Operation &op, Function &function, bool &needToDelete) const;
    Status RemoveDummyExpand(Function &function) const;
    Status DeleteRedundantOps(Function &function) const;
    Status RemoveViewAssemble(Function &function) const;
    void ProcessPerfectMatch (Function &function,LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const;
    bool IsNotSameViewInput (LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const;
    bool IsDataReplace (LogicalTensorPtr &endTensor) const;
    void GenerateNewView(Function &function,Operation &op,LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const;
    void EraseRedundantAssemble(Function &function) const;
};
} // namespace npu::tile_fwk
#endif  // REMOVE_REDUNDANT_OP_H