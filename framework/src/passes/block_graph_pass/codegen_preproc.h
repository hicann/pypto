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
 * \file codegen_preproc.h
 * \brief
 */

#ifndef PASS_CODEGEN_PREPROC_H
#define PASS_CODEGEN_PREPROC_H

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu {
namespace tile_fwk {
class CodegenPreproc : public Pass {
public:
    CodegenPreproc() : Pass("CodegenPreproc") {}
    ~CodegenPreproc() override = default;
    Status RunOnFunction(Function &function) override;

private:
    /*
     * update oOperand of alloc node for codegen, gm tensor param layout:
     * [gm1 addr|gm1 offset0|gm1 offset1|...|gm1 tile shape0|gm1 tile shape1|...|gm1 raw shape0|gm1 raw shape1|...] ...
     * [gm1 addr|gm2 offset0|gm2 offset1|...|gm2 tile shape0|gm2 tile shape1|...|gm2 raw shape0|gm2 raw shape1|...] ...
     * so addr is always the first param of this gm tensor
     */
    Status SaveGmTensorParamIdxToOp(Function &func) const;
    // force combine axis
    Status ForceCombineAxis(Function &func) const;
    Status ForceCombineAxisForAxisCombine(Function &func) const;
    bool IsNeedSave(const Operation &op) const;
    void CombineTailAxis(std::vector<int64_t> &shape, size_t shapeSize) const;
    void CombineLastAxis(std::vector<SymbolicScalar> &shape, size_t shapeSize) const;
    Status ProcessAxis(Operation &op, std::vector<bool> attr, bool isInput) const;
    void SetNeedAllocAttr(Function &function);
    std::string DumpOpList(Function &function);
};

} // namespace tile_fwk
} // namespace npu
#endif // PASS_CODEGEN_PREPROC_H