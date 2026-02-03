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
 * \file loopaxes_proc.h
 * \brief
 */

#ifndef PASS_LOOPAXES_PROC_H
#define PASS_LOOPAXES_PROC_H

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"

namespace npu {
namespace tile_fwk {
class LoopaxesProc : public Pass {
public:
    LoopaxesProc() : Pass("LoopaxesProc") {
        SetSupportedArches({NPUArch::DAV_3510});
    }
    ~LoopaxesProc() override = default;
    Status RunOnFunction(Function &function) override;
private:
    Status UpdateFuncLoopAxes(Function &function);
    Status UpdateOpLoopAxes(Operation &op);
    bool SameLoopAxes(const std::vector<SymbolicScalar> &curLoopAxes);
    void ClearStatus();

    int64_t groupIdx{INVALID_LOOP_GROUPID};
    int64_t lastGroupIdx{INVALID_LOOP_GROUPID};
    int64_t previousOutputMagic{INVALID_LOOP_GROUPID};
    std::shared_ptr<Operation> lastOpInLoop{nullptr};
    std::vector<SymbolicScalar> previousLoopAxes;
};

} // namespace tile_fwk
} // namespace npu
#endif // PASS_LOOPAXES_PROC_H