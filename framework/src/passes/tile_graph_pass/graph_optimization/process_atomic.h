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
 * \file process_atomic.h
 * \brief Process atomic operations including ReduceAcc and AtomicRMW
 */

#ifndef PROCESS_ATOMIC_H
#define PROCESS_ATOMIC_H

#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

class ProcessAtomic : public Pass {
public:
    ProcessAtomic() : Pass("ProcessAtomic") {}
    ~ProcessAtomic() override = default;

    Status PreCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    Status EliminateReduceAcc(Function& function);
    Status EliminateAtomicRMW(Function& function);
    Status EliminateVecDupBranch(Function& function, bool& hasReduceAccCascade);

private:
    Status CheckAtomicRMWUnsupportedMode(Function& function);
    std::string GetRmwAttrKey(AtomicRMWMode mode);
    Status CheckAndSetRmwAttr(Operation& producerOp, AtomicRMWMode rmwMode, const std::string& rmwAttrKey);
    void AccumulateAssembleOffset(
        std::shared_ptr<AssembleOpAttribute> producerAttr, const std::vector<int64_t>& rmwOffset,
        const std::vector<SymbolicScalar>& rmwDynOffset);
    Status ProcessAssembleProducer(
        Operation& producerOp, std::shared_ptr<LogicalTensor> rmwOut, AtomicRMWMode rmwMode,
        const std::vector<int64_t>& rmwOffset, const std::vector<SymbolicScalar>& rmwDynOffset);
    bool HasAssembleProducer(const std::shared_ptr<LogicalTensor>& input) const;
    bool HasConsumerExcept(const std::shared_ptr<LogicalTensor>& input, const Operation& op) const;
    Status PrepareAtomicRMWSharedInputs(Function& function, const std::vector<Operation*>& atomicRmwOps) const;
    std::shared_ptr<LogicalTensor> PrepareExclusiveAtomicInput(
        Function& function, Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input) const;
    Status ProcessSingleAtomicRMW(Operation& op);
    void CollectReduceAccUpstream(Operation& op, std::set<int>& visited,
        std::vector<Operation*>& result) const;
    Status TraceBackAndRemoveVecDup(Function& function, Operation& op, std::set<int>& visited, bool& anyRemoved);
    Status RemoveVecDupBranchFromCubeOp(Operation& cubeOp, bool& anyRemoved);
    bool IsVecDupAssembleInput(const Operation& assembleOp) const;
};
} // namespace npu::tile_fwk
#endif // PROCESS_ATOMIC_H
