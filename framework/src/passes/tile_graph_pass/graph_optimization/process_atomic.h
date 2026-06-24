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

const std::string RMW_MODE_ATTR_ADD = OP_ATTR_PREFIX + "atomic_add";
const std::string RMW_MODE_ATTR_MIN = OP_ATTR_PREFIX + "atomic_min";
const std::string RMW_MODE_ATTR_MAX = OP_ATTR_PREFIX + "atomic_max";

class ProcessAtomic : public Pass {
public:
    ProcessAtomic() : Pass("ProcessAtomic") {}
    ~ProcessAtomic() override = default;

    Status PreCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    Status EliminateReduceAcc(Function& function);
    Status EliminateAtomicRMW(Function& function);

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
    Status ProcessSingleAtomicRMW(Operation& op);
};
} // namespace npu::tile_fwk
#endif // PROCESS_ATOMIC_H