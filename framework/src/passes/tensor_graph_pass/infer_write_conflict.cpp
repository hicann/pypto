/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "infer_write_conflict.h"

#include "interface/operation/opcode.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

bool InferWriteConflict::MayOverlap(const Operation* op0, const Operation* op1)
{
    auto attr0 = std::static_pointer_cast<AssembleOpAttribute>(op0->GetOpAttribute());
    auto attr1 = std::static_pointer_cast<AssembleOpAttribute>(op1->GetOpAttribute());
    ASSERT(attr0 != nullptr && attr1 != nullptr) << "missing assemble attribute";

    auto getOffset = [](auto attr) {
        auto dynoffset = attr->GetToDynOffset();
        if (dynoffset.empty()) {
            dynoffset = SymbolicScalar::FromConcrete(attr->GetToOffset());
        }
        return dynoffset;
    };

    auto check = [](SymbolicScalar cond) {
        cond = cond.Simplify();
        return cond.ConcreteValid() && cond.Concrete() == true;
    };

    auto offset0 = getOffset(attr0);
    auto offset1 = getOffset(attr1);
    auto& shape0 = op0->GetIOperands()[0]->GetShape();
    auto& shape1 = op1->GetIOperands()[0]->GetShape();
    ASSERT(shape0.size() == shape1.size()) << "shape0 and shape1 must have the same size";

    for (size_t i = 0; i < shape0.size(); i++) {
        if (check(offset0[i] + shape0[i] <= offset1[i]) || check(offset1[i] + shape1[i] <= offset0[i])) {
            return false;
        }
    }
    return true;
}

bool InferWriteConflict::MayOverlap(const std::vector<Operation*>& prods)
{
    for (size_t i = 0; i < prods.size(); i++) {
        for (size_t j = i + 1; j < prods.size(); j++) {
            if (MayOverlap(prods[i], prods[j])) {
                return true;
            }
        }
    }
    return false;
}

Status InferWriteConflict::RunOnFunction(Function& function)
{
    for (auto& outcast : function.GetOutcast()) {
        auto& prodSet = outcast->GetProducers();
        bool hasConflict = std::any_of(prodSet.begin(), prodSet.end(), [](auto& prod) {
            return prod->GetOpcode() == Opcode::OP_ATOMIC_RMW;
        });
        if (hasConflict) {
            outcast->SetAttr(OpAttributeKey::writeConflict, true);
            continue;
        }

        std::vector<Operation*> prods;
        for (auto& prod : prodSet) {
            if (prod->GetOpcode() == Opcode::OP_ASSEMBLE || prod->GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
                prods.push_back(prod);
            }
        }
        if (MayOverlap(prods)) {
            outcast->SetAttr(OpAttributeKey::writeConflict, true);
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
