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
 * \file operation.cpp
 * \brief
 */

#include "ir/operation.h"

namespace pto {

Operation::Operation()
    : Object(ObjectType::Operation),
      opcode_(Opcode::OP_INVALID) {}

Operation::Operation(Opcode opcode)
    : Object(ObjectType::Operation),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode, std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode,
                     ValuePtrs inputs,
                     ValuePtrs outputs,
                     std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      ioperands_(std::move(inputs)),
      ooperands_(std::move(outputs)),
      opcode_(opcode) {
    for (size_t i = 0; i < inputs.size(); i++) {
        if (std::dynamic_pointer_cast<ScalarValue>(inputs[i])) {
            iScalarIndex_ = i;
            break;
        }
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        if (std::dynamic_pointer_cast<ScalarValue>(outputs[i])) {
            oScalarIndex_ = i;
            break;
        }
    }
}


void Operation::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);

    // ===== results =====
    if (GetNumOutputOperand()) {
        for (size_t i = 0; i < GetNumOutputOperand(); ++i) {
            os << GetOutputOperand(i)->GetSSAName();
            if (i + 1 < GetNumOutputOperand())
                os << ", ";
        }
        os << " = ";
    }

    // ===== op name =====
    // Determine dialect prefix based on output type
    // - Tensor outputs -> tensor. prefix (Python frontend operations)
    // - Tile outputs -> tile. prefix (compiler optimization operations)
    // - Scalar outputs -> tensor. prefix (element-wise operations)
    std::string opcodeName = GetOpcodeName(GetOpcode());
    if (GetNumOutputOperand()) {
        const auto& firstOutput = GetOutputOperand(0);
        if (firstOutput->GetValueKind() == ValueKind::Tile) {
            os << "tile." << opcodeName << "";
        } else {
            // Tensor or Scalar -> use tensor. prefix
            os << "tensor." << opcodeName << "";
        }
    } else {
        // No outputs (shouldn't happen, but fallback to tensor.)
        os << "tensor." << opcodeName << "";
    }

    // ===== operands =====
    os << " ";
    for (size_t i = 0; i < GetNumInputOperand(); ++i) {
        os << GetInputOperand(i)->GetSSAName();
        if (i + 1 < GetNumInputOperand())
            os << ", ";
    }

    // ===== type signature =====
    os << " : (";
    for (size_t i = 0; i < GetNumInputOperand(); ++i) {
        GetInputOperand(i)->Print(os, 0);
        if (i + 1 < GetNumInputOperand())
            os << ", ";
    }
    os << ")";

    os << " -> ";
    if (GetNumOutputOperand() == 1) {
        GetOutputOperand(0)->Print(os, 0);
    } else {
        os << "(";
        for (size_t i = 0; i < GetNumOutputOperand(); ++i) {
            GetOutputOperand(i)->Print(os, 0);
            if (i + 1 < GetNumOutputOperand())
                os << ", ";
        }
        os << ")";
    }

    os << "\n";
}

} // namespace pto
