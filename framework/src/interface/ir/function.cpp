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
 * \file function.cpp
 * \brief
 */

#include "ir/function.h"

namespace pto {

Function::Function(std::string name, FunctionKind kind, FunctionSignature signature)
    : Object(ObjectType::Function, std::move(name)),
      kind_(kind),
      signature_(std::move(signature)) {
    inputCompound_ = std::make_shared<CompoundStatement>();
    // Make the function body scope a child of the input scope.
    compound_ = std::make_shared<CompoundStatement>(inputCompound_);

    // Register function arguments into the dedicated input scope so that
    // Function::scope_ can see them via GetAncestorValues().
    for (const auto& arg : signature_.arguments) {
        if (arg) {
            // Use SSA name as the key in environment table
            inputCompound_->SetEnvVar(arg->GetName(), arg);
        }
    }
}

void Function::AddStatement(StatementPtr stmt) {
    compound_->AddStatement(std::move(stmt));
}

static const char* toString(FunctionKind kind) {
    switch (kind) {
    case FunctionKind::ControlFlow:
        return "control_flow";
    case FunctionKind::DataFlow:
        return "data_flow";
    case FunctionKind::Block:
        return "block";
    default:
        return "unknown";
    }
    return "unknown";
}

static void PrintAttributes(std::ostream& os, const AttributeMap& attrs, const std::string& prefix) {
    for (const auto& kv : attrs) {
        os << prefix << kv.first << " = " << kv.second << "\n";
    }
}

void Function::Print(std::ostream& os, int indent) const {
    // Print indentation for the function header.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "func.func " << GetPrefixedName() << "(";
    for (size_t i = 0; i < signature_.arguments.size(); ++i) {
        const auto& arg = signature_.arguments[i];
        if (arg) {
            os << arg->GetSSAName() << ": ";
            arg->Print(os, 0);
        }
        if (arg->Attributes().find("io") != arg->Attributes().end()) {
            os << " #" << arg->Attributes().at("io");
        }
        if (i + 1 < signature_.arguments.size()) {
            os << ", ";
        }
    }
    os << ")";

    if (!signature_.results.empty()) {
        os << " -> (";
        for (size_t i = 0; i < signature_.results.size(); ++i) {
            if (signature_.results[i]) {
                signature_.results[i]->GetType()->Print(os);
            }
            if (i + 1 < signature_.results.size()) {
                os << ", ";
            }
        }
        os << ")";
    }

    os << " {\n";

    // Print structured statement body if present.
    for (size_t i = 0; i < compound_->GetStatementsNum(); ++i) {
        auto stmt = compound_->GetStatement(i);
        if (stmt) {
            stmt->Print(os, indent + 1);
        }
    }

    // Closing brace.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "}\n";

    // Function attributes can be printed as separate lines following MLIR style.
    std::string attrPrefix;
    for (int i = 0; i < indent; ++i) {
        attrPrefix += "  ";
    }
    attrPrefix += "func.attr ";
    PrintAttributes(os, attributes_, attrPrefix);

    // Optionally print a comment giving the function kind.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "// func.kind = " << toString(kind_) << "\n";
}

std::ostream& operator<<(std::ostream& os, const Function& func) {
    func.Print(os, 0);
    return os;
}

uint64_t Function::ComputeHash() {
    // wait for IRVisitor to implement
    std::stringstream ss;
    ss << GetName() << "_" << static_cast<int>(GetKind());
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

bool Function::isFromInCast(const ValuePtr &value) const {
    for (auto arg : signature_.arguments) {
        if (arg == value && arg->Attributes().at("io") == "in") {
            return true;
        }
    }
    return false;
}

bool Function::isFromOutCast(const ValuePtr &value) const {
    for (auto result : signature_.arguments) {
        if (result == value && result->Attributes().at("io") == "out") {
            return true;
        }
    }
    return false;
}

int Function::GetIncastIndex(const ValuePtr &value) const {
    for (size_t i = 0; i < signature_.arguments.size(); i++) {
        if (signature_.arguments[i] == value && signature_.arguments[i]->Attributes().at("io") == "in") {
            return i;
        }
    }
    return -1;
}

int Function::GetOutcastIndex(const ValuePtr &value) const {
    for (size_t i = 0; i < signature_.arguments.size(); i++) {
        if (signature_.arguments[i] == value && signature_.arguments[i]->Attributes().at("io") == "out") {
            return i;
        }
    }
    return -1;
}

} // namespace pto


