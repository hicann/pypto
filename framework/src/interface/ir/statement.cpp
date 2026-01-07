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
 * \file statement.cpp
 * \brief
 */

#include "ir/statement.h"
#include "ir/utils.h"

#include <ostream>
#include <sstream>

namespace pto {

ValuePtr CompoundStatement::FindValue(const std::string& name) const {
    // Use GetEnvVar which already searches through the scope chain
    return GetEnvVar(name);
}

void CompoundStatement::RemoveValue(ValuePtr val) {
    for (auto it = envTable_.begin(); it != envTable_.end();) {
        if (it->second == val) {
            it = envTable_.erase(it);
            return;
        } else {
            ++it;
        }
    }
}

std::unordered_map<std::string, ValuePtr> CompoundStatement::GetAncestorValues() const {
    std::unordered_map<std::string, ValuePtr> ancestor_values;

    // Traverse all ancestor scopes (parent, grandparent, etc.)
    auto currentParent = parent_.lock();
    while (currentParent) {
        // Collect all values from current ancestor scope's environment table
        const auto& parent_env = currentParent->GetEnvTable();
        for (const auto& pair : parent_env) {
            if (pair.second) {
                // If a variable with the same name exists in multiple ancestor scopes,
                // the closer one (more recent ancestor) takes precedence
                if (ancestor_values.find(pair.first) == ancestor_values.end()) {
                    ancestor_values[pair.first] = pair.second;
                }
            }
        }

        // Move to next ancestor
        currentParent = currentParent->GetParent().lock();
    }

    return ancestor_values;
}

void CompoundStatement::SetEnvVar(const std::string& name, ValuePtr value) {
    if (!value) {
        throw std::runtime_error("Scope::SetEnvVar: value is null");
    }
    envTable_[name] = value;
}

ValuePtr CompoundStatement::GetEnvVar(const std::string& name) const {
    // First, search in current scope
    auto it = envTable_.find(name);
    if (it != envTable_.end()) {
        return it->second;
    }

    // If not found, search in parent scope (recursively)
    if (auto parentPtr = parent_.lock()) {
        return parentPtr->GetEnvVar(name);
    }

    // Not found in any scope
    return nullptr;
}

void OpStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "statement.op {\n";

    // Print linear operations.
    for (const auto& op : operations_) {
        if (op) {
            op->Print(os, indent + 2);
        }
    }

    PrintIndent(os, indent);
    os << "}\n";
}

void ForStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);

    // Print result variables if results_ is non-empty
    if (!results_.empty()) {
        for (size_t i = 0; i < results_.size(); ++i) {
            if (results_[i]) {
                os << results_[i]->GetSSAName();
            }
            if (i + 1 < results_.size()) {
                os << ", ";
            }
        }
        os << " = ";
    } 
    
    // Print loop header: statement.for %iv = %lb to %ub step %step
    os << "statement.for ";
    if (iterationVar_) {
        iterationVar_->Print(os, 0);
    } else {
        os << "<null>";
    }
    os << " = ";
    if (range_ && range_->GetStart()) {
        range_->GetStart()->Print(os, 0);
    } else {
        os << "<null>";
    }
    os << " to ";
    if (range_ && range_->GetEnd()) {
        range_->GetEnd()->Print(os, 0);
    } else {
        os << "<null>";
    }
    os << " step ";
    if (range_ && range_->GetStep()) {
        range_->GetStep()->Print(os, 0);
    } else {
        os << "<null>";
    }

    // Print iter_args if present.
    if (!iterArgs_.empty()) {
        os << "\n";
        PrintIndent(os, indent + 2);
        os << "iter_args(";
        for (size_t i = 0; i < iterArgs_.size(); ++i) {
            const auto& arg = iterArgs_[i];
            // Use value's SSA name as iter_arg identifier
            if (arg.value) {
                os << arg.value->GetSSAName() << " = ";
            } else {
                os << "<no-value> = ";
            }
            if (arg.initValue) {
                os << arg.initValue->GetSSAName() << " : ";
                // Print type information
                std::ostringstream typeStream;
                arg.initValue->Print(typeStream, 0);
                std::string typeStr = typeStream.str();
                // Remove leading indentation if any
                size_t firstNonSpace = typeStr.find_first_not_of(" \t");
                if (firstNonSpace != std::string::npos) {
                    typeStr = typeStr.substr(firstNonSpace);
                }
                os << typeStr;
            } else {
                os << "<null> : <unknown>";
            }
            if (i + 1 < iterArgs_.size()) {
                os << ", ";
            }
        }
        os << ")";
    }

    // Print attributes if present.
    if (!Attributes().empty()) {
        os << "\n";
        PrintIndent(os, indent + 2);
        os << "attributes {";
        bool first = true;
        for (const auto& kv : Attributes()) {
            if (!first) {
                os << ", ";
            }
            os << kv.first << " = " << kv.second;
            first = false;
        }
        os << "}";
    }

    os << " {\n";

    // Print loop body.
    for (size_t i = 0; i < compound_->GetStatementsNum(); ++i) {
        auto stmt = compound_->GetStatement(i);
        if (stmt) {
            stmt->Print(os, indent + 2);
        }
    }

    PrintIndent(os, indent);
    os << "}\n";
}

std::shared_ptr<YieldStatement> ForStatement::Yield() {
    if (compound_->GetStatementsNum() > 0) {
        return std::dynamic_pointer_cast<YieldStatement>(compound_->GetStatement(compound_->GetStatementsNum() - 1));
    }
    return nullptr;
}

const std::shared_ptr<YieldStatement> ForStatement::Yield() const {
    if (compound_->GetStatementsNum() > 0) {
        return std::dynamic_pointer_cast<YieldStatement>(compound_->GetStatement(compound_->GetStatementsNum() - 1));
    }
    return nullptr;
}

void ForStatement::BuildResult() {
    results_.clear();

    // Find terminal yield in loop body.
    auto loopYield = Yield();
    if (!loopYield) {
        return;
    }

    const auto& yieldVals = loopYield->Values();

    // If no iterArgs, no results to build.
    if (iterArgs_.empty() || yieldVals.empty()) {
        return;
    }

    // Require the same number of yielded values as iterArgs.
    if (yieldVals.size() != iterArgs_.size()) {
        return;
    }

    // Build result values, ensuring matching Data types with iterArgs.
    for (size_t i = 0; i < iterArgs_.size(); ++i) {
        auto yieldVal = yieldVals[i];
        auto initVal = iterArgs_[i].initValue;

        if (!yieldVal || !initVal) {
            results_.clear();
            return;
        }

        // Check type compatibility.
        if (yieldVal->GetValueKind() != initVal->GetValueKind() ||
            yieldVal->GetDataType() != initVal->GetDataType()) {
            results_.clear();
            return;
        }

        // For tiles, create a new tile with the same shape/element type.
        auto yieldTile = std::dynamic_pointer_cast<TileValue>(yieldVal);
        auto initTile = std::dynamic_pointer_cast<TileValue>(initVal);
        if (yieldTile && initTile) {
            auto res = std::make_shared<TileValue>(yieldTile->GetShape(),
                                               yieldTile->GetDataType(), yieldTile->GetName());
            results_.push_back(res);
        } else {
            // For tensors, create a new tensor with the same shape/element type.
            auto yieldTensor = std::dynamic_pointer_cast<TensorValue>(yieldVal);
            auto initTensor = std::dynamic_pointer_cast<TensorValue>(initVal);
            if (yieldTensor && initTensor) {
                auto res = std::make_shared<TensorValue>(yieldTensor->GetShape(), yieldTensor->GetDataType(),
                                                    yieldTensor->GetName(), yieldTensor->GetFormat());
                results_.push_back(res);
            } else {
                // For other types (e.g., Scalar), create a new scalar with the same type.
                auto yieldScalar = std::dynamic_pointer_cast<ScalarValue>(yieldVal);
                auto initScalar = std::dynamic_pointer_cast<ScalarValue>(initVal);
                if (yieldScalar && initScalar) {
                    auto res = std::make_shared<ScalarValue>(yieldScalar->GetDataType(),
                                                         yieldScalar->GetName(),
                                                         yieldScalar->GetScalarValueKind());
                    results_.push_back(res);
                } else {
                    // Fallback: reuse the yield value.
                    results_.push_back(yieldVal);
                }
            }
        }
    }
}

void IfStatement::BuildResult() {
    results_.clear();

    // Find terminal yields in then/else scopes.
    const YieldStatement* thenYield = nullptr;
    const YieldStatement* elseYield = nullptr;

    if (thenCompound_->GetStatementsNum() > 0) {
        auto lastThenStmt = thenCompound_->GetStatement(thenCompound_->GetStatementsNum() - 1);
        thenYield = dynamic_cast<const YieldStatement*>(lastThenStmt.get());
    }
    if (elseCompound_->GetStatementsNum() > 0) {
        auto lastElseStmt = elseCompound_->GetStatement(elseCompound_->GetStatementsNum() - 1);
        elseYield = dynamic_cast<const YieldStatement*>(lastElseStmt.get());
    }

    if (!thenYield || !elseYield) {
        return;
    }

    const auto& thenVals = thenYield->Values();
    const auto& elseVals = elseYield->Values();

    // Require the same number of yielded values.
    if (thenVals.size() != elseVals.size() || thenVals.empty()) {
        return;
    }

    // Build result tensors, ensuring matching Data types.
    for (size_t i = 0; i < thenVals.size(); ++i) {
        auto t = thenVals[i];
        auto e = elseVals[i];
        if (!t || !e) {
            results_.clear();
            return;
        }
        if (t->GetValueKind() != e->GetValueKind() ||
            t->GetDataType() != e->GetDataType()) {
            results_.clear();
            return;
        }

        // For tiles, create a new tiles with the same shape/element type/layout.
        auto tTile = std::dynamic_pointer_cast<TileValue>(t);
        auto eTile = std::dynamic_pointer_cast<TileValue>(e);
        if (tTile && eTile) {
            auto res = std::make_shared<TileValue>(tTile->GetShape(),
                                            eTile->GetDataType(), tTile->GetName());
            results_.push_back(res);
        } else {
            // For tensors, create a new tensor with the same shape/element type.
            auto tTensor = std::dynamic_pointer_cast<TensorValue>(t);
            auto eTensor = std::dynamic_pointer_cast<TensorValue>(e);
            if (tTensor && eTensor) {
                auto res = std::make_shared<TensorValue>(tTensor->GetShape(), eTensor->GetDataType(),
                                                    tTensor->GetName(), tTensor->GetFormat());
                results_.push_back(res);
            } else {
                // For other types (e.g., Scalar), just reuse the then-branch value.
                results_.push_back(t);
            }
        }
    }
}

void IfStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    // If results_ is non-empty, print them as the result variables:
    //   %r0, %r1 = statement.if ...
    // Otherwise, treat this as a pure statement-level conditional.
    if (!results_.empty()) {
        for (size_t i = 0; i < results_.size(); ++i) {
            if (results_[i]) {
                os << results_[i]->GetSSAName();
            }
            if (i + 1 < results_.size()) {
                os << ", ";
            }
        }
        os << " = ";
    }

    os << "statement.if ";
    condition_->Print(os, 0);
    os << " {\n";

    for (size_t i = 0; i < thenCompound_->GetStatementsNum(); ++i) {
        auto stmt = thenCompound_->GetStatement(i);
        if (stmt) {
            stmt->Print(os, indent + 2);
        }
    }

    PrintIndent(os, indent);
    os << "} else {\n";

    for (size_t i = 0; i < elseCompound_->GetStatementsNum(); ++i) {
        auto stmt = elseCompound_->GetStatement(i);
        if (stmt) {
            stmt->Print(os, indent + 2);
        }
    }

    PrintIndent(os, indent);
    os << "}\n";
}

void YieldStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "statement.yield";
    if (!values_.empty()) {
        os << " ";
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i]) {
                os << values_[i]->GetSSAName();
            }
            if (i + 1 < values_.size()) {
                os << ", ";
            }
        }
    }
    os << "\n";
}

void ReturnStatement::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "statement.return";
    if (!values_.empty()) {
        os << " ";
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i]) {
                os << values_[i]->GetSSAName();
            }
            if (i + 1 < values_.size()) {
                os << ", ";
            }
        }
    }
    os << "\n";
}

} // namespace pto


