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
 * \file ir_builder_stmt.cpp
 * \brief
 */

#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <functional>

namespace pto {

OpStatementPtr IRBuilder::CreateOpStmt(IRBuilderContext& ctx) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::CreateOpStmt: ctx.compound is null");

    auto opStmt = std::make_shared<OpStatement>();
    ctx.compound->AddStatement(opStmt);
    ctx.activeOpStmt = opStmt;
    return opStmt;
}

ForStatementPtr IRBuilder::CreateForStmt(IRBuilderContext& ctx,
                                        ScalarValuePtr iv,
                                        ScalarValuePtr start,
                                        ScalarValuePtr end,
                                        ScalarValuePtr step) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::CreateForStmt: ctx.compound is null");

    auto st = std::make_shared<ForStatement>(std::move(iv), std::move(start), std::move(end), std::move(step));
    ctx.compound->AddStatement(st);
    return st;
}

IfStatementPtr IRBuilder::CreateIfStmt(IRBuilderContext& ctx, ScalarValuePtr cond) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::CreateIfStmt: ctx.compound is null");

    auto st = std::make_shared<IfStatement>(std::move(cond));
    ctx.compound->AddStatement(st);
    return st;
}

YieldStatementPtr IRBuilder::CreateYield(IRBuilderContext& ctx, ValuePtrs values) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::CreateYield: ctx.compound is null");

    auto st = std::make_shared<YieldStatement>();
    st->Values() = std::move(values);

    ctx.compound->AddStatement(st);
    return st;
}

ReturnStatementPtr IRBuilder::CreateReturn(IRBuilderContext& ctx, ValuePtrs values) {
    if (!ctx.compound) throw std::runtime_error("IRBuilder::CreateReturn: ctx.compound is null");

    auto st = std::make_shared<ReturnStatement>();
    st->Values() = std::move(values);

    ctx.compound->AddStatement(st);
    return st;
}

// ===== Enter nested scopes (explicit push only) =====

void IRBuilder::EnterFunctionBody(IRBuilderContext& ctx, std::shared_ptr<Function> func) {
    if (!func) throw std::runtime_error("IRBuilder::EnterFunctionBody: func is null");

    CompoundStatementPtr compound = func->GetCompound();
    if (!compound) throw std::runtime_error("IRBuilder::EnterFunctionBody: func compound is null");

    // Push function scope; caller will PopScope.
    ctx.PushScope(compound, func);
}

void IRBuilder::EnterForBody(IRBuilderContext& ctx, ForStatementPtr st) {
    if (!ctx.func) throw std::runtime_error("IRBuilder::EnterForBody: ctx.func is null");
    if (!ctx.compound) throw std::runtime_error("IRBuilder::EnterForBody: ctx.compound is null");
    if (!st) throw std::runtime_error("IRBuilder::EnterForBody: st is null");

    CompoundStatementPtr new_compound = st->GetCompound();
    new_compound->SetParent(ctx.compound);

    ctx.PushScope(new_compound);
}

void IRBuilder::EnterIfThen(IRBuilderContext& ctx, IfStatementPtr st) {
    if (!ctx.func) throw std::runtime_error("IRBuilder::EnterIfThen: ctx.func is null");
    if (!ctx.compound) throw std::runtime_error("IRBuilder::EnterIfThen: ctx.compound is null");
    if (!st) throw std::runtime_error("IRBuilder::EnterIfThen: st is null");

    CompoundStatementPtr new_compound = st->GetThenCompound();
    new_compound->SetParent(ctx.compound);

    ctx.PushScope(new_compound);
}

void IRBuilder::EnterIfElse(IRBuilderContext& ctx, IfStatementPtr st) {
    if (!ctx.func) throw std::runtime_error("IRBuilder::EnterIfElse: ctx.func is null");
    if (!ctx.compound) throw std::runtime_error("IRBuilder::EnterIfElse: ctx.compound is null");
    if (!st) throw std::runtime_error("IRBuilder::EnterIfElse: st is null");

    CompoundStatementPtr new_compound = st->GetElseCompound();
    new_compound->SetParent(ctx.compound);

    ctx.PushScope(new_compound);
}

// ===== Exit helpers (do not pop) =====

void IRBuilder::ExitIfStatement(IRBuilderContext& ctx, IfStatementPtr st) {
    (void)ctx;
    if (!st) throw std::runtime_error("IRBuilder::ExitIfStatement: st is null");

    auto parentCompound = st->GetThenCompound()->GetParent().lock();
    if (!parentCompound) {
        throw std::runtime_error("IRBuilder::ExitIfStatement: then compound has no parent");
    }

    std::unordered_map<std::string, ValuePtr> envBeforeIf = st->GetThenCompound()->GetAncestorValues();
    std::unordered_map<std::string, ValuePtr> envAfterThen = st->GetThenCompound()->GetEnvTable();
    std::unordered_map<std::string, ValuePtr> envAfterElse = st->GetElseCompound()->GetEnvTable();

    bool hasElseBranch = st->GetElseCompound()->GetStatementsNum() > 0 || !envAfterElse.empty();

    std::vector<std::string> modifiedVars;

    if (hasElseBranch) {
        std::unordered_set<std::string> candidateVars;
        for (const auto& [varName, _] : envAfterThen) candidateVars.insert(varName);
        for (const auto& [varName, _] : envAfterElse) candidateVars.insert(varName);
        for (const auto& [varName, _] : envBeforeIf) candidateVars.insert(varName);

        for (const std::string& varName : candidateVars) {
            auto itAfterThen = envAfterThen.find(varName);
            auto itAfterElse = envAfterElse.find(varName);

            bool existsInThen = (itAfterThen != envAfterThen.end());
            bool existsInElse = (itAfterElse != envAfterElse.end());

            ValuePtr valueBeforeIf = parentCompound->GetEnvVar(varName);
            bool existsBefore = (valueBeforeIf != nullptr);

            if (existsBefore) {
                bool modifiedInThen = existsInThen && (itAfterThen->second != valueBeforeIf);
                bool modifiedInElse = existsInElse && (itAfterElse->second != valueBeforeIf);
                if (modifiedInThen || modifiedInElse) {
                    modifiedVars.push_back(varName);
                }
            } else {
                if (existsInThen && existsInElse) {
                    modifiedVars.push_back(varName);
                }
            }
        }
    } else {
        for (const auto& [varName, valueAfterThen] : envAfterThen) {
            ValuePtr valueBeforeIf = parentCompound->GetEnvVar(varName);
            if (valueBeforeIf && valueBeforeIf != valueAfterThen) {
                modifiedVars.push_back(varName);
            }
        }
    }

    if (modifiedVars.empty()) return;

    ValuePtrs thenValues;
    ValuePtrs elseValues;

    for (const std::string& varName : modifiedVars) {
        ValuePtr originalValue = parentCompound->GetEnvVar(varName);

        ValuePtr thenValue = nullptr;
        if (auto itThen = envAfterThen.find(varName); itThen != envAfterThen.end()) thenValue = itThen->second;
        else thenValue = originalValue;

        ValuePtr elseValue = nullptr;
        if (hasElseBranch) {
            if (auto itElse = envAfterElse.find(varName); itElse != envAfterElse.end()) elseValue = itElse->second;
            else elseValue = originalValue;
        } else {
            elseValue = originalValue;
        }

        if (thenValue && elseValue) {
            thenValues.push_back(thenValue);
            elseValues.push_back(elseValue);
        }
    }

    auto addOrUpdateYield = [](CompoundStatementPtr compound, const ValuePtrs& values) {
        bool hasYield = false;
        if (compound->GetStatementsNum() > 0) {
            auto lastStmt = compound->GetStatement(compound->GetStatementsNum() - 1);
            hasYield = dynamic_cast<YieldStatement*>(lastStmt.get()) != nullptr;
        }
        if (!hasYield) {
            auto yield = std::make_shared<YieldStatement>();
            yield->Values() = values;
            compound->AddStatement(yield);
        } else {
            auto lastStmt = compound->GetStatement(compound->GetStatementsNum() - 1);
            auto yield = dynamic_cast<YieldStatement*>(lastStmt.get());
            if (yield) yield->Values() = values;
        }
    };

    addOrUpdateYield(st->GetThenCompound(), thenValues);
    addOrUpdateYield(st->GetElseCompound(), elseValues);

    st->BuildResult();

    const auto& results = st->Results();
    for (size_t i = 0; i < modifiedVars.size() && i < results.size(); ++i) {
        const std::string& varName = modifiedVars[i];
        if (results[i]) parentCompound->SetEnvVar(varName, results[i]);
    }
}

void IRBuilder::ExitForStatement(IRBuilderContext& ctx, ForStatementPtr st) {
    if (!st) throw std::runtime_error("IRBuilder::ExitForStatement: st is null");

    auto parentCompound = st->GetCompound()->GetParent().lock();
    if (!parentCompound) {
        throw std::runtime_error("IRBuilder::ExitForStatement: loop scope has no parent");
    }

    std::unordered_map<std::string, ValuePtr> envBeforeFor = st->GetCompound()->GetAncestorValues();
    std::unordered_map<std::string, ValuePtr> envAfterFor  = st->GetCompound()->GetEnvTable();

    std::vector<std::string> loopCarriedVars;
    for (const auto& [varName, valueAfterFor] : envAfterFor) {
        ValuePtr valueBeforeFor = parentCompound->GetEnvVar(varName);
        if (valueBeforeFor && valueBeforeFor != valueAfterFor) {
            loopCarriedVars.push_back(varName);
        }
    }

    if (loopCarriedVars.empty()) return;

    for (const std::string& varName : loopCarriedVars) {
        auto itBefore = envBeforeFor.find(varName);
        if (itBefore != envBeforeFor.end() && itBefore->second) {
            st->AddIterArg(itBefore->second);
        }
    }

    // Create value for each iter_arg and replace initValue usage in loop body
    auto loopCompound = st->GetCompound();

    // Temporarily switch insertion scope to loop scope using ctx stack.
    ctx.PushScope(loopCompound);

    auto createIterArgValue = [this, &ctx](ValuePtr initValue) -> ValuePtr {
        if (!initValue) return nullptr;

        ValueKind kind = initValue->GetValueKind();
        DataType dt = initValue->GetDataType();

        if (kind == ValueKind::Tensor) {
            auto tensor = std::dynamic_pointer_cast<TensorValue>(initValue);
            if (tensor) return CreateTensor(ctx, tensor->GetShape(), dt, tensor->GetName());
        } else if (kind == ValueKind::Tile) {
            auto tile = std::dynamic_pointer_cast<TileValue>(initValue);
            if (tile) return CreateTile(ctx, tile->GetShape(), dt, tile->GetName());
        } else if (kind == ValueKind::Scalar) {
            return CreateScalar(ctx, dt, initValue->GetName());
        }
        return nullptr;
    };

    std::unordered_map<ValuePtr, ValuePtr> initValueToValue;
    for (auto& iterArg : st->IterArgs()) {
        if (iterArg.initValue) {
            ValuePtr newValue = createIterArgValue(iterArg.initValue);
            if (newValue) {
                iterArg.value = newValue;
                initValueToValue[iterArg.initValue] = newValue;
            }
        }
    }

    // Restore insertion scope.
    ctx.PopScope();

    // Replace initValue with iter_arg value in loop body operations (keep your original logic)
    auto replaceValueInOperations = [&initValueToValue](OpStatement& opStmt) {
        for (auto& op : opStmt.Operations()) {
            if (!op) continue;
            for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
                auto input = op->GetInputOperand(k);
                auto it = initValueToValue.find(input);
                if (it != initValueToValue.end()) {
                    input = it->second;
                }
            }
        }
    };

    std::function<void(StatementPtr)> replaceValueInStatement = [&](StatementPtr stmt) {
        if (!stmt) return;

        if (auto opStmt = std::dynamic_pointer_cast<OpStatement>(stmt)) {
            replaceValueInOperations(*opStmt);
        } else if (auto yield = std::dynamic_pointer_cast<YieldStatement>(stmt)) {
            auto& yieldValues = yield->Values();
            for (auto& val : yieldValues) {
                auto it = initValueToValue.find(val);
                if (it != initValueToValue.end()) {
                    val = it->second;
                }
            }
        } else if (auto ifStmt = std::dynamic_pointer_cast<IfStatement>(stmt)) {
            for (size_t i = 0; i < ifStmt->ThenBranchStmtsNum(); ++i) {
                auto thenStmt = ifStmt->GetThenBranchStatement(i);
                replaceValueInStatement(thenStmt);
            }
            for (size_t i = 0; i < ifStmt->ElseBranchStmtsNum(); ++i) {
                auto elseStmt = ifStmt->GetElseBranchStatement(i);
                replaceValueInStatement(elseStmt);
            }
        } else if (auto forStmt = std::dynamic_pointer_cast<ForStatement>(stmt)) {
            for (size_t i = 0; i < forStmt->BodyStmtsNum(); ++i) {
                auto nestedStmt = forStmt->GetBodyStatement(i);
                replaceValueInStatement(nestedStmt);
            }
        }
    };

    for (size_t i = 0; i < st->GetCompound()->GetStatementsNum(); ++i) {
        auto stmt = st->GetCompound()->GetStatement(i);
        replaceValueInStatement(stmt);
    }

    ValuePtrs yieldValues;
    for (const std::string& varName : loopCarriedVars) {
        auto itAfter = envAfterFor.find(varName);
        if (itAfter != envAfterFor.end()) {
            yieldValues.push_back(itAfter->second);
        }
    }

    for (size_t i = 0; i < loopCarriedVars.size() && i < st->IterArgs().size(); ++i) {
        const std::string& varName = loopCarriedVars[i];
        auto& iterArg = st->IterArgs()[i];
        if (iterArg.value) {
            loopCompound->SetEnvVar(varName, iterArg.value);
        }
    }

    auto addOrUpdateYield = [](CompoundStatementPtr compound, const ValuePtrs& values) {
        bool hasYield = false;
        if (compound->GetStatementsNum() > 0) {
            auto lastStmt = compound->GetStatement(compound->GetStatementsNum() - 1);
            hasYield = dynamic_cast<YieldStatement*>(lastStmt.get()) != nullptr;
        }
        if (!hasYield) {
            auto yield = std::make_shared<YieldStatement>();
            yield->Values() = values;
            compound->AddStatement(yield);
        } else {
            auto lastStmt = compound->GetStatement(compound->GetStatementsNum() - 1);
            auto yield = dynamic_cast<YieldStatement*>(lastStmt.get());
            if (yield) yield->Values() = values;
        }
    };

    addOrUpdateYield(st->GetCompound(), yieldValues);

    st->BuildResult();

    const auto& results = st->Results();
    for (size_t i = 0; i < loopCarriedVars.size() && i < results.size(); ++i) {
        const std::string& varName = loopCarriedVars[i];
        if (results[i]) {
            parentCompound->SetEnvVar(varName, results[i]);
        }
    }
}

} // namespace pto
