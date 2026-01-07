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
 * \file ir_builder_core.cpp
 * \brief
 */

#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>

namespace pto {

IRBuilder::IRBuilder(std::shared_ptr<ProgramModule> module) : module_(module) {}

void IRBuilder::SetModule(std::shared_ptr<ProgramModule> m) {
    module_ = std::move(m);
}

std::shared_ptr<Function> IRBuilder::CreateFunction(
    std::string name,
    FunctionKind kind,
    FunctionSignature sig,
    bool setAsEntry) {

    if (!module_) {
        throw std::runtime_error("IRBuilder::CreateFunction: module is null");
    }

    auto fn = std::make_shared<Function>(std::move(name), kind, std::move(sig));
    module_->AddFunction(fn);

    if (setAsEntry) {
        module_->SetProgramEntry(fn);
    }

    return fn;
}

OpStatementPtr IRBuilder::GetOrCreateActiveOpStmt(IRBuilderContext& ctx) {
    if (!ctx.compound) {
        throw std::runtime_error("IRBuilder::GetOrCreateActiveOpStmt: ctx.compound is null");
    }
    if (ctx.activeOpStmt) {
        return ctx.activeOpStmt;
    }

    auto opStmt = std::make_shared<OpStatement>();
    ctx.compound->AddStatement(opStmt);
    ctx.activeOpStmt = opStmt;
    return opStmt;
}

} // namespace pto
