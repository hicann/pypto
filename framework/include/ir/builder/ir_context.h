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
 * \file ir_context.h
 * \brief
 */

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "ir/function.h"
#include "ir/statement.h"

namespace pto {

// A single stack frame for restoring insertion state.
struct ScopeFrame {
    std::shared_ptr<CompoundStatement> compound{nullptr};
    std::shared_ptr<Function> func{nullptr};
    std::shared_ptr<OpStatement> activeOpStmt{nullptr};
};

// Explicit, stack-based builder context.
// All mutable insertion state lives here.
struct IRBuilderContext {
    std::shared_ptr<Function> func{nullptr};
    std::shared_ptr<CompoundStatement> compound{nullptr};
    std::shared_ptr<OpStatement> activeOpStmt{nullptr};

    std::vector<ScopeFrame> scopeStack;

    void ResetInsertionPoint() { activeOpStmt.reset(); }

    void PushScope(std::shared_ptr<CompoundStatement> newCompound,
                   std::shared_ptr<Function> newFunc = nullptr) {
        scopeStack.push_back(ScopeFrame{compound, func, activeOpStmt});

        compound = std::move(newCompound);
        if (newFunc) {
            func = std::move(newFunc);
        }
        activeOpStmt.reset();  // always reset insertion point when entering a new scope
    }

    void PopScope() {
        ScopeFrame frame = scopeStack.back();
        scopeStack.pop_back();

        compound = std::move(frame.compound);
        func = std::move(frame.func);
        activeOpStmt = std::move(frame.activeOpStmt);
    }
};

} // namespace pto
