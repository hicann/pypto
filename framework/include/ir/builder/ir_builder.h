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
 * \file ir_builder.h
 * \brief
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "ir/utils_defop.h"
#include "ir/builder/ir_context.h"

namespace pto {

class IRBuilder {
public:
    IRBuilder() = default;
    explicit IRBuilder(std::shared_ptr<ProgramModule> module);

    // ===== Module =====
    void SetModule(std::shared_ptr<ProgramModule> m);
    std::shared_ptr<ProgramModule> GetModule() const { return module_; }

    // ===== Function (stateless: no current func stored in builder) =====
    std::shared_ptr<Function> CreateFunction(
        std::string name,
        FunctionKind kind,
        FunctionSignature sig,
        bool setAsEntry = false);

    // ===== Insertion point =====
    OpStatementPtr GetOrCreateActiveOpStmt(IRBuilderContext& ctx);

    // ===== Scope registration =====
    ValuePtr AddToCompound(IRBuilderContext& ctx, ValuePtr v);

    // Optional convenience: create values (explicit creation only)
    std::shared_ptr<TensorValue> CreateTensor(IRBuilderContext& ctx,
        const std::vector<ScalarValuePtr>& shape, DataType dt, std::string name = "");
    std::shared_ptr<TileValue> CreateTile(IRBuilderContext& ctx,
        const std::vector<uint64_t>& shape, DataType dt, std::string name = "");
    std::shared_ptr<ScalarValue> CreateScalar(IRBuilderContext& ctx,
        DataType dt, std::string name = "");
    std::shared_ptr<ScalarValue> CreateConst(IRBuilderContext& ctx,
        int64_t v, std::string name = "");
    std::shared_ptr<ScalarValue> CreateConst(IRBuilderContext& ctx,
        double v, std::string name = "");

    // ===== Emit op (used by schema build) =====
    OperationPtr Emit(IRBuilderContext& ctx, OperationPtr op);

    // ===== The ONLY op-building entry =====
    // NOTE: your operation.def/tile_graph.def macros must call Emit(ctx, ...)
    // If your current DEFOP_IRBUILDER assumes Emit(op) without ctx, you need to
    // update that macro to include IRBuilderContext& ctx as the first argument.

#define DEFOP DEFOP_IRBUILDER
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

    // ===== Statement building =====
    OpStatementPtr CreateOpStmt(IRBuilderContext& ctx);

    ForStatementPtr CreateForStmt(IRBuilderContext& ctx,
        ScalarValuePtr iv,
        ScalarValuePtr start,
        ScalarValuePtr end,
        ScalarValuePtr step);

    IfStatementPtr CreateIfStmt(IRBuilderContext& ctx, ScalarValuePtr cond);

    YieldStatementPtr CreateYield(IRBuilderContext& ctx, ValuePtrs values);

    ReturnStatementPtr CreateReturn(IRBuilderContext& ctx, ValuePtrs values);

    // ===== Enter nested scopes (explicit, stack-based) =====
    // These helpers only push; caller decides when to pop.
    void EnterFunctionBody(IRBuilderContext& ctx, std::shared_ptr<Function> func);
    void EnterForBody(IRBuilderContext& ctx, ForStatementPtr st);
    void EnterIfThen(IRBuilderContext& ctx, IfStatementPtr st);
    void EnterIfElse(IRBuilderContext& ctx, IfStatementPtr st);

    // ===== Exit helpers =====
    // These do not pop scopes; they only finalize merge/yield/results and update parent env.
    void ExitIfStatement(IRBuilderContext& ctx, IfStatementPtr st);
    void ExitForStatement(IRBuilderContext& ctx, ForStatementPtr st);

private:
    std::shared_ptr<ProgramModule> module_{nullptr};
};

} // namespace pto
