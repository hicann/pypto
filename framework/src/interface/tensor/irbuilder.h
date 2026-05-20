/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "ir/builder.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"

using namespace pypto;

namespace npu::tile_fwk {

class IRContext;

class IRBuilder : public ir::IRBuilder {
public:
    IRBuilder();

    // Disable copying and moving since we have unique_ptr members
    IRBuilder(const IRBuilder&) = delete;
    IRBuilder& operator=(const IRBuilder&) = delete;
    IRBuilder(IRBuilder&&) = delete;
    IRBuilder& operator=(IRBuilder&&) = delete;

    /* create raw tensor with static shape */
    std::shared_ptr<RawTensor> CreateRawTensor(
        DataType t, Shape shape, TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create raw tensor with dynamic shape */
    std::shared_ptr<RawTensor> CreateRawTensor(
        DataType t, std::vector<SymbolicScalar> shape, TileOpFormat format = TileOpFormat::TILEOP_ND,
        std::string name = "");

    /* create logical tensor with static shape */
    LogicalTensorPtr CreateTensorVar(
        DataType t, Shape shape, TileOpFormat format = TileOpFormat::TILEOP_ND, std::string name = "");

    /* create logical tensor with dynamic shape */
    LogicalTensorPtr CreateTensorVar(
        DataType t, Shape shape, std::vector<SymbolicScalar> validShape, TileOpFormat format = TileOpFormat::TILEOP_ND,
        std::string name = "");

    /* create logical tensor from raw tensor */
    LogicalTensorPtr CreateTensorVar(
        std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape, std::vector<SymbolicScalar> validShape = {});

    /* create tensor operation statement */
    Operation& CreateTensorOpStmt(
        Function& f, const Opcode opCode, const LogicalTensors& iOperands, const LogicalTensors& oOperands,
        ir::Span span = ir::Span::Unknown());

    ir::TensorOpStmtPtr CreateTensorOpStmt(
        std::vector<ir::VarPtr> result, ir::VarPtr result_token, std::string opcode, std::vector<ir::ExprPtr> args,
        std::vector<ir::ExprPtr> tokens, std::vector<std::pair<std::string, std::any>> attrs, ir::Span span);

    /* create function */
    std::shared_ptr<Function> CreateFunction(
        std::string name, LogicalTensors params, ir::StmtPtr body, ir::Span span = ir::Span::Unknown());

    /* create symbolic scalar */
    SymbolicScalar CreateConstInt(int64_t value);

    SymbolicScalar CreateScalarVar(std::string sym);

    /* ==== scf statement ==== */

    ir::AssignStmtPtr CreateAssignStmt(ir::VarPtr var, ir::ExprPtr value, ir::Span span);

    ir::SeqStmtsPtr CreateSeqStmts(std::vector<ir::StmtPtr> stmts, ir::Span span);

    ir::IfStmtPtr CreateIfStmt(
        ir::ExprPtr cond, ir::StmtPtr thenBody, std::optional<ir::StmtPtr> elseBody, std::vector<ir::VarPtr> returnVars,
        ir::Span span);

    ir::YieldStmtPtr CreateYieldStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ReturnStmtPtr CreateReturnStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ForStmtPtr CreateForStmt(
        ir::VarPtr loopVar, ir::ExprPtr start, ir::ExprPtr stop, ir::ExprPtr step, std::vector<ir::IterArgPtr> iterArgs,
        ir::StmtPtr body, std::vector<ir::VarPtr> returnVars, ir::Span span);

    ir::WhileStmtPtr CreateWhileStmt(
        ir::ExprPtr cond, std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body, std::vector<ir::VarPtr> returnVars,
        ir::Span span);

    ir::BreakStmtPtr CreateBreakStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::ContinueStmtPtr CreateContinueStmt(std::vector<ir::ExprPtr> values, ir::Span span);

    ir::FunctionPtr CreateFunction(
        std::string name, std::vector<ir::VarPtr> params, std::vector<ir::TypePtr> returnTypes, ir::StmtPtr body,
        ir::Span span);

    ir::ProgramPtr CreateProgram(std::vector<ir::FunctionPtr> functions, std::string name, ir::Span span);

    ir::VarPtr CreateTokenVar(ir::Span span);

private:
    IRContext& irContext_;
};
} // namespace npu::tile_fwk
