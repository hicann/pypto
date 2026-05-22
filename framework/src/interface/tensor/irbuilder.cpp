/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "irbuilder.h"

#include "logical_tensor.h"
#include "raw_tensor.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"

#include "ir/expr.h"

using namespace pypto;

namespace npu::tile_fwk {

class IRContext {
public:
    IRContext(const IRContext&) = delete;
    IRContext& operator=(const IRContext&) = delete;
    IRContext(IRContext&&) = delete;
    IRContext& operator=(IRContext&&) = delete;

    ir::VarPtr MakeVar(std::string name, ir::TypePtr type, ir::Span span)
    {
        auto var_name = GetVarName(name);
        type_map_[var_name] = type;
        return std::make_shared<ir::Var>(var_name, type, span);
    }

    ir::VarPtr MakeTempVar(ir::TypePtr type, ir::Span span)
    {
        auto var_name = GetVarName();
        return MakeVar(var_name, type, span);
    }

    ir::IterArgPtr MakeIterArg(std::string name, ir::TypePtr type, ir::ExprPtr initVal, ir::Span span)
    {
        auto var_name = GetVarName(name);
        type_map_[var_name] = type;
        return std::make_shared<ir::IterArg>(var_name, type, initVal, span);
    }

    ir::VarPtr MakeToken() { return MakeTempVar(ir::GetTokenType(), ir::Span::Unknown()); }

    ir::TypePtr GetType(ir::VarPtr var) { return type_map_[var->name_]; }

    void SetType(ir::VarPtr var, ir::TypePtr type) { type_map_[var->name_] = type; }

    std::string GetOriginName(ir::VarPtr var) { return all_vars_[var->name_]; }

    std::string GetVarName(const std::string& name = "")
    {
        auto var_name = name;
        if (var_name.empty()) {
            auto idx = temp_counter_++;
            var_name = "%" + std::to_string(idx);
        } else {
            while (all_vars_.count(var_name)) {
                auto idx = var_counter_[var_name]++;
                var_name = name + "." + std::to_string(idx);
            }
        }
        all_vars_[var_name] = name;
        return var_name;
    }

    void Reset()
    {
        temp_counter_ = 0;
        type_map_.clear();
        var_counter_.clear();
        all_vars_.clear();
    }

    static IRContext& Get()
    {
        static IRContext ctx;
        return ctx;
    }

private:
    IRContext() = default;
    int64_t temp_counter_{0};                     // counter for temporary variables
    std::map<std::string, ir::TypePtr> type_map_; // type for each variable
    std::map<std::string, int64_t> var_counter_;  // counter for named variable
    std::map<std::string, std::string> all_vars_; // unique var name -> var name
};

Function& DummyFunc()
{
    static Function func(Program::GetInstance(), "Dummy", "Dummy", nullptr);
    return func;
}

IRBuilder::IRBuilder() : irContext_(IRContext::Get()) {}

LogicalTensorPtr IRBuilder::CreateTensorVar(DataType t, Shape shape, TileOpFormat format, std::string name)
{
    auto tensorName = irContext_.GetVarName(name);
    auto tensor = std::make_shared<LogicalTensor>(DummyFunc(), t, std::move(shape), format, tensorName);
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(
    DataType t, Shape shape, std::vector<SymbolicScalar> validShape, TileOpFormat format, std::string name)
{
    auto tensorName = irContext_.GetVarName(name);
    auto tensor = std::make_shared<LogicalTensor>(
        DummyFunc(), t, std::move(shape), std::move(validShape), format, tensorName);
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(
    std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape, std::vector<SymbolicScalar> validShape)
{
    LogicalTensorPtr tensor;
    auto tensorName = irContext_.GetVarName(rawTensor->GetSymbol());
    if (validShape.empty()) {
        tensor =
            std::make_shared<LogicalTensor>(DummyFunc(), std::move(rawTensor), std::move(offset), std::move(shape));
    } else {
        tensor = std::make_shared<LogicalTensor>(
            DummyFunc(), std::move(rawTensor), std::move(offset), std::move(shape), std::move(validShape));
    }
    tensor->name_ = tensorName;
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(Function& f, DataType t, Shape shape, TileOpFormat format, std::string name)
{
    auto tensorName = irContext_.GetVarName(name);
    auto tensor = std::make_shared<LogicalTensor>(f, t, std::move(shape), format, tensorName);
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(
    Function& f, DataType t, Shape shape, std::vector<SymbolicScalar> validShape, TileOpFormat format, std::string name)
{
    auto tensorName = irContext_.GetVarName(name);
    auto tensor =
        std::make_shared<LogicalTensor>(f, t, std::move(shape), std::move(validShape), format, tensorName);
    return tensor;
}

LogicalTensorPtr IRBuilder::CreateTensorVar(
    Function& f, std::shared_ptr<RawTensor> rawTensor, Offset offset, Shape shape,
    std::vector<SymbolicScalar> validShape)
{
    LogicalTensorPtr tensor;
    auto tensorName = irContext_.GetVarName(rawTensor->GetSymbol());
    if (validShape.empty()) {
        tensor = std::make_shared<LogicalTensor>(f, std::move(rawTensor), std::move(offset), std::move(shape));
    } else {
        tensor = std::make_shared<LogicalTensor>(
            f, std::move(rawTensor), std::move(offset), std::move(shape), std::move(validShape));
    }
    tensor->name_ = tensorName;
    return tensor;
}

Operation& IRBuilder::CreateTensorOpStmt(
    Function& f, const Opcode opCode, const LogicalTensors& iOperands, const LogicalTensors& oOperands, ir::Span span)
{
    return f.AddRawOperation(opCode, iOperands, oOperands, true, span);
}

std::shared_ptr<RawTensor> IRBuilder::CreateRawTensor(DataType t, Shape shape, TileOpFormat format, std::string name)
{
    auto magic = IdGen<IdType::RAW_TENSOR>::Inst().NewId();
    return std::make_shared<RawTensor>(t, std::move(shape), format, std::move(name), magic);
}

std::shared_ptr<RawTensor> IRBuilder::CreateRawTensor(
    DataType t, std::vector<SymbolicScalar> dynShape, TileOpFormat format, std::string name)
{
    auto shape = SymbolicScalar::Concrete(dynShape, -1);
    auto magic = IdGen<IdType::RAW_TENSOR>::Inst().NewId();
    auto raw = std::make_shared<RawTensor>(t, shape, format, std::move(name), magic);
    raw->UpdateDynRawShape(dynShape);
    return raw;
}

ir::TensorOpStmtPtr IRBuilder::CreateTensorOpStmt(
    std::vector<ir::VarPtr> result, ir::VarPtr result_token, std::string opcode, std::vector<ir::ExprPtr> args,
    std::vector<ir::ExprPtr> tokens, std::vector<std::pair<std::string, std::any>> attrs, ir::Span span)
{
    return std::make_shared<ir::TensorOpStmt>(result, result_token, opcode, args, tokens, attrs, span);
}

/* create function */
std::shared_ptr<Function> CreateFunction(
    std::string name, LogicalTensors params, ir::StmtPtr body, ir::Span span = ir::Span::Unknown());

/* create symbolic scalar */
SymbolicScalar IRBuilder::CreateConstInt(int64_t value) { return SymbolicScalar(value); }

SymbolicScalar IRBuilder::CreateScalarVar(std::string sym)
{
    auto name = irContext_.GetVarName(sym);
    return SymbolicScalar(name);
}

/* scf statement */
ir::AssignStmtPtr IRBuilder::CreateAssignStmt(ir::VarPtr var, ir::ExprPtr value, ir::Span span)
{
    return std::make_shared<ir::AssignStmt>(var, value, span);
}

ir::SeqStmtsPtr IRBuilder::CreateSeqStmts(std::vector<ir::StmtPtr> stmts, ir::Span span)
{
    return std::make_shared<ir::SeqStmts>(stmts, span);
}

ir::IfStmtPtr IRBuilder::CreateIfStmt(
    ir::ExprPtr cond, ir::StmtPtr thenBody, std::optional<ir::StmtPtr> elseBody, std::vector<ir::VarPtr> returnVars,
    ir::Span span)
{
    return std::make_shared<ir::IfStmt>(cond, thenBody, elseBody, returnVars, span);
}

ir::YieldStmtPtr IRBuilder::CreateYieldStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::YieldStmt>(values, span);
}

ir::ReturnStmtPtr IRBuilder::CreateReturnStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::ReturnStmt>(values, span);
}

ir::ForStmtPtr IRBuilder::CreateForStmt(
    ir::VarPtr loopVar, ir::ExprPtr start, ir::ExprPtr stop, ir::ExprPtr step, std::vector<ir::IterArgPtr> iterArgs,
    ir::StmtPtr body, std::vector<ir::VarPtr> returnVars, ir::Span span)
{
    return std::make_shared<ir::ForStmt>(loopVar, start, stop, step, iterArgs, body, returnVars, span);
}

ir::WhileStmtPtr IRBuilder::CreateWhileStmt(
    ir::ExprPtr cond, std::vector<ir::IterArgPtr> iterArgs, ir::StmtPtr body, std::vector<ir::VarPtr> returnVars,
    ir::Span span)
{
    return std::make_shared<ir::WhileStmt>(cond, iterArgs, body, returnVars, span);
}

ir::BreakStmtPtr IRBuilder::CreateBreakStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::BreakStmt>(values, span);
}

ir::ContinueStmtPtr IRBuilder::CreateContinueStmt(std::vector<ir::ExprPtr> values, ir::Span span)
{
    return std::make_shared<ir::ContinueStmt>(values, span);
}

ir::FunctionPtr IRBuilder::CreateFunction(
    std::string name, std::vector<ir::VarPtr> params, std::vector<ir::TypePtr> returnTypes, ir::StmtPtr body,
    ir::Span span)
{
    return std::make_shared<ir::Function>(name, params, returnTypes, body, span);
}

ir::ProgramPtr IRBuilder::CreateProgram(std::vector<ir::FunctionPtr> functions, std::string name, ir::Span span)
{
    return std::make_shared<ir::Program>(functions, name, span);
}

ir::VarPtr IRBuilder::CreateTokenVar(ir::Span span) { return irContext_.MakeTempVar(ir::GetTokenType(), span); }
} // namespace npu::tile_fwk
