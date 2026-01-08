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
 * \file symbolic_scalar_evaluate.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/interpreter/function.h"
#include "interface/operation/attribute.h"
#include "tensor/symbolic_scalar.h"

namespace npu::tile_fwk {
namespace {
ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDimSize(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == 1);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];

    auto ret = input->GetShape().size();
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputShapeDim(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_TWO);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto n = dataList[1];

    auto ret = input->GetShape()[n];
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim1(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_TWO);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto off0 = dataList[1];
    ASSERT(input->GetShape().size() == 1);

    int index = off0;
    auto elt = input->GetElement(index);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim2(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_THREE);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto off0 = dataList[1];
    auto off1 = dataList[2];
    ASSERT(input->GetShape().size() == SIZE_TWO);

    int index = off0 * input->GetShape()[1] + off1;
    auto elt = input->GetElement(index);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim3(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_FOUR);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto off0 = dataList[1];
    auto off1 = dataList[2];
    auto off2 = dataList[3];
    ASSERT(input->GetShape().size() == SIZE_THREE);

    int index = off0 * input->GetShape()[1] * input->GetShape()[2] + off1 * input->GetShape()[2] + off2;
    auto elt = input->GetElement(index);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetInputDataInt32Dim4(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_FOUR);
    auto inputIndex = dataList[0];
    auto input = evaluateSymbol->GetInputDataViewList()[inputIndex];
    auto off0 = dataList[1];
    auto off1 = dataList[2];
    auto off2 = dataList[3];
    auto off3 = dataList[4];
    ASSERT(input->GetShape().size() == SIZE_THREE);

    int index = ((off0 * input->GetShape()[1] + off1) * input->GetShape()[2] + off2) * input->GetShape()[3] + off3;
    auto elt = input->GetElement(index);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32(
        EvaluateSymbol *evaluateSymbol,
        int ioType, int ioTypeIndex,
        const std::vector<ScalarImmediateType> &offsetList) {
    UNUSED(offsetList);
    auto inoutDataPair = evaluateSymbol->GetInoutDataPair();
    std::shared_ptr<LogicalTensorData> view;
    if (ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST) {
        view = inoutDataPair->GetIncastDataViewList()[ioTypeIndex];
    } else if (ioType == GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST) {
        view = inoutDataPair->GetOutcastDataViewList()[ioTypeIndex];
    } else {
        ASSERT(false);
    }
    auto elt = view->GetElement(0);
    auto ret = static_cast<ScalarImmediateType>(elt.Cast<int64_t>());
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim1(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1, dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim2(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1, dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim3(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    auto ioType = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE - 1];
    auto ioTypeIndex = dataList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX - 1];
    std::vector<ScalarImmediateType> offsetList(dataList.begin() + GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS - 1 + 1, dataList.end());

    auto ret = EvaluateSymbolicCallRuntimeGetTensorDataInt32(evaluateSymbol, ioType, ioTypeIndex, offsetList);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallGetParaAddr(EvaluateSymbol *, const std::vector<ScalarImmediateType> &) {
    // not used by getTensorData
    return 0;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopBegin(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_TWO);
    auto ret = evaluateSymbol->RuntimeIsLoopBegin(dataList[0], dataList[1]);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeIsLoopEnd(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    ASSERT(dataList.size() == SIZE_TWO);
    auto ret = evaluateSymbol->RuntimeIsLoopEnd(dataList[0], dataList[1]);
    return ret;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeGetViewValidShapeDim(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList) {
    UNUSED(evaluateSymbol);
    ASSERT(dataList.size() == SIZE_THREE);
    auto validshape = dataList[0];
    auto viewOffset = dataList[1];
    auto viewshape = dataList[2];
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetValidShape(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList,
        const std::vector<SymbolicScalar> &linearArgList) {
    ASSERT(linearArgList.size()) << "linearArgList is null";
    auto coaIndex = dataList[1] + COA_INDEX_DIM_BASE + dataList[0] * 3 + dataList[2];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}

ScalarImmediateType EvaluateSymbolicCallRuntimeCoaGetOffset(
        EvaluateSymbol *evaluateSymbol,
        const std::vector<ScalarImmediateType> &dataList,
        const std::vector<SymbolicScalar> &linearArgList) {
    ASSERT(linearArgList.size()) << "linearArgList is null";

    auto coaIndex = dataList[1] + COA_INDEX_DIM_BASE + dataList[2];
    return evaluateSymbol->EvaluateSymbolicScalar(linearArgList[coaIndex]);
}
}

ScalarImmediateType EvaluateSymbol::EvaluateSymbolicCall(
        const std::string &name, const std::vector<ScalarImmediateType> &dataList,
        const std::vector<SymbolicScalar> &linearArgList) {
    using CallEntry = ScalarImmediateType (*)(EvaluateSymbol *, const std::vector<ScalarImmediateType> &dataList);
    static std::unordered_map<std::string, CallEntry> callEntryDict = {
        {"RUNTIME_GetInputShapeDimSize",        EvaluateSymbolicCallRuntimeGetInputShapeDimSize},
        {"RUNTIME_GetInputShapeDim",            EvaluateSymbolicCallRuntimeGetInputShapeDim},
        {"RUNTIME_GetInputDataInt32Dim1",       EvaluateSymbolicCallRuntimeGetInputDataInt32Dim1},
        {"RUNTIME_GetInputDataInt32Dim2",       EvaluateSymbolicCallRuntimeGetInputDataInt32Dim2},
        {"RUNTIME_GetInputDataInt32Dim3",       EvaluateSymbolicCallRuntimeGetInputDataInt32Dim3},
        {"RUNTIME_GetInputDataInt32Dim4",       EvaluateSymbolicCallRuntimeGetInputDataInt32Dim4},
        {"RUNTIME_IsLoopBegin",                 EvaluateSymbolicCallRuntimeIsLoopBegin},
        {"RUNTIME_IsLoopEnd",                   EvaluateSymbolicCallRuntimeIsLoopEnd},
        {"RUNTIME_GetViewValidShapeDim",        EvaluateSymbolicCallRuntimeGetViewValidShapeDim},
        {"RUNTIME_GetTensorDataInt32Dim1",      EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim1},
        {"RUNTIME_GetTensorDataInt32Dim2",      EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim2},
        {"RUNTIME_GetTensorDataInt32Dim3",      EvaluateSymbolicCallRuntimeGetTensorDataInt32Dim3},
        {"RUNTIME_COA_GET_PARAM_ADDR",          EvaluateSymbolicCallGetParaAddr},
    };
    using CallWithLinerArgsEntry = ScalarImmediateType (*)(EvaluateSymbol *, const std::vector<ScalarImmediateType> &dataList, 
        const std::vector<SymbolicScalar> &linearArgList);
    static std::unordered_map<std::string, CallWithLinerArgsEntry> CallWithLinerArgsEntryDict = {
        {"RUNTIME_COA_GET_PARAM_VALID_SHAPE",   EvaluateSymbolicCallRuntimeCoaGetValidShape},
        {"RUNTIME_COA_GET_PARAM_OFFSET",        EvaluateSymbolicCallRuntimeCoaGetOffset},
    };
    ScalarImmediateType ret{0};
    if (callEntryDict.count(name)) {
        auto callEntry = callEntryDict[name];
        ret = callEntry(this, dataList);
    } else if (CallWithLinerArgsEntryDict.count(name)) {
        auto callEntry = CallWithLinerArgsEntryDict[name];
        ret = callEntry(this, dataList, linearArgList);
    } else {
        ASSERT(false) << "Symbolic call not found: " << name;
    }
    return ret;
}

ScalarImmediateType EvaluateSymbol::EvaluateSymbolicScalar(const RawSymbolicScalarPtr &ss, const std::vector<SymbolicScalar> &linearArgList) {
    ScalarImmediateType result{0};
    switch (ss->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            std::shared_ptr<RawSymbolicImmediate> imm = std::static_pointer_cast<RawSymbolicImmediate>(ss);
            result = imm->Immediate();
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::shared_ptr<RawSymbolicSymbol> sym = std::static_pointer_cast<RawSymbolicSymbol>(ss);
            ASSERT(symbolDict_.count(sym->Name()));
            result = symbolDict_[sym->Name()];
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            std::shared_ptr<RawSymbolicExpression> expr = std::static_pointer_cast<RawSymbolicExpression>(ss);
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 1; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i], linearArgList));
                }
                std::string name = std::static_pointer_cast<RawSymbolicSymbol>(expr->OperandList()[0])->Name();
                result = EvaluateSymbolicCall(name, dataList, linearArgList);
            } else {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 0; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i], linearArgList));
                }
                if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
                    result = RawSymbolicExpression::GetSymbolicCalcUnary(expr->Opcode())(dataList[0]);
                } else if (SymbolicOpcode::T_BOP_BEGIN <= expr->Opcode() &&
                           expr->Opcode() < SymbolicOpcode::T_BOP_END) {
                    result = dataList[0];
                    for (size_t i = 1; i < dataList.size(); i++) {
                        result = RawSymbolicExpression::GetSymbolicCalcBinary(expr->Opcode())(result, dataList[i]);
                    }
                }
            }
        } break;
        default: ASSERT(false); break;
    }
    return result;
}

} // namespace npu::tile_fwk