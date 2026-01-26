/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_call.cpp
 * \brief
 */

#include "ir/block_call.h"

namespace pto {
using namespace npu::tile_fwk;

static void NormalizeCoaForScalar(const std::vector<SymbolicScalar>& scalarArgs,
                                  std::vector<std::vector<SymbolicScalar>> &coaArgsList, int &coaIndex) {
    std::vector<SymbolicScalar> coaArgs;
    for (auto &value: scalarArgs) {
        coaArgs.push_back(value * 1);
    }

    coaArgsList.push_back(coaArgs);
    coaIndex += (int)coaArgs.size();
}

static void NormalizeCoaForTensor(LogicalTensorPtr tensor,
                                  std::vector<std::vector<SymbolicScalar>> &coaArgsList, int &coaIndex) {
    auto rawshape = OpImmediate::Specified(tensor->GetRawTensor()->GetRawShape());
    int dim = rawshape.size();
    int curIndex = COA_INDEX_DIM_BASE;
    coaIndex += COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> coaArgs(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    // offset
    curIndex += dim;
    coaIndex += dim;

    // shape
    OpImmediate::NormalizeValue(coaArgs, curIndex, rawshape, coaIndex, false);
    curIndex += dim;
    coaIndex += dim;

    // raw shape
    OpImmediate::NormalizeValue(coaArgs, curIndex, rawshape, coaIndex, false);
    curIndex += dim;
    coaIndex += dim;

    // valid shape
    curIndex += dim;
    coaIndex += dim;

    coaArgsList.push_back(coaArgs);
}

static void NormalizeCoaForBlockFunc(std::vector<LogicalTensorPtr> &inCasts, std::vector<LogicalTensorPtr> &outCasts,
                                     const std::vector<SymbolicScalar>& scalarArgs,
                                     std::vector<std::vector<SymbolicScalar>> &coaArgsList,
                                     std::vector<int> &iOffset, std::vector<int> &oOffset) {
    int coaIndex = COA_INDEX_BASE;
    std::unordered_map<LogicalTensorPtr, int> processedOperands;

    NormalizeCoaForScalar(scalarArgs, coaArgsList, coaIndex);

    for (auto &tensor: inCasts) {
        if (processedOperands.find(tensor) != processedOperands.end()) {
            iOffset.push_back(processedOperands[tensor]);
        } else {
            iOffset.push_back(coaIndex);
            processedOperands[tensor] = coaIndex;
            NormalizeCoaForTensor(tensor, coaArgsList, coaIndex);
        }
    }

    for (auto &tensor: outCasts) {
        if (processedOperands.find(tensor) != processedOperands.end()) {
            oOffset.push_back(processedOperands[tensor]);
        } else {
            oOffset.push_back(coaIndex);
            processedOperands[tensor] = coaIndex;
            NormalizeCoaForTensor(tensor, coaArgsList, coaIndex);
        }
    }
}

std::vector<Tensor> CallBlock(const pto::FunctionPtr &blockFuncPtr,
    const std::vector<std::reference_wrapper<const Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputTensorArgs,
    const std::vector<SymbolicScalar>& indices) {
    ASSERT(blockFuncPtr != nullptr) << "Block function pointer should not be nullptr";
    std::vector<LogicalTensorPtr> inputLogicTensors;
    for (const auto &tensorRef : inputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<uint64_t> tensorShape;
        for (int64_t ele : tensor.GetShape()) {
            tensorShape.emplace_back(static_cast<uint64_t>(ele));
        }
        inputLogicTensors.emplace_back(tensor.GetStorage(true));
    }

    std::vector<Tensor> result;
    std::vector<LogicalTensorPtr> outputLogicTensors;
    size_t i = 0;
    for (const auto &tensorRef : outputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<uint64_t> tensorShape;
        for (auto ele : tensor.GetShape()) {
            tensorShape.emplace_back(static_cast<uint64_t>(ele));
        }
        result.emplace_back(tensor.GetDataType(), tensor.GetShape(), "result" + std::to_string(i), TileOpFormat::TILEOP_ND);
        outputLogicTensors.emplace_back(result.back().GetStorage(false));
        i++;
    }

    auto function = npu::tile_fwk::Program::GetInstance().GetCurrentFunction();
    ALOG_INFO_F("In block call Function name %s, type %d %d", function->GetMagicName().c_str(),
        (int)function->GetFunctionType(), (int)function->GetGraphType());
    // 1. Here we put program module into TensorGraph and create a CallOp in TensorGraph
    if (function->programModule_ == nullptr) {
        function->programModule_ = std::make_shared<ProgramModule>(function->GetMagicName() + "_IR");
    }
    std::vector<ScalarValuePtr> index;
    ASSERT(blockFuncPtr->GetKind() == FunctionKind::Block);
    function->programModule_->AddFunction(blockFuncPtr);
    function->programModule_->SetProgramEntry(blockFuncPtr);
    auto &callOp = function->AddRawOperation(npu::tile_fwk::Opcode::OP_BLOCK_CALL,
        inputLogicTensors, outputLogicTensors, false);

    // 2 Compute hash of program module
    FunctionHash hash = blockFuncPtr->ComputeHash();
    auto &functionCache = npu::tile_fwk::Program::GetInstance().GetFunctionCache();
    auto cacheValue = functionCache.Get(hash);
    if (cacheValue == std::nullopt) {
        functionCache.Insert(hash, blockFuncPtr.get());
    }

    // 3 Create Call op attribute
    std::vector<std::vector<SymbolicScalar>> argList;
    std::vector<int> iOffset;
    std::vector<int> oOffset;

    NormalizeCoaForBlockFunc(inputLogicTensors, outputLogicTensors, indices, argList, iOffset, oOffset);

    auto opAttribute = std::make_shared<CallOpAttribute>(hash, argList,
        function->programModule_->GetFunctions().back()->GetName());
    callOp.SetOpAttribute(opAttribute);
    callOp.SetOpOffset(iOffset, oOffset);
    return result;
}

std::vector<npu::tile_fwk::Tensor> CallBlock(const BlockFunctionType &blockFunc,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const npu::tile_fwk::Tensor>> &outputTensorArgs,
    const std::vector<npu::tile_fwk::SymbolicScalar> &indices)
{
    std::vector<TensorValuePtr> inputArgs;
    for (const auto &tensorRef : inputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<uint64_t> tensorShape;
        for (int64_t ele : tensor.GetShape()) {
            tensorShape.emplace_back(static_cast<uint64_t>(ele));
        }
        auto tensorValue = std::make_shared<TensorValue>((pto::DataType)tensor.GetDataType(), tensorShape,
            tensor.GetName());
        inputArgs.emplace_back(tensorValue);
    }

    std::vector<TensorValuePtr> outputArgs;
    for (const auto &tensorRef : outputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<uint64_t> tensorShape;
        for (auto ele : tensor.GetShape()) {
            tensorShape.emplace_back(static_cast<uint64_t>(ele));
        }
        auto tensorValue = std::make_shared<TensorValue>((pto::DataType)tensor.GetDataType(), tensorShape,
            tensor.GetName());
        outputArgs.emplace_back(tensorValue);
    }
    std::vector<ScalarValuePtr> index;
    auto blockFuncPtr = blockFunc(inputArgs, outputArgs, index);
    ASSERT(blockFuncPtr != nullptr) << "Block function pointer should not be nullptr";
    ASSERT(blockFuncPtr->GetKind() == FunctionKind::Block);
    return CallBlock(blockFuncPtr, inputTensorArgs, outputTensorArgs, indices);
}
}