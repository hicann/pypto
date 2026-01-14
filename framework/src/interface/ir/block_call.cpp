/**
 * Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
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
void CallBlock(BlockFunctionType blockFunction,
    const std::vector<std::reference_wrapper<const Tensor>> &inputTensorArgs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputTensorArgs,
    const std::vector<SymbolicScalar>& indices) {
    // 根据Tensor生成TileValue
    std::vector<TileValuePtr> inputArgs;
    std::vector<LogicalTensorPtr> inputLogicTensors;
    for (const auto &tensorRef : inputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<int64_t> tensorShape;
        std::vector<ScalarValuePtr> validShapes;
        for (int64_t ele : tensor.GetShape()) {
            tensorShape.emplace_back(ele);
            validShapes.emplace_back(std::make_shared<ScalarValue>(ele));
        }
        auto tileValue = std::make_shared<TileValue>(tensorShape, 
            (pto::DataType)tensor.GetDataType(), validShapes, tensor.GetName());
        inputArgs.emplace_back(tileValue);
        inputLogicTensors.emplace_back(tensor.GetStorage(true));
    }

    std::vector<TileValuePtr> outputArgs;
    std::vector<LogicalTensorPtr> outputLogicTensors;
    for (const auto &tensorRef : outputTensorArgs) {
        const auto &tensor = tensorRef.get();
        std::vector<int64_t> tensorShape;
        std::vector<ScalarValuePtr> validShapes;
        for (int64_t ele : tensor.GetShape()) {
            tensorShape.emplace_back(ele);
            validShapes.emplace_back(std::make_shared<ScalarValue>(ele));
        }
        auto tileValue = std::make_shared<TileValue>(tensorShape, 
            (pto::DataType)tensor.GetDataType(), validShapes, tensor.GetName());
        outputArgs.emplace_back(tileValue);
        outputLogicTensors.emplace_back(tensor.GetStorage(false));
        // Handle slot: simulate the behavior in Tensor::operation=
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(tensor);
    }

    auto function = npu::tile_fwk::Program::GetInstance().GetCurrentFunction();
    ALOG_INFO_F("In block call Function name %s, type %d %d", function->GetMagicName().c_str(), 
        (int)function->GetFunctionType(), (int)function->GetGraphType());
    // 1. Here we put program module into TensorGraph and create a CallOp in TensorGraph
    if (function->programModule_ == nullptr) {
        function->programModule_ = std::make_shared<ProgramModule>(function->GetMagicName() + "_IR");
    }
    std::vector<ScalarValuePtr> index;
    auto irFunc = blockFunction(inputArgs, outputArgs, index);
    function->programModule_->AddFunction(irFunc);
    function->programModule_->SetProgramEntry(irFunc);
    auto &callOp = function->AddRawOperation(npu::tile_fwk::Opcode::OP_BLOCK_CALL, 
        inputLogicTensors, outputLogicTensors, false);

    // 2 Compute hash of program module
    FunctionHash hash = irFunc->ComputeHash();
    // IR block function hash
    // 3 Create Call op attribute
    std::vector<std::vector<SymbolicScalar>> argList;
    argList.emplace_back(indices);
    auto opAttribute = std::make_shared<CallOpAttribute>(hash, argList,
        function->programModule_->GetFunctions().back()->GetName());
    callOp.SetOpAttribute(opAttribute);
}
}