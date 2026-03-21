/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_codegen_utils.h"

#include <iostream>

#include "gtest/gtest.h"

#include "codegen/utils/codegen_error.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo &info) {
    if (info.memType == MemoryType::MEM_DEVICE_DDR) {
        std::shared_ptr<RawTensor> ddrRawTensor =
            std::make_shared<RawTensor>(info.dType, info.shape, TileOpFormat::TILEOP_ND, info.tensorName, info.magic);
        std::vector<int64_t> offset = std::vector<int64_t>(info.shape.size(), 0);
        auto ddrTensor = std::make_shared<LogicalTensor>(info.function, ddrRawTensor, offset, info.shape);
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
        ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
        if (!info.dynValidShape.empty()) {
            ddrTensor->UpdateDynValidShape(info.dynValidShape);
        }
        return ddrTensor;
    }

    auto localTensor = std::make_shared<LogicalTensor>(
        info.function, info.dType, info.shape, TileOpFormat::TILEOP_ND, info.tensorName);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(info.memType);
    localTensor->SetMemoryTypeToBe(info.memType);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memoryrange.memId = 0;
    localTensor->memoryrange.start = 0;
    localTensor->memoryrange.end = 0;
    if (info.magic != -1) {
        localTensor->SetMagic(info.magic);
    }
    if (!info.dynValidShape.empty()) {
        localTensor->UpdateDynValidShape(info.dynValidShape);
    }
    info.function.GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    return localTensor;
}

std::string GetResultFromCpp(const Function &function) {
    const auto &subFunc = function.rootFunc_->programs_[0];
    auto leafFuncAttr = subFunc->GetLeafFuncAttribute();
    ASSERT(FwkErr::INVALID_FUNCTION, leafFuncAttr != nullptr) << "can not find leaf func attribute";
    std::string binPath = leafFuncAttr->binPath;
    std::string cppFile = binPath.substr(0, binPath.rfind('.')) + ".cpp";
    std::ifstream ifs(cppFile);
    std::string res((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    ifs.close();
    return res;
}

void CheckStringExist(const std::string &target, const std::string &content) {
    bool res = content.find(target) != std::string::npos;
    EXPECT_TRUE(res) << "target: \n" << target << "\n\n ---- not found in content ---- \n\n" << content << std::endl;
}

Function *GenMockFuncDyn(const std::string &funcName, const std::vector<int64_t> &shape) {
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(funcName, {inputA, inputB}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);
    return function;
}

std::shared_ptr<LogicalTensor> CreateConvTensor(Function &function, const DataType &dtype,
    const std::vector<int64_t> &shape, const MemoryType &memType, const bool &isCopyIn) {
    std::shared_ptr<LogicalTensor> tensorPtr = nullptr;
    if (isCopyIn) {
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            tensorPtr = std::make_shared<LogicalTensor>(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                TileOpFormat::TILEOP_ND, "GmTensor", NodeType::INCAST);
        } else {
            tensorPtr = std::make_shared<LogicalTensor>(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                TileOpFormat::TILEOP_NZ, "L1Tensor", NodeType::LOCAL);
            tensorPtr->UpdateSubgraphID(0);
            tensorPtr->SetAttr(OpAttributeKey::needAlloc, true);
            tensorPtr->memoryrange.memId = 0;
            tensorPtr->memoryrange.start = 0;
            tensorPtr->memoryrange.end = 0;
        }
    } else {
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            tensorPtr = std::make_shared<LogicalTensor>(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                TileOpFormat::TILEOP_ND, "GmTensor", NodeType::OUTCAST);
        } else {
            tensorPtr = std::make_shared<LogicalTensor>(function, dtype, shape, SymbolicScalar::FromConcrete(shape),
                TileOpFormat::TILEOP_NZ, "L0CTensor", NodeType::LOCAL);
            tensorPtr->UpdateSubgraphID(0);
            tensorPtr->SetAttr(OpAttributeKey::needAlloc, true);
            tensorPtr->memoryrange.memId = 0;
            tensorPtr->memoryrange.start = 0;
            tensorPtr->memoryrange.end = 0;
        }
    }
    tensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(shape));
    tensorPtr->SetMemoryTypeOriginal(memType);
    tensorPtr->SetMemoryTypeToBe(memType);
    function.GetTensorMap().inverseMap_[tensorPtr->GetMagic()] = tensorPtr;
    return tensorPtr;
}

} // namespace npu::tile_fwk
