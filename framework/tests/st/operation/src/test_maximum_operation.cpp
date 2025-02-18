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
 * \file test_maximum_operation.cpp
 * \brief
 */

#include <nlohmann/json.hpp>
#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct MaximumOpFuncArgs : public OpFuncArgs {
    MaximumOpFuncArgs(const Element &value, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : value_(value), viewShape_(viewShape), tileShape_(tileShape) {}

    Element value_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct MaximumOpMetaData {
    explicit MaximumOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};


Shape GetBroadCastViewShape(const Tensor &self, const Tensor &other, const Shape &viewShape) {
    ASSERT(self.GetShape().size() == other.GetShape().size());
    Shape result = viewShape;
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 1;
        } else {
            result[i] = std::min(selfDim, viewShape[i]);
        }
    }
    return result;
}

std::vector<int64_t> GetBroadCastOffsetRatio(const Tensor &self, const Tensor &other, const Shape &viewShape) {
    ASSERT(self.GetShape().size() == other.GetShape().size());
    Shape result(viewShape.size(), 1);
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        int64_t selfDim = self.GetShape()[i];
        int64_t otherDim = other.GetShape()[i];
        if (selfDim != otherDim && selfDim == 1 && otherDim != 1) {
            result[i] = 0;
        }
    }
    return result;
}

void MaximumOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                const Shape &tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                const std::vector<int64_t> &tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1], args->viewShape_);
                const Shape &tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                const std::vector<int64_t> &tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0], args->viewShape_);
                Tensor tileTensor0 = View(inputs[0], {tile0ViewShape[0], tile0ViewShape[1]},
                    {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                        std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1])},
                    {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0], sIdx * tile0ViewShape[1] * tile0OffsetRatio[1]});
                Tensor tileTensor1 = View(inputs[1], {tile1ViewShape[0], tile1ViewShape[1]},
                    {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                        std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1])},
                    {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0], sIdx * tile1ViewShape[1] * tile1OffsetRatio[1]});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Maximum(tileTensor0, tileTensor1);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

void MaximumOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    const Shape &tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                    const std::vector<int64_t> &tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1], args->viewShape_);
                    const Shape &tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                    const std::vector<int64_t> &tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0], args->viewShape_);
                    Tensor tileTensor0 = View(inputs[0], {tile0ViewShape[0], tile0ViewShape[1], tile0ViewShape[2]},
                        {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                            std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1]),
                            std::min(thirdDim - nIdx * tile0ViewShape[2], tile0ViewShape[2])},
                        {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0],
                            sIdx * tile0ViewShape[1] * tile0OffsetRatio[1],
                            nIdx * tile0ViewShape[2] * tile0OffsetRatio[2]});
                    Tensor tileTensor1 = View(inputs[1], {tile1ViewShape[0], tile1ViewShape[1], tile1ViewShape[2]},
                        {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                            std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1]),
                            std::min(thirdDim - nIdx * tile1ViewShape[2], tile1ViewShape[2])},
                        {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0],
                            sIdx * tile1ViewShape[1] * tile1OffsetRatio[1],
                            nIdx * tile1ViewShape[2] * tile1OffsetRatio[2]});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Maximum(tileTensor0, tileTensor1);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

void MaximumOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = std::max(inputs[0].GetShape()[0], inputs[1].GetShape()[0]);
        SymbolicScalar secondDim = std::max(inputs[0].GetShape()[1], inputs[1].GetShape()[1]);
        SymbolicScalar thirdDim = std::max(inputs[0].GetShape()[2], inputs[1].GetShape()[2]);
        SymbolicScalar fourthDim = std::max(inputs[0].GetShape()[3], inputs[1].GetShape()[3]);
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);
        const int kloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1)) {
                        const Shape &tile0ViewShape = GetBroadCastViewShape(inputs[0], inputs[1], args->viewShape_);
                        const std::vector<int64_t> &tile0OffsetRatio = GetBroadCastOffsetRatio(inputs[0], inputs[1], args->viewShape_);
                        const Shape &tile1ViewShape = GetBroadCastViewShape(inputs[1], inputs[0], args->viewShape_);
                        const std::vector<int64_t> &tile1OffsetRatio = GetBroadCastOffsetRatio(inputs[1], inputs[0], args->viewShape_);
                        Tensor tileTensor0 = View(inputs[0], {tile0ViewShape[0], tile0ViewShape[1], tile0ViewShape[2], tile0ViewShape[3]},
                            {std::min(firstDim - bIdx * tile0ViewShape[0], tile0ViewShape[0]),
                                std::min(secondDim - sIdx * tile0ViewShape[1], tile0ViewShape[1]),
                                std::min(thirdDim - nIdx * tile0ViewShape[2], tile0ViewShape[2]),
                                std::min(fourthDim - kIdx * tile0ViewShape[3], tile0ViewShape[3])},
                            {bIdx * tile0ViewShape[0] * tile0OffsetRatio[0],
                                sIdx * tile0ViewShape[1] * tile0OffsetRatio[1],
                                nIdx * tile0ViewShape[2] * tile0OffsetRatio[2],
                                kIdx * tile0ViewShape[3] * tile0OffsetRatio[3]});
                        Tensor tileTensor1 = View(inputs[1], {tile1ViewShape[0], tile1ViewShape[1], tile1ViewShape[2], tile1ViewShape[3]},
                            {std::min(firstDim - bIdx * tile1ViewShape[0], tile1ViewShape[0]),
                                std::min(secondDim - sIdx * tile1ViewShape[1], tile1ViewShape[1]),
                                std::min(thirdDim - nIdx * tile1ViewShape[2], tile1ViewShape[2]),
                                std::min(fourthDim - kIdx * tile1ViewShape[3], tile1ViewShape[3])},
                            {bIdx * tile1ViewShape[0] * tile1OffsetRatio[0],
                                sIdx * tile1ViewShape[1] * tile1OffsetRatio[1],
                                nIdx * tile1ViewShape[2] * tile1OffsetRatio[2],
                                kIdx * tile1ViewShape[3] * tile1OffsetRatio[3]});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Maximum(tileTensor0, tileTensor1);
                        Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, kIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

void MaxSOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Maximum(tileTensor0, args->value_);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

void MaxSOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Maximum(tileTensor0, args->value_);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

void MaxSOperationExeFuncQuadrupleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const MaximumOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 =
                            View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    qIdx * fourthViewShape});
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Maximum(tileTensor0, args->value_);
                        Assemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

class MaximumOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MaximumOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestMaximum, MaximumOperationTest,
    ::testing::ValuesIn(GetOpMetaData<MaximumOpMetaData>(
        {MaximumOperationExeFunc2Dims, MaximumOperationExeFunc3Dims, MaximumOperationExeFunc4Dims}, "Maximum")));

Element GetElementByType(DataType dataType, nlohmann::json test_data, string name) {
    if (dataType == DT_FP32 || dataType == DT_FP16 || dataType == DT_BF16) {
        Element element(dataType, GetValueByName<float>(test_data, name));
        return element;
    } else if (dataType == DT_INT8) {
        Element element(dataType, GetValueByName<int8_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT16) {
        Element element(dataType, GetValueByName<int16_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT32) {
        Element element(dataType, GetValueByName<int32_t>(test_data, name));
        return element;
    } else if (dataType == DT_INT64) {
        Element element(dataType, GetValueByName<int64_t>(test_data, name));
        return element;
    } else {
        std::string errorMessage = "UnSupport Type in MaxS ST Test" + DataType2String(dataType);
        throw std::invalid_argument(errorMessage.c_str());
    }
}

TEST_P(MaximumOperationTest, TestMaximum) {
    auto test_data = GetParam().test_data_;

    bool isElementMode = test_data.at("input_tensors").size() <= 1;
    Element value = {};
    if (isElementMode) {
        auto dtype = GetDataType(GetValueByName<std::string>(test_data, "scalar_type"));
        value = GetElementByType(dtype, test_data, "scalar");
    }
    Shape viewShape = GetViewShape(test_data);
    Shape tileShape = GetTileShape(test_data);

    auto args = MaximumOpFuncArgs(value, viewShape, tileShape);
    auto testCase = CreateTestCaseDesc<MaximumOpMetaData>(GetParam(), &args);

    std::vector<OpFunc> opFuncs = {};
    if (isElementMode) {
        opFuncs = {
            MaxSOperationExeFuncDoubleCut,
            MaxSOperationExeFuncTripleCut,
            MaxSOperationExeFuncQuadrupleCut
        };
    }
    else {
        opFuncs = {
            MaximumOperationExeFunc2Dims,
            MaximumOperationExeFunc3Dims,
            MaximumOperationExeFunc4Dims
        };
    }
    testCase.opFunc = opFuncs[viewShape.size() - 2];

    TestExecutor::runTest(testCase);
}
} // namespace
