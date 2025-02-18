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
 * \file test_cumsum_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct CumSumOpFuncArgs : public OpFuncArgs {
    CumSumOpFuncArgs(int axis, const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape, bool flag)
        : axis_(axis), viewShape_(viewShape), tileShape_(tileShape), flag_(flag) {}

    int axis_;
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    bool flag_;
};

struct CumSumOpMetaData {
    explicit CumSumOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static std::vector<int64_t> GetCumSumViewShape(Tensor tensor, const std::vector<int64_t> &viewShape, int axis) {
    std::vector<int64_t> resultViewShape = {};
    for (size_t i = 0; i < viewShape.size(); i++) {
        if (static_cast<int>(i) != axis) {
            resultViewShape.push_back(std::min(viewShape[i], tensor.GetShape()[i]));
        } else {
            resultViewShape.push_back(tensor.GetShape()[i]);
        }
    }
    return resultViewShape;
}

static std::vector<SymbolicScalar> GetNoAxisDims(const Tensor &input, int axis) {
    std::vector<SymbolicScalar> dimensions = {};
    for (size_t i = 0; i < input.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        dimensions.push_back(input.GetShape()[i]);
    }
    return dimensions;
}

static std::vector<int64_t> GetNoAxisViewShapes(const std::vector<int64_t> &viewShapes, int axis) {
    std::vector<int64_t> shapes = {};
    for (size_t i = 0; i < viewShapes.size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        shapes.push_back(viewShapes[i]);
    }
    return shapes;
}

static void CumSumOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const CumSumOpFuncArgs *>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        if (inputs[0].GetShape().size() == 1) {
            int axis = args->axis_;
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, 1, 1)) {
                (void)bIdx;
                SymbolicScalar validShape0 = SymbolicScalar(inputs[0].GetShape()[0]);
                auto tileTensor = View(inputs[0], {validShape0}, {validShape0}, {0});

                TileShape::Current().SetVecTile({validShape0});
                auto res = CumSum(tileTensor, axis);
                Assemble(res, {0}, outputs[0]);
            }
        } else {
            int axis = args->axis_ < 0 ? args->axis_ + inputs[0].GetShape().size() : args->axis_;
            std::vector<SymbolicScalar> noAxisDims = GetNoAxisDims(inputs[0], axis);
            std::vector<int64_t> noAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
            const int bloop = CeilDiv(noAxisDims[0], noAxisViewShapes[0]);
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
                std::vector<SymbolicScalar> indices = {bIdx};
                indices.insert(indices.begin() + axis, 0);
                const std::vector<int64_t> tensorViewShape = GetCumSumViewShape(inputs[0], args->viewShape_, axis);

                SymbolicScalar validShape0 = std::min(
                    SymbolicScalar(inputs[0].GetShape()[0]) - indices[0] * tensorViewShape[0], tensorViewShape[0]);
                SymbolicScalar validShape1 = std::min(
                    SymbolicScalar(inputs[0].GetShape()[1]) - indices[1] * tensorViewShape[1], tensorViewShape[1]);
                auto tileTensor = View(inputs[0], tensorViewShape, {validShape0, validShape1},
                    {indices[0] * tensorViewShape[0], indices[1] * tensorViewShape[1]});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = CumSum(tileTensor, axis);
                Assemble(res, {indices[0] * args->viewShape_[0], indices[1] * args->viewShape_[1]}, outputs[0]);
            }
        }
    }
}

static void CumSumOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const CumSumOpFuncArgs *>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        int axis = args->axis_ < 0 ? args->axis_ + inputs[0].GetShape().size() : args->axis_;
        std::vector<SymbolicScalar> noAxisDims = GetNoAxisDims(inputs[0], axis);
        std::vector<int64_t> noAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
        const int bloop = CeilDiv(noAxisDims[0], noAxisViewShapes[0]);
        const int sloop = CeilDiv(noAxisDims[1], noAxisViewShapes[1]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                std::vector<SymbolicScalar> indices = {bIdx, sIdx};
                indices.insert(indices.begin() + axis, 0);

                const std::vector<int64_t> tensorViewShape = GetCumSumViewShape(inputs[0], args->viewShape_, axis);
                SymbolicScalar validShape0 = std::min(
                    SymbolicScalar(inputs[0].GetShape()[0]) - indices[0] * tensorViewShape[0], tensorViewShape[0]);
                SymbolicScalar validShape1 = std::min(
                    SymbolicScalar(inputs[0].GetShape()[1]) - indices[1] * tensorViewShape[1], tensorViewShape[1]);
                SymbolicScalar validShape2 = std::min(
                    SymbolicScalar(inputs[0].GetShape()[2]) - indices[2] * tensorViewShape[2], tensorViewShape[2]);
                auto tileTensor = View(inputs[0], tensorViewShape, {validShape0, validShape1, validShape2},
                    {
                        indices[0] * tensorViewShape[0],
                        indices[1] * tensorViewShape[1],
                        indices[2] * tensorViewShape[2],
                    });

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = CumSum(tileTensor, axis);
                Assemble(res,
                    {
                        indices[0] * args->viewShape_[0],
                        indices[1] * args->viewShape_[1],
                        indices[2] * args->viewShape_[2],
                    },
                    outputs[0]);
            }
        }
    }
}

static void CumSumOperationExeFuncQuadraticCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    auto args = static_cast<const CumSumOpFuncArgs *>(opArgs);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        int axis = args->axis_ < 0 ? args->axis_ + inputs[0].GetShape().size() : args->axis_;
        std::vector<SymbolicScalar> noAxisDims = GetNoAxisDims(inputs[0], axis);
        std::vector<int64_t> noAxisViewShapes = GetNoAxisViewShapes(args->viewShape_, axis);
        const int bloop = CeilDiv(noAxisDims[0], noAxisViewShapes[0]);
        const int sloop = CeilDiv(noAxisDims[1], noAxisViewShapes[1]);
        const int kloop = CeilDiv(noAxisDims[2], noAxisViewShapes[2]);
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L1_kIdx", FunctionType::DYNAMIC_LOOP, kIdx, LoopRange(0, kloop, 1)) {
                    std::vector<SymbolicScalar> indices = {bIdx, sIdx, kIdx};
                    indices.insert(indices.begin() + axis, 0);

                    const std::vector<int64_t> tensorViewShape = GetCumSumViewShape(inputs[0], args->viewShape_, axis);
                    SymbolicScalar validShape0 = std::min(
                        SymbolicScalar(inputs[0].GetShape()[0]) - indices[0] * tensorViewShape[0], tensorViewShape[0]);
                    SymbolicScalar validShape1 = std::min(
                        SymbolicScalar(inputs[0].GetShape()[1]) - indices[1] * tensorViewShape[1], tensorViewShape[1]);
                    SymbolicScalar validShape2 = std::min(
                        SymbolicScalar(inputs[0].GetShape()[2]) - indices[2] * tensorViewShape[2], tensorViewShape[2]);
                    SymbolicScalar validShape3 = std::min(
                        SymbolicScalar(inputs[0].GetShape()[3]) - indices[3] * tensorViewShape[3], tensorViewShape[3]);
                    auto tileTensor =
                        View(inputs[0], tensorViewShape, {validShape0, validShape1, validShape2, validShape3},
                            {
                                indices[0] * tensorViewShape[0],
                                indices[1] * tensorViewShape[1],
                                indices[2] * tensorViewShape[2],
                                indices[3] * tensorViewShape[3],
                            });

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = CumSum(tileTensor, axis);
                    Assemble(res,
                        {
                            indices[0] * args->viewShape_[0],
                            indices[1] * args->viewShape_[1],
                            indices[2] * args->viewShape_[2],
                            indices[3] * args->viewShape_[3],
                        },
                        outputs[0]);
                }
            }
        }
    }
}

class CumSumOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<CumSumOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestCumSum, CumSumOperationTest,
    ::testing::ValuesIn(GetOpMetaData<CumSumOpMetaData>(
        {CumSumOperationExeFuncDoubleCut, CumSumOperationExeFuncTripleCut, CumSumOperationExeFuncQuadraticCut},
        "CumSum")));

TEST_P(CumSumOperationTest, TestCumSum) {
    TestCaseDesc testCase;
    auto test_data = GetParam().test_data_;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    bool flag = false;
    int axis = GetValueByName<int>(test_data, "axis");
    auto args = CumSumOpFuncArgs(axis, GetViewShape(test_data), GetTileShape(test_data), flag);
    testCase.args = &args;
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/" + testCase.inputTensors[0].GetStorage()->Symbol() + ".bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/" + testCase.outputTensors[0].GetStorage()->Symbol() + ".bin"};
    TestExecutor::runTest(testCase);
}
} // namespace