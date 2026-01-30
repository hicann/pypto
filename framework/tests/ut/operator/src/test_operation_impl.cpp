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
 * \file test_operation_impl.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
using namespace npu::tile_fwk;

class OperationImplTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override { 
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    }

    void TearDown() override {}
};

TEST_F(OperationImplTest, test_CumSum_dim2_1) {
    int axis = 1;
    TileShape::Current().SetVecTile(9, 9);
    Tensor input(DT_FP32, {13, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") {
        result = CumSum(input, axis);
    }
}

TEST_F(OperationImplTest, test_CumSum_dim2_0) {
    int axis = 0;
    TileShape::Current().SetVecTile(4, 3);
    Tensor input(DT_FP16, {11, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") {
        result = CumSum(input, axis);
    }
}

TEST_F(OperationImplTest, test_CumSum_dim1) {
    int axis = 0;
    TileShape::Current().SetVecTile(5);
    Tensor input(DT_INT32, {13}, "input");
    Tensor result;
    FUNCTION("TestCumSum") {
        result = CumSum(input, axis);
    }
}

TEST_F(OperationImplTest, test_CumSum_dim3) {
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 3);
    Tensor input(DT_FP32, {8, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") {
        result = CumSum(input, axis);
    }
}

TEST_F(OperationImplTest, test_CumSum_dim4) {
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 5, 3);
    Tensor input(DT_FP32, {7, 7, 7, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") {
        result = CumSum(input, axis);
    }
}

TEST_F(OperationImplTest, test_Compare_BOOL) {
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor operand2(DT_FP32, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") {
        result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL);
    }
}

TEST_F(OperationImplTest, test_Compare_BIT) {
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    Tensor operand2(DT_FP16, {16, 16}, "operand2");
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") {
        result = Compare(operand1, operand2, OpType::EQ, OutType::BIT);
    }
}

TEST_F(OperationImplTest, Test_Compare_BF16) {
    PROGRAM("Compare") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        Tensor input_b(DT_BF16, shape, "B");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Compare_BF16") {
            output = Compare(input_a, input_b, npu::tile_fwk::OpType::EQ, npu::tile_fwk::OutType::BOOL);
        }
    }
}

TEST_F(OperationImplTest, test_Cmps_BOOL) {
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP32, scalar);
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") {
        result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL);
    }
}

TEST_F(OperationImplTest, test_Cmps_BIT) {
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP16, scalar);
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") {
        result = Compare(operand1, operand2, OpType::EQ, OutType::BIT);
    }
}

TEST_F(OperationImplTest, test_Cmps_BF16) {
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_BF16, {8, 8}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_BF16, scalar);
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") {
        result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL);
    }
}

TEST_F(OperationImplTest, Test_IndexAdd_BF16) {
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_BF16, {10, 16}, "operand0");
    Tensor src(DT_BF16, {8, 16}, "operand1");
    Tensor index(DT_INT32, {8}, "operand2");
    Element alpha(DT_BF16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") {
        result = IndexAdd(self, src, index, axis, alpha);
    }
}

TEST_F(OperationImplTest, Test_IndexAdd_INT8) {
    int scalar = 2;
    int axis = 1;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_INT8, {10, 16}, "operand0");
    Tensor src(DT_INT8, {10, 18}, "operand1");
    Tensor index(DT_INT32, {18}, "operand2");
    Element alpha(DT_INT8, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") {
        result = IndexAdd(self, src, index, axis, alpha);
    }
}

TEST_F(OperationImplTest, Test_IndexAdd_INT16) {
    int scalar = 2;
    int axis = 1;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_INT16, {10, 5}, "operand0");
    Tensor src(DT_INT16, {10, 2}, "operand1");
    Tensor index(DT_INT64, {2}, "operand2");
    Element alpha(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") {
        result = IndexAdd(self, src, index, axis, alpha);
    }
}

TEST_F(OperationImplTest, Test_IndexAdd_FP32) {
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 16});
    Tensor self(DT_FP32, {10, 10, 16}, "operand0");
    Tensor src(DT_FP32, {15, 10, 16}, "operand1");
    Tensor index(DT_INT32, {15}, "operand2");
    Element alpha(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") {
        result = IndexAdd(self, src, index, axis, alpha);
    }
}

TEST_F(OperationImplTest, Test_IndexAdd_FP16) {
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 8, 16});
    Tensor self(DT_FP16, {10, 10, 10, 16}, "operand0");
    Tensor src(DT_FP16, {8, 10, 10, 16}, "operand1");
    Tensor index(DT_INT32, {8}, "operand2");
    Element alpha(DT_FP16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") {
        result = IndexAdd(self, src, index, axis, alpha);
    }
}

void TestPow(DataType dataType, double exponent) {
    PROGRAM("POWS") {
        std::vector<int64_t> shape = {32, 32};
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(dataType, shape, "input");
        auto output = Tensor(dataType, shape, "res");
        FUNCTION("POWS_FUC") {
            output = Pow(input_a, Element(dataType, exponent));
        }
    }
}

TEST_F(OperationImplTest, Test_Pows__1_5_FP32) {
    constexpr double EXP = -1.5;
    TestPow(DataType::DT_FP32, EXP);
}

TEST_F(OperationImplTest, Test_Pows_1_5_FP32) {
    constexpr double EXP = 1.5;
    TestPow(DataType::DT_FP32, EXP);
}

TEST_F(OperationImplTest, Test_Pows_1_5_FP16) {
    constexpr double EXP = 1.5;
    TestPow(DataType::DT_FP16, EXP);
}

TEST_F(OperationImplTest, Test_Pows_2_FP32) {
    constexpr double EXP = 2;
    TestPow(DataType::DT_FP32, EXP);
}

TEST_F(OperationImplTest, Test_Pows_3_FP32) {
    constexpr double EXP = 3;
    TestPow(DataType::DT_FP32, EXP);
}

TEST_F(OperationImplTest, Test_LogicalNot_BF16) {
    PROGRAM("LogicalNot") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("LogicalNot_BF16") {
            output = LogicalNot(input_a);
        }
    }
}

TEST_F(OperationImplTest, Test_WhereTT_BF16) {
    PROGRAM("Where") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") {
            output = Where(input_c, input_a, input_b);
        }
    }
}

TEST_F(OperationImplTest, Test_WhereTS_BF16) {
    PROGRAM("Where") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        float scalar = 10.0;
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") {
            output = Where(input_c, input_a, operand2);
        }
    }
}

TEST_F(OperationImplTest, Test_WhereSS_BF16) {
    PROGRAM("Where") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") {
            output = Where(input_c, operand1, operand2);
        }
    }
}

TEST_F(OperationImplTest, Test_WhereST_BF16) {
    PROGRAM("Where") {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") {
            output = Where(input_c, operand1, input_b);
        }
    }
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false, bool isTransB = false>
void TestNZFormatBatch(int bs, int m, int k, int n) {
    std::vector<int64_t> batch_shape_a = {bs * m, k};
    auto nLen = isTransB ? bs * n : bs * k;
    auto kLen = isTransB ? k : n;
    std::vector<int64_t> batch_shape_b = {nLen, kLen};
    std::vector<int64_t> batch_shape_c = {bs * m, n};
    PROGRAM("BATCHMATMUL") {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, batch_shape_a, "MatA", afmt);
        Tensor matB(inputType, batch_shape_b, "MatB", bfmt);
        Tensor matC(outputType, batch_shape_c, "MatC");
        std::vector<Tensor> matrixVec;
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            std::vector<std::pair<Tensor, std::vector<int64_t>>> assembleVec;
            for (size_t index = 0; index < (size_t)bs; ++index) {
                auto inputA = View(matA, {m, k}, {(int)index * m, 0});
                auto inputB =
                    isTransB ? View(matB, {n, k}, {(int)index * n, 0}) : View(matB, {k, n}, {(int)index * k, 0});
                TileShape::Current().SetMatrixSize({m, k, n});
                auto outTensor = npu::tile_fwk::Matrix::Matmul(outputType, inputA, inputB, false, isTransB);
                std::vector<int64_t> pairSecond = {(int)index * m, 0};
                auto pair = std::make_pair(outTensor, pairSecond);
                assembleVec.emplace_back(pair);
            }
            matC = Assemble(assembleVec);
        }
    }
}

TEST_F(OperationImplTest, test_Range_FP16) {
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP16, startValue);
    Element end(DT_FP16, endValue);
    Element step(DT_FP16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") {
        result = Range(start, end, step);
    }
}

TEST_F(OperationImplTest, test_Range_BF16) {
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_BF16, startValue);
    Element end(DT_BF16, endValue);
    Element step(DT_BF16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") {
        result = Range(start, end, step);
    }
}

TEST_F(OperationImplTest, test_Range_FP32) {
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP32, startValue);
    Element end(DT_FP32, endValue);
    Element step(DT_FP32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") {
        result = Range(start, end, step);
    }
}

TEST_F(OperationImplTest, test_Range_INT32) {
    int startValue = 1;
    int endValue = 10;
    int stepValue = 3;
    int NUM_Eight = 8;
    Element start(DT_INT32, startValue);
    Element end(DT_INT32, endValue);
    Element step(DT_INT32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") {
        result = Range(start, end, step);
    }
}

TEST_F(OperationImplTest, test_Rsqrt_FP16) {
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP16, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") {
        result = Rsqrt(operand1);
    }
}

TEST_F(OperationImplTest, test_Rsqrt_FP32) {
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") {
        result = Rsqrt(operand1);
    }
}

TEST_F(OperationImplTest, test_Ceil_FP32) {
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestCeil") {
        result = Ceil(operand1);
    }
}

TEST_F(OperationImplTest, test_Floor_FP32) {
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestFloor") {
        result = Floor(operand1);
    }
}

TEST_F(OperationImplTest, test_Trunc_FP32) {
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestTrunc") {
        result = Trunc(operand1);
    }
}

 TEST_F(OperationImplTest, test_Reciprocal_FP32) {
 	     constexpr int TILE_SHAPE = 32;
 	     constexpr int SHAPE = 128;
 	     TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
 	     Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
 	     Tensor result;
 	     FUNCTION("TestReciprocal") {
 	         result = Reciprocal(operand1);
 	     }
}

TEST_F(OperationImplTest, TestIndexPut_) {
    constexpr int TILE_SHAPE = 8;
    TileShape::Current().SetVecTile(TILE_SHAPE);
    Shape shapeSelf({128, 8, 8});
    Shape shapeValues({128, 8});
    Shape shapeIndices({128});
    Tensor self(DT_INT32, shapeSelf, "self");
    Tensor values(DT_INT32, shapeValues, "values");
    Tensor indices0(DT_INT32, shapeIndices, "indices0");
    Tensor indices1(DT_INT32, shapeIndices, "indices1");
    std::vector<Tensor> indices{indices0, indices1};
    bool accumulate = false;
    Tensor result;
    FUNCTION("TestIndexPut_") {
        IndexPut_(self, indices, values, accumulate);
    }
}

TEST_F(OperationImplTest, test_Expand_8_1_to_8_8) {
    TileShape::Current().SetVecTile({4, 4});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") {
        result = Expand(operand1, dstShape);
    }
}

TEST_F(OperationImplTest, test_Expand_8_1_to_8_8_dyn) {
    TileShape::Current().SetVecTile({3, 3});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") {
        result = Expand(operand1, dstShape);
    }
}

TEST_F(OperationImplTest, test_Clip_FP16) {
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") {
        result = Clip(src, min, max);
    }
}

TEST_F(OperationImplTest, test_Clip_FP32_VS) {
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Element min(DT_FP32, minValue);
    Element max(DT_FP32, maxValue);

    Tensor result;
    FUNCTION("TestClip") {
        result = Clip(src, min, max);
    }
}

TEST_F(OperationImplTest, test_Clip_FP16_VS) {
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") {
        result = Clip(src, min, max);
    }
}

TEST_F(OperationImplTest, test_Clip_FP32_VV) {
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 16, 16}, "min");
    Tensor max(DT_FP32, {8, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") {
        result = Clip(src, min, max);
    }
}

TEST_F(OperationImplTest, test_Clip_FP32_VV_BRC) {
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 1, 16}, "min");
    Tensor max(DT_FP32, {1, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") {
        result = Clip(src, min, max);
    }
}

TEST_F(OperationImplTest, Test_Amax) {
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmax") {
        result = Amax(operand, -1, true);
    }
}

TEST_F(OperationImplTest, Test_Amin) {
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmin") {
        result = Amin(operand, -1, true);
    }
}

TEST_F(OperationImplTest, test_Gather) {
    TileShape::Current().SetVecTile(8, 8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT32, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGather") {
        result = Gather(operand1, operand2, -1);
    }
}

TEST_F(OperationImplTest, test_Scatter_FP16) {
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Element operand3(DT_FP16, 1.0);
    Tensor result;
    FUNCTION("TestScatter") {
        result = Scatter(operand1, operand2, operand3, 0);
    }
}

TEST_F(OperationImplTest, test_ScatterTensor_FP16) {
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Tensor operand3(DT_FP16, {2, 16}, "operand3");
    Tensor result;
    FUNCTION("TestScatter") {
        result = Scatter(operand1, operand2, operand3, 0);
    }
}

TEST_F(OperationImplTest, test_Where) {
    TileShape::Current().SetVecTile(8, 8);
    Tensor condition(DT_UINT8, {8, 2}, "condition");
    Tensor input(DT_FP32, {8, 16}, "input");
    Tensor other(DT_FP32, {8, 16}, "other");
    Tensor result;
    FUNCTION("TestWhere") {
        result = Where(condition, input, other);
    }
}

TEST_F(OperationImplTest, test_Add_Brcb) {
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input0");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestAddBrcb") {
        result = Add(input0, input1);
    }
}

TEST_F(OperationImplTest, test_Fmod) {
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 16}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, false);
    FUNCTION("TestFmod") {
        result = Fmod(input0, input1);
    }
}

TEST_F(OperationImplTest, test_Fmod_Brcb) {
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestFmodBrcb") {
        result = Fmod(input0, input1);
    }
}

TEST_F(OperationImplTest, test_FmodS) {
    TileShape::Current().SetVecTile({4, 4});
    Tensor input0(DT_FP32, {8, 8}, "input0");
    float scalar = 10.0;
    Element input1(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestFmodS") {
        result = Fmod(input0, input1);
    }
}

TEST_F(OperationImplTest, Test_TopK_01) {
    std::vector<int64_t> inputShape = {1, 16384};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") {
        output = TopK(input_a, 2048, -1);
    }
}

TEST_F(OperationImplTest, Test_TopK_02) {
    std::vector<int64_t> inputShape = {1, 24576};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") {
        output = TopK(input_a, 2048, -1);
    }
}

TEST_F(OperationImplTest, Test_TopK_03) {
    std::vector<int64_t> inputShape = {1, 49152};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") {
        output = TopK(input_a, 2048, -1);
    }
}

TEST_F(OperationImplTest, Test_TopK_04) {
    std::vector<int64_t> inputShape = {1, 40960};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") {
        output = TopK(input_a, 2048, -1);
    }
}

TEST_F(OperationImplTest, Test_BitwiseRightShift) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") {
        result = BitwiseRightShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_BitwiseRightShift_brc) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {1, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") {
        result = BitwiseRightShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_BitwiseRightShifts) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseRightShift") {
        result = BitwiseRightShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_SBitwiseRightShift) {
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") {
        result = BitwiseRightShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShift) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") {
        result = BitwiseLeftShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShift_brc) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {1, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") {
        result = BitwiseLeftShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_BitwiseLeftShifts) {
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") {
        result = BitwiseLeftShift(self, other);
    }
}

TEST_F(OperationImplTest, Test_SBitwiseLeftShift) {
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") {
        result = BitwiseLeftShift(self, other);
    }
}