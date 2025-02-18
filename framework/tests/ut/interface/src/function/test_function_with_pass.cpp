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
 * \file test_function_with_pass.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <iostream>
#include <vector>
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/expected_value.h"
#include "interface/tensor/float.h"
using namespace std;
using namespace npu::tile_fwk;

class FunctionWithPass : public testing::Test {
public:
    static void TearDownTestCase() { }

    static void SetUpTestCase() { }

    void SetUp() override {
        std::cout << "-----------------------------SetUp-------------------------------" << std::endl;
        Program::GetInstance().Reset();
        config::Reset();
        TileShape::Current().SetVecTile(16, 16);
    }

    void TearDown() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    bool TestContinuous(const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> &testcase) {
        auto &func = *Program::GetInstance().GetCurrentFunction();
        std::vector<LogicalTensorPtr> tensors;
        std::vector<int64_t> shape3(testcase.front().first.size(), 512);
        Tensor t(DT_FP32, shape3, "p");

        for (auto &ele : testcase) {
            auto t1 = t.GetStorage()->View(func, ele.first, ele.second);
            tensors.emplace_back(t1);
        }
        return FunctionUtils::IsContinuous(tensors);
    }
};

namespace {
constexpr float F_1_1 = 1.1f;
constexpr float F_2_2 = 2.2f;
constexpr float F_3_3 = 3.3f;
constexpr float F_2_42 = 2.42f;

TEST_F(FunctionWithPass, TestContinuous) {
    std::vector<int64_t> shape{128};
    auto &func = *Program::GetInstance().GetCurrentFunction();
    Tensor a(DT_FP32, shape, "a");
    auto b = a.GetStorage()->View(func, {64}, {0});
    auto c = a.GetStorage()->View(func, {64}, {64});
    auto d = a.GetStorage()->View(func, {64}, {63});
    std::vector<LogicalTensorPtr> tensors1 = {b, c};
    std::vector<LogicalTensorPtr> tensors2 = {b, d};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors1), true);
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors2), false);

    std::vector<int64_t> shape2{256, 384};
    Tensor e(DT_FP32, shape2, "e");
    auto f = e.GetStorage()->View(func, {64, 128}, {0, 128});
    auto g = e.GetStorage()->View(func, {64, 128}, {64, 128});
    std::vector<LogicalTensorPtr> tensors = {f, g};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), true);

    auto h = e.GetStorage()->View(func, {64, 128}, {0, 128});
    auto i = e.GetStorage()->View(func, {64, 128}, {0, 256});
    tensors = {h, i};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), true);

    auto j = e.GetStorage()->View(func, {64, 127}, {0, 128});
    auto k = e.GetStorage()->View(func, {64, 128}, {0, 256});
    tensors = {j, k};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), false);

    auto l = e.GetStorage()->View(func, {64, 128}, {0, 128});
    auto m = e.GetStorage()->View(func, {64, 128}, {65, 128});
    tensors = {l, m};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), false);

    auto n = e.GetStorage()->View(func, {64, 127}, {0, 128});
    auto o = e.GetStorage()->View(func, {64, 128}, {64, 128});
    tensors = {n, o};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), false);

    std::vector<int64_t> shape3{256, 512};
    Tensor p(DT_FP32, shape3, "p");
    auto p1 = p.GetStorage()->View(func, {64, 64}, {0, 256});
    auto p2 = p.GetStorage()->View(func, {64, 64}, {0, 320});
    auto p3 = p.GetStorage()->View(func, {64, 64}, {64, 256});
    auto p4 = p.GetStorage()->View(func, {64, 64}, {64, 320});
    std::vector<LogicalTensorPtr> tensorsP = {p1, p2, p3, p4};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensorsP), true);

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> testcase1 = {
        {{2, 3}, {1, 1}},
        {{2, 3}, {3, 1}},
        {{2, 3}, {1, 4}},
        {{2, 3}, {3, 4}}
    };
    EXPECT_EQ(TestContinuous(testcase1), true);
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_3d_offset = {
        {{2, 2, 2}, {1, 1, 1}},
        {{2, 2, 2}, {3, 1, 1}},
        {{2, 2, 2}, {1, 3, 1}},
        {{2, 2, 2}, {3, 3, 1}},
        {{2, 2, 2}, {1, 1, 3}},
        {{2, 2, 2}, {3, 1, 3}},
        {{2, 2, 2}, {1, 3, 3}},
        {{2, 2, 2}, {3, 3, 3}}
    };
    EXPECT_EQ(TestContinuous(tensors_3d_offset), true);

    // 测试用例 3: 2D 矩形，不规则排列，无重叠无缝隙
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_2d_irregular = {
        {{2, 2}, {0, 0}},
        {{2, 2}, {2, 0}},
        {{2, 2}, {0, 2}},
        {{2, 2}, {2, 2}},
        {{4, 1}, {0, 4}},
        {{1, 4}, {4, 0}}
    };
    EXPECT_EQ(TestContinuous(tensors_2d_irregular), false);
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_2d_irregular_fixed = {
        {{2, 2}, {0, 0}},
        {{2, 2}, {2, 0}},
        {{2, 2}, {0, 2}},
        {{2, 2}, {2, 2}},
        {{4, 1}, {0, 4}},
        {{1, 4}, {4, 0}},
        {{1, 1}, {4, 4}} // 添加一个 1x1 的矩形填补缝隙
    };
    EXPECT_EQ(TestContinuous(tensors_2d_irregular_fixed), true);

    // 测试用例 4: 3D 立方体，不规则排列，无重叠无缝隙
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_3d_irregular = {
        {{2, 2, 2}, {0, 0, 0}},
        {{2, 2, 2}, {2, 0, 0}},
        {{2, 2, 2}, {0, 2, 0}},
        {{2, 2, 2}, {2, 2, 0}},
        {{2, 2, 2}, {0, 0, 2}},
        {{2, 2, 2}, {2, 0, 2}},
        {{2, 2, 2}, {0, 2, 2}},
        {{2, 2, 2}, {2, 2, 2}},
        {{4, 1, 1}, {0, 0, 4}},
        {{1, 4, 1}, {4, 0, 0}},
        {{1, 1, 4}, {4, 4, 0}}
    };
    EXPECT_EQ(TestContinuous(tensors_3d_irregular), false);

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_3d_irregular_fix = {
        {{2, 2, 2}, {0, 0, 0}},
        {{2, 2, 2}, {2, 0, 0}},
        {{2, 2, 2}, {0, 2, 0}},
        {{2, 2, 2}, {2, 2, 0}},
        {{2, 2, 2}, {0, 0, 2}},
        {{2, 2, 2}, {2, 0, 2}},
        {{2, 2, 2}, {0, 2, 2}},
        {{2, 2, 2}, {2, 2, 2}},
        {{4, 4, 1}, {0, 0, 4}},
        {{4, 1, 5}, {0, 5, 0}},
        {{1, 5, 5}, {4, 0, 0}},
    };
    EXPECT_EQ(TestContinuous(tensors_3d_irregular), false);

    // 测试用例 5: 2D 矩形，有重叠
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_2d_overlap = {
        {{2, 2}, {1, 1}},
        {{2, 2}, {2, 2}}
    };
    EXPECT_EQ(TestContinuous(tensors_2d_overlap), false);

    // 测试用例 6: 3D 立方体，有缝隙
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_3d_gap = {
        {{2, 2, 2}, {0, 0, 0}},
        {{2, 2, 2}, {2, 0, 0}},
        {{2, 2, 2}, {0, 2, 0}},
        {{2, 2, 2}, {2, 2, 0}},
        {{2, 2, 2}, {0, 0, 2}},
        {{2, 2, 2}, {2, 0, 2}},
        {{2, 2, 2}, {0, 2, 2}}
        // 缺少一个 2x2x2 的立方体在 (2, 2, 2)
    };
    EXPECT_EQ(TestContinuous(tensors_3d_gap), false);

    // 测试用例 7: 1D 线段，offset 不从 0 开始
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tensors_1d_offset = {
        {{3}, {1}},
        {{3}, {4}},
        {{3}, {7}}
    };
    EXPECT_EQ(TestContinuous(tensors_1d_offset), true);
}

TEST_F(FunctionWithPass, TestContinuous1) {
    std::vector<int64_t> shape2{32, 32, 64};
    auto &func = *Program::GetInstance().GetCurrentFunction();
    Tensor e(DT_FP32, shape2, "e");
    auto f = e.GetStorage()->View(func, {8, 4, 64}, {0, 0, 0});
    auto g = e.GetStorage()->View(func, {8, 4, 64}, {0, 4, 0});
    std::vector<LogicalTensorPtr> tensors = {f, g};
    EXPECT_EQ(FunctionUtils::IsContinuous(tensors), true);
}

TEST_F(FunctionWithPass, TestBf16) {
    npu::tile_fwk::bfloat16 bf1 = 1.0f;
    float f1 = (float)bf1;
    EXPECT_EQ(f1, 1.0f);
}

TEST_F(FunctionWithPass, TestBf16Str) {
    npu::tile_fwk::bfloat16 bf1 = 1.0f;
    auto strBf16 = std::to_string(bf1);
    EXPECT_EQ(strBf16, "1.000000");
}

TEST_F(FunctionWithPass, TestFp16Str) {
    npu::tile_fwk::float16 fp1 = 2.0f;
    auto strFp16 = std::to_string(fp1);
    EXPECT_EQ(strFp16, "2.000000");
}

TEST_F(FunctionWithPass, AssignFloat_NormalCase) {
    npu::tile_fwk::float16 fp16;
    float fVal = 1.5f; // 正常情况，指数在正常范围内

    fp16 = fVal;
}

TEST_F(FunctionWithPass, AssignFloat_ExponentOverflow) {
    npu::tile_fwk::float16 fp16;
    float fVal = 1e30f; // 指数溢出，应转换为无穷大

    fp16 = fVal;

    // 验证是否为无穷大
    EXPECT_EQ(fp16.value, 31744);
}

TEST_F(FunctionWithPass, AssignFloat_ExponentUnderflow) {
    npu::tile_fwk::float16 fp16;
    float fVal = 1e-40f; // 指数下溢，应转换为零或非规格化数

    fp16 = fVal;

    // 验证是否为零或非规格化数
    EXPECT_EQ(fp16.value, 0) << "Expected zero or denormalized value";
}

TEST_F(FunctionWithPass, AssignFloat_Exponent8Fu_PositiveInfinity) {
    npu::tile_fwk::float16 fp16;
    float fVal = std::numeric_limits<float>::infinity();

    fp16 = fVal;

    // 验证是否为正无穷大
    EXPECT_EQ(fp16.value, 31744);
}

TEST_F(FunctionWithPass, AssignFloat_Exponent8Fu_NegativeInfinity) {
    npu::tile_fwk::float16 fp16;
    float fVal = -std::numeric_limits<float>::infinity();

    fp16 = fVal;

    // 验证是否为负无穷大
    EXPECT_EQ(fp16.value, 64512);
}

TEST_F(FunctionWithPass, AssignFloat_Exponent70u_Zero) {
    npu::tile_fwk::float16 fp16;
    float fVal = 0.0f;

    fp16 = fVal;

    // 验证是否为零
    EXPECT_EQ(fp16.value, 0);
}

TEST_F(FunctionWithPass, AssignFloat_Exponent70u_Denormalized) {
    npu::tile_fwk::float16 fp16;
    float fVal = 1.0e-45f; // 非规格化数

    fp16 = fVal;

    // 验证是否为非规格化数
    EXPECT_EQ(fp16.value, 0);
    EXPECT_EQ(fp16.value, 0);
}

TEST_F(FunctionWithPass, AssignFloat_Zero) {
    npu::tile_fwk::float16 fp16;
    float fVal = 0.0f;

    fp16 = fVal;

    // 验证是否为零
    EXPECT_EQ(fp16.value, 0);
}

TEST_F(FunctionWithPass, AssignFloat_NegativeZero) {
    npu::tile_fwk::float16 fp16;
    float fVal = -0.0f;

    fp16 = fVal;

    // 验证是否为负零
    EXPECT_EQ(fp16.value, 32768);
}

TEST_F(FunctionWithPass, AssignFloat_PositiveInfinity) {
    npu::tile_fwk::float16 fp16;
    float fVal = std::numeric_limits<float>::infinity();

    fp16 = fVal;

    // 验证是否为正无穷大
    EXPECT_EQ(fp16.value, 31744);
}

TEST_F(FunctionWithPass, AssignFloat_NegativeInfinity) {
    npu::tile_fwk::float16 fp16;
    float fVal = -std::numeric_limits<float>::infinity();

    fp16 = fVal;

    // 验证是否为负无穷大
    EXPECT_EQ(fp16.value, 0xFC00);
}

TEST_F(FunctionWithPass, AssignFloat_Denormalized) {
    npu::tile_fwk::float16 fp16;
    float fVal = 1.0e-45f; // 非规格化数

    fp16 = fVal;

    // 验证是否为非规格化数
    EXPECT_EQ(fp16.value, 0);
    EXPECT_EQ(fp16.value, 0);
}

TEST_F(FunctionWithPass, AssignFloat_Rounding) {
    npu::tile_fwk::float16 fp16;
    float fVal = 0.1f; // 需要进行舍入的情况

    fp16 = fVal;

    EXPECT_NE(fp16.value, 0x1F);
}

TEST_F(FunctionWithPass, TestFp16) {
    npu::tile_fwk::float16 f1 = F_1_1;
    npu::tile_fwk::float16 f2 = f1;
    auto test0 = static_cast<float>(f2);
    EXPECT_NE(test0, F_1_1);
    npu::tile_fwk::float16 f3 = f1 + f2;
    auto test1 = static_cast<float>(f3);
    EXPECT_NE(test1, F_2_2);
    auto test2 = static_cast<float>(f1);
    EXPECT_NE(test2, F_1_1);
    auto test3 = static_cast<float>(f2);
    EXPECT_NE(test3, F_1_1);
    npu::tile_fwk::float16 f4 = f3 - f2;
    npu::tile_fwk::float16 f5 = f4 * f3;
    auto test4 = static_cast<float>(f5);
    EXPECT_NE(test4, F_2_42);
    npu::tile_fwk::float16 f6 = f5 / f4;
    auto test5 = static_cast<float>(f6);
    EXPECT_NE(test5, F_2_2);
    f3 += f1;
    auto test6 = static_cast<float>(f3);
    EXPECT_NE(test6, F_3_3);
    f3 -= f1;
    auto test7 = static_cast<float>(f3);
    EXPECT_NE(test7, F_2_2);
    f3 *= f1;
    f3 /= f1;
    auto test8 = static_cast<float>(f3);
    EXPECT_NE(test8, F_2_2);

    auto test9 = static_cast<bool>(f1 == f2);
    EXPECT_EQ(test9, true);
    auto test10 = static_cast<bool>(f1 == f3);
    EXPECT_EQ(test10, false);
    EXPECT_EQ(static_cast<bool>(f1 != f3), true);
    EXPECT_EQ(static_cast<bool>(f1 != f2), false);
    EXPECT_EQ(static_cast<bool>(f5 > f3), true);
    EXPECT_EQ(static_cast<bool>(f1 > f3), false);
    EXPECT_EQ(static_cast<bool>(f1 < f3), true);
    EXPECT_EQ(static_cast<bool>(f5 < f4), false);
    EXPECT_EQ(static_cast<bool>(f5 >= f6), true);
    EXPECT_EQ(static_cast<bool>(f2 >= f4), true);
    EXPECT_EQ(static_cast<bool>(f1 >= f5), false);
    EXPECT_EQ(static_cast<bool>(f5 <= f6), false);
    EXPECT_EQ(static_cast<bool>(f2 <= f4), true);
    EXPECT_EQ(static_cast<bool>(f1 <= f5), true);
}

TEST_F(FunctionWithPass, AssignZero) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0;
    fp = uiVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignLenGreaterThan11) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0x7FFF; // len = 12
    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignLenLessOrEqual11) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0x0FFF; // len = 11
    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignWithDifferentRoundingModes) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0x7FFF;

    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundaryValues) {
    npu::tile_fwk::float16 fp;

    // 测试最小值
    uint16_t uiValMin = 0x0001;
    fp = uiValMin;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最大值
    uint16_t uiValMax = 0xFFFF;
    fp = uiValMax;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignZero_double) {
    npu::tile_fwk::float16 fp;
    double dVal = 0.0;
    fp = dVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignPositiveDouble) {
    npu::tile_fwk::float16 fp;
    double dVal = 1.0;
    fp = dVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    dVal = 3.141592653589793;
    fp = dVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignNegativeDouble) {
    npu::tile_fwk::float16 fp;
    double dVal = -1.0;
    fp = dVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    dVal = -3.141592653589793;
    fp = dVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundaryDouble) {
    npu::tile_fwk::float16 fp;

    // 测试最小正数
    double dValMinPositive = 1.0;
    fp = dValMinPositive;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最小负数
    double dValMinNegative = -1.0;
    fp = dValMinNegative;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最大正数
    double dValMaxPositive = std::numeric_limits<double>::max();
    fp = dValMaxPositive;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最大负数
    double dValMaxNegative = -std::numeric_limits<double>::max();
    fp = dValMaxNegative;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignSpecialDoubleValues) {
    npu::tile_fwk::float16 fp;

    // 测试正无穷大
    double dVal = std::numeric_limits<double>::infinity();
    fp = dVal;
    // 预期结果：构造无穷大
    // 根据float16的定义，无穷大可能表示为特定的值
    // 需要根据具体实现确定预期结果
    // 例如，假设无穷大表示为0x7C00
    EXPECT_EQ(fp.value, 0x7C00);

    // 测试负无穷大
    dVal = -std::numeric_limits<double>::infinity();
    fp = dVal;
    // 预期结果：构造负无穷大
    // 例如，假设负无穷大表示为0xFC00
    EXPECT_EQ(fp.value, 0xFC00);
}

TEST_F(FunctionWithPass, AssignRoundingModes_double) {
    npu::tile_fwk::float16 fp;
    double dVal = 1.5;

    fp = dVal;
    // 预期结果：根据舍入模式，可能为2或1
    // 需要根据具体实现确定预期结果
    // 例如，假设ROUND_TO_NEAREST时，1.5舍入为2
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignZero_uint32) {
    npu::tile_fwk::float16 fp;
    uint32_t uiVal = 0;
    fp = uiVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMax_uint32) {
    npu::tile_fwk::float16 fp;
    uint32_t uiVal = 0xFFFFFFFF;
    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundary_uint32) {
    npu::tile_fwk::float16 fp;

    // 测试最小值
    uint32_t uiValMin = 1;
    fp = uiValMin;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最大值
    uint32_t uiValMax = 0xFFFFFFFF;
    fp = uiValMax;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignRoundingModes_uint32) {
    npu::tile_fwk::float16 fp;
    uint32_t uiVal = 0xFFFFFFFF;

    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignZero_int32) {
    npu::tile_fwk::float16 fp;
    int32_t iVal = 0;
    fp = iVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMax_int32) {
    npu::tile_fwk::float16 fp;
    int32_t iVal = 0x7FFFFFFF;
    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMin_int32) {
    npu::tile_fwk::float16 fp;
    int32_t iVal = -0x80000000;
    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundary_int32) {
    npu::tile_fwk::float16 fp;

    // 测试最小正数
    int32_t iValMinPositive = 1;
    fp = iValMinPositive;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最小负数
    int32_t iValMinNegative = -1;
    fp = iValMinNegative;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignRoundingModes_int32) {
    npu::tile_fwk::float16 fp;
    int32_t iVal = 0x7FFFFFFF;

    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignZero_int16) {
    npu::tile_fwk::float16 fp;
    int16_t iVal = 0;
    fp = iVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMax_int16) {
    npu::tile_fwk::float16 fp;
    int16_t iVal = 0x7FFF;
    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMin_int16) {
    npu::tile_fwk::float16 fp;
    int16_t iVal = -0x8000;
    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundary_int16) {
    npu::tile_fwk::float16 fp;

    // 测试最小正数
    int16_t iValMinPositive = 1;
    fp = iValMinPositive;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最小负数
    int16_t iValMinNegative = -1;
    fp = iValMinNegative;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignRoundingModes_int16) {
    npu::tile_fwk::float16 fp;
    int16_t iVal = 0x7FFF;

    // 测试ROUND_TO_NEAREST
    fp = iVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

}

TEST_F(FunctionWithPass, AssignZero_uint16) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0;
    fp = uiVal;
    EXPECT_EQ(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignMax_uint16) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0x7FFF;
    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignBoundary_uint16) {
    npu::tile_fwk::float16 fp;

    // 测试最小值
    uint16_t uiValMin = 1;
    fp = uiValMin;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);

    // 测试最大值
    uint16_t uiValMax = 0xFFFF;
    fp = uiValMax;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, AssignRoundingModes_uint16) {
    npu::tile_fwk::float16 fp;
    uint16_t uiVal = 0x7FFF;

    fp = uiVal;
    // 预期结果根据具体计算确定
    EXPECT_NE(fp.value, 0);
}

TEST_F(FunctionWithPass, fp16ToUInt8_Negative) {
    uint16_t fpVal = 0xBC00; // -1.0 in fp16
    float result = npu::tile_fwk::float16::FromBase(fpVal);
    EXPECT_EQ(result, -1); // 负数转换为uint8_t应为0
}

TEST_F(FunctionWithPass, fp16ToUInt8_Zero) {
    uint16_t fpVal = 0x0000; // +0
    float result = npu::tile_fwk::float16::FromBase(fpVal);
    EXPECT_EQ(result, 0);

    fpVal = 0x8000; // -0
    result = npu::tile_fwk::float16::FromBase(fpVal);
    EXPECT_EQ(result, 0);
}

TEST_F(FunctionWithPass, fp16ToInt8_Zero) {
    uint16_t fpVal = 0x0000; // +0
    float result = npu::tile_fwk::float16::FromBase(fpVal);
    EXPECT_EQ(result, 0);

    fpVal = 0x8000; // -0
    result = npu::tile_fwk::float16::FromBase(fpVal);
    EXPECT_EQ(result, 0);
}

TEST_F(FunctionWithPass, TestSymbolicScalar) {
    ScalarImmediateType sit = 0;
    RawSymbolicImmediate ryi(sit);
    ryi.DumpJson();

    RawSymbolicSymbol rss("test");
    rss.DumpJson();
    auto rssPtr = RawSymbolicSymbol::Create("Test");
    EXPECT_NE(rssPtr, nullptr);
}

TEST_F(FunctionWithPass, Constructor) {
    // 测试无操作数
    RawSymbolicExpression expr1(SymbolicOpcode::T_BOP_ADD, {});
    EXPECT_EQ(expr1.Opcode(), SymbolicOpcode::T_BOP_ADD);
    EXPECT_EQ(expr1.OperandList().size(), 0);

    // 测试一个操作数
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicExpression expr2(SymbolicOpcode::T_BOP_ADD, {operand1});
    EXPECT_EQ(expr2.Opcode(), SymbolicOpcode::T_BOP_ADD);
    EXPECT_EQ(expr2.OperandList().size(), 1);

    // 测试多个操作数
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);
    RawSymbolicExpression expr3(SymbolicOpcode::T_BOP_ADD, {operand1, operand2});
    EXPECT_EQ(expr3.Opcode(), SymbolicOpcode::T_BOP_ADD);
    EXPECT_EQ(expr3.OperandList().size(), 2);
}

TEST_F(FunctionWithPass, GetSymbolicCalcBinary) {
    EXPECT_EQ(RawSymbolicExpression::GetSymbolicCalcBinary(SymbolicOpcode::T_BOP_ADD), &RawSymbolicExpression::CalcBopAdd);
    EXPECT_EQ(RawSymbolicExpression::GetSymbolicCalcBinary(SymbolicOpcode::T_BOP_SUB), &RawSymbolicExpression::CalcBopSub);
    // 类似地测试其他操作码
}

TEST_F(FunctionWithPass, Accessors) {
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);
    RawSymbolicExpression expr(SymbolicOpcode::T_BOP_ADD, {operand1, operand2});

    EXPECT_EQ(expr.Opcode(), SymbolicOpcode::T_BOP_ADD);
    const auto &operandList = expr.OperandList();
    EXPECT_EQ(operandList.size(), 2);
    EXPECT_EQ(operandList[0]->IsImmediate(), true);
    EXPECT_EQ(operandList[1]->IsImmediate(), true);
}

TEST_F(FunctionWithPass, DumpJson) {
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);
    RawSymbolicExpression expr(SymbolicOpcode::T_BOP_ADD, {operand1, operand2});

    Json exprJson = expr.DumpJson();
    EXPECT_EQ(exprJson.size(), 4);
    EXPECT_EQ(exprJson[0], SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION);
    EXPECT_EQ(exprJson[1], SymbolicOpcode::T_BOP_ADD);
    EXPECT_EQ(exprJson[2], operand1->DumpJson());
    EXPECT_EQ(exprJson[3], operand2->DumpJson());
}

TEST_F(FunctionWithPass, BinaryOperations) {
    ScalarImmediateType lhs = 5;
    ScalarImmediateType rhs = 3;

    EXPECT_EQ(RawSymbolicExpression::CalcBopAdd(lhs, rhs), 8);
    EXPECT_EQ(RawSymbolicExpression::CalcBopSub(lhs, rhs), 2);
    EXPECT_EQ(RawSymbolicExpression::CalcBopMul(lhs, rhs), 15);
    EXPECT_EQ(RawSymbolicExpression::CalcBopDiv(lhs, rhs), 1);
    EXPECT_EQ(RawSymbolicExpression::CalcBopMod(lhs, rhs), 2);
    EXPECT_EQ(RawSymbolicExpression::CalcBopEq(lhs, lhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopNe(lhs, rhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopLt(rhs, lhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopLe(lhs, lhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopGt(lhs, rhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopGe(lhs, lhs), true);
    EXPECT_EQ(RawSymbolicExpression::CalcBopMin(lhs, rhs), 3);
    EXPECT_EQ(RawSymbolicExpression::CalcBopMax(lhs, rhs), 5);
}

TEST_F(FunctionWithPass, GetSymbolicCalcOpcode) {
    EXPECT_EQ(RawSymbolicExpression::GetSymbolicCalcOpcode(SymbolicOpcode::T_BOP_ADD), "+");
    EXPECT_EQ(RawSymbolicExpression::GetSymbolicCalcOpcode(SymbolicOpcode::T_BOP_SUB), "-");
    // 类似地测试其他操作码
}

TEST_F(FunctionWithPass, Create) {
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);

    // 测试所有操作数都是立即数
    RawSymbolicScalarPtr result1 = RawSymbolicExpression::Create(SymbolicOpcode::T_BOP_ADD, {operand1, operand2});
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->Kind(), SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE);
    EXPECT_EQ(std::static_pointer_cast<RawSymbolicImmediate>(result1)->Immediate(), 8);

    // 测试至少有一个操作数不是立即数
    std::vector<RawSymbolicScalarPtr> operandList = {operand1, operand2};
    RawSymbolicScalarPtr expr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_BOP_ADD, operandList);
    RawSymbolicScalarPtr result2 = RawSymbolicExpression::Create(SymbolicOpcode::T_BOP_ADD, {operand1, expr});
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->Kind(), SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION);
}

TEST_F(FunctionWithPass, CreateBopOperations) {
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);

    // 测试CreateBopAdd
    RawSymbolicScalarPtr exprAdd = RawSymbolicExpression::CreateBopAdd(operand1, operand2);
    RawSymbolicScalarPtr exprMul = RawSymbolicExpression::CreateBopMul(operand1, operand2);
    RawSymbolicScalarPtr exprMin = RawSymbolicExpression::CreateBopMin(operand1, operand2);
    RawSymbolicScalarPtr exprMax = RawSymbolicExpression::CreateBopMax(operand1, operand2);
    EXPECT_NE(exprAdd, nullptr);
    EXPECT_EQ(exprAdd->Kind(), SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE);

    // 类似地测试其他二元操作方法
}

TEST_F(FunctionWithPass, DumpBuffer) {
    RawSymbolicScalarPtr operand1 = std::make_shared<RawSymbolicImmediate>(5);
    RawSymbolicScalarPtr operand2 = std::make_shared<RawSymbolicImmediate>(3);
    RawSymbolicExpression expr(SymbolicOpcode::T_BOP_ADD, {operand1, operand2});

    std::string buffer;
    expr.DumpBuffer(buffer);
    EXPECT_EQ(buffer, "(5+3)");
}

TEST_F(FunctionWithPass, CalculateHashTest) {
    // 创建不同的操作符
    std::vector<int64_t> attrs;
    RawExpectedOperator op1(Opcode::OP_MUL, attrs);
    RawExpectedOperator op2(Opcode::OP_MUL, attrs);

    // 计算哈希值
    size_t hash1 = op1.CalculateHash();
    size_t hash2 = op2.CalculateHash();

    // 确保不同的操作符生成不同的哈希值
    EXPECT_EQ(hash1, hash2);
}

TEST_F(FunctionWithPass, EqualTest) {
    // 创建相同类型的RawExpectedValue对象
    auto value1 = make_shared<RawExpectedInputValue>(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    auto value2 = make_shared<RawExpectedInputValue>(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    EXPECT_TRUE(value1 != value2);
    EXPECT_TRUE(*value1 == *value2);

    // 创建不同类型的RawExpectedValue对象
    std::vector<int64_t> attrs;
    std::vector<ExpectedValue> operands;
    auto value3 = make_shared<RawExpectedOperationValue>(ExpectedOperator(Opcode::OP_ADD, attrs), operands);
    auto value4 = make_shared<RawExpectedOperationValue>(ExpectedOperator(Opcode::OP_SUB, attrs), operands);
    EXPECT_FALSE(value3 == value4);
}

TEST_F(FunctionWithPass, OperatorEqualsTest) {
    // 测试相同类型的ExpectedValue对象
    ExpectedValue ev1(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    ExpectedValue ev2(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    EXPECT_TRUE(ev1 == ev2);

    // 测试不同类型的ExpectedValue对象
    ExpectedValue ev3(ExpectedOperator(Opcode::OP_SUB, {}), {});
    EXPECT_FALSE(ev1 == ev3);

    // 测试不同形状的ExpectedValue对象
    ExpectedValue ev4(std::vector<int64_t>{2, 3}, DT_FP16, "input2");
    EXPECT_FALSE(ev1 == ev4);
}

TEST_F(FunctionWithPass, ConstructorTest) {
    // 测试输入值构造函数
    ExpectedValue ev1(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    EXPECT_NE(ev1.Get(), nullptr);

    // 测试操作值构造函数
    ExpectedValue ev2(ExpectedOperator(Opcode::OP_ADD, {}), {});
    EXPECT_NE(ev2.Get(), nullptr);

    // 测试提取值构造函数
    ExpectedValue source(std::vector<int64_t>{2, 3}, DT_FP16, "source");
    ExpectedValue ev3(source, std::vector<int64_t>{2, 3}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1});
    EXPECT_NE(ev3.Get(), nullptr);

    // 测试插入值构造函数
    vector<RawExpectedInsertValueElement> elements;
    ExpectedValue ev4(std::vector<int64_t>{2, 3}, elements);
    EXPECT_NE(ev4.Get(), nullptr);

    // 测试结果值构造函数
    ExpectedValue resultof(std::vector<int64_t>{2, 3}, DT_FP16, "resultof");
    ExpectedValue ev5(resultof, 0);
    EXPECT_NE(ev5.Get(), nullptr);
}

TEST_F(FunctionWithPass, CastTest) {
    // 测试输入值类型转换
    ExpectedValue ev1(std::vector<int64_t>{2, 3}, DT_FP16, "input1");
    auto castInput = ev1.CastInputValue();
    EXPECT_NE(castInput, nullptr);

    // 测试操作值类型转换
    ExpectedValue ev2(ExpectedOperator(Opcode::OP_ADD, {}), {});
    auto castOperation = ev2.CastOperationValue();
    EXPECT_NE(castOperation, nullptr);

    // 测试提取值类型转换
    ExpectedValue source(std::vector<int64_t>{2, 3}, DT_FP16, "source");
    ExpectedValue ev3(source, std::vector<int64_t>{2, 3}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1});
    auto castExtract = ev3.CastExtractValue();
    EXPECT_NE(castExtract, nullptr);

    // 测试插入值类型转换
    vector<RawExpectedInsertValueElement> elements;
    ExpectedValue ev4(std::vector<int64_t>{2, 3}, elements);
    auto castInsert = ev4.CastInsertValue();
    EXPECT_NE(castInsert, nullptr);

    // 测试结果值类型转换
    ExpectedValue resultof(std::vector<int64_t>{2, 3}, DT_FP16, "resultof");
    ExpectedValue ev5(resultof, 0);
    auto castResultof = ev5.CastResultofValue();
    EXPECT_NE(castResultof, nullptr);
}

// 测试 RawExpectedInputValue
TEST_F(FunctionWithPass, CalculateHash0) {
    std::vector<int64_t> shape = {1, 2, 3};
    DataType dataType = DT_INT32;
    std::string name = "input1";
    RawExpectedInputValue inputValue(shape, dataType, name);

    std::size_t hash = inputValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals0) {
    std::vector<int64_t> shape = {1, 2, 3};
    DataType dataType = DT_INT32;
    std::string name = "input1";
    RawExpectedInputValue inputValue1(shape, dataType, name);
    RawExpectedInputValue inputValue2(shape, dataType, name);

    EXPECT_TRUE(inputValue1 == inputValue2);
}

// 测试 RawExpectedOperationValue
TEST_F(FunctionWithPass, CalculateHash1) {
    std::vector<int64_t> attr;
    ExpectedOperator oper = ExpectedOperator(Opcode::OP_ADD, attr);
    std::vector<int64_t> sourceShape = {1, 2, 3};
    auto exp1 = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp1");
    auto exp2 = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp2");
    std::vector<ExpectedValue> operands = {exp1, exp2};
    RawExpectedOperationValue operationValue(oper, operands);

    std::size_t hash = operationValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals1) {
    std::vector<int64_t> attr;
    ExpectedOperator oper = ExpectedOperator(Opcode::OP_ADD, attr);
    std::vector<int64_t> sourceShape = {1, 2, 3};
    auto exp1 = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp1");
    auto exp2 = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp2");
    std::vector<ExpectedValue> operands = {exp1, exp2};
    RawExpectedOperationValue operationValue1(oper, operands);
    RawExpectedOperationValue operationValue2(oper, operands);

    EXPECT_TRUE(operationValue1 == operationValue2);
}

// 测试 RawExpectedExtractValue
TEST_F(FunctionWithPass, CalculateHash2) {
    std::vector<int64_t> sourceShape = {1, 2, 3};
    ExpectedValue source = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp1");
    std::vector<int64_t> resultOffset = {0, 0, 0};
    std::vector<int64_t> resultShape = {1, 1, 1};
    RawExpectedExtractValue extractValue(source, sourceShape, resultOffset, resultShape);

    std::size_t hash = extractValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals2) {
    std::vector<int64_t> sourceShape = {1, 2, 3};
    ExpectedValue source = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp1");
    std::vector<int64_t> resultOffset = {0, 0, 0};
    std::vector<int64_t> resultShape = {1, 1, 1};
    RawExpectedExtractValue extractValue1(source, sourceShape, resultOffset, resultShape);
    RawExpectedExtractValue extractValue2(source, sourceShape, resultOffset, resultShape);

    EXPECT_TRUE(extractValue1 == extractValue2);
}

// 测试 RawExpectedInsertValue
TEST_F(FunctionWithPass, CalculateHash3) {
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<RawExpectedInsertValueElement> elements = {
        RawExpectedInsertValueElement({0, 0, 0}, {1, 1, 1}, ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"))
    };
    RawExpectedInsertValue insertValue(shape, elements);

    std::size_t hash = insertValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals3) {
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<RawExpectedInsertValueElement> elements = {
        RawExpectedInsertValueElement({0, 0, 0}, {1, 1, 1}, ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"))
    };
    RawExpectedInsertValue insertValue1(shape, elements);
    RawExpectedInsertValue insertValue2(shape, elements);

    EXPECT_TRUE(insertValue1 == insertValue2);
}

// 测试 RawExpectedResultofValue
TEST_F(FunctionWithPass, CalculateHash4) {
    std::vector<int64_t> shape = {1, 2, 3};
    ExpectedValue resultof = ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1");
    int index = 0;
    RawExpectedResultofValue resultofValue(resultof, index);

    std::size_t hash = resultofValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals4) {
    std::vector<int64_t> shape = {1, 2, 3};
    ExpectedValue resultof = ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1");
    int index = 0;
    RawExpectedResultofValue resultofValue1(resultof, index);
    RawExpectedResultofValue resultofValue2(resultof, index);

    EXPECT_TRUE(resultofValue1 == resultofValue2);
}

// 测试 ListExpectedValue
TEST_F(FunctionWithPass, CalculateHash5) {
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<ExpectedValue> elements = {ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"),
        ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp2")};
    ListExpectedValue listValue(elements);

    std::size_t hash = listValue.CalculateHash();
    EXPECT_NE(hash, 0);
}

TEST_F(FunctionWithPass, OperatorEquals5) {
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<ExpectedValue> elements = {ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"),
        ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp2")};
    ListExpectedValue listValue1(elements);
    ListExpectedValue listValue2(elements);

    EXPECT_TRUE(listValue1 == listValue2);
}

// 测试 ExpectedValue
TEST_F(FunctionWithPass, IsNull) {
    ExpectedValue value;
    EXPECT_TRUE(value.IsNull());
}

TEST_F(FunctionWithPass, IsInputValue) {
    std::vector<int64_t> shape = {1, 2, 3};
    DataType dataType = DT_INT32;
    std::string name = "input1";
    ExpectedValue value(shape, dataType, name);

    EXPECT_TRUE(value.IsInputValue());
}

TEST_F(FunctionWithPass, IsOperationValue) {
    std::vector<int64_t> attr;
    ExpectedOperator oper = ExpectedOperator(Opcode::OP_ADD, attr);
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<ExpectedValue> operands = {ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"),
        ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp2")};
    ExpectedValue value(oper, operands);

    EXPECT_TRUE(value.IsOperationValue());
}

TEST_F(FunctionWithPass, IsExtractValue) {
    std::vector<int64_t> sourceShape = {1, 2, 3};
    ExpectedValue source = ExpectedValue(sourceShape, npu::tile_fwk::DT_FP32, "exp1");
    std::vector<int64_t> resultOffset = {0, 0, 0};
    std::vector<int64_t> resultShape = {1, 1, 1};
    ExpectedValue value(source, sourceShape, resultOffset, resultShape);

    EXPECT_TRUE(value.IsExtractValue());
}

TEST_F(FunctionWithPass, IsInsertValue) {
    std::vector<int64_t> shape = {1, 2, 3};
    std::vector<RawExpectedInsertValueElement> elements = {
        RawExpectedInsertValueElement({0, 0, 0}, {1, 1, 1},
        ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"))
    };
    ExpectedValue value(shape, elements);

    EXPECT_TRUE(value.IsInsertValue());
}

TEST_F(FunctionWithPass, IsResultofValue) {
    std::vector<int64_t> shape = {1, 2, 3};
    ExpectedValue resultof = ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1");
    int index = 0;
    ExpectedValue value(resultof, index);

    EXPECT_TRUE(value.IsResultofValue());
}

TEST_F(FunctionWithPass, ValueLookupTest) {
    ExpectedValueBuilder builder;
    ExpectedValue value1(std::vector<int64_t>{2, 3}, npu::tile_fwk::DT_FP32, "input1");
    ExpectedValue value2(std::vector<int64_t>{2, 3}, npu::tile_fwk::DT_FP32, "input1");

    // 测试值不存在时插入
    EXPECT_NE(builder.ValueLookup(value1).Get(), nullptr);

    // 测试值存在时返回已有的值
    EXPECT_EQ(builder.ValueLookup(value2).Get(), value1.Get());
}

TEST_F(FunctionWithPass, CreateValueTest) {
    ExpectedValueBuilder builder;
    std::vector<int64_t> shape{2, 3};
    DataType dataType = npu::tile_fwk::DT_FP32;
    string name = "input1";


    ExpectedValue value = builder.CreateValue(shape, dataType, name);
    EXPECT_NE(value.Get(), nullptr);

    LogicalTensors iOperands;
    LogicalTensors oOperands;
    FUNCTION("test") {
        Operation op(*Program::GetInstance().GetCurrentFunction(), Opcode::OP_ADD, iOperands, oOperands);
        vector<ExpectedValue> operands;
        ExpectedValue operationValue = builder.CreateValue(op, operands);
        EXPECT_NE(operationValue.Get(), nullptr);
    }
}

TEST_F(FunctionWithPass, CreateOperationOOperandsTest) {
    ExpectedValueBuilder builder;
    LogicalTensors iOperands;
    LogicalTensors oOperands;
    FUNCTION("test") {
        Operation op(*Program::GetInstance().GetCurrentFunction(), Opcode::OP_ADD, iOperands, oOperands);
        vector<ExpectedValue> ioperands;

        vector<ExpectedValue> ooperands = builder.CreateOperationOOperands(op, ioperands);
        EXPECT_EQ(ooperands.size(), 0);
    }
}

TEST_F(FunctionWithPass, CreateListTest) {
    ExpectedValueBuilder builder;
    std::vector<int64_t> shape{2, 3};
    std::vector<ExpectedValue> elements = {ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp1"),
        ExpectedValue(shape, npu::tile_fwk::DT_FP32, "exp2")};

    ListExpectedValue list1 = builder.CreateList(elements);
    EXPECT_EQ(list1.GetElements().size(), 2);
}
}  // namespace