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
 * \file test_interp_calc.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
class TorchAdaptorTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t> &shape, const T &val) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int64_t> &shape, const std::vector<T> &vals) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

#define ASSERT_ALLCLOSE(self, other) \
    ASSERT(calc::AllClose(self, other)) << "lhs:\n" << self->ToString() << "\nrhs:\n" << other->ToString() << "\n"

TEST_F(TorchAdaptorTest, LogicalNot) {
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_BOOL, {16, 16}, true);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, false);
        calc::LogicalNot(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        auto out = makeTensorData(DT_BOOL, {16, 16}, true);
        auto golden = makeTensorData(DT_BOOL, {16, 16}, false);
        calc::LogicalNot(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, LogicalAnd) {
    auto self = makeTensorData(DT_BOOL, {16, 16}, true);
    auto other = makeTensorData(DT_BOOL, {16, 16}, true);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden = makeTensorData(DT_BOOL, {16, 16}, true);
    calc::LogicalAnd(out, self, other);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Range) {
    std::vector<float> gdata = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    auto out = makeTensorData(DT_FP32, {7}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {7}, gdata);
    calc::Range(out, Element(DT_FP32, 1.1f), Element(DT_INT32, 8), Element(DT_FP32, 1.1f));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Compare) {
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden_true = makeTensorData(DT_BOOL, {16, 16}, true);
    auto golden_false = makeTensorData(DT_BOOL, {16, 16}, false);

    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BOOL, true},
        {CmpOperationType::NE, CmpModeType::BOOL, false},
        {CmpOperationType::LT, CmpModeType::BOOL, false},
        {CmpOperationType::LE, CmpModeType::BOOL, true},
        {CmpOperationType::GT, CmpModeType::BOOL, false},
        {CmpOperationType::GE, CmpModeType::BOOL, true},
    };
    for (const auto &test : cases) {
        calc::Compare(out, self, other, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_true);
        } else {
            ASSERT_ALLCLOSE(out, golden_false);
        }
    }
}

TEST_F(TorchAdaptorTest, CompareBit) {
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto out = makeTensorData(DT_UINT8, {16, 2}, false);
    auto golden_1 = makeTensorData(DT_UINT8, {16, 2}, (uint8_t)0xFF);
    auto golden_0 = makeTensorData(DT_UINT8, {16, 2}, (uint8_t)0);

    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BIT, true},
        {CmpOperationType::NE, CmpModeType::BIT, false},
        {CmpOperationType::LT, CmpModeType::BIT, false},
        {CmpOperationType::LE, CmpModeType::BIT, true},
        {CmpOperationType::GT, CmpModeType::BIT, false},
        {CmpOperationType::GE, CmpModeType::BIT, true},
    };
    for (const auto &test : cases) {
        calc::Compare(out, self, other, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_1);
        } else {
            ASSERT_ALLCLOSE(out, golden_0);
        }
    }
}

TEST_F(TorchAdaptorTest, Cmps) {
    auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    auto elem = Element(DT_FP32, 4.0f);
    auto out = makeTensorData(DT_BOOL, {16, 16}, false);
    auto golden_true = makeTensorData(DT_BOOL, {16, 16}, true);
    auto golden_false = makeTensorData(DT_BOOL, {16, 16}, false);
    struct {
        CmpOperationType type;
        CmpModeType mode;
        bool expect;
    } cases[] = {
        {CmpOperationType::EQ, CmpModeType::BOOL, true},
        {CmpOperationType::NE, CmpModeType::BOOL, false},
        {CmpOperationType::LT, CmpModeType::BOOL, false},
        {CmpOperationType::LE, CmpModeType::BOOL, true},
        {CmpOperationType::GT, CmpModeType::BOOL, false},
        {CmpOperationType::GE, CmpModeType::BOOL, true},
    };
    for (const auto &test : cases) {
        calc::Cmps(out, self, elem, test.type, test.mode);
        if (test.expect) {
            ASSERT_ALLCLOSE(out, golden_true);
        } else {
            ASSERT_ALLCLOSE(out, golden_false);
        }
    }
}
TEST_F(TorchAdaptorTest, Ceil) {
    // ceil
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
    calc::Ceil(out, self);
    ASSERT_ALLCLOSE(out, golden);
}
TEST_F(TorchAdaptorTest, Floor) {
    // floor
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    calc::Floor(out, self);
    ASSERT_ALLCLOSE(out, golden);
}
TEST_F(TorchAdaptorTest, Trunc) {
    // trunc
    auto self = makeTensorData(DT_FP32, {16, 16}, 1.1f);
    auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
    calc::Trunc(out, self);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, UnaryOps) {
    {
        // rsqrt
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.5f);
        calc::Rsqrt(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sqrt
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Sqrt(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // abs
        auto self = makeTensorData(DT_FP32, {16, 16}, -4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::Abs(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // ln
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::log(2.0f));
        calc::Ln(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // exp
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::exp(2.0f));
        calc::Exp(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // neg
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, -2.0f);
        calc::Neg(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 1e8f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(static_cast<int>(1e8f)));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand scalar
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::ExpandS(out, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast and cast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast (torch modes) - integer targets with ties/non-ties
        // values: +2.5, -2.5, +2.4, -2.4, +2.6, -2.6
        std::vector<float> vals = {2.5f, -2.5f, 2.4f, -2.4f, 2.6f, -2.6f};
        auto self = makeTensorData(DT_FP32, {6, 1}, vals);

        // default (integer dst): torch.round behavior
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 3, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self);
            ASSERT_ALLCLOSE(out, golden);
        }
        // explicit CAST_ROUND
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 3, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_ROUND);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_FLOOR
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -3, 2, -3, 2, -3};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_FLOOR);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_CEIL
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {3, -2, 3, -2, 3, -2};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_CEIL);
            ASSERT_ALLCLOSE(out, golden);
        }
        // CAST_TRUNC
        {
            auto out = makeTensorData(DT_INT32, {6, 1}, 0);
            std::vector<int32_t> exp = {2, -2, 2, -2, 2, -2};
            auto golden = makeTensorData(DT_INT32, {6, 1}, exp);
            calc::Cast(out, self, CAST_TRUNC);
            ASSERT_ALLCLOSE(out, golden);
        }
        // float targets: pass-through
        {
            auto out = makeTensorData(DT_FP32, {6, 1}, 0.0f);
            auto golden = makeTensorData(DT_FP32, {6, 1}, vals);
            calc::Cast(out, self);
            ASSERT_ALLCLOSE(out, golden);
        }
        // brcb 
        {
            std::vector<float> sdata = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> gdata = {1.0f, 1.0f, 1.0f,
                                        2.0f, 2.0f, 2.0f,
                                        3.0f, 3.0f, 3.0f,
                                        4.0f, 4.0f, 4.0f};
            auto self_brcb = makeTensorData(DT_FP32, {4, 1}, sdata);
            auto out = makeTensorData(DT_FP32, {4, 3}, 0.0f);
            auto golden = makeTensorData(DT_FP32, {4, 3}, gdata);
            calc::Brcb(out, self_brcb);
            ASSERT_ALLCLOSE(out, golden);
    }
    }
}

TEST_F(TorchAdaptorTest, BinaryOps) {
    {
        // add
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }   
    {
        // sub
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::Sub(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::Sub(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // mul
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::Mul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::Mul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // div
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::Div(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::Div(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // fmod
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        calc::Fmod(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // fmod broadcast
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        calc::Fmod(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // add broadcast
        auto self = makeTensorData(DT_FP32, {1, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise max
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Max(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 6.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        calc::Max(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise min
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Min(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Min(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar min
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {1.0, 2.0, 2.0, 2.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MinS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar max
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MaxS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter update 2dim
        std::vector<float> sdata = {1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,};
        std::vector<float> gdata = {0.0f, 0.0f, 0.0f,
                                    16.0f, 17.0f, 18.0f,
                                    0.0f, 0.0f, 0.0f,
                                    1.0f, 2.0f, 3.0f,
                                    7.0f, 8.0f, 9.0f,
                                    4.0f, 5.0f, 6.0f,
                                    0.0f, 0.0f, 0.0f,
                                    13.0f, 14.0f, 15.0f,
                                    10.0f, 11.0f, 12.0f,
                                    0.0f, 0.0f, 0.0f, };
        std::vector<int64_t> idata = {3, 5, 4, 8, 7, 1};
        auto self = makeTensorData(DT_FP32, {6, 3}, sdata);
        auto dst = makeTensorData(DT_FP32, {10, 3}, 0.0f);
        auto out = makeTensorData(DT_FP32, {10, 3}, 0.0f);
        auto index = makeTensorData(DT_INT64, {2, 3}, idata);
        auto golden = makeTensorData(DT_FP32, {10, 3}, gdata);
        calc::ScatterUpdate(out, self, index, dst);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter update 4dim
        std::vector<float> sdata = {1.0f, 1.0f, 1.0f, 1.0f,
                                    2.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 3.0f,
                                    4.0f, 4.0f, 4.0f, 4.0f};
        std::vector<float> gdata = {0.0f, 0.0f, 0.0f, 0.0f,
                                    4.0f, 4.0f, 4.0f, 4.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f,
                                    3.0f, 3.0f, 3.0f, 3.0f,
                                    2.0f, 2.0f, 2.0f, 2.0f};
        std::vector<int64_t> idata = {3, 5, 4, 1};
        auto self = makeTensorData(DT_FP32, {2, 2, 1, 4}, sdata);
        auto dst = makeTensorData(DT_FP32, {3, 2, 1, 4}, 0.0f);
        auto out = makeTensorData(DT_FP32, {3, 2, 1, 4}, 0.0f);
        auto index = makeTensorData(DT_INT64, {2, 2}, idata);
        auto golden = makeTensorData(DT_FP32, {3, 2, 1, 4}, gdata);
        calc::ScatterUpdate(out, self, index, dst, -2, "PA_BSND", 2); // blocksize设置为2
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter replace
        std::vector<float> selfData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f,};
        std::vector<int64_t> indicesData = {1, 0, 1, 1,};
        std::vector<float> gdata = {1.0f, 2.0f, 1.0f, 1.0f, 1.0f,
                                    2.0f, 1.0f, 2.0f, 2.0f, 1.0f,};
        auto src = Element(DT_FP32, 2.0f);
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::ScatterElement(out, self, indices, src, 0, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter add
        std::vector<float> selfData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f,};
        std::vector<int64_t> indicesData = {1, 0, 1, 1,};
        std::vector<float> gdata = {1.0f, 3.0f, 1.0f, 1.0f, 1.0f,
                                    3.0f, 1.0f, 3.0f, 3.0f, 1.0f,};
        auto src = Element(DT_FP32, 2.0f);
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::ScatterElement(out, self, indices, src, 0, 1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scatter tensor replace
        std::vector<float> selfData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f,};
        std::vector<int64_t> indicesData = {1, 0, 1, 1,};
        std::vector<float> srcData = {10, 11, 12, 13,};
        std::vector<float> gdata = {1.0f,  11.0f, 1.0f,  1.0f,  1.0f,
                                    10.0f, 1.0f,  12.0f, 13.0f, 1.0f,};
        auto self = makeTensorData(DT_FP32, {2, 5}, selfData);
        auto indices = makeTensorData(DT_INT64, {1, 4}, indicesData);
        auto src = makeTensorData(DT_FP32, {1, 4}, srcData);
        auto out = makeTensorData(DT_FP32, {2, 5}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 5}, gdata);
        calc::Scatter(out, self, indices, src, 0, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryOpsS) {
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::AddS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::SubS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::MulS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::DivS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, -3.0f);
        calc::SubS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 0.4f);
        calc::DivS(out, self, elem, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        calc::FmodS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BitwiseShift) {
    {
        // bitwiserightshift
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::BitwiseRightShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseleftshift
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::BitwiseLeftShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiserightshifts
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = Element(DT_INT16, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::BitwiseRightShiftS(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // bitwiseleftshifts
        auto self = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(4));
        auto other = Element(DT_INT16, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::BitwiseLeftShiftS(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sbitwiserightshift
        auto self = Element(DT_INT16, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::SBitwiseRightShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sbitwiseleftshift
        auto self = Element(DT_INT16, static_cast<int16_t>(4));
        auto other = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(1));
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(8));
        calc::SBitwiseLeftShift(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Where) {
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereTT(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_FP32, {16, 16}, 6.0f);
        auto other = Element(DT_FP32, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereTS(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = Element(DT_FP32, 6.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereST(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = Element(DT_FP32, 6.0f);
        auto other = Element(DT_FP32, 4.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::WhereSS(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto out = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(0.0f));
        auto condition = makeTensorData(DT_BOOL, {16, 16}, false);
        auto input = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(6.0f));
        auto other = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        auto golden = makeTensorData(DT_BF16, {16, 16}, static_cast<bfloat16>(4.0f));
        calc::WhereTT(out, condition, input, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

LogicalTensorDataPtr makePartialGolden(int n, int p, float v1, float v2) {
    std::vector<float> ret(n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ret[i *n + j] = j < p ? v1 : v2;
        }
    }
    return makeTensorData(DT_FP32, {n, n}, ret);
}

TEST_F(TorchAdaptorTest, BinaryPairOps) {
    int n = 16, p = 5;
    {
        auto self = makeTensorData(DT_FP32, {n, n}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, p}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 7.0, 4.0);
        calc::PairSum(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {n, p}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, n}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 7.0, 3.0);
        calc::PairSum(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
       auto self = makeTensorData(DT_FP32, {n, n}, 4.0f);
        auto other = makeTensorData(DT_FP32, {n, p}, 3.0f);
        auto out = makeTensorData(DT_FP32, {n, n}, 0.0f);
        auto golden = makePartialGolden(n, p, 3.0, 4.0);
        calc::PairMin(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, MatMul) {
    {
        // matmul
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        MatMulParam param = {false, false, 4};
        calc::MatMul(out, self, other, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul bt
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        MatMulParam param = {false, true, 0};
        calc::MatMul(out, self, other, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul bt splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        MatMulParam param = {false, true, 4};
        calc::MatMul(out, self, other, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul(out, self, other, out);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        MatMulParam param = {false, false, 4};
        calc::AccMatMul(out, self, other, out, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc bt
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        MatMulParam param = {false, true, 0};
        calc::AccMatMul(out, self, other, out, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc bt splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        MatMulParam param = {false, true, 4};
        calc::AccMatMul(out, self, other, out, param);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul fp16 @ fp16 -> fp32
        auto self = makeTensorData(DT_FP16, {8, 16}, float16(1.0));
        auto other = makeTensorData(DT_FP16, {16, 8}, float16(1.0));
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul fp16 @ fp16 -> fp16
        auto self = makeTensorData(DT_FP16, {8, 16}, float16(1.0));
        auto other = makeTensorData(DT_FP16, {16, 8}, float16(1.0));
        auto out = makeTensorData(DT_FP16, {8, 8}, float16(1.0f));
        auto golden = makeTensorData(DT_FP16, {8, 8}, float16(16.0f));
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Reduce) {
    {
        // sum expand
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 16.0f);
        calc::RowSumExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }

    {
        // sum
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 1}, 16.0f);
        calc::RowSumSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // min expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 1.0, 4.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMinExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // minsingle
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMinSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // minline
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 16}, 1.0f);
        calc::RowMinLine(out, self, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // max expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMaxExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // maxsingle
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMaxSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // maxline
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {1, 16}, 1.0f);
        calc::RowMaxLine(out, self, 0);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // reduce acc
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::ReduceAcc(out, {self, self, self, self});
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Misc) {
    {
        // reshape
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Reshape(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // transpose
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Transpose(out, self, -1, -2);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {3, 2}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy trans
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // view
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                auto v = out->View({4, 4}, {i * 4, j * 4});
                calc::ExpandS(v, Element(DT_FP32, i * 2.0f + j));
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                auto v = out->View({4, 4}, {i * 4, j * 4});
                auto g = makeTensorData(DT_FP32, {4, 4}, i * 2.0f + j);
                ASSERT(calc::AllClose(v, g)) << v << "\n" << g;
            }
        }
    }
}

TEST_F(TorchAdaptorTest, BitSortDescending) {
    // 降序
    std::vector<float> sdata = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                                0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                                80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                                160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0,
                                240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata = {31.0, 31.0, 30.0, 30.0, 29.0, 29.0, 28.0, 28.0,
                                27.0, 27.0, 26.0, 26.0, 25.0, 25.0, 24.0, 24.0,
                                23.0, 23.0, 22.0, 22.0, 21.0, 21.0, 20.0, 20.0,
                                19.0, 19.0, 18.0, 18.0, 17.0, 17.0, 16.0, 16.0,
                                15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0, 12.0,
                                11.0, 11.0, 10.0, 10.0, 9.0, 9.0, 8.0, 8.0,
                                7.0, 7.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0,
                                3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0,
                                310.0, 31.0, 300.0, 30.0, 290.0, 29.0, 280.0, 28.0,
                                270.0, 27.0, 260.0, 26.0, 250.0, 25.0, 240.0, 24.0,
                                230.0, 23.0, 220.0, 22.0, 210.0, 21.0, 200.0, 20.0,
                                190.0, 19.0, 180.0, 18.0, 170.0, 17.0, 160.0, 16.0,
                                150.0, 15.0, 140.0, 14.0, 130.0, 13.0, 120.0, 12.0,
                                110.0, 11.0, 100.0, 10.0, 90.0, 9.0, 80.0, 8.0,
                                70.0, 7.0, 60.0, 6.0, 50.0, 5.0, 40.0, 4.0,
                                30.0, 3.0, 20.0, 2.0, 10.0, 1.0, 0.0, 0.0};
    auto self = makeTensorData(DT_FP32, {2, 32}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 128}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    calc::BitSort(out, self, -1, true);
    ASSERT_ALLCLOSE(out->View({2, 64}, {0, 0}), golden);
}

TEST_F(TorchAdaptorTest, BitSortAscending) {
    // 升序
    std::vector<float> sdata = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                                0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                                80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                                160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0,
                                240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata = {0.0, 0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0,
                                -4.0, 4.0, -5.0, 5.0, -6.0, 6.0, -7.0, 7.0,
                                -8.0, 8.0, -9.0, 9.0, -10.0, 10.0, -11.0, 11.0,
                                -12.0, 12.0, -13.0, 13.0, -14.0, 14.0, -15.0, 15.0,
                                -16.0, 16.0, -17.0, 17.0, -18.0, 18.0, -19.0, 19.0,
                                -20.0, 20.0, -21.0, 21.0, -22.0, 22.0, -23.0, 23.0,
                                -24.0, 24.0, -25.0, 25.0, -26.0, 26.0, -27.0, 27.0,
                                -28.0, 28.0, -29.0, 29.0, -30.0, 30.0, -31.0, 31.0,
                                0.0, 0.0, -10.0, 1.0, -20.0, 2.0, -30.0, 3.0,
                                -40.0, 4.0, -50.0, 5.0, -60.0, 6.0, -70.0, 7.0,
                                -80.0, 8.0, -90.0, 9.0, -100.0, 10.0, -110.0, 11.0,
                                -120.0, 12.0, -130.0, 13.0, -140.0, 14.0, -150.0, 15.0,
                                -160.0, 16.0, -170.0, 17.0, -180.0, 18.0, -190.0, 19.0,
                                -200.0, 20.0, -210.0, 21.0, -220.0, 22.0, -230.0, 23.0,
                                -240.0, 24.0, -250.0, 25.0, -260.0, 26.0, -270.0, 27.0,
                                -280.0, 28.0, -290.0, 29.0, -300.0, 30.0, -310.0, 31.0};
    auto self = makeTensorData(DT_FP32, {2, 32}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 128}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    calc::BitSort(out, self, -1, false);
    ASSERT_ALLCLOSE(out->View({2, 64}, {0, 0}), golden);
}

TEST_F(TorchAdaptorTest, TopkDescending) {
    // 降序
    std::vector<float> sdata1 = {31.0, 31.0, 30.0, 30.0, 29.0, 29.0, 28.0, 28.0,
                                    27.0, 27.0, 26.0, 26.0, 25.0, 25.0, 24.0, 24.0,
                                    23.0, 23.0, 22.0, 22.0, 21.0, 21.0, 20.0, 20.0,
                                    19.0, 19.0, 18.0, 18.0, 17.0, 17.0, 16.0, 16.0,
                                    15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0, 12.0,
                                    11.0, 11.0, 10.0, 10.0, 9.0, 9.0, 8.0, 8.0,
                                    7.0, 7.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0,
                                    3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    std::vector<float> sdata2 = {310.0, 31.0, 300.0, 30.0, 290.0, 29.0, 280.0, 28.0,
                                    270.0, 27.0, 260.0, 26.0, 250.0, 25.0, 240.0, 24.0,
                                    230.0, 23.0, 220.0, 22.0, 210.0, 21.0, 200.0, 20.0,
                                    190.0, 19.0, 180.0, 18.0, 170.0, 17.0, 160.0, 16.0,
                                    150.0, 15.0, 140.0, 14.0, 130.0, 13.0, 120.0, 12.0,
                                    110.0, 11.0, 100.0, 10.0, 90.0, 9.0, 80.0, 8.0,
                                    70.0, 7.0, 60.0, 6.0, 50.0, 5.0, 40.0, 4.0,
                                    30.0, 3.0, 20.0, 2.0, 10.0, 1.0, 0.0, 0.0};
    const int64_t OUTPUT_AXIS_SIZE = 64;
    const int64_t TOPK_VAL = 32;
    std::vector<float> gdata = sdata1;
    gdata.insert(gdata.end(), sdata2.begin(), sdata2.end());
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    std::vector<float> sdata = sdata1;
    sdata.insert(sdata.end(), OUTPUT_AXIS_SIZE, 0.0f);
    sdata.insert(sdata.end(), sdata2.begin(), sdata2.end());
    sdata.insert(sdata.end(), OUTPUT_AXIS_SIZE, 0.0f);
    auto self = makeTensorData(DT_FP32, {2, 128}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 64}, 0.0f);

    calc::Topk(out, self, -1, TOPK_VAL, true);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TopkAscending) {
    // 升序
    std::vector<float> sdata1 = {0.0, 0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0,
                                    -4.0, 4.0, -5.0, 5.0, -6.0, 6.0, -7.0, 7.0,
                                    -8.0, 8.0, -9.0, 9.0, -10.0, 10.0, -11.0, 11.0,
                                    -12.0, 12.0, -13.0, 13.0, -14.0, 14.0, -15.0, 15.0,
                                    -16.0, 16.0, -17.0, 17.0, -18.0, 18.0, -19.0, 19.0,
                                    -20.0, 20.0, -21.0, 21.0, -22.0, 22.0, -23.0, 23.0,
                                    -24.0, 24.0, -25.0, 25.0, -26.0, 26.0, -27.0, 27.0,
                                    -28.0, 28.0, -29.0, 29.0, -30.0, 30.0, -31.0, 31.0};
    std::vector<float> sdata2 = {0.0, 0.0, -10.0, 1.0, -20.0, 2.0, -30.0, 3.0,
                                    -40.0, 4.0, -50.0, 5.0, -60.0, 6.0, -70.0, 7.0,
                                    -80.0, 8.0, -90.0, 9.0, -100.0, 10.0, -110.0, 11.0,
                                    -120.0, 12.0, -130.0, 13.0, -140.0, 14.0, -150.0, 15.0,
                                    -160.0, 16.0, -170.0, 17.0, -180.0, 18.0, -190.0, 19.0,
                                    -200.0, 20.0, -210.0, 21.0, -220.0, 22.0, -230.0, 23.0,
                                    -240.0, 24.0, -250.0, 25.0, -260.0, 26.0, -270.0, 27.0,
                                    -280.0, 28.0, -290.0, 29.0, -300.0, 30.0, -310.0, 31.0};
    const int64_t OUTPUT_AXIS_SIZE = 64;
    std::vector<float> gdata = sdata1;
    gdata.insert(gdata.end(), sdata2.begin(), sdata2.end());
    auto golden = makeTensorData(DT_FP32, {2, 64}, gdata);
    std::vector<float> sdata = sdata1;
    sdata.insert(sdata.end(), OUTPUT_AXIS_SIZE, 0.0f);
    sdata.insert(sdata.end(), sdata2.begin(), sdata2.end());
    sdata.insert(sdata.end(), OUTPUT_AXIS_SIZE, 0.0f);
    auto self = makeTensorData(DT_FP32, {2, 128}, sdata);
    auto out = makeTensorData(DT_FP32, {2, 64}, 0.0f);
    const int64_t TOPK_VAL = 32;

    calc::Topk(out, self, -1, TOPK_VAL, false);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TiledMrgSortOp) {
    std::vector<float> sdata1 = {0.0, 0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0,
                                -4.0, 4.0, -5.0, 5.0, -6.0, 6.0, -7.0, 7.0};
    std::vector<float> sdata2 = {-0.5, 8.0, -10.0, 9.0, -20.0, 10.0, -30.0, 11.0,
                                -40.0, 12.0, -50.0, 13.0, -60.0, 14.0, -70.0, 15.0};
    std::vector<float> gdata = {0.0, 0.0, -0.5, 8.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0,
                                -4.0, 4.0, -5.0, 5.0, -6.0, 6.0};
    auto golden = makeTensorData(DT_FP32, {1, 16}, gdata);
    auto src1 = makeTensorData(DT_FP32, {1, 16}, sdata1);
    auto src2 = makeTensorData(DT_FP32, {1, 16}, sdata2);
    auto out = makeTensorData(DT_FP32, {1, 16}, 0.0f);
    int validBit = 2;
    int kvalue = 8;
    calc::TiledMrgSort(out, src1, src2, src2, src2, validBit, kvalue);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ExtractDescending) {
    // 降序
    std::vector<float> sdata = {31, 31, 30, 30, 29, 29, 28, 28,
                                27, 27, 26, 26, 25, 25, 24, 24,
                                23, 23, 22, 22, 21, 21, 20, 20,
                                19, 19, 18, 18, 17, 17, 16, 16,
                                15, 15, 14, 14, 13, 13, 12, 12,
                                11, 11, 10, 10, 9, 9, 8, 8,
                                7, 7, 6, 6, 5, 5, 4, 4,
                                3, 3, 2, 2, 1, 1, 0, 0,
                                310, 31, 300, 30, 290, 29, 280, 28,
                                270, 27, 260, 26, 250, 25, 240, 24,
                                230, 23, 220, 22, 210, 21, 200, 20,
                                190, 19, 180, 18, 170, 17, 160, 16,
                                150, 15, 140, 14, 130, 13, 120, 12,
                                110, 11, 100, 10, 90, 9, 80, 8,
                                70, 7, 60, 6, 50, 5, 40, 4,
                                30, 3, 20, 2, 10, 1, 0, 0};
    std::vector<float> gdata0 = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0,
                                    23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0,
                                    15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                                    7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
                                    310.0, 300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0,
                                    230.0, 220.0, 210.0, 200.0, 190.0, 180.0, 170.0, 160.0,
                                    150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                                    70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0};
    std::vector<float> gdata1 = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0,
                                    23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0,
                                    15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                                    7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
                                    31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0,
                                    23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0,
                                    15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                                    7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
    auto self = makeTensorData(DT_FP32, {2, 64}, sdata);
    auto out0 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto out1 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto golden0 = makeTensorData(DT_FP32, {2, 32}, gdata0);
    auto golden1 = makeTensorData(DT_FP32, {2, 32}, gdata1);

    calc::Extract(out0, self, 0, true);
    calc::Extract(out1, self, 1, true);
    ASSERT_ALLCLOSE(out0, golden0);
    ASSERT_ALLCLOSE(out1, golden1);
}

TEST_F(TorchAdaptorTest, ExtractAscending) {
    // 升序
    std::vector<float> sdata = {0, 0, -1, 1, -2, 2, -3, 3,
                                -4, 4, -5, 5, -6, 6, -7, 7,
                                -8, 8, -9, 9, -10, 10, -11, 11,
                                -12, 12, -13, 13, -14, 14, -15, 15,
                                -16, 16, -17, 17, -18, 18, -19, 19,
                                -20, 20, -21, 21, -22, 22, -23, 23,
                                -24, 24, -25, 25, -26, 26, -27, 27,
                                -28, 28, -29, 29, -30, 30, -31, 31,
                                0, 0, -10, 1, -20, 2, -30, 3,
                                -40, 4, -50, 5, -60, 6, -70, 7,
                                -80, 8, -90, 9, -100, 10, -110, 11,
                                -120, 12, -130, 13, -140, 14, -150, 15,
                                -160, 16, -170, 17, -180, 18, -190, 19,
                                -200, 20, -210, 21, -220, 22, -230, 23,
                                -240, 24, -250, 25, -260, 26, -270, 27,
                                -280, 28, -290, 29, -300, 30, -310, 31};
    std::vector<float> gdata0 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                                    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                                    80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                                    160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0,
                                    240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0};
    std::vector<float> gdata1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0};
    auto self = makeTensorData(DT_FP32, {2, 64}, sdata);
    auto out0 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto out1 = makeTensorData(DT_FP32, {2, 32}, 0.0f);
    auto golden0 = makeTensorData(DT_FP32, {2, 32}, gdata0);
    auto golden1 = makeTensorData(DT_FP32, {2, 32}, gdata1);

    calc::Extract(out0, self, 0, false);
    calc::Extract(out1, self, 1, false);
    ASSERT_ALLCLOSE(out0, golden0);
    ASSERT_ALLCLOSE(out1, golden1);
}

TEST_F(TorchAdaptorTest, TopkSort) {
    // Test TopkSort with 8-element input
    // Input: small array for easy verification
    std::vector<float> sdata = {3.0, 7.0, 1.0, 5.0, 9.0, 2.0, 8.0, 4.0};

    auto self = makeTensorData(DT_FP32, {1, 8}, sdata);
    auto outValue = makeTensorData(DT_FP32, {1, 64}, 0.0f);  // Output padded to 32*2
    auto outTemp = makeTensorData(DT_FP32, {1, 64}, 0.0f);

    calc::TopkSort(outValue, outTemp, self, 0);

    // Expected: pack format [v0, i0, v1, i1, ...] with 32 elements (8 real + 24 padding)
    // Values sorted descending within the 32-element group, indices from 0-7
    // The exact golden output depends on the implementation
    // We verify by extracting top values and checking they're in descending order
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, outValue->View({1, 64}, {0, 0}), 8, false);

    // Top 8 values should include all original values (9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    std::vector<float> expectedTop = {9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    auto goldenTop = makeTensorData(DT_FP32, {1, 8}, expectedTop);
    ASSERT_ALLCLOSE(extractedValues, goldenTop);
}

TEST_F(TorchAdaptorTest, TopkSortLargeInput) {
    // Test TopkSort with 32-element aligned input
    std::vector<float> sdata = {31.0, 15.0, 27.0, 8.0, 19.0, 3.0, 23.0, 11.0,
                                7.0, 28.0, 16.0, 2.0, 24.0, 9.0, 30.0, 14.0,
                                22.0, 5.0, 18.0, 1.0, 26.0, 10.0, 29.0, 13.0,
                                6.0, 20.0, 12.0, 25.0, 4.0, 21.0, 0.0, 17.0};

    auto self = makeTensorData(DT_FP32, {1, 32}, sdata);
    auto outValue = makeTensorData(DT_FP32, {1, 64}, 0.0f);
    auto outTemp = makeTensorData(DT_FP32, {1, 64}, 0.0f);

    calc::TopkSort(outValue, outTemp, self, 0);

    // Verify by extracting top-8 values
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, outValue->View({1, 64}, {0, 0}), 8, false);

    std::vector<float> expectedTop8 = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0};
    auto goldenTop8 = makeTensorData(DT_FP32, {1, 8}, expectedTop8);
    ASSERT_ALLCLOSE(extractedValues, goldenTop8);
}

TEST_F(TorchAdaptorTest, TopkMerge) {
    // Test TopkMerge with pre-sorted pack array
    // Pack format: [v0, i0, v1, i1, v2, i2, ...]
    std::vector<float> packData = {
        // First 8 packs (sorted descending)
        30.0, 0.0, 28.0, 1.0, 26.0, 2.0, 24.0, 3.0,
        22.0, 4.0, 20.0, 5.0, 18.0, 6.0, 16.0, 7.0,
        // Second 8 packs (sorted descending)
        31.0, 8.0, 29.0, 9.0, 27.0, 10.0, 25.0, 11.0,
        23.0, 12.0, 21.0, 13.0, 19.0, 14.0, 17.0, 15.0
    };

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_FP32, {1, 32}, 0.0f);

    // mergeSize = 8 means every 8 packs are already sorted
    calc::TopkMerge(out, self, 8);

    // Extract top 8 values to verify proper merging
    auto extractedValues = makeTensorData(DT_FP32, {1, 8}, 0.0f);
    calc::TopkExtract(extractedValues, out, 8, false);

    // Top 8 should be: 31, 30, 29, 28, 27, 26, 25, 24
    std::vector<float> expectedTop = {31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0};
    auto goldenTop = makeTensorData(DT_FP32, {1, 8}, expectedTop);
    ASSERT_ALLCLOSE(extractedValues, goldenTop);
}

TEST_F(TorchAdaptorTest, TopkExtractValues) {
    // Test TopkExtract for value extraction (isIndex=false)
    std::vector<float> packData = {
        // Pack format: [v0, i0, v1, i1, ...]
        // Values sorted in descending order
        100.0, 5.0, 95.0, 12.0, 90.0, 3.0, 85.0, 18.0,
        80.0, 7.0, 75.0, 21.0, 70.0, 1.0, 65.0, 14.0,
        60.0, 9.0, 55.0, 25.0, 50.0, 2.0, 45.0, 16.0,
        40.0, 11.0, 35.0, 28.0, 30.0, 4.0, 25.0, 19.0
    };

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_FP32, {1, 8}, 0.0f);

    // Extract top 8 values
    calc::TopkExtract(out, self, 8, false);

    std::vector<float> expectedValues = {100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0};
    auto golden = makeTensorData(DT_FP32, {1, 8}, expectedValues);

    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, TopkExtractIndices) {
    // Test TopkExtract for index extraction (isIndex=true)
    std::vector<float> packData = {
        // Pack format: [v0, i0, v1, i1, ...]
        100.0, 5.0, 95.0, 12.0, 90.0, 3.0, 85.0, 18.0,
        80.0, 7.0, 75.0, 21.0, 70.0, 1.0, 65.0, 14.0,
        60.0, 9.0, 55.0, 25.0, 50.0, 2.0, 45.0, 16.0,
        40.0, 11.0, 35.0, 28.0, 30.0, 4.0, 25.0, 19.0
    };

    auto self = makeTensorData(DT_FP32, {1, 32}, packData);
    auto out = makeTensorData(DT_INT32, {1, 8}, 0);

    // Extract top 8 indices
    calc::TopkExtract(out, self, 8, true);

    std::vector<int> expectedIndices = {5, 12, 3, 18, 7, 21, 1, 14};
    auto golden = makeTensorData(DT_INT32, {1, 8}, expectedIndices);

    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, Print) {
    auto t0 = makeTensorData(DT_FP32, {16, 16}, 4.0f);
    std::cout << t0->ToString() << std::endl;
    auto t1 = makeTensorData(DT_FP32, {4, 4, 4}, 4.0f);
    std::cout << t1->ToString() << std::endl;
    auto t2 = makeTensorData(DT_FP32, {4, 4, 1024, 512}, 4.0f);
    std::cout << t2->ToString() << std::endl;
    auto t3 = makeTensorData(DT_FP32, {4, 128}, 4.0f);
    std::cout << t3->ToString() << std::endl;
}

static inline int64_t alignup(int64_t x, int64_t align) {
    return (x + (align - 1)) & ~(align - 1);
}

TEST_F(TorchAdaptorTest, NDNZ) {
    for (auto m : {32, 33, 48}) {
        for (auto n : {32, 33, 48}) {
            int padm = alignup(m, 16);
            int padn = alignup(n, 8);
            std::vector<int> data(2 * m * n);
            std::iota(data.begin(), data.end(), 0);
            auto t0 = makeTensorData(DT_INT32, {2, m, n}, data);
            auto nzout = makeTensorData(DT_INT32, {2, padm, padn}, 0);
            auto ndzout = makeTensorData(DT_INT32, {2, m, n}, 0);
            auto golden = makeTensorData(DT_INT32, {2, m, n}, data);

            calc::FormatND2NZ(nzout, t0);
            calc::FormatNZ2ND(ndzout, nzout);
            ASSERT_ALLCLOSE(ndzout, golden);
        }
    }
}
} // namespace npu::tile_fwk
