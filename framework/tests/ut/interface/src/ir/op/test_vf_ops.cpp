/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_vf_ops.cpp
 * \brief Coverage tests for vf_ops.cpp type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// DeduceVFUnknownType: vf.vf_scope_enter, vf.vf_scope_exit
// ============================================================================

class VFOpsTest : public testing::Test {};

TEST_F(VFOpsTest, VfScopeEnter_NoArgs_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.vf_scope_enter", {}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(VFOpsTest, VfScopeExit_NoArgs_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.vf_scope_exit", {}, Sp());
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

// ============================================================================
// DeduceVFScalarType: vf.RegTensor
// ============================================================================

TEST_F(VFOpsTest, RegTensor_NoArgs_ReturnsFP32Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.RegTensor", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFMaskType: vf.CreateMask
// ============================================================================

TEST_F(VFOpsTest, CreateMask_NoArgs_ReturnsUINT16Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.CreateMask", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::UINT16);
}

// ============================================================================
// DeduceVFFromFirstArg: vf.Duplicate
// ============================================================================

TEST_F(VFOpsTest, Duplicate_WithArg_ReturnsFirstArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto arg = MakeScalarVar("s", DataType::FP16);
    auto call = reg.Create("vf.Duplicate", {arg}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(VFOpsTest, Duplicate_NoArgs_ReturnsFP32Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.Duplicate", {}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFFromSecondArg: 从第二个参数推导类型
// ============================================================================

TEST_F(VFOpsTest, DeduceVFFromSecondArg_WithArgs_ReturnsSecondArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::FP16);
    auto dst = MakeScalarVar("dst", DataType::FP32);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.Muls", {src, dst, mask}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, DeduceVFFromSecondArg_LessThan2Args_ReturnsFP32Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::FP16);
    auto call = reg.Create("vf.Muls", {src}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFFromDstArg: 从目标参数推导类型
// ============================================================================

TEST_F(VFOpsTest, DeduceVFFromDstArg_WithArgs_ReturnsDstArgType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src0 = MakeScalarVar("src0", DataType::FP16);
    auto src1 = MakeScalarVar("src1", DataType::FP16);
    auto dst = MakeScalarVar("dst", DataType::FP32);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.Add", {src0, src1, dst, mask}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, DeduceVFFromDstArg_LessThan3Args_ReturnsFP32Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto src0 = MakeScalarVar("src0", DataType::FP16);
    auto src1 = MakeScalarVar("src1", DataType::FP16);
    auto call = reg.Create("vf.Add", {src0, src1}, Sp());
    auto rt = As<ScalarType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

} // namespace ir
} // namespace pypto
