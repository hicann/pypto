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
 * \file test_ptr_ops.cpp
 * \brief Coverage tests for ptr_ops.cpp type deduction (ptr.addptr, ptr.make_tensor)
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// ptr.addptr
// ============================================================================

class PtrOpsTest : public testing::Test {};

TEST_F(PtrOpsTest, AddPtr_PtrAndOffset_ReturnsPtrType)
{
    auto& reg = OpRegistry::GetInstance();
    auto ptr = MakePtrVar("p", DataType::FP16);
    auto offset = MakeScalarVar("off", DataType::INDEX);
    auto call = reg.Create("ptr.addptr", {ptr, offset}, Sp());
    auto rt = As<PtrType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(PtrOpsTest, AddPtr_ChainedAddPtr_ReturnsPtrType)
{
    auto& reg = OpRegistry::GetInstance();
    auto ptr = MakePtrVar("p", DataType::FP32);
    auto off1 = std::make_shared<ConstInt>(int64_t(16), DataType::INDEX, Sp());
    auto call1 = reg.Create("ptr.addptr", {ptr, off1}, Sp());

    // Chain: addptr on the result of addptr
    auto off2 = std::make_shared<ConstInt>(int64_t(32), DataType::INDEX, Sp());
    auto call2 = reg.Create("ptr.addptr", {call1, off2}, Sp());
    auto rt = As<PtrType>(call2->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(PtrOpsTest, AddPtr_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("ptr.addptr", {MakePtrVar("p", DataType::FP16)}, Sp()), npu::tile_fwk::Error);
}

TEST_F(PtrOpsTest, AddPtr_NonPtrFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("ptr.addptr",
                                  {MakeScalarVar("s", DataType::FP32), MakeScalarVar("off", DataType::INDEX)}, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// ptr.make_tensor
// ============================================================================

TEST_F(PtrOpsTest, MakeTensor_PtrShapeStride_ReturnsTensorType)
{
    auto& reg = OpRegistry::GetInstance();
    auto ptr = MakePtrVar("p", DataType::FP16);
    auto shape = MakeIntTuple({16, 32});
    auto stride = MakeIntTuple({32, 1});
    auto call = reg.Create("ptr.make_tensor", {ptr, shape, stride}, Sp());
    auto rt = As<TensorType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
    EXPECT_EQ(rt->shape_.size(), 2u);
}

TEST_F(PtrOpsTest, MakeTensor_WrongArgCount_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto ptr = MakePtrVar("p", DataType::FP16);
    auto shape = MakeIntTuple({16});
    EXPECT_THROW((void)reg.Create("ptr.make_tensor", {ptr, shape}, Sp()), npu::tile_fwk::Error);
}

TEST_F(PtrOpsTest, MakeTensor_NonPtrFirst_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    EXPECT_THROW((void)reg.Create("ptr.make_tensor",
                                  {MakeScalarVar("s", DataType::FP32), MakeIntTuple({16}), MakeIntTuple({1})}, Sp()),
                 npu::tile_fwk::Error);
}

TEST_F(PtrOpsTest, MakeTensor_ShapeStrideMismatch_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto ptr = MakePtrVar("p", DataType::FP32);
    auto shape = MakeIntTuple({16, 32});
    auto stride = MakeIntTuple({1}); // rank mismatch
    EXPECT_THROW((void)reg.Create("ptr.make_tensor", {ptr, shape, stride}, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
