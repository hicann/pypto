/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_simt_ops.cpp
 * \brief Type-deduction tests for the SIMT operation registry.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

using Kwargs = std::vector<std::pair<std::string, std::any>>;

TEST(SimtOpsTest, ContextAndBuiltinScalarTypes)
{
    auto& registry = OpRegistry::GetInstance();
    const std::vector<std::string> context_ops = {"simt.thread_idx", "simt.block_dim", "simt.block_idx",
                                                  "simt.grid_dim"};

    for (std::size_t axis = 0; axis < context_ops.size(); ++axis) {
        auto call = registry.Create(context_ops[axis], {}, Kwargs{{"axis", static_cast<int>(axis % 3)}}, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, DataType::UINT32);
    }

    auto linear = registry.Create("simt.linear_thread_idx", {}, Sp());
    auto linear_type = As<ScalarType>(linear->GetType());
    ASSERT_NE(linear_type, nullptr);
    EXPECT_EQ(linear_type->dtype_, DataType::UINT32);

    auto warp = registry.Create("simt.warp_size", {}, Sp());
    auto warp_type = As<ScalarType>(warp->GetType());
    ASSERT_NE(warp_type, nullptr);
    EXPECT_EQ(warp_type->dtype_, DataType::INT32);
}

TEST(SimtOpsTest, LaunchType)
{
    auto& registry = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("count", DataType::UINT32);

    std::vector<ExprPtr> launch_args = {
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        scalar,
    };
    auto launch = registry.Create("simt.launch", launch_args,
                                  Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 256}}, Sp());

    EXPECT_EQ(launch->GetType(), GetUnknownType());
}

TEST(SimtOpsTest, RejectsInvalidContextAxisAndLaunchBounds)
{
    auto& registry = OpRegistry::GetInstance();
    EXPECT_THROW((void)registry.Create("simt.thread_idx", {}, Kwargs{{"axis", 3}}, Sp()), npu::tile_fwk::Error);

    std::vector<ExprPtr> too_many_threads = {
        std::make_shared<ConstInt>(2048, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(2, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
    };
    EXPECT_THROW((void)registry.Create("simt.launch", too_many_threads,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 2048}}, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SimtOpsTest, RejectsInvalidLaunchArgumentAbi)
{
    auto& registry = OpRegistry::GetInstance();
    auto invalid_tensor = MakeTensorVar("tensor", {256}, DataType::FP32);
    std::vector<ExprPtr> tensor_args = {
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        invalid_tensor,
    };
    EXPECT_THROW((void)registry.Create("simt.launch", tensor_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);

    auto invalid_tile = MakeTileVar("tile", {1, 256}, DataType::FP32);
    std::vector<ExprPtr> tile_args = {tensor_args[0], tensor_args[1], tensor_args[2], invalid_tile};
    EXPECT_THROW((void)registry.Create("simt.launch", tile_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);
}
} // namespace ir
} // namespace pypto
