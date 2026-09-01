/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tensor_utils.cpp
 * \brief Unit test for tensor_utils.
 */

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/tensor_utils.h"
#include "tilefwk/tilefwk.h"

namespace npu {
namespace tile_fwk {
namespace {
constexpr int64_t DIM = 16;

struct SharedGraph {
    std::shared_ptr<Function> function;
    std::vector<LogicalTensorPtr> sharedOutputs;
    std::vector<LogicalTensorPtr> uniqueInputs;
    std::vector<Operation*> sharedProducers;
};

SharedGraph BuildSharedGraph(const std::string& name, int sharedCount)
{
    SharedGraph g;
    g.function = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape{DIM, DIM};
    std::vector<int64_t> zeroOffset(shape.size(), 0);
    auto validShape = SymbolicScalar::FromConcrete(shape);
    auto sharedRaw = IRBuilder().CreateRawTensor(DT_FP32, shape);
    for (int i = 0; i < sharedCount; ++i) {
        auto inputRaw = IRBuilder().CreateRawTensor(DT_FP32, shape);
        auto input = IRBuilder().CreateTensorVar(*g.function, inputRaw, zeroOffset, shape, validShape);
        auto output = IRBuilder().CreateTensorVar(*g.function, sharedRaw, zeroOffset, shape, validShape);
        auto& assemble = IRBuilder().CreateTensorOpStmt(*g.function, Opcode::OP_ASSEMBLE, {input}, {output});
        assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(zeroOffset));
        g.uniqueInputs.push_back(input);
        g.sharedOutputs.push_back(output);
        g.sharedProducers.push_back(&assemble);
    }
    return g;
}

bool ContainsTensor(const std::vector<LogicalTensorPtr>& vec, const LogicalTensorPtr& t)
{
    if (t == nullptr) {
        return false;
    }
    return std::any_of(vec.begin(), vec.end(),
                       [&](const LogicalTensorPtr& e) { return e != nullptr && e->GetMagic() == t->GetMagic(); });
}

bool ContainsOp(const std::vector<Operation*>& vec, Operation* op)
{
    return std::any_of(vec.begin(), vec.end(), [op](Operation* e) { return e == op; });
}
} // namespace

class TestTensorUtils : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

// 两个共享rawMagic的output: GetSameRawMagicLogicalTensors 返回2个(含自身)
// 对应2个assemble producer(含自身的producer)
TEST_F(TestTensorUtils, TwoWayReturnsSelfAndSiblingWithProducers)
{
    auto g = BuildSharedGraph("TwoWay", 2);
    ASSERT_EQ(g.sharedOutputs.size(), 2);

    auto tensors = TensorUtils::GetSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[0]);
    EXPECT_EQ(tensors.size(), 2);
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[0]));
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[1]));

    auto producers = TensorUtils::GetProducersOfSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[0]);
    EXPECT_EQ(producers.size(), 2);
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[0]));
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[1]));
}

// 对称性: 从另一侧查询同样返回2个tensor和2个producer(含自身)
TEST_F(TestTensorUtils, QueryIsSymmetricFromOtherSide)
{
    auto g = BuildSharedGraph("TwoWaySym", 2);
    auto tensors = TensorUtils::GetSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[1]);
    EXPECT_EQ(tensors.size(), 2);
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[0]));
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[1]));

    auto producers = TensorUtils::GetProducersOfSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[1]);
    EXPECT_EQ(producers.size(), 2);
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[0]));
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[1]));
}

// 独立rawMagic的input: 桶内仅自身, 返回1个tensor(自身); input无producer故producer为空
TEST_F(TestTensorUtils, UniqueRawMagicReturnsOnlySelf)
{
    auto g = BuildSharedGraph("Unique", 2);
    ASSERT_FALSE(g.uniqueInputs.empty());
    auto tensors = TensorUtils::GetSameRawMagicLogicalTensors(*g.function, g.uniqueInputs[0]);
    EXPECT_EQ(tensors.size(), 1);
    EXPECT_TRUE(ContainsTensor(tensors, g.uniqueInputs[0]));

    // uniqueInputs 是 assemble 的输入, 自身无 producer
    auto producers = TensorUtils::GetProducersOfSameRawMagicLogicalTensors(*g.function, g.uniqueInputs[0]);
    EXPECT_TRUE(producers.empty());
}

// 三个共享rawMagic的output: 返回3个tensor(含自身)和3个producer(含自身的producer)
TEST_F(TestTensorUtils, ThreeWayReturnsAllTensorsAndProducers)
{
    auto g = BuildSharedGraph("ThreeWay", 3);
    ASSERT_EQ(g.sharedOutputs.size(), 3);

    auto tensors = TensorUtils::GetSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[0]);
    EXPECT_EQ(tensors.size(), 3);
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[0]));
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[1]));
    EXPECT_TRUE(ContainsTensor(tensors, g.sharedOutputs[2]));

    auto producers = TensorUtils::GetProducersOfSameRawMagicLogicalTensors(*g.function, g.sharedOutputs[0]);
    EXPECT_EQ(producers.size(), 3);
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[0]));
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[1]));
    EXPECT_TRUE(ContainsOp(producers, g.sharedProducers[2]));
}

TEST_F(TestTensorUtils, NullInputReturnsEmpty)
{
    auto g = BuildSharedGraph("Null", 1);
    auto tensors = TensorUtils::GetSameRawMagicLogicalTensors(*g.function, nullptr);
    EXPECT_TRUE(tensors.empty());

    auto producers = TensorUtils::GetProducersOfSameRawMagicLogicalTensors(*g.function, nullptr);
    EXPECT_TRUE(producers.empty());
}
} // namespace tile_fwk
} // namespace npu
