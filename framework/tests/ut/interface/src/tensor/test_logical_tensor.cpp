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
 * \file test_logical_tensor.cpp
 * \brief Test cases for LogicalTensor class with error codes
 */

#include <memory>

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestLogicalTensor : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "TestLogicalTensor SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "TestLogicalTensor TearDownTestCase" << std::endl; }
    void SetUp() override { std::cout << "TestLogicalTensor SetUp" << std::endl; }
    void TearDown() override { std::cout << "TestLogicalTensor TearDown" << std::endl; }
};

TEST_F(TestLogicalTensor, ViewDimensionMismatch)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewOffsetMismatch)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewShapeOutOfBounds)
{
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {20, 8, 8};
    std::vector<int64_t> newOffset = {0, 0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, PackedDataSize)
{
    Tensor input(DT_FP4_E2M1, {2, 4}, "input");

    auto storage = input.GetStorage(false);
    storage->SetMemoryTypeToBe(MEM_L2);

    EXPECT_EQ(storage->GetDataSize(), 4);
    EXPECT_EQ(storage->GetRawTensor()->GetRawDataSize(), 4);
    EXPECT_EQ(storage->MemorySize(), 4);
}

// Helper: build a T_MOP_CALL expression that GetTensorDataDict recognises
// (callee symbol name starts with "RUNTIME_GetTensorData"). By making the
// IOTYPE (operand[2]) or IOTYPE_INDEX (operand[3]) a non-immediate symbol,
// the FE_ASSERT inside UpdateGetTensorDataIOIndex fails and throws.
static SymbolicScalar MakeGetTensorDataCall(bool ioTypeImmediate, bool ioTypeIndexImmediate)
{
    auto callee = RawSymbolicSymbol::Create(AddRuntimePrefix("GetTensorDataTest"));
    auto index = RawSymbolicImmediate::Create(0); // operand[1] must be immediate for GetImmediateValue
    // operand[2] (IOTYPE): use OUTCAST so the "continue" guard is bypassed
    RawSymbolicScalarPtr ioType = ioTypeImmediate ?
                                      static_cast<RawSymbolicScalarPtr>(
                                          RawSymbolicImmediate::Create(GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST)) :
                                      static_cast<RawSymbolicScalarPtr>(RawSymbolicSymbol::Create("nonImmIOType"));
    // operand[3] (IOTYPE_INDEX): non-immediate to trigger the second FE_ASSERT
    RawSymbolicScalarPtr ioTypeIndex = ioTypeIndexImmediate ?
                                           static_cast<RawSymbolicScalarPtr>(RawSymbolicImmediate::Create(0)) :
                                           static_cast<RawSymbolicScalarPtr>(
                                               RawSymbolicSymbol::Create("nonImmIOTypeIndex"));
    std::vector<RawSymbolicScalarPtr> operands = {callee, index, ioType, ioTypeIndex};
    auto expr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL, operands);
    return SymbolicScalar(expr);
}

// FE_ASSERT(currIOType->IsImmediate()) fails because the IOTYPE operand
// is a symbol, not an immediate.
TEST_F(TestLogicalTensor, UpdateGetTensorDataIOIndexNonImmediateIOTypeThrows)
{
    // IOTYPE non-immediate triggers the first FE_ASSERT.
    SymbolicScalar scalar = MakeGetTensorDataCall(false, true);
    EXPECT_THROW(UpdateGetTensorDataIOIndex(0, 1, scalar), std::exception);
}

// FE_ASSERT(currIOTypeIndex->IsImmediate()) fails because IOTYPE is
// immediate-OUTCAST (passes the first assert and the continue guard),
// but IOTYPE_INDEX is a symbol, triggering the second FE_ASSERT.
TEST_F(TestLogicalTensor, UpdateGetTensorDataIOIndexNonImmediateIOTypeIndexThrows)
{
    // IOTYPE is immediate OUTCAST; IOTYPE_INDEX non-immediate triggers the
    // second FE_ASSERT.
    SymbolicScalar scalar = MakeGetTensorDataCall(true, false);
    EXPECT_THROW(UpdateGetTensorDataIOIndex(0, 1, scalar), std::exception);
}
