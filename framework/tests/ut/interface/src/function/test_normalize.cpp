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
 * \file test_normalize.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

class NormalizeTest : public testing::Test {
public:
    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override { Program::GetInstance().Reset(); }
};

TEST_F(NormalizeTest, ReshapeCopyOutWritesValidShapeBack)
{
    auto parent = std::make_shared<Function>(Program::GetInstance(), "reshape_copy_parent", "reshape_copy_parent",
                                             nullptr);
    parent->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    auto leaf = std::make_shared<Function>(Program::GetInstance(), "reshape_copy_leaf", "reshape_copy_leaf",
                                           parent.get());

    const std::vector<int64_t> fromShape{4, 4, 128};
    const std::vector<int64_t> toShape{16, 128};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, fromShape);
    auto output = IRBuilder().CreateTensorVar(DT_FP32, toShape);
    input->SetMemoryTypeBoth(MEM_UB);
    output->SetMemoryTypeBoth(MEM_DEVICE_DDR);

    auto& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_RESHAPE_COPY_OUT, {input}, {output});
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MEM_UB, OpImmediate::Specified({0, 0}), OpImmediate::Specified(fromShape), OpImmediate::Specified(toShape));
    copyAttr->SetToDynValidShape(OpImmediate::Specified({2, 128}));
    op.SetOpAttribute(copyAttr);

    std::vector<std::vector<SymbolicScalar>> coaLists;
    int coaIndex = COA_INDEX_BASE;
    leaf->NormalizeCoaForSpecialInfo(coaLists, coaIndex);

    ASSERT_EQ(coaLists.size(), 1U);
    ASSERT_EQ(coaLists[0].size(), 2U);
    EXPECT_EQ(coaLists[0][0].Dump(), "2");
    EXPECT_EQ(coaLists[0][1].Dump(), "128");
    EXPECT_EQ(coaIndex, COA_INDEX_BASE + 2);
    const auto& validShape = copyAttr->GetToDynValidShape();
    ASSERT_EQ(validShape.size(), 2U);
    for (const auto& dim : validShape) {
        EXPECT_NE(dim.GetSpecifiedValue().Dump().find("RUNTIME_COA_GET_PARAM"), std::string::npos);
    }
}
