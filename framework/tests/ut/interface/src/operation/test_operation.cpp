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
 * \file test_operation.cpp
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class OperationOpsTest : public testing::Test {};

TEST_F(OperationOpsTest, CheckIndexAddParamsInvalid_FP16_Overflow)
{
    std::vector<int64_t> selfShape = {10, 10};
    std::vector<int64_t> srcShape = {5, 10};
    std::vector<int64_t> indicesShape = {5};
    int axis = 0;

    Tensor self(DT_FP16, selfShape);
    Tensor src(DT_FP16, srcShape);
    Tensor indices(DT_INT32, indicesShape);
    Element alpha(DT_FP16, 65505.0f);

    EXPECT_THROW(IndexAdd_(self, src, indices, axis, alpha), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStartDataType)
{
    Element start(DT_INT8, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedEndDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT8, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStepDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT8, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedOutputDataType)
{
    Element start(DT_INT64, 0);
    Element end(DT_INT64, INT64_MAX);
    Element step(DT_INT64, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, LogicalNot_UnsupportedDataType)
{
    std::vector<int64_t> shape = {10, 10};
    Tensor input(DT_INT32, shape);

    EXPECT_THROW(LogicalNot(input), std::exception);
}
