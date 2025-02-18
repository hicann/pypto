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
 * \file test_encode.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "machine/utils/dynamic/dev_encode.h"

using namespace npu::tile_fwk::dynamic;

TEST(DevEncode, DevSymShape) {
  DevSymShape shape;
  shape.SetShape({SymInt(true, 0), SymInt(true, 2), SymInt(2)}); // 4, 8, 2
  uint64_t exprTbl[] = {4, 6, 8};
  uint64_t strides[3] = {0};
  shape.ToStride(strides, exprTbl);
  EXPECT_EQ(strides[0], 16);
  EXPECT_EQ(strides[1], 2);
  EXPECT_EQ(strides[2], 1);
}