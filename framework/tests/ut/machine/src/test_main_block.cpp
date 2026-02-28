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
 * \file test_main_block.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/pypto_fwk_log.h"

#define private public
#include "machine/host/main_block.h"
#undef private

using namespace npu::tile_fwk;

class TestMainBlockLog : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
    }
    void TearDown() override {}
};

TEST_F(TestMainBlockLog, GetValidShapeFromCoa_EmptyArgList) {
    MainBlockCondBulider builder;
    std::vector<SymbolicScalar> emptyArgList;
    Shape shape;
    std::vector<SymbolicScalar> dynValidShape;

    bool result = builder.GetValidShapeFromCoa(emptyArgList, shape, dynValidShape);
    EXPECT_FALSE(result);
    EXPECT_TRUE(shape.empty());
    EXPECT_TRUE(dynValidShape.empty());
}

