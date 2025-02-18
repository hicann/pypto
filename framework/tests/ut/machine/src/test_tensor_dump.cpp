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
 * \file test_tensor_dump.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstdint>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "machine/runtime/runtime.h"
#include "machine/device/aicore_dump.h"
#include "interface/utils/common.h"
#include "machine/utils/device_log.h"
#include "tilefwk/core_func_data.h"

#include <iostream>

using namespace npu::tile_fwk;

class TestAicoreDump : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestAicoreDump, test_aicore_dump_special) {
    npu::tile_fwk::TensorInfo *info = new npu::tile_fwk::TensorInfo();
    const int tmpShape[3] = {1, 1, 12};
    for (int i = 0; i < 3; i++) {
        info->shape[i] = tmpShape[i];
        info->stride[i] = tmpShape[i];
    }
    info->dataByte = 4;
    info->dims = 2;
    info->paramType = 1;

    std::vector<uint64_t> vec(1024, 1024);
    uint64_t tmp_tensorAddr = reinterpret_cast<uint64_t>(vec.data());

    DumpTensorData aicoreDump(info, tmp_tensorAddr);
    delete info;
}

TEST_F(TestAicoreDump, test_aicore_dump_normal) {
    npu::tile_fwk::TensorInfo *info = new npu::tile_fwk::TensorInfo();
    const int tmpShape[3] = {1, 2, 12};
    for (int i = 0; i < 3; i++) {
        info->shape[i] = tmpShape[i];
        info->stride[i] = tmpShape[i];
    }
    info->dataByte = 4;
    info->dims = 3;
    info->paramType = 1;

    std::vector<uint64_t> vec(1024, 1024);
    uint64_t tmp_tensorAddr = reinterpret_cast<uint64_t>(vec.data());

    DumpTensorData aicoreDump(info, tmp_tensorAddr);
    delete info;
}