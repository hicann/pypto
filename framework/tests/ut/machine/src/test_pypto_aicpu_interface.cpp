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
 * \file test_pypto_aicpu_interface.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/device/machine_interface/pypto_aicpu_interface.h"

using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void *args);

TEST(PyptoAicpuInterfaceTest, NullInitArgReturnsError) {
    EXPECT_EQ(DynPyptoKernelServerNull(nullptr), 1U);
}

TEST(PyptoAicpuInterfaceTest, ExecuteBeforeLoadReturnsError) {
    EXPECT_EQ(DynPyptoKernelServer(nullptr), 1U);
    EXPECT_EQ(DynPyptoKernelServerInit(nullptr), 1U);
}
