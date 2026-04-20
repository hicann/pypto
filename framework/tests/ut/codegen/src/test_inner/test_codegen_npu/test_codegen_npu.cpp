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
 * \file test_codegen_npu.cpp
 * \brief Unit test for codegen npu.
 */

#include "gtest/gtest.h"

#include <sstream>
#include "codegen/npu/codegen_npu.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

class TestCodeGenNPU : public ::testing::Test {};

TEST_F(TestCodeGenNPU, TestAppendVFOptions)
{
    std::ostringstream oss;

    // Case 1: platform != DAV_3510, should output nothing
    oss.str("");
    CodeGenNPU::AppendVFOptions(NPUArch::DAV_2201, oss);
    EXPECT_EQ(oss.str(), "");

    // Case 2: platform == DAV_3510, KEY_ENABLE_VF=false, output --cce-simd-vf-fusion=false
    oss.str("");
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);
    CodeGenNPU::AppendVFOptions(NPUArch::DAV_3510, oss);
    EXPECT_NE(oss.str().find("--cce-simd-vf-fusion=false"), std::string::npos);

    // Case 3: platform == DAV_3510, KEY_ENABLE_VF=true, KEY_ENABLE_VF_UNROLL=false
    oss.str("");
    config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
    config::SetPassGlobalConfig(KEY_ENABLE_VF_UNROLL, false);
    CodeGenNPU::AppendVFOptions(NPUArch::DAV_3510, oss);
    EXPECT_NE(oss.str().find("--enable-pto-tile-fusion"), std::string::npos);
    EXPECT_EQ(oss.str().find("-enable-unroll-after-fused=true"), std::string::npos);

    // Case 4: platform == DAV_3510, KEY_ENABLE_VF=true, KEY_ENABLE_VF_UNROLL=true
    oss.str("");
    config::SetPassGlobalConfig(KEY_ENABLE_VF_UNROLL, true);
    CodeGenNPU::AppendVFOptions(NPUArch::DAV_3510, oss);
    EXPECT_NE(oss.str().find("-enable-unroll-after-fused=true"), std::string::npos);
}

} // namespace npu::tile_fwk
