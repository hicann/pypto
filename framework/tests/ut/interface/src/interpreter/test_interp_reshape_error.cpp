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
 * \file test_interp_reshape_error.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "interface/inner/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

class ReshapeErrorTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        ProgramData::GetInstance().Reset();
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    }
};

 TEST_F(ReshapeErrorTest, ReshapeMismatchElementCount) {
 	     config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
 	     config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
 	 
 	     Tensor input(DT_FP32, {256, 1, 128}, "input");
 	     Tensor output(DT_FP32, {1, 128, 128}, "output");
 	 
 	     ProgramData::GetInstance().AppendInputs({
 	         RawTensorData::CreateConstantTensor<float>(input, 1.0f),
 	     });
 	     ProgramData::GetInstance().AppendOutputs({
 	         RawTensorData::CreateConstantTensor<float>(output, 0.0f),
 	     });
 	 
 	     FUNCTION("main", {input}, {output}) {
 	         LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
 	             (void)i;
                 TileShape::Current().SetVecTile(128, 128, 128);
 	             auto t1 = View(input, {128, 1, 128}, {30, 1, 128}, {0, 0, 0});
                 auto t2 = Reshape(t1, {128, 128}, {30, 128});
                 auto t3 = Reshape(t2, {1, 128, 128}, {1, 30, 128});
                 Assemble(t3, {0, 0, 0}, output);
 	         }
 	     }
 	 }

} // namespace npu::tile_fwk


