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
 * \file test_dynamic_op.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tilefwk_op.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicOpTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(DynamicOpTest, FullUnalign) {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    TileShape::Current().SetVecTile(16, 128);

    // [b*s,h]
    int sTile = 32;
    int s = 50;  // dynamic
    int h = 128;
    std::vector<int64_t> shape = {s, h};
    Tensor output(DT_FP32, shape, "output");
    Tensor actSeqs(DT_INT32, {1}, "actual_seq");

    FUNCTION("main", {actSeqs}, {output}) {
        SymbolicScalar curSeq = GetTensorData(actSeqs, {0});

        LOOP("L0", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange((curSeq + sTile - 1) / sTile)) {
            Tensor tmp = Full(Element(DataType::DT_FP32, 2.0f), DT_FP32, {sTile, h},
                {std::min(curSeq - sIdx * sTile, sTile), h});
            Assemble(tmp, {sIdx * sTile, 0}, output);
        }
    }

    // read data
    int sValid = 50;
    std::vector<float> golden(s * h, 2.0f);
    // auto start = golden.end() - (s - sValid) * h;
    // std::fill(start, golden.end(), 0.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(actSeqs, sValid),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
