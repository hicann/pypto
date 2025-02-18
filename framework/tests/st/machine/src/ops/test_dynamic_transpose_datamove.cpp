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
 * \file test_dynamic_transpose_datamove.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/runtime/runtime.h"
#include "cost_model/simulation/backend.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicDatamoveTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};
TEST_F(DynamicDatamoveTest, TestDynamicDatamove) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(1, 32, 64);

    int n = 1;
    int b = 1;
    int sq = 32;
    int d = 64;
    std::vector<int64_t> shape = {b * n, sq, d};
    std::vector<int64_t> outShape = {b * sq, n, d};

    Tensor input(DT_FP32, shape, "input");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 30);
    std::vector<float> inputData(b * n * sq * d, 0);
    std::vector<float> golden(b * n * sq * d, 0);
    readInput<float>(GetGoldenDir() + "/q.bin", inputData);
    readInput<float>(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(input, inputData),
        RawTensorData::CreateTensor<int>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {input, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0});

            Tensor input0 = View(input, {n, sq, d}, {n, curSeq, d}, {batchId, 0, 0});
            auto tmp = Transpose(input0, {0, 1});
            Assemble(tmp, {batchId * sq, 0, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
