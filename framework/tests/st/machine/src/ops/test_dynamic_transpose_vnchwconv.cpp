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
 * \file test_dynamic_transpose_vnchwconv.cpp
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
class DynamicTransposeTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};
TEST_F(DynamicTransposeTest, TestDynamicVnchwconv) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(64, 64);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> qShape = {b * sq, d};
    // outshape是不是应该按照整体的转置后来，还是只有有效的shape？
    std::vector<int64_t> outShape = {b * d, sq};

    Tensor q(DT_FP32, qShape, "q");
    Tensor actSeqs(DT_INT32, {b, 1}, "actual_seq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 100);
    std::vector<float> qData(b * sq * d, 0);
    std::vector<float> golden(b * sq * d, 0);
    readInput<float>(GetGoldenDir() + "/q.bin", qData);
    readInput<float>(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {q, actSeqs}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (sq))) {
            SymbolicScalar curSeq = GetTensorData(actSeqs, {batchId, 0});

            Tensor q0 = View(q, {sq, d}, {curSeq, d}, {batchId * sq, 0});
            auto tmp = Transpose(q0, {0, 1});
            // 01转置后是不是应该为{batchid * d, 0}?
            Assemble(tmp, {batchId * d, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}
