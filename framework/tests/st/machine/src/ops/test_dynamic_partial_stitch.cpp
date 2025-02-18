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
 * \file test_dynamic_partial_stitch.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/page_attention.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

using namespace npu::tile_fwk::dynamic;
class DynamicTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(DynamicTest, TestPartial) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);

    TileShape::Current().SetVecTile(16, 16);

    int b = 8;
    int blockSize = 32;
    std::vector<int64_t> qShape = {b * blockSize, blockSize}; /* 1 - b */
    std::vector<int64_t> seqShape = {b};
    std::vector<int64_t> midShape = {b * blockSize, blockSize};
    std::vector<int64_t> outShape = {b * blockSize, blockSize};
    DataType vType = DataType::DT_FP32;

    Tensor q(vType, qShape, "q");
    Tensor seq(DataType::DT_INT32, seqShape, "seq");
    Tensor out(vType, outShape, "out");

    std::vector<float> qData(b * blockSize * blockSize);
    for (int i = 0; i < b * blockSize * blockSize; i++) {
        qData[i] = i / (blockSize * blockSize);
    }

    std::vector<int> seqData(b);
    for (int i = 0; i < b; i++) {
        seqData[i] = i;
    }

    std::vector<float> goldenData(b * blockSize * blockSize);
    for (int i = 0; i < b * blockSize * blockSize; i++) {
        goldenData[i] = (float)(((i / (blockSize * blockSize))) * 2 + 1.0);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(q, qData),
        RawTensorData::CreateTensor<int>(seq, seqData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, goldenData),
    });

    FUNCTION("main", {q, seq}, {out}) {
        Tensor mid(vType, midShape, "mid");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0) / (blockSize))) {
            Tensor block = View(q, {blockSize, blockSize}, {batchId * blockSize, 0});
            SymbolicScalar curSeq = GetTensorData(seq, {batchId});
            config::SetSemanticLabel("add");
            Tensor add = Add(block, block);
            Assemble(add, {curSeq * blockSize, 0}, mid);
        }
        LOOP("SUM", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
            (void)_;
            config::SetSemanticLabel("adds");
            out = Add(mid, Element(DT_FP32, 1.0f));
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    resultCmp<float>(goldenData, &outs->Get<float>(0), 0.001f);
}
