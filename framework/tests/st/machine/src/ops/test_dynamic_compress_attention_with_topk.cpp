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
 * \file test_dynamic_compress_attention_with_topk.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/compress_attention_with_topk.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class CmpAttnTopk : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpAttnTopk(CmpAttnTopkTile &tileConfig) {

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else {
        dType = DT_FP32;
    }

    int paramsSize = 13;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/in_params.bin", input_param);

    const int b = input_param[0];
    const int s1 = input_param[1];
    const int n1 = input_param[2];
    const int dn = input_param[3];
    const int dr = input_param[4];
    const int n2 = input_param[5];
    const int blockSize = input_param[6];
    const int cmpBlockSize = input_param[7];
    const int cmpStride = input_param[8];
    const int slcBlockSize = input_param[9];
    const int topk = input_param[10];
    const int front = input_param[11];
    const int near = input_param[12];
    const float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    ALOG_EVENT_F(R"(
TestCmpAttnTopk params:
    b=%d
    s1=%d
    n1=%d
    dn=%d
    dr=%d
    n2=%d
    blockSize=%d
    cmpBlockSize=%d
    cmpStride=%d
    slcBlockSize=%d
    topk=%d
    front=%d
    near=%d
)",
        b, s1, n1, dn, dr, n2, blockSize, cmpBlockSize, cmpStride, slcBlockSize, topk, front, near);

    DataType qType = dType;
    DataType kType = dType;

    // Read actSeq
    std::vector<int> actSeqLen(b);
    readInput<int>(GetGoldenDir() + "/act_seq.bin", actSeqLen);
    std::vector<int> actCmpSeq;
    for (auto curSeq : actSeqLen) {
        auto curCmpSeq = (curSeq - cmpBlockSize) / cmpStride + 1;
        actCmpSeq.emplace_back(curCmpSeq);
    }

    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, blockSize);

    const int slcSize = slcBlockSize / cmpStride;
    const int blockSlcNum = blockSize / slcSize;

    // Construct input tensors
    Tensor qNope(qType, {b * s1 * n1, dn}, "qNope");
    Tensor qRope(qType, {b * s1 * n1, dr}, "qRope");
    Tensor cmpKvCache(kType, {cmpBlockNum, blockSize, n2, dn}, "cmpKvCache");
    Tensor cmpKrCache(kType, {cmpBlockNum, blockSize, n2, dr}, "cmpKrCache");
    Tensor cmpBlockTable(DT_INT32, {b, maxCmpBlockNum}, "cmpBlockTable");
    Tensor actSeq(DT_INT32, {b}, "actSeq");
    Tensor auxTensor(DT_FP32, {slcBlockSize / cmpStride + cmpBlockSize / cmpStride - 1, n1}, "auxTensor"); // (5, n1)
    Tensor incSeq(DT_INT32, {1, 1, maxCmpBlockNum * blockSlcNum}, "incSeq");

    // Construct out tensors
    Tensor cmpAttn(DT_FP32, {b, s1, n1, dn}, "cmpAttnOut");
    Tensor topkRes(DT_INT32, {b, s1, topk}, "topkRes");

    // Read goldens
    std::vector<float> attnGolden(b * s1 * n1 * dn, 0.0);
    std::vector<int32_t> topkGolden(b * s1 * topk, 0);

    readInput(GetGoldenDir() + "/cmp_attn_out.bin", attnGolden);
    readInput(GetGoldenDir() + "/topk_res.bin", topkGolden);

    auto qNopeData = CreateTensorData<T>(qNope, "/q_nope.bin");
    auto qRopeData = CreateTensorData<T>(qRope, "/q_rope.bin");
    auto cmpKvCacheData = CreateTensorData<T>(cmpKvCache, "/cmp_kv_cache.bin");
    auto cmpKrCacheData = CreateTensorData<T>(cmpKrCache, "/cmp_kr_cache.bin");
    auto cmpBlockTableData = CreateTensorData<int32_t>(cmpBlockTable, "/cmp_block_table.bin");
    auto actSeqData = CreateTensorData<int32_t>(actSeq, "/act_seq.bin");
    auto auxData = CreateTensorData<float>(auxTensor, "/aux_tensor.bin");

    auto cmpAttnData = RawTensorData::CreateConstantTensor<float>(cmpAttn, 0.0f);
    auto topkResData = RawTensorData::CreateConstantTensor<int32_t>(topkRes, 0);

    std::vector<RawTensorDataPtr> inputDataList = {
        qNopeData, qRopeData, cmpKvCacheData, cmpKrCacheData, cmpBlockTableData, actSeqData, auxData};
    std::vector<RawTensorDataPtr> outputDataList = {cmpAttnData, topkResData};

    FUNCTION("CompressAttentionWithTopK",
        {qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor}, {cmpAttn, topkRes}) {
        CompressAttentionWithTopK(qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor, cmpAttn,
            topkRes, blockSize, cmpBlockSize, cmpStride, slcBlockSize, softmaxScale, n1, topk, front, near, tileConfig);
    }

    auto funcop = Program::GetInstance().GetLastFunction();
    ProgramData::GetInstance().AppendInputs(inputDataList);
    ProgramData::GetInstance().AppendOutputs(outputDataList);
    DevFuncRunner::Run(funcop);

    float eps = 3e-3f; // Compare results
    std::cout << "=======================attnOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(attnGolden, (float *)cmpAttnData->data(), eps, 100));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkGolden, (int32_t *)topkResData->data(), 0, 0, 3, false, false, 128));
}

void CommonTestConfig() {

    CmpAttnTopkTile config;
    config.topkTile = {1, 1, 128};
    config.cmpTile.c1Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v1Tile = {128, 128};
    config.cmpTile.c2Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v2Tile = {128, 64};

    TestCmpAttnTopk<npu::tile_fwk::bfloat16>(config);
}

TEST_F(CmpAttnTopk, cmp_attn_with_topk_singleop_bf16) {
    CommonTestConfig();
}