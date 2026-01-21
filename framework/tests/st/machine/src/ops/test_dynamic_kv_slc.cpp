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
 * \file test_dynamic_slc.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/gen_kv_slc.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicSlcTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16, DataType tensorType = DataType::DT_FP16>
void testSlc(KvSlcTileShapeConfig& tileConfig) {
    SetInterpreterConfig();
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
    int paramsSize = 10;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    int blockSize = input_param[9];
    int b = input_param[0];
    int s = input_param[1];
    int n2 = input_param[2];
    int kv_lora_rank = input_param[3];
    int rope_dim = input_param[4];
    int front = input_param[5];
    int near = input_param[6];
    int topK = input_param[7];
    int l_prime = input_param[8];
    std::vector<int> seq(b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    int blockNum = 0;
    for (auto seq_item : seq) {
        blockNum += CeilDiv(seq_item, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    // 读数据
    Tensor topk_tensor(DT_INT32, {b, s, topK - front - near}, "topk_tensor");
    Tensor topk_tensor_shape(DT_INT32, {b, s}, "topk_tensor_shape");
    Tensor kvNopeCache(tensorType, {int(blockNum * blockSize), n2 * kv_lora_rank}, "kNopeCache");
    Tensor kRopeCache(tensorType, {int(blockNum * blockSize), n2 * rope_dim}, "vNopeCache");
    Tensor kvActSeqs(DT_INT32, {b}, "kvActSeqs");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor k_slcOut(tensorType, {b * s * n2 * topK * l_prime, rope_dim + kv_lora_rank}, "k_slcOut");
    Tensor v_slcOut(tensorType, {b * s * n2 * topK * l_prime, kv_lora_rank}, "v_slcOut");
    Tensor kvSlcActSeqs(DT_INT32, {b, s}, "kvSlcActSeqs");

    // 读数据
    std::vector<int32_t> topkTensorData(b * s * (topK - front - near), 0);
    std::vector<int32_t> topkTensorShapeData(b * s, 0);
    std::vector<T> kvNopeCacheData(blockNum * blockSize * n2 * kv_lora_rank, 0);
    std::vector<T> kRopeCacheData(blockNum * blockSize * n2 * rope_dim, 0);
    std::vector<int32_t> kvActSeqsData(b, 0);
    std::vector<int32_t> blockTableData(b * maxBlockNumPerBatch, 0);
    std::vector<int32_t> kvSlcActSeqsData(b * s, 0);

    GenKvSlc(topk_tensor, topk_tensor_shape, kvNopeCache, kRopeCache, kvActSeqs, front, near, topK, l_prime,
            n2, blockTable, blockSize, k_slcOut, v_slcOut, kvSlcActSeqs, tileConfig);

    readInput<int32_t>(GetGoldenDir() + "/topk_tensor.bin", topkTensorData);
    readInput<int32_t>(GetGoldenDir() + "/topk_tensor_shape.bin", topkTensorShapeData);
    readInput<T>(GetGoldenDir() + "/kv_nope_cache.bin", kvNopeCacheData);
    readInput<T>(GetGoldenDir() + "/k_rope_cache.bin", kRopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/actual_seq_len.bin", kvActSeqsData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", blockTableData);
    std::vector<T> k_golden(b * s * n2 * topK * l_prime * (rope_dim + kv_lora_rank), 0);
    std::vector<T> v_golden(b * s * n2 * topK * l_prime * kv_lora_rank, 0);
    std::vector<int32_t> kvSlcActSeqs_golden(b * s, 0);

    readInput(GetGoldenDir() + "/k_slc_out.bin", k_golden);
    readInput(GetGoldenDir() + "/v_slc_out.bin", v_golden);
    readInput(GetGoldenDir() + "/kv_slc_actual_seqs.bin", kvSlcActSeqs_golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(topk_tensor, topkTensorData),
        RawTensorData::CreateTensor<int32_t>(topk_tensor_shape, topkTensorShapeData),
        RawTensorData::CreateTensor<T>(kvNopeCache, kvNopeCacheData),
        RawTensorData::CreateTensor<T>(kRopeCache, kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(kvActSeqs, kvActSeqsData),
        RawTensorData::CreateTensor<int32_t>(blockTable, blockTableData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(k_slcOut, 0),
        RawTensorData::CreateConstantTensor<T>(v_slcOut, 0),
        RawTensorData::CreateConstantTensor<int32_t>(kvSlcActSeqs, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(k_slcOut, k_golden),
        RawTensorData::CreateTensor<T>(v_slcOut, v_golden),
        RawTensorData::CreateTensor<int32_t>(kvSlcActSeqs, kvSlcActSeqs_golden),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto k_Out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto v_Out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto kvSlcActSeqs_out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    EXPECT_TRUE(resultCmp(k_golden, (T *)k_Out->data(), 0.0005f));
    EXPECT_TRUE(resultCmp(v_golden, (T *)v_Out->data(), 0.0005f));
    EXPECT_TRUE(resultCmp(kvSlcActSeqs_golden, (int32_t *)kvSlcActSeqs_out->data(), 0.0005f));
}

TEST_F(DynamicSlcTest, dynamic_p_slc_fp16) {
    KvSlcTileShapeConfig tileConfig;
    tileConfig.v0TileShape = {32, 32};
    testSlc<npu::tile_fwk::float16, DT_FP16>(tileConfig);
}

TEST_F(DynamicSlcTest, dynamic_p_slc_bf16) {
    KvSlcTileShapeConfig tileConfig;
    tileConfig.v0TileShape = {32, 32};
    testSlc<npu::tile_fwk::bfloat16, DT_BF16>(tileConfig);
}