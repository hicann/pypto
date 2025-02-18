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
 * \file test_dynamic_kv_compress.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"
#include "operator/models/nsa/kv_compress.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynKVCmp : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpKv(CmpAttnTile &tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 13;
    std::vector<int32_t> input_param(paramsSize);
    readInput<int32_t>(GetGoldenDir() + "/input_param.bin", input_param);
    const int32_t blockSize = input_param[0];
    const int32_t b = input_param[1];
    const int32_t s1 = input_param[2];
    const int32_t kv_lora_rank = input_param[3];
    const int32_t rope_dim = input_param[4];
    const int32_t cmpBlockSize = input_param[5];
    const int32_t cmpStride = input_param[6];
    const int32_t stride = cmpStride;
    const int32_t dN = kv_lora_rank;
    const int32_t dQ = kv_lora_rank + rope_dim;
    const int32_t dR = rope_dim;
    const int32_t n2 = input_param[7];
    const int32_t blockNum = input_param[9];
    const int32_t cmpBlockNum = input_param[10];
    const int32_t maxBlockNum = input_param[11];
    const int32_t slcBlockSize = input_param[12];
    const int rs = slcBlockSize / stride;
    const int rc = cmpBlockSize / stride;
    const int auxVecLen = 128;

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else {
        dType = DT_FP32;
    }
    DataType kType = dType;

    ALOG_EVENT_F(R"(
KVCompress params:
    blockSize=%d
    b=%d
    s1=%d
    dN=%d
    dQ=%d
    dR=%d
    n2=%d
    blockNum=%d
    cmpBlockNum=%d
    maxBlockNum=%d
    cmpBlockSize=%d
    slcBlockSize=%d
    stride=%d
    rs=%d
    rc=%d
    auxVecLen=%d
)",
        blockSize, b, s1, dN, dQ, dR, n2, blockNum, cmpBlockNum, maxBlockNum, cmpBlockSize, slcBlockSize, stride, rs,
        rc, auxVecLen);

    // Construct input tensors
    Tensor kvCache(kType, {blockNum * blockSize, n2 * dN}, "kvCache");
    Tensor krCache(kType, {blockNum * blockSize, n2 * dR}, "krCache");
    Tensor cmpKvCache(kType, {cmpBlockNum, blockSize, n2, dN}, "cmpKvCache");
    Tensor cmpKrCache(kType, {cmpBlockNum, blockSize, n2, dR}, "cmpKrCache");
    Tensor blockTable(DT_INT32, {b, maxBlockNum}, "blockTable");
    Tensor cmpCacheIndex(DT_INT32, {b, s1}, "cmpCacheIndex");
    Tensor actSeqLen(DT_INT32, {b}, "actSeqLen");
    Tensor mlpWk1(kType, {cmpBlockSize * dQ, 2 * cmpBlockSize * dQ}, "mlpWk1", TileOpFormat::TILEOP_NZ);
    Tensor mlpWk2(kType, {2 * cmpBlockSize * dQ, dQ}, "mlpWk2", TileOpFormat::TILEOP_NZ);
    Tensor mlpCos(kType, {b, cmpBlockSize, dR}, "mlpCos");
    Tensor mlpSin(kType, {b, cmpBlockSize, dR}, "mlpSin");

    auto kvCacheInput = CreateTensorData<T>(kvCache, "/kv_cache.bin");
    auto krCacheInput = CreateTensorData<T>(krCache, "/kr_cache.bin");
    auto cmpKvCacheInput = CreateTensorData<T>(cmpKvCache, "/kv_cache_compress.bin");
    auto cmpKrCacheInput = CreateTensorData<T>(cmpKrCache, "/kr_cache_compress.bin");
    auto blockTableInput = CreateTensorData<int32_t>(blockTable, "/block_table.bin");
    auto cmpCacheIndexInput = CreateTensorData<int32_t>(cmpCacheIndex, "/cache_index_compress.bin");
    auto actSeqInput = CreateTensorData<int32_t>(actSeqLen, "/act_seq_compress.bin");
    auto wk1Input = CreateTensorData<T>(mlpWk1, "/mlp_wk1_nz.bin");
    auto wk2Input = CreateTensorData<T>(mlpWk2, "/mlp_wk2_nz.bin");
    auto cosInput = CreateTensorData<T>(mlpCos, "/mlp_cos.bin");
    auto sinInput = CreateTensorData<T>(mlpSin, "/mlp_sin.bin");

    // Construct output tensors
    Tensor cmpKvCacheOut(dType, {cmpBlockNum * blockSize, n2 * dN}, "cmpKvCacheOut");
    Tensor cmpKrCacheOut(dType, {cmpBlockNum * blockSize, n2 * dR}, "cmpKrCacheOut");
    Tensor auxTensor(DT_FP32, {rc + rs, auxVecLen}, "auxTensor");

    auto auxTensorOutput = RawTensorData::CreateConstantTensor<float>(auxTensor, 0.0f);

    std::vector<T> cmpKvCacheOutGolden(cmpBlockNum * blockSize * n2 * dN, 0.0);
    std::vector<T> cmpKrCacheOutGolden(cmpBlockNum * blockSize * n2 * dR, 0.0);
    std::vector<float> auxTensorGolden((rc + rs - 1) * auxVecLen, 0.0);
    readInput(GetGoldenDir() + "/kv_cache_out_compress.bin", cmpKvCacheOutGolden);
    readInput(GetGoldenDir() + "/kr_cache_out_compress.bin", cmpKrCacheOutGolden);
    readInput(GetGoldenDir() + "/aux_tensor.bin", auxTensorGolden);

    std::vector<RawTensorDataPtr> inputDataList = {kvCacheInput, krCacheInput, cmpKvCacheInput, cmpKrCacheInput,
        blockTableInput, cmpCacheIndexInput, actSeqInput, wk1Input, wk2Input, cosInput, sinInput};
    std::vector<RawTensorDataPtr> outputDataList = {cmpKvCacheInput, cmpKrCacheInput, auxTensorOutput};
    ProgramData::GetInstance().AppendInputs(inputDataList);
    ProgramData::GetInstance().AppendOutputs(outputDataList);
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(cmpKvCache, cmpKvCacheOutGolden),
        RawTensorData::CreateTensor<T>(cmpKrCache, cmpKrCacheOutGolden),
        RawTensorData::CreateTensor<float>(auxTensor, auxTensorGolden),
    });

    compressKv(kvCache, krCache, cmpKvCache, cmpKrCache, blockTable, cmpCacheIndex, actSeqLen, mlpWk1, mlpWk2, mlpCos,
        mlpSin, cmpKvCache, cmpKrCache, auxTensor, cmpBlockSize, cmpStride, rs, tileConfig);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto actualCmpKvCacheOutput = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto actualCmpKrCacheOutput = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto actualAuxTensorOutput = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    float eps = 0.005f;
    auto result1 = resultCmp(cmpKvCacheOutGolden, (T *)actualCmpKvCacheOutput->data(), eps);
    auto result2 = resultCmp(cmpKrCacheOutGolden, (T *)actualCmpKrCacheOutput->data(), eps);
    auto auxTensorResult = resultCmp<float>(auxTensorGolden, (float *)actualAuxTensorOutput->data(), eps);
    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);
    EXPECT_TRUE(auxTensorResult);
}

TEST_F(DynKVCmp, KVCmpBatch48float16) {
    SetInterpreterConfig();

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {128, 128};          // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {48, 128, 128};    // {b, cmpBlockSize, dR} * (1, cmpBlockSize, dR) & RotateHalf
    config.mlpRopeTile.fourDim = {1, 128, 128, 128}; // (1, cmpBlockSize, dR / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {1, 32, 192};              // (n2, cmpBlockSize, d)
    config.mlpCmpTile.c1TileShape = {48, 48, 128, 128, 256, 256}; // (b * n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {48, 128};                    // (b * n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {48, 48, 256, 256, 64, 64};   // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (b, n2, d) // not used

    TestCmpKv<npu::tile_fwk::float16>(config);
}

TEST_F(DynKVCmp, KVCmpBatch32bf16) {
    SetInterpreterConfig();

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {128, 128};          // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {32, 128, 128};    // {b, cmpBlockSize, dR} * (1, cmpBlockSize, dR) & RotateHalf
    config.mlpRopeTile.fourDim = {1, 128, 128, 128}; // (1, cmpBlockSize, dR / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {1, 32, 192};              // (n2, cmpBlockSize, d)
    config.mlpCmpTile.c1TileShape = {32, 32, 128, 128, 256, 256}; // (b * n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {32, 128};                    // (b * n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {32, 32, 256, 256, 64, 64};   // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (b, n2, d) // not used

    TestCmpKv<npu::tile_fwk::bfloat16>(config);
}

template <typename T = npu::tile_fwk::bfloat16>
void TestAuxTensor() {
    config::SetHostOption(ONLY_CODEGEN, true);

    int paramsSize = 13;
    std::vector<int32_t> input_param(paramsSize);
    readInput<int32_t>(GetGoldenDir() + "/input_param.bin", input_param);
    const int32_t cmpBlockSize = input_param[5];
    const int32_t stride = input_param[6];
    const int32_t slcBlockSize = input_param[12];
    const int rs = slcBlockSize / stride;
    const int rc = cmpBlockSize / stride;
    const int auxVecLen = 128;
    ALOG_EVENT_F("AuxTensor params: \ncmpBlockSize=%d\nslcBlockSize=%d\nstride=%d\nrs=%d\nrc=%d\nauxVecLen=%d",
        cmpBlockSize, slcBlockSize, stride, rs, rc, auxVecLen);

    DataType dType = DT_FP32;
    ASSERT((std::is_same<T, float>::value)) << "We only support FP32 for auxTensor now";
    Tensor auxTensor(dType, {rc + rs, auxVecLen}, "auxTensor");
    std::vector<T> auxTensorGolden((rc + rs - 1) * auxVecLen, 0.0);
    readInput(GetGoldenDir() + "/aux_tensor.bin", auxTensorGolden);

    FUNCTION("FuncAuxTensor", {}, {auxTensor}) {
        LOOP("COMPRESS_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1)) {
            (void)bIdx;
            TileShape::Current().SetVecTile(1, auxVecLen);
            for (int i = 0; i < rs + rc - 1; i++) {
                auto auxVector = npu::tile_fwk::Full(
                    Element(dType, float(min(i + 1, rc) - max(i - rs, 0))), dType, {1, auxVecLen});
                Assemble(auxVector, {i, 0}, auxTensor);
            }
        }
    }

    auto auxTensorOutput = RawTensorData::CreateConstantTensor<T>(auxTensor, 0.0f);
    std::vector<RawTensorDataPtr> inputDataList = {};
    std::vector<RawTensorDataPtr> outputDataList = {auxTensorOutput};
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),inputDataList, outputDataList);

    EXPECT_TRUE(resultCmp<T>(auxTensorGolden, (T *)auxTensorOutput->data(), 0.008f));
}

TEST_F(DynKVCmp, AuxVectorBuildFloat32) {
    // // 精度工具
    // config::SetVerifyOption(KEY_VERIFY_TENSOR_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_PASS, true);
    // config::SetVerifyOption(KEY_VERIFY_EXECUTE_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_CHECK_PRECISION, true);

    TestAuxTensor<float>();
}
