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

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/kv_compress.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class KvCmpUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        // skip pass, ut only execute model op code
        config::SetPassDefaultConfig(KEY_DISABLE_PASS, true);
    }

    void TearDown() override {}
};

template <typename T = npu::tile_fwk::float16>
void UTestCmpKv(CmpAttnTile& tileConfig) {
    const int32_t blockSize = 128;
    const int32_t b = 32;
    const int32_t s1 = 2;
    const int32_t kv_lora_rank = 128;
    const int32_t rope_dim = 64;
    const int32_t cmpBlockSize = 32;
    const int32_t cmpStride = 16;
    const int32_t stride = cmpStride;
    const int32_t dN = kv_lora_rank;
    const int32_t dQ = kv_lora_rank + rope_dim;
    const int32_t dR = rope_dim;
    const int32_t n2 = 1;
    const int32_t blockNum = 17116;
    const int32_t cmpBlockNum = 1083;
    const int32_t maxBlockNum = 1002;
    const int32_t slcBlockSize = 64;
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

    // Construct output tensors
    Tensor cmpKvCacheOut(dType, {cmpBlockNum * blockSize, n2 * dN}, "cmpKvCacheOut");
    Tensor cmpKrCacheOut(dType, {cmpBlockNum * blockSize, n2 * dR}, "cmpKrCacheOut");
    Tensor auxTensor(DT_FP32, {rc + rs, auxVecLen}, "auxTensor");

    auto kvCacheInput = RawTensorData::CreateConstantTensor<T>(kvCache, 0.0f);
    auto krCacheInput = RawTensorData::CreateConstantTensor<T>(krCache, 0.0f);
    auto cmpKvCacheInput = RawTensorData::CreateConstantTensor<T>(cmpKvCache, 0.0f);
    auto cmpKrCacheInput = RawTensorData::CreateConstantTensor<T>(cmpKrCache, 0.0f);
    auto blockTableInput = RawTensorData::CreateConstantTensor<int32_t>(blockTable, 0.0f);
    auto cmpCacheIndexInput = RawTensorData::CreateConstantTensor<int32_t>(cmpCacheIndex, 0.0f);
    auto actSeqInput = RawTensorData::CreateConstantTensor<int32_t>(actSeqLen, 0.0f);
    auto wk1Input = RawTensorData::CreateConstantTensor<T>(mlpWk1, 0.0f);
    auto wk2Input = RawTensorData::CreateConstantTensor<T>(mlpWk2, 0.0f);
    auto cosInput = RawTensorData::CreateConstantTensor<T>(mlpCos, 0.0f);
    auto sinInput = RawTensorData::CreateConstantTensor<T>(mlpSin, 0.0f);
    auto auxTensorOutput = RawTensorData::CreateConstantTensor<float>(auxTensor, 0.0f);

    std::vector<RawTensorDataPtr> inputDataList = {kvCacheInput, krCacheInput, cmpKvCacheInput, cmpKrCacheInput,
        blockTableInput, cmpCacheIndexInput, actSeqInput, wk1Input, wk2Input, cosInput, sinInput};
    std::vector<RawTensorDataPtr> outputDataList = {cmpKvCacheInput, cmpKrCacheInput, auxTensorOutput};
    ProgramData::GetInstance().AppendInputs(inputDataList);
    ProgramData::GetInstance().AppendOutputs(outputDataList);

    compressKv(kvCache, krCache, cmpKvCache, cmpKrCache, blockTable, cmpCacheIndex, actSeqLen, mlpWk1, mlpWk2, mlpCos,
        mlpSin, cmpKvCache, cmpKrCache, auxTensor, cmpBlockSize, cmpStride, rs, tileConfig);
}

TEST_F(KvCmpUtest, kv_compress_ut) {
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

    UTestCmpKv<npu::tile_fwk::bfloat16>(config);
}
