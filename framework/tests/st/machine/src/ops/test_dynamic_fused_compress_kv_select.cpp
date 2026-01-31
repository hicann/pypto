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
 * \file test_dynamic_fused_compress_kv_select.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/fused_compress_kv_select.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicCmpKvSel : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpKvSel(CmpAttnTile &tileConfig) {

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else {
        dType = DT_FP32;
    }

    int paramsSize = 10;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param_compress.bin", input_param);

    const int b = input_param[0];
    const int s1 = input_param[1];
    const int n1 = input_param[2];
    const int dq = input_param[3];
    const int s2 = input_param[4];
    const int n2 = input_param[5];
    const int dv = input_param[6];
    const int blockSize = input_param[7];
    const int cmpBlockSize = input_param[8];
    const int cmpStride = input_param[9];
    int dr = dq - dv;
    const float softmaxScale = static_cast<float>(1.0 / sqrtf((dq)));

    DataType qType = dType;
    DataType kType = dType;

    // Read actSeqLen_v2
    std::vector<int> actSeq(b);
    readInput<int>(GetGoldenDir() + "/act_seq_compress.bin", actSeq);
    int blockNum = 0;
    for (auto s : actSeq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable_v2: (b, maxBlockNum)
    int maxBlockNum = CeilDiv(s2, blockSize);

    // Read actCmpSeqLen_v2
    std::vector<int> actCmpSeq(b);
    readInput<int>(GetGoldenDir() + "/act_cmp_seq_compress.bin", actCmpSeq);
    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    // cmpBlockTable_v2: (b, maxCmpBlockNum)
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, blockSize);

    // Construct input tensors
    Tensor qNope_v2(qType, {b * s1 * n1, dv}, "qNope_v2");
    Tensor qRope_v2(qType, {b * s1 * n1, dr}, "qRope_v2");
    Tensor kvCache_v2(kType, {blockNum * blockSize, n2 * dv}, "kvCache_v2");
    Tensor krCache_v2(kType, {blockNum * blockSize, n2 * dr}, "krCache_v2");
    Tensor cmpKvCache_v2(kType, {cmpBlockNum * blockSize, n2 * dv}, "cmpKvCache_v2");
    Tensor cmpKrCache_v2(kType, {cmpBlockNum * blockSize, n2 * dr}, "cmpKrCache_v2");
    Tensor blockTable_v2(DT_INT32, {b, maxBlockNum}, "blockTable_v2");
    Tensor cmpBlockTable_v2(DT_INT32, {b, maxCmpBlockNum}, "cmpBlockTable_v2");
    Tensor actSeqLen_v2(DT_INT32, {b}, "actSeqLen_v2");
    Tensor actCmpSeqLen_v2(DT_INT32, {b}, "actCmpSeqLen_v2");
    Tensor mlpWk1_v2(kType, {cmpBlockSize * dq, 2 * cmpBlockSize * dq}, "mlpWk1_v2");
    Tensor mlpWk2_v2(kType, {2 * cmpBlockSize * dq, dq}, "mlpWk2_v2");
    Tensor mlpCos_v2(kType, {b, cmpBlockSize, dr}, "mlpCos_v2");
    Tensor mlpSin_v2(kType, {b, cmpBlockSize, dr}, "mlpSin_v2");
    Tensor cmpAttn(DT_FP32, {b * s1 * n1, dv}, "cmpAttnOut");
    Tensor cmpAttn16(DT_FP16, {b, s1, n1, dv}, "cmpAttnOut16");
    Tensor cmpSoftmax(DT_FP32, {b * s1 * n1, maxCmpSeq}, "cmpSoftmax");
    Tensor fullK(kType, {maxBlockNum * blockSize, n2, dq}, "fullK");
    Tensor cmpK(DT_FP32, {b, maxCmpSeq, n2, dq}, "cmpK");
    Tensor firstRope(qType, {maxCmpSeq, cmpBlockSize, n2, dr}, "firstRope");
    Tensor firstRopeInput(qType, {maxCmpSeq, cmpBlockSize, dr}, "firstRopeInput");
    Tensor topkInput(DT_FP32, {b, 16}, "topkRes");
    Tensor topkRes(DT_INT32, {b, s1, 16}, "topkRes");
    std::vector<uint32_t> topkIndicesGolden = getGoldenVec<uint32_t>({b, s1, 16}, "/topk_full.bin");

    // Read goldens
    std::vector<float> attnGolden(b * s1 * n1 * dv, 0.0);
    std::vector<T> attn16Golden(b * s1 * n1 * dv, 0.0);
    std::vector<float> softmaxGolden(b * s1 * n1 * maxCmpSeq, 0.0);
    std::vector<T> fullKGolden(maxBlockNum * blockSize * n2 * dq, 0.0);
    std::vector<float> kCmpGolden(b * maxCmpSeq * n2 * dq, 0.0);
    std::vector<T> firstRopeGolden(maxCmpSeq * cmpBlockSize * dr, 0.0);
    std::vector<T> firstRopeInputGolden(maxCmpSeq * cmpBlockSize * dr, 0.0);

    readInput(GetGoldenDir() + "/cmp_attn_compress.bin", attnGolden);
    readInput(GetGoldenDir() + "/cmp_attn16_compress.bin", attn16Golden);
    readInput(GetGoldenDir() + "/cmp_softmax_compress.bin", softmaxGolden);
    readInput(GetGoldenDir() + "/k_tensor_out_compress.bin", fullKGolden);
    readInput(GetGoldenDir() + "/k_cmp_out_compress.bin", kCmpGolden);
    readInput(GetGoldenDir() + "/first_rope_compress.bin", firstRopeGolden);
    readInput(GetGoldenDir() + "/first_rope_input_compress.bin", firstRopeInputGolden);

    auto qNopeData_v3 = CreateTensorData<T>(qNope_v2, "/q_nope_compress.bin");
    auto qRopeData_v3 = CreateTensorData<T>(qRope_v2, "/q_rope_compress.bin");
    auto kvCacheData_v3 = CreateTensorData<T>(kvCache_v2, "/kv_cache_compress.bin");
    auto krCacheData_v3 = CreateTensorData<T>(krCache_v2, "/kr_cache_compress.bin");
    auto cmpKvCacheData_v3 = CreateTensorData<T>(cmpKvCache_v2, "/cmp_kv_cache_compress.bin");
    auto cmpKrCacheData_v3 = CreateTensorData<T>(cmpKrCache_v2, "/cmp_kr_cache_compress.bin");
    auto blockTableData_v3 = CreateTensorData<int32_t>(blockTable_v2, "/block_table_compress.bin");
    auto cmpBlockTableData_v3 = CreateTensorData<int32_t>(cmpBlockTable_v2, "/cmp_block_table_compress.bin");
    auto actSeqData_v3 = CreateTensorData<int32_t>(actSeqLen_v2, "/act_seq_compress.bin");
    auto actCmpSeqData_v3 = CreateTensorData<int32_t>(actCmpSeqLen_v2, "/act_cmp_seq_compress.bin");

    auto wk1Data_v3 = CreateTensorData<T>(mlpWk1_v2, "/mlp_wk1_compress.bin");
    auto wk2Data_v3 = CreateTensorData<T>(mlpWk2_v2, "/mlp_wk2_compress.bin");
    auto cosData_v3 = CreateTensorData<T>(mlpCos_v2, "/mlp_cos_compress.bin");
    auto sinData_v3 = CreateTensorData<T>(mlpSin_v2, "/mlp_sin_compress.bin");


    auto cmpAttn_v3 = RawTensorData::CreateConstantTensor<float>(cmpAttn, 0.0f);
    auto cmpAttn16_v3 = RawTensorData::CreateConstantTensor<T>(cmpAttn16, 0.0f);
    auto cmpSoftmax_v3 = RawTensorData::CreateConstantTensor<float>(cmpSoftmax, 0.0f);
    auto fullK_v3 = RawTensorData::CreateConstantTensor<T>(fullK, 0.0f);
    auto cmpK_v3 = RawTensorData::CreateConstantTensor<float>(cmpK, 0.0f);
    auto firstRope_v3 = RawTensorData::CreateConstantTensor<T>(firstRope, 0.0f);
    auto firstRopeInput_v3 = RawTensorData::CreateConstantTensor<T>(firstRopeInput, 0.0f);
    auto topkRes_v3 = RawTensorData::CreateConstantTensor<uint32_t>(topkRes, 0.0f);
    auto topkInputData = RawTensorData::CreateConstantTensor<float>(topkInput, 0.0f);


    std::vector<RawTensorDataPtr> inputDataList = {
        qNopeData_v3, qRopeData_v3, kvCacheData_v3, krCacheData_v3, cmpKvCacheData_v3, cmpKrCacheData_v3, blockTableData_v3,
        cmpBlockTableData_v3, actSeqData_v3, actCmpSeqData_v3, wk1Data_v3, wk2Data_v3, cosData_v3, sinData_v3};
    std::vector<RawTensorDataPtr> outputDataList =
        {cmpAttn_v3, cmpAttn16_v3, cmpSoftmax_v3, fullK_v3, cmpK_v3, firstRope_v3, firstRopeInput_v3, topkRes_v3, topkInputData};

    FusedCompressKvSelect(qNope_v2, qRope_v2, kvCache_v2, krCache_v2, cmpKvCache_v2, cmpKrCache_v2, blockTable_v2,
        cmpBlockTable_v2, actSeqLen_v2, actCmpSeqLen_v2, mlpWk1_v2, mlpWk2_v2, mlpCos_v2, mlpSin_v2, cmpAttn, cmpAttn16,
        cmpSoftmax, fullK, cmpK, firstRope, firstRopeInput,topkRes, topkInput, blockSize, cmpBlockSize, cmpStride, softmaxScale, n1, n2, tileConfig);

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);

    std::cout << "========================fullKCast==============================" << std::endl;
    EXPECT_TRUE(resultCmp(fullKGolden, (T *)fullK_v3->data(), 0.005f));
    std::cout << "=======================ropeOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp<T>(firstRopeGolden, (T *)firstRope_v3->data(), 0.005f));
    std::cout << "========================kCmpCast==============================" << std::endl;
    EXPECT_TRUE(resultCmp(kCmpGolden, (float *)cmpK_v3->data(), 0.005f, 50));
    std::cout << "=======================softmaxOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(softmaxGolden, (float *)cmpSoftmax_v3->data(), 0.00001f, 50));
    std::cout << "=======================attnOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(attnGolden, (float *)cmpAttn_v3->data(), 0.005f, 100));
    std::cout << "=======================attn16Out===============================" << std::endl;
    EXPECT_TRUE(resultCmp(attn16Golden, (T *)cmpAttn16_v3->data(), 0.005f, 100));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkIndicesGolden, (uint32_t *)topkRes_v3->data(), 0.008f, 0, 16, false, false, 32));

}

TEST_F(DynamicCmpKvSel, dynamic_NSA_case_no_flash) {
    // // 精度工具
    // config::SetVerifyOption(KEY_VERIFY_TENSOR_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_PASS, true);
    // config::SetVerifyOption(KEY_VERIFY_EXECUTE_GRAPH, true);
    // config::SetVerifyOption(KEY_VERIFY_CHECK_PRECISION, true);

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                     // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)

    TestCmpKvSel<npu::tile_fwk::float16>(config);
}

TEST_F(DynamicCmpKvSel, debug_dynamic_NSA_case_no_flash) {

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                     // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)

    TestCmpKvSel<npu::tile_fwk::float16>(config);
}
