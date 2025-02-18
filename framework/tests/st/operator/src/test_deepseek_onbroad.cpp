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
 * \file test_deepseek_onbroad.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/deepseek_spec.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MoeInferOnbroadTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(MoeInferOnbroadTest, test_deepseekMoEInfer) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int32_t nRoutedExperts = 256;
    int b = 16; // 32
    int s = 1;  // 1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = {b * s, h};
    std::vector<int64_t> topKShape = {b * s, numExpertsPerTok};
    std::vector<int64_t> resShape = {b * s, numExpertsPerTok};

    int hiddenStatesSize = b * s * h;
    int topkSize = b * s * numExpertsPerTok;

    void *hiddenStatesPtr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin", hiddenStatesSize);
    void *topkIdxPtr = readToDev<int>(GetGoldenDir() + "/topk_idx.bin", topkSize);
    void *topkWeightPtr = readToDev<float>(GetGoldenDir() + "/topk_weight.bin", topkSize);

    assert(hiddenStatesPtr != nullptr && topkIdxPtr != nullptr && topkWeightPtr != nullptr);

    uint8_t *outputPtr = allocDevAddr(hiddenStatesSize * sizeof(float)); // [b*s,h]
    // alloc output
    uint8_t *outsPtr = allocDevAddr(b * s * numExpertsPerTok * h * sizeof(float));
    uint8_t *sortedTokensPtr = allocDevAddr(b * s * numExpertsPerTok * h * sizeof(float));
    uint8_t *idxsPtr = allocDevAddr(b * s * numExpertsPerTok * sizeof(float));

    void *ffnWeight1Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight1.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    void *ffnWeight2Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight2.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    void *ffnWeight3Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight3.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    Tensor ffnWeight1(DataType::DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight1Ptr, "ffnWeight1");
    Tensor ffnWeight2(DataType::DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight2Ptr, "ffnWeight2");
    Tensor ffnWeight3(DataType::DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight3Ptr, "ffnWeight3");

    Tensor outs(DataType::DT_FP32, {b * s * numExpertsPerTok, h}, outsPtr, "outs");
    Tensor sortedTokens(DataType::DT_FP32, {b * s * numExpertsPerTok, h}, sortedTokensPtr, "sortedTokens");
    Tensor idxs(DataType::DT_INT32, {b * s * numExpertsPerTok}, idxsPtr, "idxs");

    Tensor finalout(DataType::DT_FP32, { b*s, h }, (uint8_t *)outputPtr, "finalout");

    PROGRAM("MOE_INFER") {
        config::Reset();
        TileShape::Current().SetCubeTile({std::min(128, b * s), std::min(128, b * s)}, {64, 64}, {64, 64});

        TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

        Tensor hiddenStates = Tensor(DataType::DT_FP32, hiddenStatesShape, (uint8_t *)hiddenStatesPtr, "hiddenStates");
        Tensor topkIdx = Tensor(DataType::DT_INT32, topKShape, (uint8_t *)topkIdxPtr, "topkIdx");
        Tensor topkWeight = Tensor(DataType::DT_FP32, topKShape, (uint8_t *)topkWeightPtr, "topkWeight");

        config::SetBuildStatic(true);
        FUNCTION("MOE_INFER_F",
            {hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, idxs, sortedTokens, outs,
                finalout}) {
            finalout = deepseekMoEInfer.MoeInfer(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2,
                ffnWeight3, idxs, sortedTokens, outs, nRoutedExperts);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<int32_t> goldenIdxs(b * s * numExpertsPerTok);
    std::vector<float> goldenSortedTokens(b * s * numExpertsPerTok * h);
    std::vector<float> goldenOutsTensor(b * s * numExpertsPerTok * h);
    std::vector<float> goldenFinalWeight(hiddenStatesSize);

    std::vector<int32_t> devIdxs(b * s * numExpertsPerTok);
    std::vector<float> devSortedTokens(b * s * numExpertsPerTok * h);
    std::vector<float> devOutsTensor(b * s * numExpertsPerTok * h);
    std::vector<float> devFinalWeight(hiddenStatesSize);

    machine::GetRA()->CopyFromTensor(
        (uint8_t *)devIdxs.data(), (uint8_t *)idxsPtr, b * s * numExpertsPerTok * sizeof(float));
    machine::GetRA()->CopyFromTensor((uint8_t *)devSortedTokens.data(), (uint8_t *)sortedTokensPtr,
        b * s * numExpertsPerTok * h * sizeof(float));
    machine::GetRA()->CopyFromTensor(
        (uint8_t *)devOutsTensor.data(), (uint8_t *)outsPtr, b * s * numExpertsPerTok * h * sizeof(float));
    machine::GetRA()->CopyFromTensor(
        (uint8_t *)devFinalWeight.data(), (uint8_t *)outputPtr, hiddenStatesSize * sizeof(float));

    // 真值比对
    readInput(GetGoldenDir() + "/idxs.bin", goldenIdxs);
    readInput(GetGoldenDir() + "/sorted_tokens.bin", goldenSortedTokens);
    readInput(GetGoldenDir() + "/outs.bin", goldenOutsTensor);
    readInput(GetGoldenDir() + "/final_out.bin", goldenFinalWeight);

    std::cout << "compare goldenIdxs -------- " << std::endl;
    int retIdxs = resultCmp(goldenIdxs, devIdxs, 0.001f);
    EXPECT_TRUE(retIdxs);

    std::cout << "compare sortedTokens -------- " << std::endl;
    int retSortedTokens = resultCmp(goldenSortedTokens, devSortedTokens, 0.001f);
    EXPECT_TRUE(retSortedTokens);

    std::cout << "compare outs -------- " << std::endl;
    int retOuts = resultCmp(goldenOutsTensor, devOutsTensor, 0.001f);
    EXPECT_TRUE(retOuts);

    std::cout << "compare final out -------- " << std::endl;
    int retFinalWeight = resultCmp(goldenFinalWeight, devFinalWeight, 0.001f);

    EXPECT_TRUE(retFinalWeight);
}


TEST_F(MoeInferOnbroadTest, test_deepseekMoEInfer_singleout) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int32_t nRoutedExperts = 256;
    int b = 4;   // 32
    int s = 1; // 1, optimize set_tile
    int h = 256;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };
    std::vector<int64_t> resShape = { b* s, numExpertsPerTok };

    int hiddenStatesSize = b * s * h;
    int topkSize = b * s * numExpertsPerTok;

    void *hiddenStatesPtr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin", hiddenStatesSize);
    void *topkIdxPtr = readToDev<int>(GetGoldenDir() + "/topk_idx.bin", topkSize);
    void *topkWeightPtr = readToDev<float>(GetGoldenDir() + "/topk_weight.bin", topkSize);

    assert(hiddenStatesPtr != nullptr && topkIdxPtr != nullptr && topkWeightPtr != nullptr);

    uint8_t* outputPtr = allocDevAddr(hiddenStatesSize * sizeof(float)); // [b*s,h]
    // alloc output
    void *ffnWeight1Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight1.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    void *ffnWeight2Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight2.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    void *ffnWeight3Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight3.bin", h * h * 3 * sizeof(npu::tile_fwk::float16));
    Tensor ffnWeight1(DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight1Ptr, "ffnWeight1");
    Tensor ffnWeight2(DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight2Ptr, "ffnWeight2");
    Tensor ffnWeight3(DT_FP16, {h, h * 3}, (uint8_t *)ffnWeight3Ptr, "ffnWeight3");

    Tensor finalout(DT_FP32, { b*s, h }, (uint8_t *)outputPtr, "finalout");

    PROGRAM("MOE_INFER") {
        config::Reset();
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});

        TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

        Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, (uint8_t *)hiddenStatesPtr, "hiddenStates");
        Tensor topkIdx = Tensor(DT_INT32, topKShape, (uint8_t *)topkIdxPtr, "topkIdx");
        Tensor topkWeight = Tensor(DT_FP32, topKShape, (uint8_t *)topkWeightPtr, "topkWeight");

        config::SetBuildStatic(true);
        FUNCTION("MOE_INFER_F",
            { hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3,  finalout }) {
                finalout = deepseekMoEInfer.MoeInfer(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, nRoutedExperts);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> goldenFinalWeight(hiddenStatesSize);
    std::vector<float> devFinalWeight(hiddenStatesSize);

    machine::GetRA()->CopyFromTensor((uint8_t *)devFinalWeight.data(), (uint8_t *)outputPtr,
                            hiddenStatesSize  * sizeof(float));

    // 真值比对
    readInput(GetGoldenDir() + "/final_out.bin", goldenFinalWeight);

    std::cout << "compare final out -------- " << std::endl;
    int retFinalWeight = resultCmp(goldenFinalWeight, devFinalWeight, 0.001f);

    EXPECT_TRUE(retFinalWeight);
}

TEST_F(MoeInferOnbroadTest, test_deepseekMoEInfer_singleout_singlemlp) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int32_t nRoutedExperts = 256;
    int b = 4;
    int s = 1;
    int h = 7168;
    int weightN = 2048;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };
    std::vector<int64_t> resShape = { b* s, numExpertsPerTok };

    int hiddenStatesSize = b * s * h;
    int topkSize = b * s * numExpertsPerTok;

    void *hiddenStatesPtr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin", hiddenStatesSize);
    void *topkIdxPtr = readToDev<int>(GetGoldenDir() + "/topk_idx.bin", topkSize);
    void *topkWeightPtr = readToDev<float>(GetGoldenDir() + "/topk_weight.bin", topkSize);

    assert(hiddenStatesPtr != nullptr && topkIdxPtr != nullptr && topkWeightPtr != nullptr);

    uint8_t* outputPtr = allocDevAddr(hiddenStatesSize * sizeof(float)); // [b*s,h]
    // alloc output
    void *ffnWeight1Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight1.bin", h * weightN * sizeof(npu::tile_fwk::float16));
    void *ffnWeight2Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight2.bin", h * weightN * sizeof(npu::tile_fwk::float16));
    void *ffnWeight3Ptr = readToDev<npu::tile_fwk::float16>(GetGoldenDir() + "/ffnWeight3.bin", h * weightN * sizeof(npu::tile_fwk::float16));
    Tensor ffnWeight1(DT_FP16, {h, weightN}, (uint8_t *)ffnWeight1Ptr, "ffnWeight1");
    Tensor ffnWeight2(DT_FP16, {h, weightN}, (uint8_t *)ffnWeight2Ptr, "ffnWeight2");
    Tensor ffnWeight3(DT_FP16, {h, weightN}, (uint8_t *)ffnWeight3Ptr, "ffnWeight3");

    Tensor finalout(DT_FP32, { b*s, h }, (uint8_t *)outputPtr, "finalout");

    PROGRAM("MOE_INFER") {
        config::Reset();
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});

        TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

        Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, (uint8_t *)hiddenStatesPtr, "hiddenStates");
        Tensor topkIdx = Tensor(DT_INT32, topKShape, (uint8_t *)topkIdxPtr, "topkIdx");
        Tensor topkWeight = Tensor(DT_FP32, topKShape, (uint8_t *)topkWeightPtr, "topkWeight");

        config::SetBuildStatic(true);
        FUNCTION("MOE_INFER_F",
            { hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3,  finalout }) {
                finalout = deepseekMoEInfer.MoeInferSingleMlp(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, nRoutedExperts);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> goldenFinalWeight(hiddenStatesSize);
    std::vector<float> devFinalWeight(hiddenStatesSize);

    machine::GetRA()->CopyFromTensor((uint8_t *)devFinalWeight.data(), (uint8_t *)outputPtr,
                            hiddenStatesSize  * sizeof(float));

    // 真值比对
    readInput(GetGoldenDir() + "/final_out.bin", goldenFinalWeight);

    std::cout << "compare final out -------- " << std::endl;
    int retFinalWeight = resultCmp(goldenFinalWeight, devFinalWeight, 0.001f);

    EXPECT_TRUE(retFinalWeight);
}

TEST_F(MoeInferOnbroadTest, test_deepseekMoEInfer_singleout_singlemlp_withquant) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    int32_t nRoutedExperts = 256;
    int b = 32;
    int s = 1;
    int h = 7168;
    int weightN = 2048;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };
    std::vector<int64_t> resShape = { b* s, numExpertsPerTok };

    int hiddenStatesSize = b * s * h;
    int topkSize = b * s * numExpertsPerTok;

    void *hiddenStatesPtr = readToDev<float>(GetGoldenDir() + "/hidden_states.bin", hiddenStatesSize);
    void *topkIdxPtr = readToDev<int>(GetGoldenDir() + "/topk_idx.bin", topkSize);
    void *topkWeightPtr = readToDev<float>(GetGoldenDir() + "/topk_weight.bin", topkSize);

    void *ffnwightQuantPtr = readToDev<int>(GetGoldenDir() + "/ffnWeight1.bin", h * weightN);
    void *ffnwightScalePtr = readToDev<int>(GetGoldenDir() + "/ffnScale1.bin", 1 * weightN);

    void *ffnwight2QuantPtr = readToDev<int>(GetGoldenDir() + "/ffnWeight2.bin", h * weightN);
    void *ffnwight2ScalePtr = readToDev<int>(GetGoldenDir() + "/ffnScale2.bin", 1 * weightN);

    void *ffnwight3QuantPtr = readToDev<int>(GetGoldenDir() + "/ffnWeight3.bin", h * weightN);
    void *ffnwight3ScalePtr = readToDev<int>(GetGoldenDir() + "/ffnScale3.bin", h * 1);

    assert(hiddenStatesPtr != nullptr && topkIdxPtr != nullptr && topkWeightPtr != nullptr);

    uint8_t* outputPtr = allocDevAddr(hiddenStatesSize * sizeof(float)); // [b*s,h]

    Tensor ffnWeight1(DT_INT8, {h, weightN}, (uint8_t *)ffnwightQuantPtr, "ffnWeight1", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight2(DT_INT8, {h, weightN}, (uint8_t *)ffnwight2QuantPtr, "ffnWeight2", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight3(DT_INT8, {h, weightN}, (uint8_t *)ffnwight3QuantPtr, "ffnWeight3", TileOpFormat::TILEOP_NZ);
    Tensor ffnwight1Scale(DT_FP32, {1, weightN}, (uint8_t *)ffnwightScalePtr, "ffnwight1Scale");
    Tensor ffnwight2Scale(DT_FP32, {1, weightN}, (uint8_t *)ffnwight2ScalePtr, "ffnwight2Scale");
    Tensor ffnwight3Scale(DT_FP32, {h, 1}, (uint8_t *)ffnwight3ScalePtr, "ffnwight3Scale");

    Tensor finalout(DT_FP32, { b*s, h }, (uint8_t *)outputPtr, "finalout");

    PROGRAM("MOE_INFER_SINGLEMLP_QUANT") {
        config::Reset();
        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
        config::SetPassOption(CUBE_L1_REUSE_MODE, 4);

        TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

        Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, (uint8_t *)hiddenStatesPtr, "hiddenStates");
        Tensor topkIdx = Tensor(DT_INT32, topKShape, (uint8_t *)topkIdxPtr, "topkIdx");
        Tensor topkWeight = Tensor(DT_FP32, topKShape, (uint8_t *)topkWeightPtr, "topkWeight");

        config::SetBuildStatic(true);
        FUNCTION("MOE_INFER_F",
            { hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, ffnwight1Scale, ffnwight2Scale, ffnwight3Scale, finalout }) {
                finalout = deepseekMoEInfer.MoeInferSingleMlpQuant(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3,  ffnwight1Scale, ffnwight2Scale, ffnwight3Scale, nRoutedExperts);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> goldenFinalWeight(hiddenStatesSize);
    std::vector<float> devFinalWeight(hiddenStatesSize);

    machine::GetRA()->CopyFromTensor((uint8_t *)devFinalWeight.data(), (uint8_t *)outputPtr,
                            hiddenStatesSize  * sizeof(float));

    // 真值比对
    readInput(GetGoldenDir() + "/final_out.bin", goldenFinalWeight);

    std::cout << "compare final out -------- " << std::endl;
    int retFinalWeight = resultCmp(goldenFinalWeight, devFinalWeight, 0.001f);

    EXPECT_TRUE(retFinalWeight);
}
