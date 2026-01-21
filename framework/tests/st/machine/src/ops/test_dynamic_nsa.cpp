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
 * \file test_dynamic_mla.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/dynamic_nsa.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DyNsa : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
        config::SetPassOption(VEC_NBUFFER_MODE, 1);
        config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, NUM_4}});
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{NUM_3, NUM_4}});
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);
        rtSetDevice(GetDeviceIdByEnvVar());
    }
};

namespace {
template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = float16, typename outputT = float, bool nz = false>
void TestNsa(const SimpleParams &params) {
    SetInterpreterConfig();
    int b = params.b;
    int s = params.s;
    int n = params.n;
    int h = params.h;

    DataType dType = (std::is_same<T, float16>::value) ? DT_FP16 : DT_BF16;
    DataType outputDtype = (std::is_same<outputT, float>::value) ? DT_FP32 : dType;
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    std::vector<int64_t> outputShape = {b, s, n, 3};

    Tensor x(dType, {b, s, h}, "x");
    Tensor w1(dType, {h, 4 * h}, "w1", weightFormat);
    Tensor w2(dType, {4 * h, 3 * n}, "w2", weightFormat);
    Tensor simW1(dType, {h, 3 * n}, "simW1");
    Tensor output(outputDtype, outputShape, "output");

    string outputPath = (std::is_same<outputT, float>::value) ? "/gating_score_fp32.bin" : "/gating_score.bin";
    std::vector<outputT> outputGolden = getGoldenVec<outputT>(outputShape, outputPath);

    auto xData = CreateTensorData<T>(x, "/x.bin");
    auto w1Data = CreateTensorData<T>(w1, nz ? "/gate_w1_nz.bin" : "/gate_w1.bin");
    auto w2Data = CreateTensorData<T>(w2, nz ? "/gate_w2_nz.bin" : "/gate_w2.bin");
    auto simW1Data = CreateTensorData<T>(simW1, "/gate_sim_w1.bin");

    auto outputData = RawTensorData::CreateConstantTensor<outputT>(output, 0.0);

    ProgramData::GetInstance().AppendInputs({
        xData, w1Data, w2Data, simW1Data,
    });

    ProgramData::GetInstance().AppendOutputs({
        outputData,
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<outputT>(output, outputGolden),
    });

    GenGatedScoreCompute(x, w1, w2, simW1, output, GateMode::standard);
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {xData, w1Data, w2Data, simW1Data}, {outputData});
    std::cout << "======= GateScore ====== " << std::endl;
    EXPECT_TRUE(
        resultCmp<outputT>(outputGolden, (outputT *)outputData->data(), 0.001f, NUM_16, 1000, false, false, NUM_16));
#endif
}

void TestView() {
    std::vector<int64_t> input_shape = {1, 128}, output_shape = {1, 16};
    Tensor input(DT_FP32, input_shape, "x");
    Tensor output(DT_FP32, output_shape, "output");
    auto xData = CreateTensorData<float>(input, "/input.bin");
    std::vector<float> outputGolden = getGoldenVec<float>(output_shape, "/output.bin");
    auto outputData = RawTensorData::CreateConstantTensor<float>(output, 0.0);

    FUNCTION("main", {input}, {output}) {
        int b = input.GetShape()[0];
        int tileB = b;
        SymbolicScalar bLoop = b / tileB;
        LOOP("LOOP_topk3", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, bLoop, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile({1, 16});
            auto view0 = View(input, {1, 13}, {0, 1});
            auto topVal = std::get<0>(TopK(view0, 3, -1, true));
            output = topVal;
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {xData}, {outputData});
    std::cout << "trans0 ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(outputGolden, (float *)outputData->data(), 0.008f, 0, 1000, false, false, NUM_16));
#endif
}

void TestAlignRead(bool isAlign) {
    std::vector<int64_t> input_shape = {1, 128}, output_shape = {1, 3};
    Tensor input(DT_FP32, input_shape, "x");
    Tensor output(DT_FP32, output_shape, "output");
    auto xData = CreateTensorData<float>(input, "/input.bin");
    std::vector<float> outputGolden = getGoldenVec<float>(output_shape, "/output.bin");
    auto outputData = RawTensorData::CreateConstantTensor<float>(output, 0.0);

    FUNCTION("main", {input}, {output}) {
        int b = input.GetShape()[0];
        int tileB = b;
        SymbolicScalar bLoop = b / tileB;
        LOOP("LOOP_topk3", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, bLoop, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile({1, 128});
            auto view0 = View(input, {1, 128}, {0, 0});
            auto adds_res = Add(view0, Element(DT_FP32, 0.0f));
            if (isAlign) {
                auto view1 = View(adds_res, {1, 125}, {0, 0});
                auto topVal = std::get<0>(TopK(view1, 3, -1, true));
                output = topVal;
            } else {
                auto view1 = View(adds_res, {1, 125}, {0, 1});
                //                Mask  -inf 0 1 2 3 xx x 125 -inf -inf
                //                source start end pad_value
                auto topVal = std::get<0>(TopK(view1, 3, -1, true));
                output = topVal;
            }
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {xData}, {outputData});
    std::cout << "trans0 ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(outputGolden, (float *)outputData->data(), 0.008f, 0, 1000, false, false, NUM_16));
#endif
}

void TestMultiLoopAlignRead() {
    std::vector<int64_t> input_shape = {1, 128}, output_shape = {1, 32}, middle_shape = {1, 128};
    Tensor input(DT_FP32, input_shape, "x");
    Tensor output(DT_FP32, output_shape, "output");
    auto xData = CreateTensorData<float>(input, "/input.bin");
    std::vector<float> outputGolden = getGoldenVec<float>(output_shape, "/output.bin");
    auto outputData = RawTensorData::CreateConstantTensor<float>(output, 0.0);

    FUNCTION("main", {input}, {output}) {
        int b = input.GetShape()[0];
        int tileB = b;
        SymbolicScalar bLoop = b / tileB;
        Tensor middle(DT_FP32, middle_shape, "middle");
        LOOP("LOOP0", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, bLoop, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile({1, 16});
            auto view0 = View(input, {1, 128}, {0, 0});
            auto adds_res = Add(view0, Element(DT_FP32, 0.0f));
            middle = adds_res;
        }
        LOOP("LOOP1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, bLoop, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile({1, 16});
            auto view1 = View(middle, {1, 16}, {0, 1});
            auto topVal = std::get<0>(TopK(view1, 13, -1, false));
            output = topVal;
        }
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {xData}, {outputData});
    std::cout << "trans0 ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(outputGolden, (float *)outputData->data(), 0.008f, 0, 1000, false, false, NUM_16));
#endif
}

template <typename T = float16, typename wDtype = int8_t, bool splitK = false, bool nz = true, bool isSmooth = true,
    bool usePrefetch = true>
void TestGenslc(const SimpleParams &params, int topk_actual_len = 0, bool isGenSlc = false) {
    int n2 = params.n2;
    int n = params.n;
    int g = n / n2;
    int s2 = params.s2;
    int d = 16, w = 32, l_prime = 64;
    int s_cmp = (s2 - w) / d + 1;
    int out_loop = l_prime / d;
    int s_slc = (s_cmp + out_loop - 1) / out_loop;
    int tmp_s_scmp = (topk_actual_len - 32) / 16 + 1;
    int tmp_s_slc = (tmp_s_scmp + 3) / 4;

    DataType dType = (std::is_same<T, float16>::value) ? DT_FP16 : DT_BF16;

    std::vector<int64_t> x_shape = {n2, g, s_cmp};
    if (!isGenSlc) {
        x_shape = {1, s_slc};
    }
    std::vector<int64_t> trans0Shape = {n2, s_cmp, g};
    std::vector<int64_t> reduce0Shape = {n2, s_slc, g};
    std::vector<int64_t> trans1Shape = {n2, g, s_slc};
    std::vector<int64_t> reduce1Shape = {n2, 1, s_slc};
    std::vector<int64_t> resShape = {1, 13};

    Tensor x(dType, x_shape, "x");
    Tensor trans0(dType, trans0Shape, "trans0");
    Tensor reduce0(dType, reduce0Shape, "reduce0");
    Tensor trans1(dType, trans1Shape, "trans1");
    Tensor reduce1(dType, reduce1Shape, "reduce1");
    Tensor topkInd(DT_FP32, resShape, "topkInd");
    Tensor topkVal(DT_FP32, resShape, "topkVal");
    Tensor res(DT_FP32, resShape, "res");

    std::vector<float> topkIndicesGolden = getGoldenVec<float>(resShape, "/topk_indices.bin");
    std::vector<T> trans0Golden = getGoldenVec<T>(trans0Shape, "/trans0.bin");
    std::vector<T> reduce0Golden = getGoldenVec<T>(reduce0Shape, "/reduce0.bin");
    std::vector<T> trans1Golden = getGoldenVec<T>(trans1Shape, "/trans1.bin");
    std::vector<T> reduce1Golden = getGoldenVec<T>(reduce1Shape, "/reduce1.bin");

    auto xData = CreateTensorData<T>(x, isGenSlc ? "/p_cmp.bin" : "/reduce1.bin");

    auto trans0Data = RawTensorData::CreateConstantTensor<T>(trans0, 0.0);
    auto reduce0Data = RawTensorData::CreateConstantTensor<T>(reduce0, 0.0);
    auto trans1Data = RawTensorData::CreateConstantTensor<T>(trans1, 0.0);
    auto reduce1Data = RawTensorData::CreateConstantTensor<T>(reduce1, 0.0);
    auto topkIndData = RawTensorData::CreateConstantTensor<float>(topkInd, 0.0);
    auto topkValData = RawTensorData::CreateConstantTensor<float>(topkVal, 0.0);

    auto resZeroData = RawTensorData::CreateConstantTensor<float>(res, 0.0);

    auto trans0GoldenData = CreateTensorData<T>(trans0, "/trans0.bin");
    auto reduce0GoldenData = CreateTensorData<T>(reduce0, "/reduce0.bin");
    auto trans1GoldenData = CreateTensorData<T>(trans1, "/trans1.bin");
    auto reduce1GoldenData = CreateTensorData<T>(reduce1, "/reduce1.bin");
    auto resGoldenData = CreateTensorData<float>(res, "/topk_indices.bin");

    ProgramData::GetInstance().PrepareData({xData}, {trans0Data, reduce0Data, trans1Data, reduce1Data, resZeroData},
        {trans0GoldenData, reduce0GoldenData, trans1GoldenData, reduce1GoldenData, resGoldenData});
    if (isGenSlc) {
        GenSlc(x, trans0, reduce0, trans1, reduce1, topkInd, topkVal, res, tmp_s_slc);
    } else {
        GenTopkIndicesFun(x, trans0, reduce0, trans1, reduce1, topkInd, topkVal, res, tmp_s_slc);
    }

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {xData}, {trans0Data, reduce0Data, trans1Data, reduce1Data, topkIndData, topkValData, resZeroData});
    if (isGenSlc) {
        std::cout << "trans0 ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(trans0Golden, (T *)trans0Data->data(), 0.008f, NUM_16));
        std::cout << "reduce0 ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(reduce0Golden, (T *)reduce0Data->data(), 0.008f, NUM_16));
        std::cout << "trans1 ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(trans1Golden, (T *)trans1Data->data(), 0.008f, 0));
        std::cout << "reduce1 ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(reduce1Golden, (T *)reduce1Data->data(), 0.008f, NUM_16, false, false, 128));
    }

    std::cout << "topkInd ====== " << std::endl;
    EXPECT_TRUE(
        resultCmp<float>(topkIndicesGolden, (float *)topkIndData->data(), 0.008f, 32, NUM_16, false, false, NUM_20));
    std::cout << "Genslc ====== " << std::endl;
    EXPECT_TRUE(
        resultCmp<float>(topkIndicesGolden, (float *)resZeroData->data(), 0.008f, 0, NUM_16, false, false, NUM_20));
#endif
}

template <typename T = float16>
void TestGenslcV2(const SimpleParams &params, int topk_actual_len = 0) {
    int n = params.n;
    int s2 = params.s2;
    int windowStride = 16, windowSize = 32;
    int s_cmp = (s2 - windowSize) / windowStride + 1;
    int s_cmp_valid = (topk_actual_len - windowSize) / windowStride + 1;
    int validSize = (s_cmp_valid + 3) / 4;

    DataType dType = (std::is_same<T, float16>::value) ? DT_FP16 : DT_BF16;

    std::vector<int64_t> x_shape = {n, s_cmp};
    std::vector<int64_t> resShape = {1, 13};

    Tensor x(dType, x_shape, "x");
    Tensor res(DT_FP32, resShape, "res");

    std::vector<float> topkIndicesGolden = getGoldenVec<float>(resShape, "/topk_indices.bin");

    auto xData = CreateTensorData<T>(x, "/p_cmp.bin");
    auto resZeroData = RawTensorData::CreateConstantTensor<float>(res, 0.0);

    GenSlcV2(x, res, validSize);

#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {xData}, {resZeroData});
    EXPECT_TRUE(
        resultCmp<float>(topkIndicesGolden, (float *)resZeroData->data(), 0.008f, 0, NUM_16, false, false, NUM_20));
#endif
}

TEST_F(DyNsa, GateScore_b16_s1_fp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = NUM_16;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, GateScore_b16_s1_bf) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = NUM_16;
    TestNsa<bfloat16>(params);
}

TEST_F(DyNsa, GateScore_b32_s1_fp) {
    TestNsa<float16>(SimpleParams::getHighParams());
}

TEST_F(DyNsa, GateScore_b32_s2_fp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.s = NUM2;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, GateScore_b24_s1_fp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = NUM_24;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, GateScore_b48_s2_fp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = NUM_48;
    params.s = NUM2;
    TestNsa<float16>(params);
}

// IMPORTANT
TEST_F(DyNsa, GateScore_b32_s2_bf) {
    SimpleParams params = SimpleParams::getHighParams();
    params.s = NUM2;
    TestNsa<bfloat16, float, true>(params);
}

TEST_F(DyNsa, GateScore_b48_s1_fp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = NUM_48;
    TestNsa<float16, float, true>(params);
}

TEST_F(DyNsa, gateScore_mini) {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, gateScore_mini_batch16) {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    params.b = NUM_16;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, gateScore_mini_mtp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    params.s = NUM_2;
    TestNsa<float16>(params);
}

TEST_F(DyNsa, gateScore_mini_mtp_bf16) {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    params.s = NUM_2;
    TestNsa<npu::tile_fwk::bfloat16>(params);
}

TEST_F(DyNsa, GenSlc_b1_s1_fp_8k) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;

    TestGenslc<float16>(params, params.s2, true);
}

TEST_F(DyNsa, GenSlc_b1_s1_fp_4k) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslc<float16>(params, NUM_4096, true);
}

TEST_F(DyNsa, GenSlc_b1_s1_fp_6k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslcV2<float16>(params, NUM_6144 + 1);
}

TEST_F(DyNsa, GenSlc_b1_s1_bf_1k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslcV2<npu::tile_fwk::bfloat16>(params, NUM_1024 + 1);
}

TEST_F(DyNsa, GenSlc_b1_s1_fp_4k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslc<float16>(params, NUM_4096 + 1, true);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_8k_dyn) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_4k_dyn) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_4k1_dyn) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2 + 1);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_6k1_dyn) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_6144;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2 + 1);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_8k) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_4k) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_4k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096;
    params.n2 = 1;
    TestGenslc<float16>(params, params.s2 + 1);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_6k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_8192;
    params.n2 = 1;
    TestGenslc<float16>(params, NUM_6144 + 1);
}

TEST_F(DyNsa, TestView) {
    TestView();
}

TEST_F(DyNsa, TestAlignRead) {
    TestAlignRead(true);
}

TEST_F(DyNsa, TestUnAlignRead) {
    TestAlignRead(false);
}

TEST_F(DyNsa, TestMultiLoopAlignRead) {
    TestMultiLoopAlignRead();
}

} // namespace
