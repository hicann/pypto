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
#include "gtest/gtest.h"
#include "operator/models/deepseek/dynamic_nsa.h"
#include "interface/configs/config_manager.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"

using namespace npu::tile_fwk;

class DyNsa : public testing::Test {
    void SetUp() override {
        config::SetHostOption(ONLY_CODEGEN, true);
        config::SetPassOption(VEC_NBUFFER_MODE, 1);
        config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
        config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3, 4}});
        config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);

    }
};


template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestNsa(const SimpleParams &params) {
    int b = params.b;
    int s = params.s;
    int n = params.n;
    int h = params.h;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> gateW1Shape = {h, 4 * h};
    std::vector<int64_t> gateW2Shape = {4 * h, 3 * n};
    std::vector<int64_t> gateSimW1Shape = {h, 3 * n};
    //    std::vector<int64_t> gatingScoreShape = {b, n, s, 3};
    std::vector<int64_t> gatingScoreShape = {b, s, n, 3};
    std::vector<int64_t> tempShape = {b * s, n * 3};
    std::vector<int64_t> mm1Shape = {b * s, 4 * h};

    Tensor x(dType, x_shape, "x");
    Tensor gateW1(dType, gateW1Shape, "gateW1");
    Tensor gateW2(dType, gateW2Shape, "gateW2");
    Tensor gateSimW1(dType, gateSimW1Shape, "gateSimW1");
    Tensor gatingScore(dType, gatingScoreShape, "gatingScore");
    GenGatedScoreCompute(x, gateW1, gateW2, gateSimW1, gatingScore, GateMode::standard);
}


template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestGenslc(const SimpleParams &params,int topk_actual_len = 0, bool isGenSlc = false) {
    int n2 = params.n2;
    int n = params.n;
    int g = n / n2;
    int s2 = params.s2;
    int d = 16, w = 32, l_prime = 64;
    int s_cmp = (s2 - w) / d + 1;
    int out_loop = l_prime / d;
    int s_slc = (s_cmp + out_loop - 1) / out_loop;
    int tmp_s_scmp = (topk_actual_len-32)/16+1;
    int tmp_s_slc = (tmp_s_scmp+3)/4;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    std::vector<int64_t> x_shape = {n2, g, s_cmp};
    if (!isGenSlc) {
        x_shape = {1, s_slc};
    }
    std::vector<int64_t> trans0Shape = {n2, s_cmp, g};
    std::vector<int64_t> reduce0Shape = {n2, s_slc, g};
    std::vector<int64_t> trans1Shape = {n2, g, s_slc};
    std::vector<int64_t> reduce1Shape = {n2, 1, s_slc};
    std::vector<int64_t> resShape = {1, 16};

    Tensor x(dType, x_shape, "x");
    Tensor trans0(dType, trans0Shape, "trans0");
    Tensor reduce0(dType, reduce0Shape, "reduce0");
    Tensor trans1(dType, trans1Shape, "trans1");
    Tensor reduce1(dType, reduce1Shape, "reduce1");
    Tensor topkInd(DT_FP32, resShape, "topkInd");
    Tensor topkVal(DT_FP32, resShape, "topkVal");
    Tensor res(DT_FP32, resShape, "res");

    if(isGenSlc){
        GenSlc(x, trans0, reduce0, trans1, reduce1, topkInd, topkVal, res, tmp_s_slc);
    } else {
        GenTopkIndicesFun(x, trans0, reduce0, trans1, reduce1, topkInd, topkVal, res, tmp_s_slc);
    }
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

    GenSlcV2(x, res, validSize);
}

TEST_F(DyNsa, gateScore_mini_mtp) {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    params.s = NUM_2;
    TestNsa<npu::tile_fwk::float16>(params);
}

TEST_F(DyNsa, GenSlc_b1_s1_fp_6k1) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096 * 2;
    params.n2 = 1;
    TestGenslc<npu::tile_fwk::float16>(params,(4096+1024*2)+1,true);
}

TEST_F(DyNsa, GenTopk_b1_s1_fp_6k1_dyn) {
    SimpleParams params = SimpleParams::getHighParams();
    params.b = 1;
    params.s2 = NUM_4096 + NUM_1024 * 2;
    params.n2 = 1;
    TestGenslc<npu::tile_fwk::float16>(params,params.s2+1);
}