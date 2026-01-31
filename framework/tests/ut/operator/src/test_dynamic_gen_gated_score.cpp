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
 * \file test_dynamic_gen_gated_score.cpp
 * \brief
 */

#include "tilefwk/tilefwk_op.h"
#include "test_cost_macro.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"
#include "operator/models/nsa/gen_gated_score.h"

using namespace npu::tile_fwk;

class DynamicGenGatedScoreUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

void DynamicFunction(const std::string &funcName,
    void (*execFunc)(const Tensor &x, const Tensor &, const Tensor &, Tensor &)) {
    std::vector<int64_t> bnsh = {4, 128, 4, 7168};

    int64_t b = bnsh[0];
    int64_t n = bnsh[1];
    int64_t s = bnsh[2];
    int64_t h = bnsh[3];

    DataType dType = DT_FP16;

    std::vector<int64_t> xShape = {b, s, h};
    std::vector<int64_t> w1Shape = {h, h * 4};
    std::vector<int64_t> w2Shape = {h * 4, n * 3};
    std::vector<int64_t> gatingScoreShape = {b, s, 3, n};

    Tensor x(dType, xShape, "x");
    Tensor w1(dType, w1Shape, "w1");
    Tensor w2(dType, w2Shape, "w2");
    Tensor gatingScore(dType, gatingScoreShape, "gatingScore");

    FUNCTION(funcName, {x, w1, w2}, {gatingScore}) {
        execFunc(x, w1, w2, gatingScore);
    }
}

TEST_F(DynamicGenGatedScoreUtest, utest_gen_gated_score_plus_dyn) {
    DynamicFunction("UTGENGATEDSCOREPLUS", GenGatedScoreComputePrefillPlus);
}

TEST_F_WITH_COST(DynamicGenGatedScoreUtest, utest_gen_gated_score_dyn, 58) {
    DynamicFunction("UTGENGATEDSCORE", GenGatedScoreComputePrefill);
}