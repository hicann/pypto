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
 * \file dynamic_mla.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "operator/models/deepseek/dynamic_nsa.h"

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
namespace npu::tile_fwk {
std::vector<Tensor> GenTopkIndices(
    const Tensor &tmpOut, int s_slc, int actualTopk, SymbolicScalar validSize, bool isDyn) {
    std::vector<Tensor> res;
    TileShape::Current().SetVecTile({1, s_slc});
    auto view0 = View(tmpOut, {1, 128}, {1, validSize}, {0, 1});
    if (!isDyn) {
        view0 = View(tmpOut, {1, validSize}, {0, 1});
    }
    TileShape::Current().SetVecTile({1, s_slc});
    auto topk_idx = std::get<1>(TopK(view0, 16, -1, true)); // 13
    topk_idx = Cast(topk_idx, DataType::DT_FP32);
    topk_idx = Add(topk_idx, Element(DT_FP32, 1.0f));
    res.emplace_back(topk_idx);

    topk_idx = View(topk_idx, {1, 16}, {1, actualTopk}, {0, 0});
    if (!isDyn) {
        topk_idx = View(topk_idx, {1, actualTopk}, {0, 0});
    }
    auto out32 = std::get<0>(TopK(topk_idx, 16, -1, false));
    res.emplace_back(out32);
    return res;
}

std::vector<Tensor> singleTopk(const Tensor &tmpOut, int actualValidLen) {
    std::vector<Tensor> res;
    TileShape::Current().SetVecTile({1, 128});
    auto view0 = View(tmpOut, {1, 128}, {1, actualValidLen}, {0, 1});
    TileShape::Current().SetVecTile({1, 128});
    auto topk_idx = std::get<1>(TopK(view0, 16, -1, true));
    topk_idx = Cast(topk_idx, DataType::DT_FP32);
    res.emplace_back(topk_idx);
    return res;
}

void GenSlc(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int l_prime, int d, int front, int near, int topk) {
    int n2 = x.GetShape()[0]; // 1
    assert(n2 == 1);
    int g = x.GetShape()[1];         // 128
    int s_cmp = x.GetShape()[2];     // 511
    int s_slc = (s_cmp + 3) / 4; // 128
    int loop = s_slc;
    int out_loop = l_prime / d;                      // 4
    int actualTopk = topk - (front + near);          // 13
    int actualVaildLen = actualLen - (front + near); // 125

    int tileS2 = s_cmp;
    SymbolicScalar sLoop = s_cmp / tileS2;
    Tensor tmpOut(DataType::DT_FP32, {1, g}, "tmpout");
    Tensor tmpOut1(DataType::DT_FP32, {1, 16}, "tmpout1");
    Tensor tmpTrans2(DataType::DT_FP32, {1, s_cmp, 128}, "trans1");

    FUNCTION(
        "main", {x}, {trans0res, reduce0res, trans1res, reduce1res, topkInd, topkVal, out}) {
        LOOP("LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1), {}, true) {
            SymbolicScalar sOfs = sIdx * tileS2;
            TileShape::Current().SetVecTile({1, 4, s_cmp});
            auto viewer = View(x, {n2, g, s_cmp}, {0, 0, sOfs});
            auto input32 = Cast(viewer, DataType::DT_FP32); // 1,128,511
            auto tmpTrans = Transpose(input32, {1, 2});     // 1,511,128
            Assemble(tmpTrans, {0, 0, 0}, tmpTrans2);
            TileShape::Current().SetVecTile({1, 16, g});
            trans0res = Cast(tmpTrans2, DataType::DT_FP16);
            Tensor abc(DataType::DT_FP16, {n2, loop, g}, "reduce0");
            for (int i = 0; i < loop; i++) {
                auto maxLen0 = std::min(out_loop, s_cmp - i * out_loop);
                auto view0 = View(tmpTrans, {1, maxLen0, g}, {0, i * out_loop, 0}); // 1,4,128
                auto maxLen1 = std::min(out_loop, s_cmp - i * out_loop - 1);
                TileShape::Current().SetVecTile({1, 8, g});
                auto reduce0 = Sum(view0, 1, true); // 1,1,128
                if (maxLen1 > 0) {
                    auto view1 = View(tmpTrans, {1, maxLen1, g}, {0, i * out_loop + 1, 0}); // 1,4,128
                    auto reduce1 = Sum(view1, 1, true);                                  // 1,1,128
                    auto sum = Add(reduce0, reduce1);                                       // 1,1,128
                    auto sumTmp = Cast(sum, DataType::DT_FP16);
                    Assemble(sumTmp, {0, i, 0}, abc);
                } else {
                    auto reduceTmp = Cast(reduce0, DataType::DT_FP16);
                    Assemble(reduceTmp, {0, i, 0}, abc);
                }
            }
            reduce0res = abc;
            auto trans1 = Transpose(Cast(abc, DataType::DT_FP32), {1, 2}); // 1,128,128
            trans1res = Cast(trans1, DataType::DT_FP16);
            TileShape::Current().SetVecTile({1, g, 8});
            auto reduce2 = Sum(trans1, 1, true); // 1,1,128
            tmpOut = Reshape(reduce2, {1, 128});
            reduce1res = Cast(reduce2, DataType::DT_FP16);
        }
        LOOP("LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, true);
            out = res[1];
            topkInd = res[0];
        }
    }
}

void GenSlcV2(const Tensor &x, Tensor &out, int validSize, int l_prime, int d, int front, int near, int topk) {
    int n = x.GetShape()[0];         // 128
    int s_cmp = x.GetShape()[1];     // 511
    int s_slc = (s_cmp + 3) / 4; // 128
    int loop = s_slc;
    int out_loop = l_prime / d;                      // 4
    int actualTopk = topk - (front + near);          // 13
    int actualVaildLen = validSize - (front + near); // 125
    Tensor tmpOut(DataType::DT_FP32, {1, s_slc}, "tmpout");

    FUNCTION("main", {x}, {out}) {
        LOOP("LOOP_L0_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            TileShape::Current().SetVecTile({4, s_cmp});
            auto viewer = View(x, {n, s_cmp}, {0, 0});
            auto input32 = Cast(viewer, DataType::DT_FP32); // 128,511
            auto tmpTrans = Transpose(input32, {0, 1});     // 511,128
            TileShape::Current().SetVecTile({16, n});
            Tensor abc(DataType::DT_FP16, {loop, n}, "reduce0");
            for (int i = 0; i < loop; i++) {
                auto maxLen0 = std::min(out_loop, s_cmp - i * out_loop);
                auto view0 = View(tmpTrans, {maxLen0, n}, {i * out_loop, 0}); // 4,128
                auto maxLen1 = std::min(out_loop, s_cmp - i * out_loop - 1);
                TileShape::Current().SetVecTile({8, n});
                auto reduce0 = Sum(view0, 0, true); // 1,128
                if (maxLen1 > 0) {
                    auto view1 = View(tmpTrans, {maxLen1, n}, {i * out_loop + 1, 0}); // 4,128
                    auto reduce1 = Sum(view1, 0, true);                            // 1,128
                    auto sum = Add(reduce0, reduce1);                                 // 1,128
                    auto sumTmp = Cast(sum, DataType::DT_FP16);
                    Assemble(sumTmp, {i, 0}, abc);
                } else {
                    auto reduceTmp = Cast(reduce0, DataType::DT_FP16);
                    Assemble(reduceTmp, {i, 0}, abc);
                }
            }
            auto trans1 = Transpose(Cast(abc, DataType::DT_FP32), {0, 1}); // 128,128
            TileShape::Current().SetVecTile({n, 8});
            auto reduce2 = Sum(trans1, 0, true); // 1,128
            tmpOut = Reshape(reduce2, {1, s_slc});
        }
        LOOP("LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, true);
            out = res[1];
        }
    }
}

void GenTopkIndicesFun(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int front, int near) {
    int s_slc = x.GetShape()[1];                         // 128
    int actualVaildLen = actualLen - (front + near); // 125
    Tensor tmpOut(DataType::DT_FP32, {1, s_slc}, "tmpout");
    Tensor tmpOut1(DataType::DT_FP32, {1, 16}, "tmpout1");

    FUNCTION(
        "main", {x}, {trans0res, reduce0res, trans1res, reduce1res, topkInd, topkVal, out}) {
        LOOP("LOOP_topk0", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
            TileShape::Current().SetVecTile({1, s_slc});
            tmpOut = Cast(x, DT_FP32);
        }
        LOOP("LOOP_topk1", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1), {}, true) {
            (void)sIdx;
#define single_topk
#ifdef single_topk
            std::vector<Tensor> res = singleTopk(tmpOut, actualVaildLen);
            topkInd = res[0];
#else
            std::vector<Tensor> res = GenTopkIndices(tmpOut, s_slc, actualTopk, actualVaildLen, isDyn);
            out = res[1];
            topkInd = res[0];
#endif
        }
    }
}
} // namespace npu::tile_fwk
