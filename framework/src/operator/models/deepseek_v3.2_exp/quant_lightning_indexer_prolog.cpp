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
 * \file quant_lightning_indexer_prolog.cpp
 * \brief
 */

#include "quant_lightning_indexer_prolog.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

std::tuple<Tensor, Tensor> PrologQuant(const Tensor &input) {
    config::SetSemanticLabel("Prolog-Quant");
    constexpr const float s8_max_value = 127.0f;
    constexpr const float s8_one_value = 1.0f;
    auto inputFp32 = Cast(input, DataType::DT_FP32, CAST_NONE);

    auto absRes = Abs(inputFp32);
    auto maxValue = Amax(absRes, -1, true);
    auto temp127 = Full(Element(DT_FP32, s8_max_value), DT_FP32, maxValue.GetShape());

    auto scaleQuant = Div(temp127, maxValue);
    auto outFp32 = Mul(inputFp32, scaleQuant);
    auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
    auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
    auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC);
    auto temp1 = Full(Element(DT_FP32, s8_one_value), DT_FP32, scaleQuant.GetShape());
    auto scaleDeQuant = Div(temp1, scaleQuant);
    return std::tie(outInt8, scaleDeQuant);
} // namespace npu::tile_fwk

Tensor QuantLayerNorm(const Tensor &x, const Tensor &gamma, const Tensor &beta, const int dim, float epsilon) {
    config::SetSemanticLabel("Key-LayerNorm");
    ASSERT(dim == static_cast<int64_t>(x.GetShape().size()) - 1 || dim == -1)
        << "Only support Last axis QuantLayerNorm";
    int actualDim = dim < 0 ? dim + x.GetShape().size() : dim;
    auto xDtype = x.GetDataType();

    auto xFp32 = Cast(x, DT_FP32);
    // do division first to avoid overflow
    auto xScaled = Mul(xFp32, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto mean = Sum(xScaled, actualDim, true);

    auto diff = Sub(xFp32, mean);
    auto squaredDiff = Mul(diff, diff);
    auto squaredDiffScaled = Mul(squaredDiff, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto var = Sum(squaredDiffScaled, actualDim, true);
    // add epsilon to avoid division by zero
    auto varEps = Add(var, Element(DT_FP32, epsilon));
    auto stdVar = Sqrt(varEps);
    auto res32 = Div(diff, stdVar);

    auto gamma32 = Cast(gamma, DT_FP32);
    auto beta32 = Cast(beta, DT_FP32);

    return Cast(Add(Mul(res32, gamma32), beta32), xDtype);
}

Tensor LIPrologRotateHalf(const Tensor &input) {
    constexpr size_t chunk_size = 2;
    auto shape = input.GetShape();
    auto shapeSize = shape.size();
    assert(shapeSize >= 1 && "rope rotate_half input dim less than 1");
    assert(shape[shapeSize - 1] % chunk_size == 0 && "rope rotate_half last dim shape is even.");

    shape[shapeSize - 1] /= chunk_size;
    std::vector<int64_t> offset1(shapeSize, 0);
    std::vector<int64_t> offset2(shapeSize, 0);
    offset2[shapeSize - 1] = shape[shapeSize - 1];

    Tensor x1 = View(input, shape, offset1);
    Tensor x2 = View(input, shape, offset2);

    return Cat({Mul(x2, Element(x2.GetDataType(), -1.0)), Add(x1, Element(x1.GetDataType(), 0.0))}, -1);
}

Tensor QuantRope3D(const Tensor &x, const Tensor &cos, const Tensor &sin, const QuantIndexerConfigs &configs) {
    constexpr size_t query_rope_dim = 3;
    constexpr size_t head_num_axis = 1;
    constexpr size_t head_dim_axis = 2;
    ASSERT(x.GetShape().size() == query_rope_dim && cos.GetShape().size() == COS_SIN_DIM &&
           sin.GetShape().size() == COS_SIN_DIM);

    auto xDtype = x.GetDataType();
    int tTile = x.GetShape()[0];
    int headNum = x.GetShape()[head_num_axis];
    int ropeDim = x.GetShape()[head_dim_axis];

    TileShape::Current().SetVecTile(1, ropeDim);
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);

    TileShape::Current().SetVecTile(configs.tSubTile, headNum / configs.chunkSize, ropeDim);
    auto xView = Cast(x, DT_FP32);
    castCos = Reshape(castCos, {tTile, 1, ropeDim});
    castSin = Reshape(castSin, {tTile, 1, ropeDim});

    auto xEmbed = Add(Mul(xView, castCos), Mul(LIPrologRotateHalf(xView), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}

Tensor QuantRope2D(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    config::SetSemanticLabel("Key-Rope2D");
    constexpr size_t key_rope_dim = 2;
    auto xDtype = x.GetDataType();
    int tTile = x.GetShape()[0];
    int ropeDim = x.GetShape()[1];
    ASSERT(x.GetShape().size() == key_rope_dim && cos.GetShape().size() == COS_SIN_DIM &&
           sin.GetShape().size() == COS_SIN_DIM);

    TileShape::Current().SetVecTile(tTile, ropeDim);
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);
    auto xView = Cast(x, DT_FP32);

    TileShape::Current().SetVecTile(tTile, ropeDim);
    auto xEmbed = Add(Mul(xView, castCos), Mul(LIPrologRotateHalf(xView), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}

void QuantLightningIndexerPrologCompute(const QuantIndexerPrologInput &inputs, QuantIndexerPrologOutput &outputs,
    QuantIndexerPrologAttr &attrs, const QuantIndexerConfigs &configs) {
    config::SetPassOption("vec_nbuffer_mode", 0);
    config::SetPassOption("cube_l1_reuse_setting", configs.l1ReuseParam);
    config::SetPassOption("mg_copyin_upper_bound", configs.mgCopyInUpperBound);
    config::SetPassOption("pg_upper_bound", configs.pgUpperBound);

    ASSERT(inputs.x.GetShape().size() == Q_PARAM_DIM && inputs.qNorm.GetShape().size() == Q_PARAM_DIM &&
           inputs.wk.GetShape().size() == NZ_DIM && inputs.wProj.GetShape().size() == NZ_DIM &&
           inputs.cosIdxRope.GetShape().size() == Q_PARAM_DIM);
    DataType xDtype = inputs.x.GetDataType();

    // 动态轴需通过GetInputShape函数获取
    SymbolicScalar t = GetInputShape(inputs.x, 0);

    int64_t h = inputs.x.GetShape()[1];
    int64_t qLoraRank = inputs.qNorm.GetShape()[1];
    int64_t headNum = inputs.wProj.GetShape()[0] * NZ_B16_C0;
    int64_t headDim = inputs.hadamardQ.GetShape()[0];
    int64_t ropeHeadDim = inputs.cosIdxRope.GetShape()[1];

    Tensor kCacheIndex(inputs.kCacheIndex.GetDataType(), {t, 1}, "kCacheIndex");
    Tensor gamma2D(inputs.lnGammaK.GetDataType(), {1, inputs.lnGammaK.GetShape()[0]}, "gamma2D");
    Tensor beta2D(inputs.lnBetaK.GetDataType(), {1, inputs.lnBetaK.GetShape()[0]}, "beta2D");
    Tensor wQb(inputs.wQb.GetDataType(), {qLoraRank, headNum * headDim}, "wQb", TileOpFormat::TILEOP_NZ);
    Tensor wQbScale(inputs.wQbScale.GetDataType(), {1, headNum * headDim}, "wQbScale");
    Tensor wk(inputs.wk.GetDataType(), {h, headDim}, "wk", TileOpFormat::TILEOP_NZ);
    Tensor wProj(inputs.wProj.GetDataType(), {h, headNum}, "wProj", TileOpFormat::TILEOP_NZ);
    LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(1)) {
        (void)tIdx;
        kCacheIndex = Reshape(inputs.kCacheIndex, {t, 1}, true);
        wQbScale = Reshape(inputs.wQbScale, {1, headNum * headDim}, true);
        gamma2D = Reshape(inputs.lnGammaK, {1, inputs.lnGammaK.GetShape()[0]}, true);
        beta2D = Reshape(inputs.lnBetaK, {1, inputs.lnBetaK.GetShape()[0]}, true);
        // NZ Reshape
        wQb = Reshape(inputs.wQb, {qLoraRank, headNum * headDim}, true);
        wk = Reshape(inputs.wk, {h, headDim}, true);
        wProj = Reshape(inputs.wProj, {h, headNum}, true);
    }

    auto unrollList = configs.unrollList;
    LOOP("QuantIndexerPrologLoop", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(t), unrollList) {
        for (int unrollLength : unrollList) {
            UNROLL(unrollLength) {
                int tTile = unrollLength;
                // 获取query计算的各阶段Tile参数
                auto qLinear = configs.qLinear;
                auto qHd = configs.qHd;
                // 多分档内会将tTile作为档位，offset无需乘tTile
                auto qNorm = View(inputs.qNorm, {tTile, qLoraRank}, {tTile, qLoraRank}, {tIdx, 0});
                auto qNormScale = View(inputs.qNormScale, {tTile, 1}, {tTile, 1}, {tIdx, 0});
                config::SetSemanticLabel("Query-Linear");
                TileShape::Current().SetCubeTile({qLinear[L0M_INDEX], qLinear[L1M_INDEX]},
                    {qLinear[L0K_INDEX], qLinear[L1K_INDEX]}, {qLinear[L0N_INDEX], qLinear[L1N_INDEX]}, true);
                auto qS32 = Matrix::Matmul(DT_INT32, qNorm, wQb, false, false); // (tTile, headNum * headDim)

                config::SetSemanticLabel("Query-Dequant");
                TileShape::Current().SetVecTile(configs.tSubTile, headNum * headDim / configs.chunkSize);  // (tTile, headNum * headDim), fp32
                auto qF32 = Cast(qS32, DT_FP32);
                qF32 = Mul(qF32, qNormScale); // (tTile, headNum * headDim), fp32
                qF32 = Mul(qF32, wQbScale);   // (tTile, headNum * headDim), fp32
                auto qCast = Cast(qF32, xDtype);

                auto qBF16 = Reshape(qCast, {tTile, headNum, headDim}, {tTile, headNum, headDim});
                // UB View
                auto qRope = View(qBF16, {tTile, headNum, ropeHeadDim}, {tTile, headNum, ropeHeadDim}, {0, 0, 0});
                auto qNope = View(qBF16, {tTile, headNum, headDim - ropeHeadDim},
                    {tTile, headNum, headDim - ropeHeadDim}, {0, 0, ropeHeadDim});
                auto ropeCos = View(inputs.cosIdxRope, {tTile, ropeHeadDim}, {tTile, ropeHeadDim}, {tIdx, 0});
                auto ropeSin = View(inputs.sinIdxRope, {tTile, ropeHeadDim}, {tTile, ropeHeadDim}, {tIdx, 0});

                auto qRoped = QuantRope3D(qRope, ropeCos, ropeSin, configs); // {tTile, headNum, ropeHeadDim}
                TileShape::Current().SetVecTile(configs.tSubTile, headNum / configs.chunkSize, headDim);
                qNope = Cast(Cast(qNope, DT_FP32), qBF16.GetDataType());
                auto qCat = Cat({qRoped, qNope}, -1); // {tTile, headNum, headDim}
                auto hadamardQ = Reshape(inputs.hadamardQ, {1, headDim, headDim}, {1, headDim, headDim});

                config::SetSemanticLabel("Query-Hadamard");
                const int64_t cur_max_unroll = 32;
                int64_t qHdMTile = tTile < cur_max_unroll ? cur_max_unroll : qHd[L0M_INDEX];
                TileShape::Current().SetCubeTile({qHdMTile, qHdMTile}, {qHd[L0K_INDEX], qHd[L1K_INDEX]},
                    {qHd[L0N_INDEX], qHd[L1N_INDEX]});
                auto qHadamard =
                    Matrix::BatchMatmul(xDtype, qCat, hadamardQ, false, false, false); // (tTile, headNum, headDim)

                config::SetSemanticLabel("Query-Quant");
                TileShape::Current().SetVecTile(configs.tSubTile, headNum / configs.chunkSize, headDim);
                std::tuple<Tensor, Tensor> qRes = PrologQuant(qHadamard);
                auto qScale = Cast(std::get<1>(qRes), DT_FP16);

                Assemble(std::get<0>(qRes), {tIdx, 0, 0}, outputs.qInt8);
                Assemble(qScale, {tIdx, 0, 0}, outputs.qScale);

                // 获取key计算的各阶段Tile参数
                auto kLinear = configs.kLinear;
                config::SetSemanticLabel("Key-Linear");
                TileShape::Current().SetCubeTile({kLinear[L0M_INDEX], kLinear[L1M_INDEX]},
                    {kLinear[L0K_INDEX], kLinear[L1K_INDEX]}, {kLinear[L0N_INDEX], kLinear[L1N_INDEX]}, true);
                auto x = View(inputs.x, {tTile, h}, {tTile, h}, {tIdx, 0}); // 这里将tTile分档，offset不需要乘tTile
                auto k = Matrix::Matmul(DT_FP32, x, wk, false, false);      // (tTile, headDim)

                if (tTile <= VEC_TILE_32) {
                    TileShape::Current().SetVecTile(std::min(tTile, VEC_TILE_4), headDim);
                }else {
                    TileShape::Current().SetVecTile(std::min(tTile, VEC_TILE_32), headDim);
                }
                auto kBf16 = Cast(QuantLayerNorm(k, gamma2D, beta2D, -1, attrs.eps), xDtype);

                auto kRope = View(kBf16, {tTile, ropeHeadDim}, {tTile, ropeHeadDim}, {0, 0});
                auto kNope =
                    View(kBf16, {tTile, headDim - ropeHeadDim}, {tTile, headDim - ropeHeadDim}, {0, ropeHeadDim});
                auto kRoped = QuantRope2D(kRope, ropeCos, ropeSin); // (tTile, ropeHeadDim)
                TileShape::Current().SetVecTile(tTile, headDim);
                kNope = Cast(Cast(kNope, DT_FP32), kBf16.GetDataType());
                auto kCat = Cat({kRoped, kNope}, -1);

                config::SetSemanticLabel("Key-Hadamard");
                auto hadamardK =
                    Matrix::Matmul(xDtype, kCat, inputs.hadamardK, false, false); // (tTile, headDim), bf16
                config::SetSemanticLabel("Key-Quant");
                std::tuple<Tensor, Tensor> kRes = PrologQuant(hadamardK);
                auto kCache4D = Reshape(std::get<0>(kRes), {tTile, 1, 1, headDim}, {tTile, 1, 1, headDim});
                auto kScale4D = Reshape(Cast(std::get<1>(kRes), DT_FP16), {tTile, 1, 1, 1}, {tTile, 1, 1, 1});

                auto index = View(kCacheIndex, {tTile, 1}, {tTile, 1}, {tIdx, 0});
                TileShape::Current().SetVecTile(tTile, 1, 1, headDim);
                outputs.kInt8 =
                    ScatterUpdate(inputs.kCache, index, kCache4D, SCATTER_DIM, "PA_BSND", configs.blockSize);
                outputs.kScale =
                    ScatterUpdate(inputs.kCacheScale, index, kScale4D, SCATTER_DIM, "PA_BSND", configs.blockSize);

                config::SetSemanticLabel("Weight-Linear");
                auto wLinear = configs.wLinear;
                TileShape::Current().SetCubeTile({wLinear[L0M_INDEX], wLinear[L1M_INDEX]},
                    {wLinear[L0K_INDEX], wLinear[L1K_INDEX]}, {wLinear[L0N_INDEX], wLinear[L1N_INDEX]});
                TileShape::Current().SetVecTile(tTile, headNum);
                auto weights = Cast(Matrix::Matmul(xDtype, x, wProj, false, false), DT_FP32);
                weights = Mul(weights, Element(DataType::DT_FP32, 1.0f / (std::sqrt(headNum) * std::sqrt(headDim))));
                auto weightsF16 = Cast(weights, DT_FP16);
                Assemble(weightsF16, {tIdx, 0}, outputs.weights);
            }
        }
    }
}

void QuantLightningIndexerProlog(const QuantIndexerPrologInput &inputs, QuantIndexerPrologOutput &outputs,
    QuantIndexerPrologAttr &attrs, const QuantIndexerConfigs &configs) {
    // Machine Global Config
    config::SetRuntimeOption("device_sched_mode", static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));

    FUNCTION("QuantLightningIndexerProlog",
        {
            inputs.x, inputs.qNorm, inputs.qNormScale, inputs.wQb, inputs.wQbScale, inputs.wk, inputs.wProj,
            inputs.lnGammaK, inputs.lnBetaK, inputs.cosIdxRope, inputs.sinIdxRope, inputs.hadamardQ, inputs.hadamardK,
            inputs.kCache, inputs.kCacheScale, inputs.kCacheIndex
    },
        {outputs.qInt8, outputs.qScale, outputs.weights},
        {{outputs.kInt8, inputs.kCache}, {outputs.kScale, inputs.kCacheScale}}) {
        QuantLightningIndexerPrologCompute(inputs, outputs, attrs, configs);
    }
}

} // namespace npu::tile_fwk
