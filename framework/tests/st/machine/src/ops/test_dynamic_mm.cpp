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
 * \file test_dynamic_mm.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

using BiasScaleShapeTuple =
    std::tuple<std::vector<int64_t>, std::vector<SymbolicScalar>, std::vector<int64_t>, std::vector<SymbolicScalar>>;

namespace {
const size_t MM_SHAPE_SIZE = 3;
const size_t MM_VIEW_SHAPE_SIZE = 2;
const size_t MM_SHAPE_N_IDX = 2;
const int QUANT_PERTENSOR = 1;
const int QUANT_PERCHANNEL = 2;
const int NO_RELU = 0;
const int RELU = 1;
constexpr double EPSILON = 1e-9;

class DynamicMatmulTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <bool T_transA, bool T_transB, bool T_CNz>
struct MatrixInputs {
    static constexpr bool transA = T_transA;
    static constexpr bool transB = T_transB;
    static constexpr bool isCNz = T_CNz;
};

template <typename T_inputDtype, typename T_outputDtype, typename T_biasDtype, typename T_scaleDtype,
    typename T_matrixInputs>
struct MatmulImpl {
    using inputDtype = T_inputDtype;
    using outputDtype = T_outputDtype;
    using biasDtype = T_biasDtype;
    using scaleDtype = T_scaleDtype;
    using cfg = T_matrixInputs;
};

struct MatrixOpParams {
    bool has_bias;
    int quant_mode;
    int relu_type;
    std::vector<int64_t> mmShape;
    bool isANz;
    bool isBNz;
    std::vector<int64_t> viewShape;
    std::string dataPath;
    float scaleValue;
};

struct SplitFuncParam {
    Matrix::MatmulExtendParam param;
    std::vector<int64_t> viewShape;
    Tensor tensor_a;
    Tensor tensor_b;
    Tensor tensor_c;
};

inline SymbolicScalar CeilDivSymbolicScalar(SymbolicScalar a, int64_t b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename dtype>
Tensor constructMatmulTensor(const std::vector<int64_t> &shape, const string &name, bool isNz) {
    auto dataType = GetAstDtype<dtype>();
    if (isNz) {
        return Tensor(dataType, shape, name, TileOpFormat::TILEOP_NZ);
    }
    return Tensor(dataType, shape, name);
}

BiasScaleShapeTuple GetBiasAndScaleShape(const SplitFuncParam &splitFuncParam) {
    const auto parseTensor = [](const Tensor &tensor) {
        const std::vector<int64_t> shape = tensor.GetStorage() ? tensor.GetShape() : std::vector<int64_t>{};
        const std::vector<SymbolicScalar> validShape =
            shape.empty() ? std::vector<SymbolicScalar>{} : std::vector<SymbolicScalar>{shape[0], shape[1]};
        return std::make_tuple(shape, validShape);
    };

    const auto [biasShape, biasValid] = parseTensor(splitFuncParam.param.biasTensor);
    const auto [scaleShape, scaleValid] = parseTensor(splitFuncParam.param.scaleTensor);
    return std::make_tuple(biasShape, biasValid, scaleShape, scaleValid);
}

void SetMatmulExtendParams(
    SplitFuncParam &splitFuncParam, BiasScaleShapeTuple biasScaleShapeTuple, Matrix::MatmulExtendParam &extendParam) {
    const auto [biasShape, biasValidShape, scaleShape, scaleValidShape] = biasScaleShapeTuple;

    if (splitFuncParam.param.biasTensor.GetStorage() != nullptr) {
        extendParam.biasTensor = View(splitFuncParam.param.biasTensor, biasShape, biasValidShape, {0, 0});
    }

    if (splitFuncParam.param.scaleTensor.GetStorage() != nullptr) {
        extendParam.scaleTensor = View(splitFuncParam.param.scaleTensor, scaleShape, scaleValidShape, {0, 0});
    }
    extendParam.reluType = splitFuncParam.param.reluType;
    extendParam.scaleValue = splitFuncParam.param.scaleValue;
}

// 扩展bias和scale接口
template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NonSplitFuncWithBiasAndScale(SplitFuncParam &splitFuncParam) {
    const auto &aShape = splitFuncParam.tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = splitFuncParam.tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};
    const BiasScaleShapeTuple biasScaleShapeTuple = GetBiasAndScaleShape(splitFuncParam);

    FUNCTION("testNoSplit",
        {splitFuncParam.tensor_a, splitFuncParam.tensor_b, splitFuncParam.param.biasTensor,
            splitFuncParam.param.scaleTensor},
        {splitFuncParam.tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(splitFuncParam.tensor_a, aShape, aValidShape, {mIdx, 0});
            Tensor dyn_b = View(splitFuncParam.tensor_b, bShape, bValidShape, {0, 0});
            Matrix::MatmulExtendParam extendParam;
            SetMatmulExtendParams(splitFuncParam, biasScaleShapeTuple, extendParam);
            splitFuncParam.tensor_c =
                Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, extendParam, transA, transB, isCNz);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NonSplitFunc(SplitFuncParam &splitFuncParam) {
    if (splitFuncParam.param.biasTensor.GetStorage() != nullptr ||
        splitFuncParam.param.scaleTensor.GetStorage() != nullptr || fabs(splitFuncParam.param.scaleValue - 0) > EPSILON) {
        NonSplitFuncWithBiasAndScale<outputDtype, transA, transB, isCNz>(splitFuncParam);
        return;
    }
    const auto &aShape = splitFuncParam.tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = splitFuncParam.tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};

    FUNCTION("testNoSplit",
        {splitFuncParam.tensor_a, splitFuncParam.tensor_b, splitFuncParam.param.biasTensor,
            splitFuncParam.param.scaleTensor},
        {splitFuncParam.tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(1)) {
            Tensor dyn_a = View(splitFuncParam.tensor_a, aShape, aValidShape, {mIdx, 0});
            Tensor dyn_b = View(splitFuncParam.tensor_b, bShape, bValidShape, {0, 0});
            splitFuncParam.tensor_c = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MSplitFunc(SplitFuncParam &splitFuncParam) {
    const auto &aShape = splitFuncParam.tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = splitFuncParam.tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};
    const std::vector<int64_t> &viewShape = splitFuncParam.viewShape;

    FUNCTION("testMSplit",
        {splitFuncParam.tensor_a, splitFuncParam.tensor_b, splitFuncParam.param.biasTensor,
            splitFuncParam.param.scaleTensor},
        {splitFuncParam.tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[1] : aShape[0], viewShape[0]), 1)) {
            Tensor dyn_a;
            if (transA) {
                dyn_a = View(splitFuncParam.tensor_a, {aShape[0], viewShape[0]},
                    {aShape[0], std::min(aShape[1] - viewShape[0] * mIdx, viewShape[0])}, {0, mIdx * viewShape[0]});
            } else {
                dyn_a = View(splitFuncParam.tensor_a, {viewShape[0], aShape[1]},
                    {std::min(aShape[0] - viewShape[0] * mIdx, viewShape[0]), aShape[1]}, {mIdx * viewShape[0], 0});
            }
            Tensor dyn_b = View(splitFuncParam.tensor_b, bShape, bValidShape, {0, 0});
            Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {mIdx * viewShape[0], 0}, splitFuncParam.tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void NSplitFunc(SplitFuncParam &splitFuncParam) {
    const auto &aShape = splitFuncParam.tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = splitFuncParam.tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};
    const std::vector<int64_t> &viewShape = splitFuncParam.viewShape;

    FUNCTION("testNSplit",
        {splitFuncParam.tensor_a, splitFuncParam.tensor_b, splitFuncParam.param.biasTensor,
            splitFuncParam.param.scaleTensor},
        {splitFuncParam.tensor_c}) {
        LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
            LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[0] : bShape[1], viewShape[1]), 1)) {
            Tensor dyn_a = View(splitFuncParam.tensor_a, aShape, aValidShape, {0, 0});
            Tensor dyn_b;
            if (!transB) {
                dyn_b = View(splitFuncParam.tensor_b, {bShape[0], viewShape[1]},
                    {bShape[0], std::min(bShape[1] - viewShape[1] * nIdx, viewShape[1])}, {0, nIdx * viewShape[1]});
            } else {
                dyn_b = View(splitFuncParam.tensor_b, {viewShape[1], bShape[1]},
                    {std::min(bShape[0] - viewShape[1] * nIdx, viewShape[1]), bShape[1]}, {nIdx * viewShape[1], 0});
            }
            Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
            Assemble(res, {0, nIdx * viewShape[1]}, splitFuncParam.tensor_c);
        }
    }
}

template <typename outputDtype, bool transA, bool transB, bool isCNz>
static void MNSplitFunc(SplitFuncParam &splitFuncParam) {
    const auto &aShape = splitFuncParam.tensor_a.GetShape();
    std::vector<SymbolicScalar> aValidShape = {aShape[0], aShape[1]};
    const auto &bShape = splitFuncParam.tensor_b.GetShape();
    std::vector<SymbolicScalar> bValidShape = {bShape[0], bShape[1]};
    const std::vector<int64_t> &viewShape = splitFuncParam.viewShape;

    FUNCTION("testMNSplit",
        {splitFuncParam.tensor_a, splitFuncParam.tensor_b, splitFuncParam.param.biasTensor,
            splitFuncParam.param.scaleTensor},
        {splitFuncParam.tensor_c}) {
        LOOP("mLoop", FunctionType::DYNAMIC_LOOP, mIdx,
            LoopRange(0, CeilDivSymbolicScalar(transA ? aShape[1] : aShape[0], viewShape[0]), 1)) {
            LOOP("nLoop", FunctionType::DYNAMIC_LOOP, nIdx,
                LoopRange(0, CeilDivSymbolicScalar(transB ? bShape[0] : bShape[1], viewShape[1]), 1)) {
                Tensor dyn_a;
                if (transA) {
                    dyn_a = View(splitFuncParam.tensor_a, {aShape[0], viewShape[0]},
                        {aShape[0], std::min(aShape[1] - viewShape[0] * mIdx, viewShape[0])}, {0, mIdx * viewShape[0]});
                } else {
                    dyn_a = View(splitFuncParam.tensor_a, {viewShape[0], aShape[1]},
                        {std::min(aShape[0] - viewShape[0] * mIdx, viewShape[0]), aShape[1]}, {mIdx * viewShape[0], 0});
                }
                Tensor dyn_b;
                if (!transB) {
                    dyn_b = View(splitFuncParam.tensor_b, {bShape[0], viewShape[1]},
                        {bShape[0], std::min(bShape[1] - viewShape[1] * nIdx, viewShape[1])}, {0, nIdx * viewShape[1]});
                } else {
                    dyn_b = View(splitFuncParam.tensor_b, {viewShape[1], bShape[1]},
                        {std::min(bShape[0] - viewShape[1] * nIdx, viewShape[1]), bShape[1]}, {nIdx * viewShape[1], 0});
                }
                Tensor res = Matrix::Matmul(GetAstDtype<outputDtype>(), dyn_a, dyn_b, transA, transB, isCNz);
                Assemble(res, {mIdx * viewShape[0], nIdx * viewShape[1]}, splitFuncParam.tensor_c);
            }
        }
    }
}

template <typename MatmulImplType>
void TestDynMatmul(MatrixOpParams &opParams) {
    SetInterpreterConfig();

    if (opParams.mmShape.size() != MM_SHAPE_SIZE || opParams.viewShape.size() != MM_VIEW_SHAPE_SIZE) {
        return;
    }

    using inputDtype = typename MatmulImplType::inputDtype;
    using outputDtype = typename MatmulImplType::outputDtype;
    using biasDtype = typename MatmulImplType::biasDtype;
    using scaleDtype = typename MatmulImplType::scaleDtype;

    int64_t m = opParams.mmShape[0];
    int64_t k = opParams.mmShape[1];
    int64_t n = opParams.mmShape[MM_SHAPE_N_IDX];

    Tensor tensor_a = MatmulImplType::cfg::transA ?
                          constructMatmulTensor<inputDtype>({k, m}, "tensor_a", opParams.isANz) :
                          constructMatmulTensor<inputDtype>({m, k}, "tensor_a", opParams.isANz);
    Tensor tensor_b = MatmulImplType::cfg::transB ?
                          constructMatmulTensor<inputDtype>({n, k}, "tensor_b", opParams.isBNz) :
                          constructMatmulTensor<inputDtype>({k, n}, "tensor_b", opParams.isBNz);
    Tensor tensor_c = constructMatmulTensor<outputDtype>({m, n}, "tensor_c", MatmulImplType::cfg::isCNz);
    Tensor tensor_bias = opParams.has_bias ? constructMatmulTensor<biasDtype>({1, n}, "tensor_bias", false) : Tensor();
    Tensor tensor_scale = opParams.quant_mode == QUANT_PERCHANNEL ?
                              Tensor(DT_UINT64, {1, n}, "tensor_scale", TileOpFormat::TILEOP_ND) :
                              Tensor();

    std::vector<inputDtype> aData(m * k, 0);
    std::vector<inputDtype> bData(k * n, 0);
    std::vector<outputDtype> golden(m * n, 0);
    std::vector<biasDtype> biasData(opParams.has_bias ? 1 * n : 0);
    std::vector<scaleDtype> scaleData(opParams.quant_mode == QUANT_PERCHANNEL ? 1 * n : 0);

    readInput<inputDtype>(opParams.dataPath + "/mat_a.bin", aData);
    readInput<inputDtype>(opParams.dataPath + "/mat_b.bin", bData);
    readInput<outputDtype>(opParams.dataPath + "/mat_c.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<inputDtype>(tensor_a, aData),
        RawTensorData::CreateTensor<inputDtype>(tensor_b, bData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<outputDtype>(tensor_c, 0.0f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<outputDtype>(tensor_c, golden),
    });

    if (opParams.has_bias) {
        readInput<biasDtype>(opParams.dataPath + "/mat_bias.bin", biasData);
        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateTensor<biasDtype>(tensor_bias, biasData),
        });
    } else {
        ProgramData::GetInstance().AppendInputs({nullptr});
    }
    if (opParams.quant_mode == QUANT_PERCHANNEL) {
        readInput<scaleDtype>(opParams.dataPath + "/mat_scale.bin", scaleData);
        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateTensor<scaleDtype>(tensor_scale, scaleData),
        });
    } else {
        ProgramData::GetInstance().AppendInputs({nullptr});
    }

    int64_t viewM = opParams.viewShape[0];
    int64_t viewN = opParams.viewShape[1];

    float scaleValue = (opParams.quant_mode == QUANT_PERTENSOR) ? opParams.scaleValue : 0.0f;
    Matrix::ReLuType reluType = (opParams.relu_type == NO_RELU) ? Matrix::ReLuType::NoReLu : Matrix::ReLuType::ReLu;
    SplitFuncParam funcParam = {
        Matrix::MatmulExtendParam(tensor_bias, tensor_scale, scaleValue, reluType),
        opParams.viewShape, tensor_a, tensor_b, tensor_c};

    if (viewM > 0 && viewN > 0) {
        MNSplitFunc<outputDtype, MatmulImplType::cfg::transA, MatmulImplType::cfg::transB, MatmulImplType::cfg::isCNz>(
            funcParam);
    } else if (viewM > 0) {
        MSplitFunc<outputDtype, MatmulImplType::cfg::transA, MatmulImplType::cfg::transB, MatmulImplType::cfg::isCNz>(
            funcParam);
    } else if (viewN > 0) {
        NSplitFunc<outputDtype, MatmulImplType::cfg::transA, MatmulImplType::cfg::transB, MatmulImplType::cfg::isCNz>(
            funcParam);
    } else {
        NonSplitFunc<outputDtype, MatmulImplType::cfg::transA, MatmulImplType::cfg::transB, MatmulImplType::cfg::isCNz>(
            funcParam);
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (outputDtype *)outs->data(), 0.001f));
}

// intput:fp16 output:fp16 bias:fp16
TEST_F(DynamicMatmulTest, mm_A_Bt_ND_fp16_BIAS) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t m = 128;
    int64_t k = 257;
    int64_t n = 511;
    bool isBNz = false;
    bool isANz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        true, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<npu::tile_fwk::float16, npu::tile_fwk::float16, npu::tile_fwk::float16, float,
        MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

// intput:fp16 output:fp32 bias:fp32
TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_fp16_BIAS) {
    int64_t m = 1;
    int64_t k = 512;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        true, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<npu::tile_fwk::float16, float, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

// intput:fp32 output:fp32 bias:fp32
TEST_F(DynamicMatmulTest, mm_A_B_NZ_fp32_BIAS) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t m = 16;
    int64_t k = 32;
    int64_t n = 512;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        true, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<float, float, float, float, MatrixInputs<false, false, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

// intput:int8 output:int32 bias:int32
TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_int8_BIAS) {
    int64_t m = 1;
    int64_t k = 512;
    int64_t n = 256;
    bool isBNz = true;
    bool isANz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    MatrixOpParams opParams = {
        true, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<int8_t, int32_t, int32_t, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_B_ND_bf16_BIAS) {
    TileShape::Current().SetCubeTile({64, 64}, {256, 256}, {128, 128});
    int64_t m = 129;
    int64_t k = 257;
    int64_t n = 513;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        true, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<npu::tile_fwk::float16, float, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

// PERCHANNEL
TEST_F(DynamicMatmulTest, mm_A_Bt_ND_int8_channel) {
    TileShape::Current().SetCubeTile({32, 32}, {512, 512}, {32, 32});
    int64_t m = 240;
    int64_t k = 512;
    int64_t n = 64;
    bool isBNz = false;
    bool isANz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, QUANT_PERCHANNEL, RELU, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<int8_t, npu::tile_fwk::float16, float, uint64_t, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_B_NZ_int8_tensor) {
    int64_t m = 16;
    int64_t k = 32;
    int64_t n = 512;
    bool isANz = false;
    bool isBNz = true;
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {512, 512});
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, QUANT_PERTENSOR, RELU, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 2.0f
    };
    using TestMatmulType = MatmulImpl<int8_t, npu::tile_fwk::float16, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_Bt_ND_fp16) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t m = 128;
    int64_t k = 257;
    int64_t n = 511;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, 256};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_fp16) {
    int64_t m = 1;
    int64_t k = 512;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<npu::tile_fwk::float16, float, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_B_NZ_fp32) {
    int64_t m = 16;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t k = 32;
    int64_t n = 512;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<float, float, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_int8) {
    int64_t m = 1;
    int64_t k = 512;
    int64_t n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<int8_t, int32_t, int32_t, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_B_ND_bf16) {
    int64_t m = 128;
    int64_t k = 256;
    int64_t n = 512;
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    TileShape::Current().SetCubeTile({64, 64}, {256, 256}, {128, 128});
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType = MatmulImpl<npu::tile_fwk::float16, float, float, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_Bt_NZ_int8_tile4) {
    TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {32, 32});
    int64_t m = 1;
    int64_t k = 512;
    int64_t n = 256;
    bool isANz = false;
    bool isBNz = true;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_A_ND_B_ND_C_NZ) {
    int64_t m = 16;
    int64_t k = 192;
    int64_t n = 128;
    bool isANz = false;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_AT_B_ANZ_BND_bf16) {
    int64_t m = 128;
    int64_t k = 256;
    int64_t n = 512;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    bool isANz = true;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_AT_BT_AND_BND_bf16) {
    int64_t m = 128;
    int64_t k = 256;
    bool isBNz = false;
    int64_t n = 512;
    bool isANz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_AT_B_AND_BND_fp32_UNALIGN) {
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    int64_t m = 127;
    int64_t k = 255;
    bool isBNz = false;
    int64_t n = 511;
    bool isANz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<false, true, false>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, mm_AT_BT_AND_BND_fp32) {
    int64_t m = 128;
    int64_t k = 256;
    int64_t n = 512;
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    bool isANz = false;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {-1, -1};
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<true, true, true>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

TEST_F(DynamicMatmulTest, test1_fp32) {
    int64_t m = 128;
    int64_t k = 256;
    int64_t n = 513;
    bool isANz = true;
    bool isBNz = false;
    std::vector<int64_t> viewShape = {32, 32};
    TileShape::Current().SetCubeTile({256, 256}, {64, 64}, {64, 64});
    MatrixOpParams opParams = {
        false, 0, 0, {m, k, n},
           isANz, isBNz, viewShape, GetGoldenDir(), 0.0f
    };
    using TestMatmulType =
        MatmulImpl<npu::tile_fwk::float16, float, npu::tile_fwk::float16, float, MatrixInputs<true, false, true>>;
    TestDynMatmul<TestMatmulType>(opParams);
}

} // namespace