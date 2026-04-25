/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_quantize.cpp
 * \brief Unit test for quantize and dequantize codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynQuantize : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}

protected:
    static void RunCodeGenAndCheck(const std::string& funcName, const std::string& expect)
    {
        auto function = Program::GetInstance().GetFunctionByRawName(
            FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
        npu::tile_fwk::CodeGenCtx ctx;
        npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
        codeGen.GenCode(*function, {});
        CheckStringExist(expect, GetResultFromCpp(*function));
    }

    static void RunQuantizeTestSymmetric(
        const std::string& funcName, const std::string& expect,
        DataType inputType, DataType outputType,
        const std::vector<int64_t>& inputShape,
        const std::vector<int64_t>& scaleShape,
        const std::vector<int64_t>& outputShape,
        const std::vector<int64_t>& vecTile, int axis)
    {
        TileShape::Current().SetVecTile(vecTile);
        Tensor input(inputType, inputShape, "input");
        Tensor scale(DataType::DT_FP32, scaleShape, "scale");
        Tensor zeroPoints;
        Tensor output(outputType, outputShape, "output");

        FUNCTION(funcName, {input, scale, output})
        {
            LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = Quantize(input, scale, outputType, axis, zeroPoints);
            }
        }

        RunCodeGenAndCheck(funcName, expect);
    }

    static void RunQuantizeTestAsymmetric(
        const std::string& funcName, const std::string& expect,
        DataType inputType, DataType outputType,
        const std::vector<int64_t>& inputShape,
        const std::vector<int64_t>& scaleShape,
        const std::vector<int64_t>& zeroPointShape,
        const std::vector<int64_t>& outputShape,
        const std::vector<int64_t>& vecTile, int axis)
    {
        TileShape::Current().SetVecTile(vecTile);
        Tensor input(inputType, inputShape, "input");
        Tensor scale(DataType::DT_FP32, scaleShape, "scale");
        Tensor zeroPoints(DataType::DT_INT32, zeroPointShape, "zeroPoints");
        Tensor output(outputType, outputShape, "output");

        FUNCTION(funcName, {input, scale, zeroPoints, output})
        {
            LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = Quantize(input, scale, outputType, axis, zeroPoints);
            }
        }

        RunCodeGenAndCheck(funcName, expect);
    }

    static void RunDequantizeTest(
        const std::string& funcName, const std::string& expect,
        DataType inputType,
        const std::vector<int64_t>& inputShape,
        const std::vector<int64_t>& scaleShape,
        const std::vector<int64_t>& outputShape,
        const std::vector<int64_t>& vecTile, int axis)
    {
        TileShape::Current().SetVecTile(vecTile);
        Tensor input(inputType, inputShape, "input");
        Tensor scale(DataType::DT_FP32, scaleShape, "scale");
        Tensor zeroPoints(DataType::DT_FP32, scaleShape, "zeroPoints");
        Tensor output(DataType::DT_FP32, outputShape, "output");

        FUNCTION(funcName, {input, scale, zeroPoints, output})
        {
            LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = Dequantize(input, scale, DataType::DT_FP32, axis, zeroPoints);
            }
        }

        RunCodeGenAndCheck(funcName, expect);
    }
};

// Symmetric quantization: FP32 -> INT8, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, QuantizeSymmetricToInt8)
{
    RunQuantizeTestSymmetric(
        "QUANTIZE_SYMMETRIC_INT8",
        R"!!!(TQuant<pto::QuantType::INT8_SYM>(ubTensor_4, ubTensor_0, ubTensor_2);)!!!",
        DataType::DT_FP32, DataType::DT_INT8,
        {8, 128}, {8}, {8, 128}, {8, 128}, -1);
}

// Asymmetric quantization: FP32 -> UINT8, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, QuantizeAsymmetricToUInt8)
{
    RunQuantizeTestAsymmetric(
        "QUANTIZE_ASYMMETRIC_UINT8",
        R"!!!(TQuant<pto::QuantType::INT8_ASYM>(ubTensor_6, ubTensor_0, ubTensor_2, ubTensor_4);)!!!",
        DataType::DT_FP32, DataType::DT_UINT8,
        {16, 64}, {16}, {16}, {16, 64}, {16, 64}, -1);
}

// Symmetric quantization with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, QuantizeSymmetricAxisM2)
{
    RunQuantizeTestSymmetric(
        "QUANTIZE_SYMMETRIC_AXIS_M2",
        R"!!!(TQuant<pto::QuantType::INT8_SYM>(ubTensor_7, ubTensor_2, ubTensor_5);)!!!",
        DataType::DT_FP32, DataType::DT_INT8,
        {4, 256}, {256}, {4, 256}, {4, 128}, -2);
}

// Asymmetric quantization with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, QuantizeAsymmetricAxisM2)
{
    RunQuantizeTestAsymmetric(
        "QUANTIZE_ASYMMETRIC_AXIS_M2",
        R"!!!(TQuant<pto::QuantType::INT8_ASYM>(ubTensor_9, ubTensor_2, ubTensor_5, ubTensor_7);)!!!",
        DataType::DT_FP32, DataType::DT_UINT8,
        {4, 256}, {256}, {256}, {4, 256}, {4, 128}, -2);
}

// Dequantize: INT8 -> FP32, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, DequantizeInt8ToFP32)
{
    RunDequantizeTest(
        "DEQUANTIZE_INT8_FP32",
        R"!!!(TDequant<pto::DequantType::INT8>(ubTensor_6, ubTensor_0, ubTensor_2, ubTensor_4);)!!!",
        DataType::DT_INT8,
        {8, 128}, {8}, {8, 128}, {8, 128}, -1);
}

// Dequantize: INT16 -> FP32, axis=-1 (per-row)
TEST_F(TestCodegenDynQuantize, DequantizeInt16ToFP32)
{
    RunDequantizeTest(
        "DEQUANTIZE_INT16_FP32",
        R"!!!(TDequant<pto::DequantType::INT16>(ubTensor_6, ubTensor_0, ubTensor_2, ubTensor_4);)!!!",
        DataType::DT_INT16,
        {16, 64}, {16}, {16, 64}, {16, 64}, -1);
}

// Dequantize INT8 with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, DequantizeInt8AxisM2)
{
    RunDequantizeTest(
        "DEQUANTIZE_INT8_AXIS_M2",
        R"!!!(TDequant<pto::DequantType::INT8>(ubTensor_9, ubTensor_2, ubTensor_5, ubTensor_7);)!!!",
        DataType::DT_INT8,
        {4, 256}, {256}, {4, 256}, {4, 128}, -2);
}

// Dequantize INT16 with axis=-2 (per-column)
TEST_F(TestCodegenDynQuantize, DequantizeInt16AxisM2)
{
    RunDequantizeTest(
        "DEQUANTIZE_INT16_AXIS_M2",
        R"!!!(TDequant<pto::DequantType::INT16>(ubTensor_9, ubTensor_2, ubTensor_5, ubTensor_7);)!!!",
        DataType::DT_INT16,
        {4, 256}, {256}, {4, 256}, {4, 128}, -2);
}

// Quantize-Dequantize chain (symmetric, axis=-1 per-row)
TEST_F(TestCodegenDynQuantize, QuantizeDequantizeSymmetricChain)
{
    TileShape::Current().SetVecTile({8, 128});
    Tensor input(DataType::DT_FP32, {8, 128}, "input");
    Tensor scale(DataType::DT_FP32, {8}, "scale");
    Tensor zeroPointsQ;
    Tensor zeroPointsD(DataType::DT_FP32, {8}, "zeroPointsD");
    Tensor int8Tensor(DataType::DT_INT8, {8, 128}, "int8Tensor");
    Tensor output(DataType::DT_FP32, {8, 128}, "output");

    constexpr const char* funcName = "QUANTIZE_DEQUANTIZE_SYMMETRIC";
    FUNCTION(funcName, {input, scale, zeroPointsD, int8Tensor, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            int8Tensor = Quantize(input, scale, DataType::DT_INT8, -1, zeroPointsQ);
            output = Dequantize(int8Tensor, scale, DataType::DT_FP32, -1, zeroPointsD);
        }
    }

    RunCodeGenAndCheck(funcName,
        R"!!!(TDequant<pto::DequantType::INT8>(ubTensor_9, ubTensor_4, ubTensor_2, ubTensor_7);)!!!");
}

} // namespace npu::tile_fwk
