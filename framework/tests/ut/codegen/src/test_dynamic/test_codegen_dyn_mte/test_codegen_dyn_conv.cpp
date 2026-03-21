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
 * \file test_codegen_dyn_conv.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

constexpr int64_t N0 = 16;

class TestCodegenDynConv : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

Function *GetFunctionConv(const std::string &funcName) {
    const std::vector<int64_t> shape = {64, 64};
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);

    auto function = GenMockFuncDyn(funcName, shape);
    function->SetUnderDynamicFunction(true);
    return function;
}

void SetConvL1CopyInOpAttr(Operation &op, const bool &isConv3D, std::vector<int64_t> gmShape,
    std::vector<int64_t> dstL1Shape) {
    std::vector<int64_t> offset = {0, 0, 0, 0};
    if (isConv3D) {
        offset = {0, 0, 0, 0, 0};
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset),
        MemoryType::MEM_L1, OpImmediate::Specified(gmShape), OpImmediate::Specified(dstL1Shape),
        OpImmediate::Specified(dstL1Shape));
    op.SetOpAttribute(copyAttr);
}

std::string TestConvL1CopyInBody(const std::string &funcName, const std::vector<int64_t> &gmShape, bool isFmap = true,
    int64_t copyInMode = 3, DataType dtype = DataType::DT_FP16) {
    std::map<DataType, int64_t> k0Map = {{DataType::DT_FP16, 16}, {DataType::DT_BF16, 16}, {DataType::DT_FP32, 8}};
    bool isConv3D = gmShape.size() == 5;
    auto function = GetFunctionConv(funcName);
    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR);
    std::vector<int64_t> dstL1Shape;
    if (isConv3D) {
        if (isFmap) {
            dstL1Shape = {gmShape[0], gmShape[2], CeilDiv(gmShape[1], k0Map[dtype]), gmShape[3], gmShape[4],
                          k0Map[dtype]};
        } else {
            dstL1Shape = {CeilDiv(gmShape[1], k0Map[dtype]) * gmShape[2] * gmShape[3] * gmShape[4],
                          CeilDiv(gmShape[0], N0), N0, k0Map[dtype]};
        }
    } else {
        if (isFmap) {
            dstL1Shape = {gmShape[0], CeilDiv(gmShape[1], k0Map[dtype]), gmShape[2], gmShape[3], k0Map[dtype]};
        } else {
            dstL1Shape = {CeilDiv(gmShape[1], k0Map[dtype]) * gmShape[2] * gmShape[3], CeilDiv(gmShape[0], N0), N0,
                          k0Map[dtype]};
        }
    }
    auto localTensor = CreateConvTensor(*function, dtype, dstL1Shape, MemoryType::MEM_L1);

    auto &op = function->AddOperation(Opcode::OP_L1_COPY_IN_CONV, {gmTensor}, {localTensor});
    op.SetAttribute("IS_FMAP", isFmap);
    op.SetAttribute("IS_CONV3D", isConv3D);
    op.SetAttribute("COPY_IN_MODE", copyInMode);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    SetConvL1CopyInOpAttr(op, isConv3D, gmShape, dstL1Shape);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cgop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});
    return cgop.GenOpCode();
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorFmapConv2D) {
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorFmapConv2D", {1, 16, 1, 16});
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 0, 1>(l1Tensor_10, gmTensor_11, 0, 0, 0, 0, 0, 1, 16, 0, 1, 16);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorWeightConv2D) {
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorWeightConv2D", {1, 16, 1, 1}, false);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 0, 0>(l1Tensor_10, gmTensor_11, 0, 0, 0, 0, 0, 1, 16, 0, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorFmapConv3D) {
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorFmapConv3D", {1, 16, 1, 1, 16});
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 1, 1>(l1Tensor_10, gmTensor_11, 0, 0, 0, 0, 0, 1, 16, 1, 1, 16);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, L1CopyInTileTensorWeightConv3D) {
    std::string res = TestConvL1CopyInBody("L1CopyInTileTensorWeightConv3D", {1, 16, 1, 1, 1}, false);
    std::string expect =
        R"!!!(TLoadConv<CopyInMode::DN2NZ, 1, 0>(l1Tensor_10, gmTensor_11, 0, 0, 0, 0, 0, 1, 16, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestConvL0COutBody(const std::string &funcName, const std::vector<int64_t> &l0cShape,
    std::vector<int64_t> &gmShape, int64_t copyOutMode = 3, DataType dtype = DataType::DT_FP32) {
    bool isConv3D = gmShape.size() == 5;
    std::vector<int64_t> offset = {0, 0, 0, 0};
    if (isConv3D) {
        offset = {0, 0, 0, 0, 0};
    }
    auto function = GetFunctionConv(funcName);
    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR, false);
    auto l0cTensor = CreateConvTensor(*function, DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, false);

    auto &op = function->rootFunc_->programs_[0]->AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {l0cTensor}, {gmTensor});
    auto shapeImme = OpImmediate::Specified(l0cShape);
    op.SetAttribute("COPY_OUT_MODE", copyOutMode);
    op.SetAttribute("IS_CONV3D", isConv3D);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified(offset), shapeImme, shapeImme, shapeImme));

    CodeGenCtx ctx;
    CodeGenCloudNPU codegen(ctx);
    codegen.GenCode(*function, {});
    return GetResultFromCpp(*function);
}

TEST_F(TestCodegenDynConv, L0COutTileTensorConv2D) {
    std::vector<int64_t> l0cShape = {16, 16};
    std::vector<int64_t> gmShape = {1, 16, 1, 16};
    std::string res = TestConvL0COutBody("L0COutTileTensorConv2D", l0cShape, gmShape);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2DN, 0>(gmTensor_19, l0cTensor_20, 0, 0, 0, 0, 0, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynConv, L0COutTileTensorConv3D) {
    std::vector<int64_t> l0cShape = {16, 16};
    std::vector<int64_t> gmShape = {1, 16, 1, 1, 16};
    std::string res = TestConvL0COutBody("L0COutTileTensorConv3D", l0cShape, gmShape);
    std::string expect =
        R"!!!(TStoreConv<CopyOutMode::NZ2DN, 1>(gmTensor_19, l0cTensor_20, 0, 0, 0, 0, 0, 16, 16);)!!!";
    CheckStringExist(expect, res);
}

void SetConvLoad3DAttributes(Operation &op, const bool &isConv3D) {
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::postM, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::postK, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::paddingLeft, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::paddingRight, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::paddingTop, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::paddingBottom, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::padValue, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::filterH, (int64_t)1);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::filterW, (int64_t)1);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::dilationH, (int64_t)1);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::dilationW, (int64_t)1);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::strideH, (int64_t)1);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::strideW, (int64_t)1);
    op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
}

std::string TestConvLoad3DBody(const std::string &funcName, const bool &isConv3D) {
    auto function = GetFunctionConv(funcName);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L1, {16, 16}, dynValidShape});
    auto l0Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0A, {16, 16}, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    l1Tensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto &op = function->AddOperation(Opcode::OP_LOAD3D_CONV, {l1Tensor}, {l0Tensor});
    SetConvLoad3DAttributes(op, isConv3D);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});

    return cop.GenOpCode();
}

TEST_F(TestCodegenDynConv, Load3DConv2D) {
    std::string res = TestConvLoad3DBody("Load3DConv2D", false);
    std::string expect = R"!!!(TLoad3D<0>(l0aTensor_10, l1Tensor_11, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConv, Load3DConv3D) {
    std::string res = TestConvLoad3DBody("Load3DConv3D", true);
    std::string expect = R"!!!(TLoad3D<1>(l0aTensor_10, l1Tensor_11, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
)!!!";
    EXPECT_EQ(res, expect);
}

void SetConvLoad2DAttributes(Operation &op) {
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::postK, (int64_t)0);
    op.SetAttribute(Conv::L12L0ConvOpAttributeKey::postN, (int64_t)0);
}

std::string TestConvLoad2DBody(const std::string &funcName) {
    auto function = GetFunctionConv(funcName);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L1, {16, 16}, dynValidShape});
    auto l0Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0B, {16, 16}, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    l1Tensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto &op = function->AddOperation(Opcode::OP_LOAD2D_CONV, {l1Tensor}, {l0Tensor});
    SetConvLoad2DAttributes(op);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});

    return cop.GenOpCode();
}

TEST_F(TestCodegenDynConv, Load2DConv) {
    std::string res = TestConvLoad2DBody("Load2DConv");
    std::string expect = R"!!!(TLoad2D(l0bTensor_10, l1Tensor_11, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
