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
 * \file test_codegen_dyn_copy.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

constexpr const int dummyRawMagic = 123;

class TestCodegenDynCopy : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

std::string TestL0COutBody(bool isDynamicAligned) {
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    if (isDynamicAligned) {
        config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    }
    
    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L0CToOut"});
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_COPY_OUT, {localTensor}, {ddrTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetOOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    if (!isDynamicAligned) {
        op.SetAttribute("op_attr_is_nz", 1);
    }

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;
    return cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L0CToOut) {
    std::string res = TestL0COutBody(true);
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<float, float, 64, 64, 64, 64>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L0CToOutUnalign) {
    std::string res = TestL0COutBody(false);
    std::string expect =
        R"!!!(TileOp::DynL0CCopyOut<float, float, false, 0>((__gm__ float*)GET_PARAM_ADDR(param, 0, 0), (__cc__ float*)L0C_S0_E0, 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1), 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1ToFB) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {64, 64}, {64, 64});
    Tensor input0(DT_FP32, shape, "A");
    Tensor input1(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    FUNCTION(funcName, {input0, input1, output}) {
        output = Add(input0, input1);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localInTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_FIX, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localInTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto &op = function->AddOperation(Opcode::OP_L1_TO_FIX_QUANT_PRE, {localInTensor}, {localOutTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_FIX, shapeImme, shapeImme));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localInTensor->GetMagic()] = localInTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynL1ToFB<float, 0>((__fbuf__ float*)FIXBUF_S0_E0, (__cbuf__ float*)L1_S0_E0, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestL1CopyInBody(
    bool isNz = false, int outerValueForNz = 0, int innerValueForNz = 0, bool isTileTensor = false) {
    if (isTileTensor) {
        InsertTileTensorOp(Opcode::OP_L1_COPY_IN, "TLoad");
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    const std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "TestL1CopyIn";
    if (isTileTensor) {
        funcName.append("TileTensor");
    }

    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);

    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "L1CopyIn"});
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetIOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    if (isNz) {
        op.SetAttribute(OP_ATTR_PREFIX + "is_nz", 1);
        op.SetAttribute(OP_ATTR_PREFIX + "outer_value", outerValueForNz);
        op.SetAttribute(OP_ATTR_PREFIX + "inner_value", innerValueForNz);
    }

    if (isTileTensor) {
        op.SetAttribute(OP_ATTR_PREFIX + "copy_in_mode", 2);
    }

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    return cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L1CopyIn) {
    std::string res = TestL1CopyInBody();
    std::string expect =
        R"!!!(TileOp::DynL1CopyIn<float, float>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInTileTensor) {
    std::string res = TestL1CopyInBody(false, 0, 0, true);
    std::string expect =
        R"!!!(TLoad<CopyInMode::NZ2NZ>(l1Tensor_1, gmTensor_2, Coord2Dim(GET_PARAM_OFFSET_2(param, 0, 0)), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1));
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInDynAligned) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string res = TestL1CopyInBody();
    std::string expect =
        R"!!!(TileOp::DynL1CopyIn<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZWithZeroAligned) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string res = TestL1CopyInBody(true);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float, 64, 64>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 0, 2, 1), 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, L1CopyInNZWithValue) {
    std::string res = TestL1CopyInBody(true, 1, 1);
    std::string expect =
        R"!!!(TileOp::DynL1CopyInNZ2NZ<float, float>((__cbuf__ float*)L1_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 64, 64, GET_PARAM_RAWSHAPE_2(param, 0, 0), GET_PARAM_OFFSET_2(param, 0, 0), 1, 1, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, TestGatherInL1TileTensor) {
    std::vector<int64_t> gatherShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(gatherShape);
    TileShape::Current().SetVecTile(gatherShape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    InsertTileTensorOp(Opcode::OP_GATHER_IN_L1, "TGatherInL1");
    Tensor inputA(DT_FP32, gatherShape, "A");
    Tensor inputB(DT_FP32, gatherShape, "B");
    Tensor output(DT_FP32, gatherShape, "C");

    std::string funcName = "GatherInL1TileTensor";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto gatherTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, gatherShape, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, gatherShape, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    gatherTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    localOutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    LogicalTensors inputs = {gatherTensor, gatherTensor, gatherTensor};
    LogicalTensors outputs = {localOutTensor};

    auto &gatherL1Op = function->AddOperation(Opcode::OP_GATHER_IN_L1, inputs, outputs);
    gatherL1Op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    int64_t blocksize{0};
    gatherL1Op.SetAttribute("op_attr_blocksize", blocksize);
    gatherL1Op.SetAttribute(OpAttributeKey::startOffset, blocksize);
    gatherL1Op.SetOOpAttrOffset(0, 0);
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(gatherL1Op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[gatherTensor->GetMagic()] = gatherTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(gatherL1Op);
    cop.UpdateTileTensorInfo();
    cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L1ToBt) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";

    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_BT, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto &op = function->AddOperation(Opcode::OP_L1_TO_BT, {localTensor}, {localOutTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_BT, shapeImme, shapeImme));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynL1ToBT<float, float, 0>((uint64_t)BIAS_S0_E0, (__cbuf__ float*)L1_S0_E0, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

void TestMatmulMteBody(Opcode opcode, MemoryType inType, MemoryType outType, bool isTileTensor = false) {
    if (isTileTensor) {
        InsertTileTensorOp(Opcode::OP_L0C_COPY_OUT, "TStore");
        InsertTileTensorOp(Opcode::OP_L1_TO_BT, "TExtract");
        InsertTileTensorOp(Opcode::OP_L1_TO_L0A, "TExtract");
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "MatmulMteBody";
    if (isTileTensor) {
        funcName.append("TileTensor");
    }

    FUNCTION(funcName, {inputA, inputB, output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = Add(inputA, inputB);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    function->SetUnderDynamicFunction(true);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_INT32, inType, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP16, outType, shape, dynValidShape});
    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));

    LogicalTensors inputs = {localTensor};
    LogicalTensors outputs = {localOutTensor};
    if (opcode == Opcode::OP_COPY_OUT) {
        inputs.emplace_back(localTensor);
    }
    auto &op = function->AddOperation(opcode, inputs, outputs);

    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    if (opcode == Opcode::OP_COPY_OUT) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_BT) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_BT, shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_COPY_IN) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_L1, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_FIX_QUANT_PRE) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_FIX, shapeImme, shapeImme));
    } else if (opcode == Opcode::OP_L1_TO_L0A) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0A, shapeImme, shapeImme));
    }

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop({symbolManager, *function, *function->rootFunc_->programs_[0], op, {}});
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, L1CopyInTensor) {
    TestMatmulMteBody(Opcode::OP_L1_COPY_IN, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1);
}
TEST_F(TestCodegenDynCopy, L1CopyL0Tensor) {
    TestMatmulMteBody(Opcode::OP_L1_TO_L0A, MemoryType::MEM_L1, MemoryType::MEM_L0A, true);
}
TEST_F(TestCodegenDynCopy, L1CopyFBTensor) {
    TestMatmulMteBody(Opcode::OP_L1_TO_FIX_QUANT_PRE, MemoryType::MEM_L1, MemoryType::MEM_FIX);
}
TEST_F(TestCodegenDynCopy, L1CopyBTTensor) {
    TestMatmulMteBody(Opcode::OP_L1_TO_BT, MemoryType::MEM_L1, MemoryType::MEM_BT, true);
}
TEST_F(TestCodegenDynCopy, L0CopyOutTensor) {
    TestMatmulMteBody(Opcode::OP_COPY_OUT, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR);
}
TEST_F(TestCodegenDynCopy, L0CopyOutTensorTileTensor) {
    TestMatmulMteBody(Opcode::OP_COPY_OUT, MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR, true);
}
TEST_F(TestCodegenDynCopy, L0CopyUBTensor) {
    TestMatmulMteBody(Opcode::OP_L0C_COPY_UB, MemoryType::MEM_L0C, MemoryType::MEM_UB);
}

std::string TestCopyL1Body(Opcode opcode, MemoryType inputType, MemoryType outputType) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    InsertTileTensorOp(Opcode::OP_UB_COPY_L1, "TExtract");
    InsertTileTensorOp(Opcode::OP_UB_COPY_ND2NZ, "TMoveND2NZ");
    InsertTileTensorOp(Opcode::OP_L0C_TO_L1, "TExtract");
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, inputType, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, outputType, shape, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    localTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    localOutTensor->UpdateOffset(TensorOffset(offset, dynoffset));
    LogicalTensors inputs = {localTensor};
    LogicalTensors outputs = {localOutTensor};
    if (opcode == Opcode::OP_L0C_TO_L1) {
        auto localTensor1 = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_FIX, shape, dynValidShape});
        localTensor1->UpdateOffset(TensorOffset(offset, dynoffset));
        inputs.emplace_back(localTensor1);
    }
    auto &op = function->AddOperation(opcode, inputs, outputs);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    if (opcode == Opcode::OP_L0C_TO_L1) {
        op.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        copyAttr->SetToDynValidShape(OpImmediate::Specified(shape));
        copyAttr->SetFromOffset(OpImmediate::Specified({0, 0}));
    }

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    cop.UpdateTileTensorInfo();
    return cop.GenOpCode();
}

TEST_F(TestCodegenDynCopy, UB2L1TileTensor) {
    TestCopyL1Body(Opcode::OP_UB_COPY_L1, MemoryType::MEM_UB, MemoryType::MEM_L1);
}

TEST_F(TestCodegenDynCopy, UB2UBND2NZTileTensor) {
    TestCopyL1Body(Opcode::OP_UB_COPY_ND2NZ, MemoryType::MEM_UB, MemoryType::MEM_UB);
}

TEST_F(TestCodegenDynCopy, L0CToL1TileTensor) {
    TestCopyL1Body(Opcode::OP_L0C_TO_L1, MemoryType::MEM_L0C, MemoryType::MEM_L1);
}

void TestUBCopyInBody(const std::string funcName, const std::string &expect) {
    const std::vector<int64_t> shape = {64, 64};
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);

    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);

    auto ddrTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_DEVICE_DDR, shape, "UBCopyIn"});
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_COPY_IN, {ddrTensor}, {localTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    op.SetIOpAttrOffset(0, 0);
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;

    cop.Init(op);
    cop.originShape[0] = shape;
    cop.originShape[1] = shape;

    std::string res = cop.GenOpCode();
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynCopy, UBCopyIn) {
    std::string expect =
        R"!!!(TileOp::DynUBCopyIn<float, 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 1, 1, 1, 64, 64, 1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, GET_PARAM_OFFSET_2(param, 0, 0));
)!!!";
    TestUBCopyInBody("UBCopyIn", expect);
}

TEST_F(TestCodegenDynCopy, UBCopyInAligned) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
    std::string expect =
        R"!!!(TileOp::DynUBCopyIn<float, 1, 1, 1, 64, 64, 1, 1, 64, 64>((__ubuf__ float*)UB_S0_E0, (__gm__ float*)GET_PARAM_ADDR(param, 0, 0), 1, 1, 1, GET_PARAM_RAWSHAPE_2(param, 0, 0), 0, 0, 0, GET_PARAM_OFFSET_2(param, 0, 0));
)!!!";
    TestUBCopyInBody("UBCopyInAligned", expect);
}

TEST_F(TestCodegenDynCopy, L0CToL1) {
    std::vector<int64_t> shape = {64, 64};
    auto shapeImmeL0C = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "L0CToL1";

    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L0C, shape, dynValidShape});
    auto localOutTensor = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_L1, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_L0C_TO_L1, {localTensor}, {localOutTensor});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_L0C, OpImmediate::Specified({0, 0}), shapeImmeL0C, shapeImmeL0C));
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    copyAttr->SetFromOffset(OpImmediate::Specified({0, 0}));
    copyAttr->SetToDynValidShape(OpImmediate::Specified(shape));
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynL0CToL1<float, float, 0>((__cbuf__ float*)L1_S0_E0, (__cc__ float*)L0C_S0_E0, 64, 64, 64, 64, 0, 0, 64, 64, 0, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
