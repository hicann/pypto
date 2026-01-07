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
 * \file test_pad_local_buffer.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#include "computational_graph_builder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;
constexpr size_t K_1 = 1;
constexpr size_t K_4 = 4;
constexpr size_t K_8 = 8;
constexpr size_t K_64 = 64;
constexpr size_t K_128 = 128;
class TestPadLocalBuffer : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {
    }
};

inline void ConstructGraph1(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 15};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto &copy_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")), OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& add_op = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph2(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto &copy_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")), OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& add_op = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph3(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> AShape = {8, 15};
    std::vector<int64_t> BShape = {15, 15};
    std::vector<int64_t> CShape = {8, 16};
    std::vector<int64_t> expOriShape = {16, 16};
    auto AShapeImme = OpImmediate::Specified(AShape);
    auto BShapeImme = OpImmediate::Specified(BShape);
    auto CShapeImme = OpImmediate::Specified(CShape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, AShape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, BShape);
    auto l0Atensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, AShape);
    l0Atensor0->SetMemoryTypeBoth(MEM_L0A);
    auto l0Btensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, BShape);
    l0Btensor0->SetMemoryTypeBoth(MEM_L0B);
    auto l0Ctensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, CShape);
    l0Ctensor0->SetMemoryTypeBoth(MEM_L0C);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, expOriShape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {l0Atensor0});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0A, AShapeImme, AShapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto &copy_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {l0Btensor0});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L0B, BShapeImme, BShapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")), OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);
    auto& matmul_op = currFunctionPtr->AddRawOperation(Opcode::OP_A_MULACC_B, {l0Atensor0, l0Btensor0}, {l0Ctensor0});
    (void) matmul_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {l0Ctensor0}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph4(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> expOriShape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reduce_op = currFunctionPtr->AddRawOperation(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {ubTensor1}, {ubTensor2});
    (void) reduce_op;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor2}, {ubTensor3});
    (void) abs_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph5(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> expInShape = {7, 16};
    std::vector<int64_t> reduce_shape = {7, 1};
    std::vector<int64_t> expOriShape = {7, 8};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reduce_op = currFunctionPtr->AddRawOperation(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {ubTensor1}, {ubTensor2});
    (void) reduce_op;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor2}, {ubTensor3});
    (void) abs_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph6(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> expOriShape = {8, 8};
    std::vector<int64_t> expandShape = {8, 15};
    std::vector<int64_t> expectExpandShape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, expandShape);
    ubTensor4->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, expandShape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reduce_op = currFunctionPtr->AddRawOperation(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {ubTensor1}, {ubTensor2});
    (void) reduce_op;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor2}, {ubTensor3});
    (void) abs_op;
    auto& expand_op = currFunctionPtr->AddRawOperation(Opcode::OP_EXPAND, {ubTensor3}, {ubTensor4});
    (void) expand_op;
    expand_op.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 1);
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor4}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph7(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor4->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor5->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast2->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reduce_op = currFunctionPtr->AddRawOperation(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {ubTensor1}, {ubTensor2});
    (void) reduce_op;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor2}, {ubTensor3});
    (void) abs_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    auto& copy_in_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {outCast}, {ubTensor4});
    (void) copy_in_op2;
    auto& exp_op = currFunctionPtr->AddRawOperation(Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void) exp_op;
    auto& copy_out_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast2});
    (void) copy_out_op2;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
}

inline void ConstructGraph8(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> reshape_shape = {1, 8};
    std::vector<int64_t> expect_shape = {8, 8};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor4->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    ubTensor5->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reduce_shape);
    outCast2->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reduce_op = currFunctionPtr->AddRawOperation(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {ubTensor1}, {ubTensor2});
    (void) reduce_op;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor2}, {ubTensor3});
    (void) abs_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    auto& copy_in_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {outCast}, {ubTensor4});
    (void) copy_in_op2;
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor4}, {ubTensor5});
    (void) reshape_op;
    auto& copy_out_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast2});
    (void) copy_out_op2;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
}

inline void ConstructGraph9(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {16, 6};
    std::vector<int64_t> trans_shape = {6, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, trans_shape);
    ubTensor3->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, trans_shape);
    ubTensor4->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, trans_shape);
    ubTensor5->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, trans_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast2->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& abs_op = currFunctionPtr->AddRawOperation(Opcode::OP_ABS, {ubTensor1}, {ubTensor2});
    (void) abs_op;
    auto& transpose_op = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {ubTensor2}, {ubTensor3, ubTensor4});
    (void) transpose_op;
    transpose_op.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{1, 0});
    auto& exp_op = currFunctionPtr->AddRawOperation(Opcode::OP_EXP, {ubTensor3}, {ubTensor5});
    (void) exp_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
}

/*
before:
  copyin  copyin
  [8,15]  [8,15]
    \       /
       add
      [8,15]
        |
      copyout

after:
  copyin  copyin
  [8,16]  [8,16]
    \       /
       add
      [8,16]
        |
      copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_all_vec_last_dim_unpadded) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expOriShape = {8, 16};
    ConstructGraph1(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == shape) {
                    EXPECT_EQ(out->shape, expOriShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                    EXPECT_EQ(out->tensor->oriRawshape, shape);
                }
            }
        }
    }
}

/*
before:
    copyin  copyin
    [8,16]  [8,16]
      \       /
       add
      [8,16]
        |
      copyout

after:
    copyin  copyin
    [8,16]  [8,16]
      \       /
       add
      [8,16]
        |
      copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_all_vec_last_dim_padded) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> expOriShape = {8, 16};
    ConstructGraph2(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == shape) {
                    EXPECT_EQ(out->shape, expOriShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                    EXPECT_EQ(out->tensor->oriRawshape, shape);
                }
            }
        }
    }
}


/*
before:
    copyin  copyin
    [8,15]  [15,15]
      \       /
       matmul
       [8,15]
          |
       copyout

after:
    copyin  copyin
    [16,16]  [16,16]
      \       /
       matmul
       [16,16]
          |
       copyout
*/
TEST_F(TestPadLocalBuffer, no_reduce_last_dim_mm) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> AShape = {8, 15};
    std::vector<int64_t> BShape = {15, 15};
    std::vector<int64_t> CShape = {8, 16};
    std::vector<int64_t> expOriShape = {16, 16};
    ConstructGraph3(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == AShape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                    EXPECT_EQ(in->tensor->oriRawshape, AShape);
                }
                if (in->oriShape == BShape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                    EXPECT_EQ(in->tensor->oriRawshape, BShape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == CShape) {
                    EXPECT_EQ(out->shape, expOriShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                    EXPECT_EQ(out->tensor->oriRawshape, CShape);
                }
            }
        }
    }
}

/*
before:
    copyin
    [8,15]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout

after:
    copyin
    [8,16]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_padding) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> expOriShape = {8, 16};
    ConstructGraph4(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expOriShape);
                    EXPECT_EQ(in->tensor->rawshape, expOriShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, reduce_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
    }
}


/*
before:
    copyin
    [7,15]
      |
    reduce
    [7,1]
      |
    abs
    [7,1]
      |
    copyout

after:
    copyin
    [7,16]
      |
    reduce
    [7,8]
      |
    abs
    [7,8]
      |
    copyout
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_no_padding) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> expInShape = {7, 16};
    std::vector<int64_t> reduce_shape = {7, 1};
    std::vector<int64_t> expOriShape = {7, 8};
    ConstructGraph5(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expInShape);
                    EXPECT_EQ(in->tensor->rawshape, expInShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, expOriShape);
                    EXPECT_EQ(out->tensor->rawshape, expOriShape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
    }
}

/*
before:
    copyin
    [8,15]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    expand
    [8, 15]
      |
    copyout

after:
    copyin
    [8,16]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    expand
    [8, 16]
      |
    copyout
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_with_brc) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> expOriShape = {8, 8};
    std::vector<int64_t> expandShape = {8, 15};
    std::vector<int64_t> expectExpandShape = {8, 16};
    ConstructGraph6(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expInShape);
                    EXPECT_EQ(in->tensor->rawshape, expInShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, reduce_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            for (auto &out : op.oOperand) {
                if (out->oriShape == expandShape) {
                    EXPECT_EQ(out->shape, expectExpandShape);
                    EXPECT_EQ(out->tensor->rawshape, expectExpandShape);
                    EXPECT_EQ(out->tensor->oriRawshape, expandShape);
                }
            }
        }
    }
}

/*
before:
    copyin
    [8,15]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout
    [8,1]
      |
    copyin
    [8,1]
      |
     exp
    [8,1]
      |
    copyout
    [8,1]

after:
    copyin
    [8,16]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout
    [8,1]
      |
    copyin
    [8,1]
      |
     exp
    [8,1]
      |
    copyout
    [8,1]
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_with_copyout_copyin_elementwise) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    ConstructGraph7(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expInShape);
                    EXPECT_EQ(in->tensor->rawshape, expInShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, reduce_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
        if (op.GetOpcode() == Opcode::OP_EXP) {
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, reduce_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
    }
}

/*
before:
    copyin
    [8,15]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout
    [8,1]
      |
    copyin
    [8,1]
      |
    reshape
    [1,8]
      |
    copyout
    [1,8]

after:
    copyin
    [8,16]
      |
    reduce
    [8,1]
      |
    abs
    [8,1]
      |
    copyout
    [8,1]
      |
    copyin
    [8,8]
      |
    reshape
    [8,8]
      |
    copyout
    [8,1]
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_with_copyout_copyin_reshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 15};
    std::vector<int64_t> expInShape = {8, 16};
    std::vector<int64_t> reduce_shape = {8, 1};
    std::vector<int64_t> reshape_shape = {1, 8};
    std::vector<int64_t> expect_shape = {8, 8};
    ConstructGraph8(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expInShape);
                    EXPECT_EQ(in->tensor->rawshape, expInShape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                if (out->oriShape == reduce_shape) {
                    EXPECT_EQ(out->shape, reduce_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            for (auto &out : op.oOperand) {
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, expect_shape);
                    EXPECT_EQ(out->tensor->rawshape, reduce_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reduce_shape);
                }
            }
        }
    }
}

/*
before:
    copyin
    [16,6]
      |
    abs
    [16,6]
      |
    transpose
    [6,16]
      |
    exp
    [6,16]
      |
    copyout

after:
    copyin
    [16,8]
      |
    abs
    [16,8]
      |
    transpose
    [8,16]
      |
    exp
    [6,16]
      |
    copyout
*/
TEST_F(TestPadLocalBuffer, reduce_last_dim_with_transpose) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {16, 6};
    std::vector<int64_t> expect_shape = {16, 8};
    std::vector<int64_t> trans_shape = {6, 16};
    std::vector<int64_t> expect_trans_shape = {8, 16};
    ConstructGraph9(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest("PadLocalBuffer", true);
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV) {
            for (auto &in : op.iOperand) {
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, expect_shape);
                    EXPECT_EQ(in->tensor->rawshape, expect_shape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            auto &out = op.oOperand[0];
            if (out->oriShape == trans_shape) {
                EXPECT_EQ(out->shape, expect_trans_shape);
                EXPECT_EQ(out->tensor->rawshape, expect_trans_shape);
                EXPECT_EQ(out->tensor->oriRawshape, trans_shape);
            }
        }
        if (op.GetOpcode() == Opcode::OP_EXP) {
            for (auto &out : op.oOperand) {
                if (out->oriShape == trans_shape) {
                    EXPECT_EQ(out->shape, trans_shape);
                    EXPECT_EQ(out->tensor->rawshape, trans_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, trans_shape);
                }
            }
        }
    }
}

TEST_F(TestPadLocalBuffer, axiscombine) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4,127}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4,1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4,127}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t1","t2"}, {"t3"}, "add", true), true);
    auto *rootFuncPtr = graph.GetFunction();
    config::SetOperationConfig("COMBINE_AXIS", true);
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*rootFuncPtr);
    // ================== Verify Pass Effect ==================
    auto updatedOps = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto &op : updatedOps) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
            auto outputTensor = op.GetOOperands()[0];
            EXPECT_TRUE(outputTensor->GetConsumers().size() != 0);
        }
        if (op.HasAttr(OpAttributeKey::brcbIdx)) {
            auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
            auto tensor = op.GetIOperands()[idx];
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_EQ(tensor->shape[0], K_4);
            EXPECT_EQ(tensor->shape[1], K_8);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[0], K_8);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[1], K_8);
        }
    }
    EXPECT_EQ(cnt, K_1);
}

TEST_F(TestPadLocalBuffer, axiscombine2) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32,4,1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32,4,1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32,4,1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_DIV, {"t1","t2"}, {"t3"}, "div", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {32,4,1,1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t3"}, {"t4"}, "reshape", true), true);
    config::SetOperationConfig("COMBINE_AXIS", true);
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto &op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    auto t3 = graph.GetTensor("t3");
    auto shape = t3->GetRawTensor()->GetRawShape();
    EXPECT_EQ(shape[shape.size()-1], K_1);
    EXPECT_EQ(shape[shape.size()-2], K_8);
}
