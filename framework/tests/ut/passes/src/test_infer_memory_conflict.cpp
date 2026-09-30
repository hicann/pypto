/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"

namespace npu::tile_fwk {

// FUNCTION DSL 用例共享进程级 Program 单例, 用例间不复位会残留函数表,
// 导致后续用例按 calleeHash 查找函数失败, 参照 test_infermemoryconflict.cpp 的做法在 SetUp 中复位
class PassTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(PassTest, InferMemoryConflictInferOutputWriteConflict)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {a, b, c})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            Assemble(t, {idx * 32, 0}, a); // assemble of a not overlap, no conflict
            Assemble(t, {(idx + 1) * 32, 0}, a);

            AtomicRMW(t, {idx * 32, 0}, c, AtomicRMWMode::ADD); // atomic always mark as conflict

            Assemble(t, {idx * 32, 0}, b); // assemble of b overlaped, should be conflict
            Assemble(t, {idx * 32 + 16, 0}, b);
        }
    }
    int cnt = 0;
    PassManager::Instance().RegisterStrategy("InferOutputWriteConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, "InferOutputWriteConflictTestStrategy"),
        SUCCESS);
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "a") {
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "b") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "c") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        }
    }
    EXPECT_EQ(cnt, 3);
}

TEST_F(PassTest, InferMemoryConflictAssembleParallelFalseMarksNormalAndCloneKeepsIt)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor serialOut(DT_FP32, shape, "serial_out");
    Tensor parallelOut(DT_FP32, shape, "parallel_out");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {serialOut, parallelOut})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            // Cross-loop WAW serial write: parallel=false should mark outcast NORMAL.
            Assemble(t, {idx * 32, 0}, serialOut, false);
            // Default parallel=true should not mark NORMAL.
            Assemble(t, {idx * 32, 0}, parallelOut, true);
        }
    }

    const std::string strategyName = "InferAssembleNormalAttrTestStrategy";
    PassManager::Instance().RegisterStrategy(strategyName, {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, strategyName), SUCCESS);

    int matched = 0;
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "serial_out") {
            EXPECT_TRUE(out->HasAttr("NORMAL"));
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            auto cloned = out->Clone(*loopFunc, true);
            ASSERT_NE(cloned, nullptr);
            EXPECT_TRUE(cloned->HasAttr("NORMAL")) << "NORMAL should be inherited by LogicalTensor::Clone";
            matched++;
        } else if (out->tensor->GetSymbol() == "parallel_out") {
            EXPECT_FALSE(out->HasAttr("NORMAL"));
            matched++;
        }
    }
    EXPECT_EQ(matched, 2);
}

namespace {
int CountRegisterCopy(Function* function)
{
    int cnt = 0;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            cnt += 1;
        }
    }
    return cnt;
}
} // namespace

/*
self(incast, mem0) -> INDEX_PUT(self, values, indices) -> dst(与self同物理地址) -> ASSEMBLE -> out(outcast, mem1)
内存信息仅经 INDEX_PUT 的第1输入/第1输出对传播, 传播至 assemble 后检测到与 outcast 的地址冲突,
应在 assemble 前插入 register_copy; values/indices 不参与传播
*/
TEST_F(PassTest, InferMemoryConflictIndexPutInsertsRegisterCopy)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexPut", "TestIndexPut", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {4, 4};
    std::vector<int64_t> valuesShape = {2, 4};
    std::vector<int64_t> indicesShape = {2};
    std::vector<int64_t> offset = {0, 0};

    std::shared_ptr<RawTensor> selfRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    selfRaw->SetSymbol("self");
    selfRaw->memoryId = 0;
    auto self = npu::tile_fwk::IRBuilder().CreateTensorVar(selfRaw, offset, shape, CreateTestConstIntVector(shape));
    auto values = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, valuesShape,
                                                             CreateTestConstIntVector(valuesShape));
    auto indices = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, indicesShape,
                                                              CreateTestConstIntVector(indicesShape));

    std::shared_ptr<RawTensor> dstRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    dstRaw->SetSymbol("index_put_dst");
    dstRaw->memoryId = 0; // 与 self 同物理地址
    auto dst = npu::tile_fwk::IRBuilder().CreateTensorVar(dstRaw, offset, shape, CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    outRaw->SetSymbol("out");
    outRaw->memoryId = 1;
    auto out = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, offset, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->inCasts_.push_back(self);
    currFunctionPtr->outCasts_.push_back(out);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_PUT, {self, values, indices}, {dst});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {dst}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    PassManager::Instance().RegisterStrategy("IndexPutMemoryConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "IndexPutMemoryConflictTestStrategy"),
        SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr.get()), 1);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), dst);
    EXPECT_EQ(assembleOp.GetIOperands().front(), copy->GetOOperands().front());
}

/*
self(incast, mem0) -> INDEX_ADD(self, src, indices) -> dst(与self同物理地址) -> ASSEMBLE -> out(outcast, mem1)
                                              |-> tmp(第2输出, 临时buffer) -> ASSEMBLE -> out2(outcast, mem2)
内存信息仅经第1输入/第1输出对传播: dst 分支检测到冲突应在 assemble 前插入 register_copy;
tmp 分支不传播内存信息, 不插入 register_copy, INDEX_ADD 前也不插入
*/
TEST_F(PassTest, InferMemoryConflictIndexAddInsertsRegisterCopy)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexAdd", "TestIndexAdd", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {4, 4};
    std::vector<int64_t> srcShape = {2, 4};
    std::vector<int64_t> indicesShape = {2};
    std::vector<int64_t> offset = {0, 0};

    std::shared_ptr<RawTensor> selfRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    selfRaw->SetSymbol("self");
    selfRaw->memoryId = 0;
    auto self = npu::tile_fwk::IRBuilder().CreateTensorVar(selfRaw, offset, shape, CreateTestConstIntVector(shape));
    auto src = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, srcShape, CreateTestConstIntVector(srcShape));
    auto indices = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, indicesShape,
                                                              CreateTestConstIntVector(indicesShape));

    std::shared_ptr<RawTensor> dstRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    dstRaw->SetSymbol("index_add_dst");
    dstRaw->memoryId = 0; // 与 self 同物理地址
    auto dst = npu::tile_fwk::IRBuilder().CreateTensorVar(dstRaw, offset, shape, CreateTestConstIntVector(shape));
    auto tmp = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    outRaw->SetSymbol("out");
    outRaw->memoryId = 1;
    auto out = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, offset, shape, CreateTestConstIntVector(shape));
    std::shared_ptr<RawTensor> out2Raw = std::make_shared<RawTensor>(DT_FP32, shape);
    out2Raw->SetSymbol("out2");
    out2Raw->memoryId = 2;
    auto out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(out2Raw, offset, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->inCasts_.push_back(self);
    currFunctionPtr->outCasts_.push_back(out);
    currFunctionPtr->outCasts_.push_back(out2);

    auto& indexAddOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_ADD, {self, src, indices},
                                                        {dst, tmp});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {dst}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));
    auto& assembleTmpOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tmp}, {out2});
    assembleTmpOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    PassManager::Instance().RegisterStrategy("IndexAddMemoryConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "IndexAddMemoryConflictTestStrategy"),
        SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr.get()), 1);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), dst);
    EXPECT_EQ(assembleOp.GetIOperands().front(), copy->GetOOperands().front());
    EXPECT_EQ(indexAddOp.GetIOperands().front(), self);
    EXPECT_EQ(assembleTmpOp.GetIOperands().front(), tmp);
}
} // namespace npu::tile_fwk
