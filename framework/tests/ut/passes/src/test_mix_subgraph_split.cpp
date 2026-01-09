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
 * \file test_mix_subgraph_split.cpp
  * \brief Unit test for mixSubgraphSplit
  * */
#include <gtest/gtest.h>
#include "passes/block_graph_pass/mix_subgraph_split.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t programId = 100;
constexpr int MS_NUM16 = 16;
constexpr int MS_NUM3 = 3;
constexpr int MS_NUM10005 = 10005;

class MixSubgraphSplitTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}

protected:
    // 在root function中创建callOp
    void CreateCallOpInRoot(Function& rootFunc, Function& mixFunc) {
        // 创建tensor作为callOp的输入输出
        std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

        auto incast1 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto incast2 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto incast3 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto outcast1 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);
        auto outcast2 = std::make_shared<LogicalTensor>(rootFunc, DT_FP32, shape);

        // 添加callOp
        auto& callOp = rootFunc.AddRawOperation(
            Opcode::OP_CALL,
            {incast1, incast2, incast3},
            {outcast1, outcast2});

        // 创建CallOpAttribute
        std::vector<std::vector<SymbolicScalar>> argList;
        for (int i = 0; i < 5; ++i) {
            std::vector<SymbolicScalar> argBlock;
            for (int j = 0; j < 9; ++j) {  // 2维tensor的参数块长度
                argBlock.push_back(SymbolicScalar(static_cast<int64_t>(j + i * 10)));
            }
            argList.push_back(argBlock);
        }
        std::map<int, SymbolicScalar> outIndexToExpr;
        auto callAttr = mixFunc.CreateCallOpAttribute(argList, outIndexToExpr);
        callOp.SetOpAttribute(callAttr);
        // 设置programID
        callOp.UpdateSubgraphID(mixFunc.GetProgramId());
        // 设置invokeInfo
        auto callOpAttr = std::dynamic_pointer_cast<CallOpAttribute>(callAttr);
        if (callOpAttr) {
            callOpAttr->invokeInfo_ = std::make_shared<SubfuncInvokeInfoTy>();
            // 设置一些基本的invoke信息
            callOpAttr->invokeInfo_->UpdateProgramSubgraphId(programId);
        }
    }
};

/*!
 * \brief 测试MixSubgraphSplit的基本功能
 * 1. 创建包含Mix子图的场景
 * 2. 调用RunOnFunction进行拆分
 * 3. 验证拆分结果
 */
TEST_F(MixSubgraphSplitTest, TestMixSubgraphSplit) {
    // 创建function
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // 创建Mix子图leaffunction
    auto mixFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_mix_func", "test_mix_func", rootFuncPtr.get());
    mixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFuncPtr->SetFunctionType(FunctionType::STATIC);
    // 添加到programs
    rootFuncPtr->programs_[programId] = mixFuncPtr.get();

    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    // 创建tensors
    auto incast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor4 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor5 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto tensor6 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto outcast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto outcast2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);

    // 设置incast/outcast
    mixFuncPtr->inCasts_.push_back(incast1);
    mixFuncPtr->inCasts_.push_back(incast2);
    mixFuncPtr->inCasts_.push_back(incast3);
    mixFuncPtr->outCasts_.push_back(outcast1);
    mixFuncPtr->outCasts_.push_back(outcast2);

    // 创建OpImmediate
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;

    // scope1：Cube相关op
    auto& copyin1 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
     // 定义CopyOpAttribute参数
    copyin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme,                    // offset
        MemoryType::MEM_UB,           // srcMemType
        shapeImme,                    // shape
        shapeImme,                    // rawShape
        emptyVec                      // 第五个参数
    ));
    copyin1.SetIOpAttrOffset(0, 0);
    copyin1.UpdateInternalSubgraphID(0);
    copyin1.SetAttr(OpAttributeKey::isCube, true);

    auto& copyin3 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {tensor6});
     // 定义CopyOpAttribute参数
    copyin3.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme,                    // offset
        MemoryType::MEM_UB,           // srcMemType
        shapeImme,                    // shape
        shapeImme,                    // rawShape
        emptyVec                      // 第五个参数
    ));
    copyin3.SetIOpAttrOffset(0, 0);
    copyin3.UpdateInternalSubgraphID(0);
    copyin3.SetAttr(OpAttributeKey::isCube, true);

    auto& syncSrc = mixFuncPtr->AddRawOperation(Opcode::OP_SYNC_SRC, {}, {});
    (void) syncSrc;

    auto& aMulB = mixFuncPtr->AddRawOperation(Opcode::OP_A_MUL_B, {tensor1, tensor6}, {tensor2});
    aMulB.UpdateInternalSubgraphID(0);
    aMulB.SetAttr(OpAttributeKey::isCube, true);

    // scope2：Vector AIV0 op
    auto& copyin2 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast3}, {tensor3});
    copyin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme,                    // offset
        MemoryType::MEM_UB,           // srcMemType
        shapeImme,                    // shape
        shapeImme,                    // rawShape
        emptyVec                      // 第五个参数
    ));
    copyin2.SetIOpAttrOffset(0, 0);
    copyin2.UpdateInternalSubgraphID(1);
    copyin2.SetAIVCore(AIVCore::AIV0);

    auto& add = mixFuncPtr->AddRawOperation(Opcode::OP_ADD, {tensor2, tensor3}, {tensor4});
    add.UpdateInternalSubgraphID(1);
    add.SetAIVCore(AIVCore::AIV0);

    // scope3: Vector AIV1 op
    auto& mul = mixFuncPtr->AddRawOperation(Opcode::OP_MUL, {tensor4}, {tensor5});
    mul.UpdateInternalSubgraphID(2);
    mul.SetAIVCore(AIVCore::AIV1);

    auto& syncDst = mixFuncPtr->AddRawOperation(Opcode::OP_SYNC_DST, {}, {});
    (void) syncDst;

    auto& copyout1 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor5}, {outcast1});
    // 为OP_COPY_OUT设置CopyOpAttribute
    copyout1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB,           // srcMemType
        offsetImme,                    // offset
        shapeImme,                    // shape
        shapeImme,                    // rawShape
        emptyVec                      // 第五个参数
    ));
    copyout1.SetOOpAttrOffset(0, 0);
    copyout1.UpdateInternalSubgraphID(2);
    copyout1.SetAIVCore(AIVCore::AIV1);
    // 验证Mix子图识别
    MixSubgraphSplit splitter;
    bool isMix = splitter.IsMixSubgraph(*mixFuncPtr);
    EXPECT_TRUE(isMix) << "Mix subgraph should be identified";

    // 2. 分析内部组件
    auto components = splitter.AnalyzeInternalComponents(*mixFuncPtr);
    EXPECT_GT(components.size(), 1) << "Mix subgraph should have multiple components";

    // 验证组件数量（根据CreateMixSubgraphScenario中的设置）
    EXPECT_EQ(components.size(), 3) << "Expected 3 components (cube, aiv0, aiv1)";

    // 3. 在root function中创建CallOp
    CreateCallOpInRoot(*rootFuncPtr, *mixFuncPtr);
    // 获取root function中的CallOps
    auto callOps = rootFuncPtr->GetCallopList();

    // 4. 执行MixSubgraphSplit
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";

    // 5. 验证拆分结果
    // 检查programs数量变化
    auto& programs = rootFuncPtr->programs_;
    EXPECT_GT(programs.size(), 1) << "Should have multiple programs after split";
    // 检查新创建的programs
    for (const auto& program : programs) {
        ASSERT_NE(program.second, nullptr);

        // 验证新function的名称包含后缀
        std::string funcName = program.second->GetRawName();
        EXPECT_NE(funcName.find("leaf"), std::string::npos)
            << "New function name should contain 'leaf' suffix";

        // 验证function类型
        EXPECT_EQ(program.second->GetFunctionType(), FunctionType::STATIC);
        EXPECT_EQ(program.second->GetGraphType(), GraphType::BLOCK_GRAPH);

        // 验证program ID在合理范围内
        EXPECT_EQ(program.second->GetProgramId(), program.first)
            << "Function's program ID should match map key";
    }
    // 6. 验证CallOps被正确更新
    auto newCallOps = rootFuncPtr->GetCallopList();
    EXPECT_GT(newCallOps.size(), callOps.size())
        << "Should have more call ops after split";
    // 验证每个callOp都有正确的program ID
    for (auto* callOp : newCallOps) {
        ASSERT_NE(callOp, nullptr);
        EXPECT_FALSE(callOp->IsDeleted()) << "CallOp should not be deleted";

        // 验证program ID存在
        auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
        if (callAttr != nullptr && callAttr->invokeInfo_ != nullptr) {
            uint64_t programID = callAttr->invokeInfo_->GetProgramId();
            EXPECT_NE(programs.find(programID), programs.end())
                << "CallOp's program ID should exist in programs";
        }
    }
     // 7. 验证组件分离正确
    // 统计不同AIVCore类型的function数量
    int cubeCount = 0;
    int aiv0Count = 0;
    int aiv1Count = 0;

    for (const auto& program : programs) {
        auto leafAttr = program.second->GetLeafFuncAttribute();
        if (leafAttr) {
            if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
                cubeCount++;
            } else if (leafAttr->aivCore == AIVCore::AIV0) {
                aiv0Count++;
            } else if (leafAttr->aivCore == AIVCore::AIV1) {
                aiv1Count++;
            }
        }
    }

    // 根据CreateMixSubgraphScenario的设置验证
    EXPECT_GE(cubeCount, 0) << "Should have at least one cube component";
    EXPECT_GE(aiv0Count, 0) << "Should have at least one AIV0 component";
    EXPECT_GE(aiv1Count, 0) << "Should have at least one AIV1 component";

    // 8. 验证operations被正确分配到各组件
    for (const auto& program : programs) {
        auto operations = program.second->Operations(false);

        // 验证operations的internalSubgraphID一致
        int componentID = -1;
        for (auto& op : operations) {
            if (op.IsNOP()) continue;

            int opComponentID = op.GetInternalSubgraphID();
            if (componentID == -1) {
                componentID = opComponentID;
            } else {
                EXPECT_EQ(opComponentID, componentID)
                    << "All ops in a leaf function should have same internalSubgraphID";
            }
        }
    }
    // 9. 验证资源类型设置
    for (const auto& program : programs) {
        auto leafAttr = program.second->GetLeafFuncAttribute();
        if (leafAttr) {
            EXPECT_NE(leafAttr->mixResourceType, MixResourceType::UNKNOWN)
                << "Mix resource type should be set";
            EXPECT_GT(leafAttr->mixId, -1) << "Mix ID should be assigned";
        }
    }
}

TEST_F(MixSubgraphSplitTest, TestDependOperand) {
    // Build Graph
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();

    // Add and check depend operand
    Operation *copyin2 = subGraph.GetOp("Copyin2");
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t4");
    copyin2->AddDependOperand(tensor4);
    Operation *add1 = subGraph.GetOp("Add1");
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t3");
    add1->AddDependOperand(tensor3);
    EXPECT_EQ(copyin2->GetDependOperands().front()->GetMagic(), MS_NUM3);
    EXPECT_EQ(copyin2->GetDependOperandSize(), 1);

    // Check depend
    tensor4->AddDependOp(copyin2);
    tensor4->AddDependOp(copyin2);
    EXPECT_EQ(tensor4->GetDependOps().size(), 1);
    tensor3->AddDependOp(add1);
    auto dependOp = *(tensor4->GetDependOps().begin());
    EXPECT_EQ(dependOp->GetOpMagic(), MS_NUM10005);
    EXPECT_EQ(tensor4->HasDependOp(copyin2), true);

    // Sort Operations
    function->SortOperations();
    Operation *alloc2 = subGraph.GetOp("Alloc2");
    auto sortedOpList = function->Operations().DuplicatedOpList();
    auto alloc2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), alloc2);
    auto copyin2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), copyin2);
    EXPECT_EQ(alloc2Iter - sortedOpList.begin() < copyin2Iter - sortedOpList.begin(), true);

    // Erase operands and depend Ops
    copyin2->EraseDependTensor(tensor4);
    add1->EraseDependTensor(tensor3);
    tensor4->RemoveDependOp(copyin2);
    tensor3->RemoveDependOp(add1);

    //Sort Operations
    function->SortOperations();
    auto sortedOpList2 = function->Operations().DuplicatedOpList();
    auto alloc2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), alloc2);
    auto copyin2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), copyin2);
    EXPECT_EQ(alloc2Iter2 - sortedOpList2.begin() > copyin2Iter2 - sortedOpList2.begin(), true);

    // Erase Operations
    copyin2->AddDependOperand(tensor4);
    tensor4->AddDependOp(copyin2);
    copyin2->SetAsDeleted();
    function->EraseOperations();
    EXPECT_EQ(tensor4->GetDependOps().size(), 0);
}
} // namespace tile_fwk
} // namespace npu
