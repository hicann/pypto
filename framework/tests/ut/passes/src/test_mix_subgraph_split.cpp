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

// 辅助函数声明
namespace test_utils {
    void VerifyBasicChecks(Status status, Function& rootFunc);
    void VerifyProgramProperties(Function& rootFunc);
    void VerifyCallOpsAfterSplit(Function& rootFunc);
    void VerifyScopeTypes(Function& rootFunc, int expectedCubeCount, int expectedVectorCount);
    void VerifyCleanup(Function& rootFunc, Function* originalMixFunc, Operation* originalCallOp);
} // namespace test_utils

class MixSubgraphSplitTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

// 辅助函数实现
namespace test_utils {
void VerifyBasicChecks(Status status, Function& rootFunc) {
    // 状态验证
    ASSERT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";    
    // program数量验证
    auto& programs = rootFunc.programs_;
    EXPECT_EQ(programs.size(), 2) 
        << "Should have 2 programs after split (originally 1 Mix, split to 2 leaves)";
    // ID连续性验证
    EXPECT_NE(programs.find(0), programs.end()) << "Should have program ID 0";
    EXPECT_NE(programs.find(1), programs.end()) << "Should have program ID 1";
}   

void VerifyProgramProperties(Function& rootFunc) {
    auto& programs = rootFunc.programs_;
    for (const auto& [progId, func] : programs) {
        ASSERT_NE(func, nullptr) << "Function should not be null";
        // 验证名称包含leaf后缀
        std::string funcName = func->GetRawName();
        EXPECT_NE(funcName.find("leaf"), std::string::npos)
            << "Function name should contain 'leaf' suffix: " << funcName;      
        // 验证类型设置
        EXPECT_EQ(func->GetFunctionType(), FunctionType::STATIC);
        EXPECT_EQ(func->GetGraphType(), GraphType::BLOCK_GRAPH);
        // 验证programID一致性
        EXPECT_EQ(func->GetProgramId(), progId) 
            << "Function's program ID should match map key";
        // 验证LeafFuncAttribute
        auto leafAttr = func->GetLeafFuncAttribute();
        ASSERT_NE(leafAttr, nullptr) << "LeafFuncAttribute should be set";
        EXPECT_NE(leafAttr->mixId, static_cast<uint64_t>(-1)) << "mixId should be assigned";
        EXPECT_NE(leafAttr->mixResourceType, MixResourceType::UNKNOWN)
            << "mixResourceType should be set";
    }
} 

void VerifyCallOpsAfterSplit(Function& rootFunc) {
    // callOp数量验证
    auto newCallOps = rootFunc.GetCallopList();
    EXPECT_EQ(newCallOps.size(), 2) 
        << "Should have 2 call ops after split (1 original * 2 components)";
    // callOp属性验证
    auto& programs = rootFunc.programs_;
    for (auto* newCallOp : newCallOps) {
        ASSERT_NE(newCallOp, nullptr) << "CallOp should not be null";
        EXPECT_FALSE(newCallOp->IsDeleted()) << "CallOp should not be deleted";
        
        auto newCallAttr = dynamic_cast<CallOpAttribute*>(newCallOp->GetOpAttribute().get());
        ASSERT_NE(newCallAttr, nullptr) << "CallOpAttribute should exist";
        
        if (newCallAttr && newCallAttr->invokeInfo_) {
            uint64_t progId = newCallAttr->invokeInfo_->GetProgramId();
            EXPECT_TRUE(progId == 0 || progId == 1) 
                << "CallOp's program ID should be 0 or 1, got: " << progId;
            
            // 验证对应函数存在
            auto it = programs.find(progId);
            EXPECT_NE(it, programs.end()) 
                << "CallOp references non-existent program ID: " << progId;
            
            // 验证wrapId设置
            EXPECT_NE(newCallAttr->wrapId, static_cast<uint64_t>(-1)) 
                << "wrapId should be set";
        }
    }
}   

void VerifyScopeTypes(Function& rootFunc, int expectedCubeCount, int expectedVectorCount) {
    // scope类型验证
    auto& programs = rootFunc.programs_;
    int cubeCount = 0;
    int vectorCount = 0;
    
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr) {
            if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
                cubeCount++;
            } else if (leafAttr->aivCore == AIVCore::AIV0 || leafAttr->aivCore == AIVCore::AIV1) {
                vectorCount++;
            }
        }
    }
    
    EXPECT_EQ(cubeCount, expectedCubeCount) << "Cube component count mismatch";
    EXPECT_EQ(vectorCount, expectedVectorCount) << "Vector component count mismatch";
}  

void VerifyCleanup(Function& rootFunc, Function* originalMixFunc, Operation* originalCallOp) {
    auto& programs = rootFunc.programs_;
    // 原始Mix子图验证
    bool originalMixFuncStillExists = false;
    for (const auto& [progId, func] : programs) {
        if (func == originalMixFunc) {
            originalMixFuncStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalMixFuncStillExists) << "Original mix function should be removed";
    
    // 原始callOp清理验证
    auto newCallOps = rootFunc.GetCallopList();
    bool originalCallOpStillExists = false;
    for (auto* callOpPtr : newCallOps) {
        if (callOpPtr == originalCallOp) {
            originalCallOpStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalCallOpStillExists) << "Original callOp should be deleted";    
    // 验证programs映射正确性
    std::set<uint64_t> programIdsFromCallOps;
    for (auto* callOpPtr : newCallOps) {
        auto newCallOpAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
        if (newCallOpAttr && newCallOpAttr->invokeInfo_) {
            programIdsFromCallOps.insert(newCallOpAttr->invokeInfo_->GetProgramId());
        }
    }
    for (const auto& [progId, func] : programs) {
        EXPECT_NE(programIdsFromCallOps.find(progId), programIdsFromCallOps.end())
            << "Program ID " << progId << " should have corresponding callOp";
    }
}
} // namespace test_utils

// 在测试类定义之后，测试用例之前，先定义所有辅助函数

// 辅助函数1：验证结果
void VerifyBasicSplitResult(Status status, Function& rootFunc, 
                           Function* originalMixFunc, Operation* originalCallOp) {
    using namespace test_utils;
    VerifyBasicChecks(status, rootFunc);
    VerifyProgramProperties(rootFunc);
    VerifyCallOpsAfterSplit(rootFunc);
    VerifyScopeTypes(rootFunc, 1, 1);
    VerifyCleanup(rootFunc, originalMixFunc, originalCallOp);
}

// 辅助函数2：创建CallOp
Operation& CreateCallOp(std::shared_ptr<Function>& rootFuncPtr,
                       const uint64_t mixProgramId,
                       const FunctionHash& mixFuncHash) {
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    
    auto callInTensor1 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callInTensor2 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callInTensor3 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    auto callOutTensor = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, tensorShape);
    
    auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, 
        {callInTensor1, callInTensor2, callInTensor3}, 
        {callOutTensor});
    
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(mixProgramId);
    callAttr->SetCalleeHash(mixFuncHash);
    callAttr->invokeInfo_ = invokeInfo;
    
    std::vector<SymbolicScalar> linearArgs;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 9; j++) {
            linearArgs.push_back(SymbolicScalar(static_cast<int64_t>(j + i * 10)));
        }
    }
    callAttr->linearArgList_ = linearArgs;
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(mixProgramId);
    
    return callOp;
}

// 辅助函数3：创建Vector scope
void CreateVectorScope(std::shared_ptr<Function>& mixFuncPtr,
                      std::shared_ptr<LogicalTensor>& incast3,
                      std::shared_ptr<LogicalTensor>& outcast1) {
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    
    // 获取cubeTensor3（最后一个cube操作的输出）
    auto operations = mixFuncPtr->Operations(false);
    auto cubeTensor3 = operations[operations.size() - 1].GetOOperands()[0];
    
    auto vectorTensor1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto vectorTensor2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;
    
    // Vector scope op（internalSubgraphID=1）
    auto& vectorAdd = mixFuncPtr->AddRawOperation(Opcode::OP_ADD, {cubeTensor3, incast3}, {vectorTensor1});
    vectorAdd.SetIOpAttrOffset(1, 5);
    vectorAdd.UpdateInternalSubgraphID(1);
    vectorAdd.SetAIVCore(AIVCore::AIV0);

    auto& vectorSqrt = mixFuncPtr->AddRawOperation(Opcode::OP_SQRT, {vectorTensor1}, {vectorTensor2});
    vectorSqrt.UpdateInternalSubgraphID(1);
    vectorSqrt.SetAIVCore(AIVCore::AIV0);

    auto& vectorCopyOut = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_OUT, {vectorTensor2}, {outcast1});
    vectorCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
    vectorCopyOut.SetOOpAttrOffset(0, 0);
    vectorCopyOut.UpdateInternalSubgraphID(1);
    vectorCopyOut.SetAIVCore(AIVCore::AIV0);
}

// 辅助函数4：创建Cube scope
void CreateCubeScope(std::shared_ptr<Function>& mixFuncPtr,
                     std::shared_ptr<LogicalTensor>& incast1,
                     std::shared_ptr<LogicalTensor>& incast2) {
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    auto cubeTensor1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto cubeTensor2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto cubeTensor3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;
    
    // Cube scope op（internalSubgraphID=0）
    auto& cubeCopyIn1 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {cubeTensor1});
    cubeCopyIn1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn1.SetIOpAttrOffset(0, 0);
    cubeCopyIn1.UpdateInternalSubgraphID(0);
    cubeCopyIn1.SetAttr(OpAttributeKey::isCube, true);

    auto& cubeCopyIn2 = mixFuncPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {cubeTensor2});
    cubeCopyIn2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn2.SetIOpAttrOffset(0, 0);
    cubeCopyIn2.UpdateInternalSubgraphID(0);
    cubeCopyIn2.SetAttr(OpAttributeKey::isCube, true);
    
    auto& cubeMul = mixFuncPtr->AddRawOperation(Opcode::OP_A_MUL_B, {cubeTensor1, cubeTensor2}, {cubeTensor3});
    cubeMul.UpdateInternalSubgraphID(0);
    cubeMul.SetAttr(OpAttributeKey::isCube, true);
}

// 辅助函数5：设置Mix子图结构
void SetupMixSubgraphStructure(std::shared_ptr<Function>& mixFuncPtr,
                               FunctionHash& mixFuncHash) {
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    
    // 创建incast和outcast
    auto incast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast2 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto incast3 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    auto outcast1 = std::make_shared<LogicalTensor>(*mixFuncPtr, DT_FP32, tensorShape);
    
    mixFuncPtr->inCasts_.push_back(incast1);
    mixFuncPtr->inCasts_.push_back(incast2);
    mixFuncPtr->inCasts_.push_back(incast3);
    mixFuncPtr->outCasts_.push_back(outcast1);
    
    // 创建Cube scope
    CreateCubeScope(mixFuncPtr, incast1, incast2);
    
    // 创建Vector scope
    CreateVectorScope(mixFuncPtr, incast3, outcast1);
    
    // 计算哈希并注册
    mixFuncPtr->ComputeHash();
    mixFuncHash = mixFuncPtr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *mixFuncPtr);
}

// 辅助函数6：创建Mix子图
std::shared_ptr<Function> CreateMixSubgraph(std::shared_ptr<Function>& rootFuncPtr,
                                           FunctionHash& mixFuncHash) {
    auto mixFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_mix_func", "test_mix_func", rootFuncPtr.get());
    mixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFuncPtr->SetFunctionType(FunctionType::STATIC);
    
    // 添加到programs，使用programId=100
    const uint64_t mixProgramId = 100;
    rootFuncPtr->programs_[mixProgramId] = mixFuncPtr.get();
    
    // 创建Mix子图内部结构
    SetupMixSubgraphStructure(mixFuncPtr, mixFuncHash);
    
    return mixFuncPtr;
}

// 辅助函数7：创建根函数
std::shared_ptr<Function> CreateTestRootFunction() {
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    return rootFuncPtr;
}

TEST_F(MixSubgraphSplitTest, TestSingleMixSubgraphBasicSplit) {
    // 1. 创建测试场景
    auto rootFuncPtr = CreateTestRootFunction();
    
    FunctionHash mixFuncHash;
    auto mixFuncPtr = CreateMixSubgraph(rootFuncPtr, mixFuncHash);
    
    // 2. 创建callOp
    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(rootFuncPtr, mixProgramId, mixFuncHash);
    
    // 3. 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    
    // 4. 验证结果
    VerifyBasicSplitResult(status, *rootFuncPtr, mixFuncPtr.get(), &callOp);
}

// 辅助函数1：为非Mix子图创建callOp（这个函数会被CreateNonMixFunctions调用，所以要放在前面）
void CreateCallOpForNonMix(std::shared_ptr<Function>& rootFuncPtr,
                          uint64_t programIdx,
                          FunctionHash hash,
                          const std::vector<int64_t>& shape) {
    auto callInTensor = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, shape);
    auto callOutTensor = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, shape);
    
    auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {callInTensor}, {callOutTensor});
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(programIdx);
    callAttr->SetCalleeHash(hash);
    callAttr->invokeInfo_ = invokeInfo;
    
    std::vector<SymbolicScalar> linearArgs;
    for (int argIdx = 0; argIdx < 2; argIdx++) {
        for (int j = 0; j < 9; j++) {
            linearArgs.push_back(SymbolicScalar(static_cast<int64_t>(j + argIdx * 10)));
        }
    }
    callAttr->linearArgList_ = linearArgs;
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(programIdx);
}

// 辅助函数2：创建非Mix子图（这个函数调用CreateCallOpForNonMix）
void CreateNonMixFunctions(std::shared_ptr<Function>& rootFuncPtr,
                          std::vector<std::shared_ptr<Function>>& nonMixFunctions,
                          const std::vector<uint64_t>& nonMixProgramIds) {
    for (int i = 0; i < 2; i++) {
        auto nonMixFunc = std::make_shared<Function>(
            Program::GetInstance(),
            "test_non_mix_" + std::to_string(i),
            "test_non_mix_" + std::to_string(i),
            rootFuncPtr.get());
        nonMixFunc->SetGraphType(GraphType::BLOCK_GRAPH);
        nonMixFunc->SetFunctionType(FunctionType::STATIC);

        std::vector<int64_t> shape = {8, 8};
        auto incastTensor = std::make_shared<LogicalTensor>(*nonMixFunc, DT_FP32, shape);
        auto outcastTensor = std::make_shared<LogicalTensor>(*nonMixFunc, DT_FP32, shape);
        
        nonMixFunc->inCasts_.push_back(incastTensor);
        nonMixFunc->outCasts_.push_back(outcastTensor);

        auto internalTensor1 = std::make_shared<LogicalTensor>(*nonMixFunc, DT_FP32, shape);
        auto& copyInOp = nonMixFunc->AddRawOperation(Opcode::OP_COPY_IN, {incastTensor}, {internalTensor1});
        copyInOp.SetIOpAttrOffset(0, 0);
        
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;
        auto copyInAttr = std::make_shared<CopyOpAttribute>(
            offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec);
        copyInOp.SetOpAttribute(copyInAttr);

        auto internalTensor2 = std::make_shared<LogicalTensor>(*nonMixFunc, DT_FP32, shape);
        auto& expOp = nonMixFunc->AddRawOperation(Opcode::OP_EXP, {internalTensor1}, {internalTensor2});
        (void) expOp;
        auto& copyOutOp = nonMixFunc->AddRawOperation(Opcode::OP_COPY_OUT, {internalTensor2}, {outcastTensor});
        copyOutOp.SetOOpAttrOffset(0, 0);
        
        auto copyOutAttr = std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec);
        copyOutOp.SetOpAttribute(copyOutAttr);

        nonMixFunc->ComputeHash();
        FunctionHash hash = nonMixFunc->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *nonMixFunc);
        rootFuncPtr->programs_[nonMixProgramIds[i]] = nonMixFunc.get();
        nonMixFunctions.push_back(nonMixFunc);

        // 调用CreateCallOpForNonMix创建对应的callOp
        CreateCallOpForNonMix(rootFuncPtr, nonMixProgramIds[i], hash, shape);
    }
}

void CreateCallOpsForMixFunction(std::shared_ptr<Function>& rootFuncPtr,
                                uint64_t programIdx,
                                FunctionHash hash,
                                const std::vector<int64_t>& shape,
                                int mixIdx) {
    int callOpCount = (mixIdx % 2 == 0) ? 1 : 2;
    
    for (int callIdx = 0; callIdx < callOpCount; callIdx++) {
        auto callInTensor1 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, shape);
        auto callInTensor2 = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, shape);
        auto callOutTensor = std::make_shared<LogicalTensor>(*rootFuncPtr, DT_FP32, shape);
        
        auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, 
            {callInTensor1, callInTensor2}, 
            {callOutTensor});
        
        auto callAttr = std::make_shared<CallOpAttribute>();
        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programIdx);
        callAttr->SetCalleeHash(hash);
        callAttr->invokeInfo_ = invokeInfo;
        
        std::vector<SymbolicScalar> linearArgs;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 9; j++) {
                linearArgs.push_back(SymbolicScalar(static_cast<int64_t>(j + i * 10)));
            }
        }
        callAttr->linearArgList_ = linearArgs;
        callOp.SetOpAttribute(callAttr);
        callOp.UpdateSubgraphID(programIdx);
    }
}

// 辅助函数4：创建额外的scope（这个函数会被CreateMixFunctions调用）
void CreateAdditionalScopes(std::shared_ptr<Function>& mixFunc,
                           int componentCount,
                           const std::vector<int64_t>& shape) {
    // 获取最后一个tensor
    auto operations = mixFunc->Operations(false);
    auto lastTensor = operations.back().GetOOperands()[0];
    
    for (int compIdx = 1; compIdx < componentCount; compIdx++) {
        auto newTensor = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        Opcode opcode = (compIdx % 2 == 0) ? Opcode::OP_NEG : Opcode::OP_SQRT;
        
        if (compIdx % 2 == 0) {
            auto& cubeOp = mixFunc->AddRawOperation(opcode, {lastTensor}, {newTensor});
            cubeOp.UpdateInternalSubgraphID(compIdx);
            cubeOp.SetAttr(OpAttributeKey::isCube, true);
        } else {
            auto& vectorOp = mixFunc->AddRawOperation(opcode, {lastTensor}, {newTensor});
            vectorOp.UpdateInternalSubgraphID(compIdx);
            vectorOp.SetAIVCore(AIVCore::AIV0);
        }
        lastTensor = newTensor;
    }
    
    // 创建COPY_OUT op
    auto& copyOut = mixFunc->AddRawOperation(Opcode::OP_COPY_OUT, {lastTensor}, {mixFunc->outCasts_[0]});
    copyOut.UpdateInternalSubgraphID(componentCount - 1);
    copyOut.SetOOpAttrOffset(0, 0);
    
    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;
    auto copyOutAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec);
    copyOut.SetOpAttribute(copyOutAttr);
    
    if ((componentCount - 1) % 2 == 0) {
        copyOut.SetAttr(OpAttributeKey::isCube, true);
    } else {
        copyOut.SetAIVCore(AIVCore::AIV0);
    }
}

void CreateMixFunctions(std::shared_ptr<Function>& rootFuncPtr,
                       std::vector<std::shared_ptr<Function>>& mixFunctions,
                       const std::vector<uint64_t>& mixProgramIds,
                       const std::vector<int>& componentCounts) {
    for (int mixIdx = 0; mixIdx < 3; mixIdx++) {
        auto mixFunc = std::make_shared<Function>(Program::GetInstance(),"test_mix_" + std::to_string(mixIdx),
            "test_mix_" + std::to_string(mixIdx), rootFuncPtr.get());
        mixFunc->SetGraphType(GraphType::BLOCK_GRAPH);
        mixFunc->SetFunctionType(FunctionType::STATIC);

        std::vector<int64_t> shape = {16, 16};
        auto incast1 = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        auto incast2 = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        auto outcast = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        
        mixFunc->inCasts_.push_back(incast1);
        mixFunc->inCasts_.push_back(incast2);
        mixFunc->outCasts_.push_back(outcast);

        // 创建Cube scope
        auto cubeTensor1 = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        auto cubeTensor2 = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        auto cubeTensor3 = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        auto cubeOutput = std::make_shared<LogicalTensor>(*mixFunc, DT_FP32, shape);
        
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;
        
        auto& copyIn1 = mixFunc->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {cubeTensor1});
        copyIn1.UpdateInternalSubgraphID(0);
        copyIn1.SetIOpAttrOffset(0, 0);
        copyIn1.SetAttr(OpAttributeKey::isCube, true);
        auto copyIn1Attr = std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec);
        copyIn1.SetOpAttribute(copyIn1Attr);

        auto& copyIn2 = mixFunc->AddRawOperation(Opcode::OP_COPY_IN, {incast2}, {cubeTensor2});
        copyIn2.UpdateInternalSubgraphID(0);
        copyIn2.SetIOpAttrOffset(0, 0);
        copyIn2.SetAttr(OpAttributeKey::isCube, true);
        auto copyIn2Attr = std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec);
        copyIn2.SetOpAttribute(copyIn2Attr);
        
        auto& cubeMul = mixFunc->AddRawOperation(Opcode::OP_A_MUL_B, {cubeTensor1, cubeTensor2}, {cubeTensor3});
        cubeMul.UpdateInternalSubgraphID(0);
        cubeMul.SetAttr(OpAttributeKey::isCube, true);
        
        auto& cubeExp = mixFunc->AddRawOperation(Opcode::OP_EXP, {cubeTensor3}, {cubeOutput});
        cubeExp.UpdateInternalSubgraphID(0);
        cubeExp.SetAttr(OpAttributeKey::isCube, true);
        
        // 调用CreateAdditionalScopes创建后续scope
        CreateAdditionalScopes(mixFunc, componentCounts[mixIdx], shape);
        
        mixFunc->ComputeHash();
        FunctionHash hash = mixFunc->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *mixFunc);
        rootFuncPtr->programs_[mixProgramIds[mixIdx]] = mixFunc.get();
        mixFunctions.push_back(mixFunc);
        
        // 调用CreateCallOpsForMixFunction为Mix子图创建callOp
        CreateCallOpsForMixFunction(rootFuncPtr, mixProgramIds[mixIdx], hash, shape, mixIdx);
    }
}

// 辅助函数6：验证多个Mix子图拆分结果（最后定义，因为它不调用其他辅助函数）
void VerifyMultipleMixSplitResults(std::shared_ptr<Function>& rootFuncPtr,
                                  Status status,
                                  const std::vector<std::shared_ptr<Function>>& mixFunctions,
                                  const std::vector<std::shared_ptr<Function>>& nonMixFunctions,
                                  const std::vector<int>& componentCounts) {
    EXPECT_EQ(status, SUCCESS) << "Multiple mix subgraphs split should succeed";
    
    // 计算预期的新子图总数
    size_t expectedNewProgramCount = 2;  // 2个非Mix子图保留
    for (int count : componentCounts) {
        expectedNewProgramCount += count;  // 每个Mix子图的scope数
    }
    
    auto& programs = rootFuncPtr->programs_;
    EXPECT_EQ(programs.size(), expectedNewProgramCount)
        << "Program count mismatch. Expected: " << expectedNewProgramCount
        << ", Actual: " << programs.size();
    
    // ID连续性验证
    uint64_t expectedMaxId = expectedNewProgramCount - 1;
    for (uint64_t i = 0; i <= expectedMaxId; i++) {
        EXPECT_NE(programs.find(i), programs.end())
            << "Missing continuous program ID: " << i;
    }
    
    // 非Mix子图ID重映射验证
    for (size_t i = 0; i < nonMixFunctions.size(); i++) {
        bool found = false;
        for (const auto& [progId, func] : programs) {
            if (func == nonMixFunctions[i].get()) {
                EXPECT_LT(progId, 2) << "Non-mix function should have ID < 2";
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Non-mix function " << i << " not found after split";
    }
    
    // Mix子图scope验证
    int totalSplitFunctions = 0;
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr && leafAttr->mixId != -1) {
            totalSplitFunctions++;
        }
    }
    
    EXPECT_EQ(totalSplitFunctions, 2 + 3 + 4)
        << "Should have " << (2+3+4) << " split functions from 3 mix subgraphs";
    
    // callOp数量验证
    auto newCallOps = rootFuncPtr->GetCallopList();
    size_t expectedNewCallOpCount = 2 * 1 + 1 * 2 + 2 * 3 + 1 * 4;
    EXPECT_EQ(newCallOps.size(), expectedNewCallOpCount)
        << "CallOp count mismatch. Expected: " << expectedNewCallOpCount
        << ", Actual: " << newCallOps.size();
    
    // 资源清理验证
    for (const auto& mixFunc : mixFunctions) {
        bool stillExists = false;
        for (const auto& [progId, func] : programs) {
            if (func == mixFunc.get()) {
                stillExists = true;
                break;
            }
        }
        EXPECT_FALSE(stillExists) << "Original mix function should be removed";
    }
}

/**
 * 测试多个Mix子图拆分处理
 */
TEST_F(MixSubgraphSplitTest, TestMultipleMixSubgraphsSplit) {
    // 1. 创建rootFunction
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root_multi", "test_root_multi", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    
    // 2. 创建3个Mix子图和2个非Mix子图
    std::vector<std::shared_ptr<Function>> mixFunctions;
    std::vector<std::shared_ptr<Function>> nonMixFunctions;
    std::vector<uint64_t> mixProgramIds = {100, 101, 102};
    std::vector<uint64_t> nonMixProgramIds = {200, 201};
    
    // 2.1 创建2个非Mix子图
    CreateNonMixFunctions(rootFuncPtr, nonMixFunctions, nonMixProgramIds);
    
    // 2.2 创建3个Mix子图，scope数量分别为2,3,4
    std::vector<int> componentCounts = {2, 3, 4};
    CreateMixFunctions(rootFuncPtr, mixFunctions, mixProgramIds, componentCounts);
    
    // 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    
    // 验证结果
    VerifyMultipleMixSplitResults(rootFuncPtr, status, mixFunctions, nonMixFunctions, componentCounts);
}

/**
 * 测试rootFunction无Mix子图时的处理逻辑
 */
TEST_F(MixSubgraphSplitTest, TestNoMixSubgraphScenario) {
    // 1. 创建仅包含非Mix子图的rootFunction
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_root_no_mix", "test_root_no_mix", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // 2. 创建3个普通（非Mix）子图
    std::vector<std::shared_ptr<Function>> nonMixFunctions;
    std::vector<uint64_t> programIds = {10, 20, 30};
    for (int i = 0; i < 3; i++) {
        auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_" + std::to_string(i),
            "test_func_" + std::to_string(i), rootFuncPtr.get());
        func->SetGraphType(GraphType::BLOCK_GRAPH);
        func->SetFunctionType(FunctionType::STATIC);
        // 创建简单op（无internalSubgraphID标记）
        std::vector<int64_t> shape = {8, 8};
        auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        func->inCasts_.push_back(inputTensor);
        func->outCasts_.push_back(outputTensor);
        auto& expOp = func->AddRawOperation(Opcode::OP_EXP, {inputTensor}, {outputTensor});
        (void) expOp;
        func->ComputeHash();
        FunctionHash hash = func->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *func);
        
        rootFuncPtr->programs_[programIds[i]] = func.get();
        nonMixFunctions.push_back(func);
         // 创建callOp
        auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
        auto callAttr = std::make_shared<CallOpAttribute>();
        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programIds[i]);
        callAttr->SetCalleeHash(hash);
        callAttr->invokeInfo_ = invokeInfo;
        callOp.SetOpAttribute(callAttr);
    }
      // 3. 记录原始状态
    auto originalPrograms = rootFuncPtr->programs_;
    auto originalCallOps = rootFuncPtr->GetCallopList();
    size_t originalProgramCount = originalPrograms.size();  // 应该为3
    size_t originalCallOpCount = originalCallOps.size();    // 应该为3
    // 4. 执行拆分（应该提前退出）
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    // 5. 验证结果
    EXPECT_EQ(status, SUCCESS) << "Should succeed even with no mix subgraphs";
    // 5.1 验证IsMixSubgraph返回false
    for (const auto& func : nonMixFunctions) {
        bool isMix = splitter.IsMixSubgraph(*func);
        EXPECT_FALSE(isMix) << "Non-mix function should not be identified as mix";
    }
    // 5.2 验证programs保持不变
    auto& newPrograms = rootFuncPtr->programs_;
    EXPECT_EQ(newPrograms.size(), originalProgramCount) << "Program count should not change when no mix subgraphs";
    // 5.3 验证callOps数量不变
    auto newCallOps = rootFuncPtr->GetCallopList();
    EXPECT_EQ(newCallOps.size(), originalCallOpCount) << "CallOp count should not change when no mix subgraphs";
}

// 辅助函数：创建外部Mix子图
void CreateExternalMixFunction(std::shared_ptr<Function>& externalRootFuncPtr,
                              std::shared_ptr<Function>& externalMixFuncPtr) {
    // 创建外部root function
    externalRootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "external_root", "external_root", nullptr);
    externalRootFuncPtr->rootFunc_ = externalRootFuncPtr.get();
    
    // 创建Mix子图
    const uint64_t externalMixProgramId = 999;
    externalMixFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "external_mix", "external_mix", externalRootFuncPtr.get());
    externalMixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    externalMixFuncPtr->SetFunctionType(FunctionType::STATIC);
    
    // 添加到programs
    externalRootFuncPtr->programs_[externalMixProgramId] = externalMixFuncPtr.get();
    
    // 创建Mix子图内部结构
    std::vector<int64_t> shape = {16, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*externalMixFuncPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*externalMixFuncPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*externalMixFuncPtr, DT_FP32, shape);

    externalMixFuncPtr->inCasts_.push_back(incast1);
    externalMixFuncPtr->inCasts_.push_back(incast2);
    externalMixFuncPtr->outCasts_.push_back(outcast1);
    
    // 创建3个scope的op
    for (int compIdx = 0; compIdx < 3; compIdx++) {
        auto inputTensor = std::make_shared<LogicalTensor>(*externalMixFuncPtr, DT_FP32, shape);
        auto outputTensor = std::make_shared<LogicalTensor>(*externalMixFuncPtr, DT_FP32, shape);
        Opcode opcode = Opcode::OP_EXP;
        auto& op = externalMixFuncPtr->AddRawOperation(opcode, {inputTensor}, {outputTensor});
        op.UpdateInternalSubgraphID(compIdx);
        
        // 交替设置Cube和Vector
        if (compIdx % 2 == 0) {
            op.SetAttr(OpAttributeKey::isCube, true);
        } else {
            op.SetAIVCore((compIdx == 1) ? AIVCore::AIV0 : AIVCore::AIV1);
        }
    }
    
    // 计算哈希并注册到缓存
    externalMixFuncPtr->ComputeHash();
    FunctionHash mixFuncHash = externalMixFuncPtr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *externalMixFuncPtr);
}

// 辅助函数：验证跨function调用结果
void VerifyCrossFunctionResults(MixSubgraphSplit& splitter,
                               std::shared_ptr<Function>& rootFuncPtr,
                               std::shared_ptr<Function>& externalRootFuncPtr,
                               std::shared_ptr<Function>& externalMixFuncPtr,
                               Operation* crossCallOp,
                               Status status) {
    EXPECT_EQ(status, FAILED) << "Fresh external mix function without preceding processing should fail";
    
    // 9.1 验证Mix子图识别
    bool isMix = splitter.IsMixSubgraph(*externalMixFuncPtr);
    EXPECT_TRUE(isMix) << "External mix function should be identified as mix";
    
    // 9.2 验证当前rootFunction的programs
    auto& programs = rootFuncPtr->programs_;
    // 跨function场景：新创建的leafFunction不加入当前rootFunction的programs
    EXPECT_EQ(programs.size(), 0);
    
    // 9.3 验证callOp创建
    auto newCallOps = rootFuncPtr->GetCallopList();
    // 应该只有原始的callOp，没有新的callOp被创建
    EXPECT_EQ(newCallOps.size(), 1) << "Should only have original call op on failure";
    
    // 9.4 验证原始外部Mix子图状态不变
    // 外部Mix子图不应该被修改
    EXPECT_EQ(externalRootFuncPtr->programs_.size(), 1)
        << "External root function's programs should remain unchanged";
    
    auto it = externalRootFuncPtr->programs_.find(999);
    EXPECT_NE(it, externalRootFuncPtr->programs_.end())
        << "External mix function should still exist in external root";
    EXPECT_EQ(it->second, externalMixFuncPtr.get())
        << "External mix function pointer should be unchanged";

    // 9.5 验证原始的callOp仍然存在
    bool originalCallOpExists = false;
    for (auto* callOp : newCallOps) {
        if (callOp == crossCallOp) {
            originalCallOpExists = true;
            break;
        }
    }
    EXPECT_TRUE(originalCallOpExists) << "Original cross-function callOp should still exist on failure";
}

/**
 * 测试Mix子图为跨function调用时的特殊处理
 */
TEST_F(MixSubgraphSplitTest, TestCrossFunctionMixSubgraph) {
    // 1. 创建场景：Mix子图不在当前rootFunc的programs中
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root_cross", "test_root_cross", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    
    // 2. 创建外部Mix子图（保持为shared_ptr，避免析构）
    std::shared_ptr<Function> externalRootFuncPtr;
    std::shared_ptr<Function> externalMixFuncPtr;
    CreateExternalMixFunction(externalRootFuncPtr, externalMixFuncPtr);
    
    // 3. 获取Mix函数hash
    FunctionHash mixFuncHash = externalMixFuncPtr->GetFunctionHash();
    
    // 4. 创建指向外部Mix子图的callOp
    auto& crossCallOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto crossCallAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    crossCallAttr->SetCalleeHash(mixFuncHash);
    crossCallAttr->invokeInfo_ = invokeInfo;
    crossCallOp.SetOpAttribute(crossCallAttr);
    
    // 5. 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    
    // 6. 验证结果
    VerifyCrossFunctionResults(splitter, rootFuncPtr, externalRootFuncPtr, 
                              externalMixFuncPtr, &crossCallOp, status);
}
    
TEST_F(MixSubgraphSplitTest, TestDependencyRebuilding) {
    // 创建root function
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // ==================== 创建2个非Mix子图 ====================
    // 非Mix子图1：纯CUBE子图
    auto nonMixFunc1Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_non_mix_func1", "test_non_mix_func1", rootFuncPtr.get());
    nonMixFunc1Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    nonMixFunc1Ptr->SetFunctionType(FunctionType::STATIC);    
    // 非Mix子图2：纯VECTOR子图
    auto nonMixFunc2Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_non_mix_func2", "test_non_mix_func2", rootFuncPtr.get());
    nonMixFunc2Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    nonMixFunc2Ptr->SetFunctionType(FunctionType::STATIC);
    // ==================== 创建Mix子图3 ====================
    auto mixFunc3Ptr = std::make_shared<Function>(
        Program::GetInstance(), "test_mix_func3", "test_mix_func3", rootFuncPtr.get());
    mixFunc3Ptr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFunc3Ptr->SetFunctionType(FunctionType::STATIC);

    // 添加到programs 
    uint64_t nonMixProgramId1 = 0;
    uint64_t nonMixProgramId2 = 1;
    uint64_t mixProgramId = 2;
    rootFuncPtr->programs_[nonMixProgramId1] = nonMixFunc1Ptr.get();
    rootFuncPtr->programs_[nonMixProgramId2] = nonMixFunc2Ptr.get();
    rootFuncPtr->programs_[mixProgramId] = mixFunc3Ptr.get();

    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};

    // 为非Mix子图创建输入tensors
    auto nonMixInput1 = std::make_shared<LogicalTensor>(*nonMixFunc1Ptr, DT_FP32, tensorShape);
    auto nonMixInput2 = std::make_shared<LogicalTensor>(*nonMixFunc2Ptr, DT_FP32, tensorShape);

    // 创建Mix子图外部输入tensor（来自非Mix子图1和2）
    auto logicalTensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto logicalTensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 设置incast/outcast
    nonMixFunc1Ptr->inCasts_.push_back(nonMixInput1);
    nonMixFunc1Ptr->outCasts_.push_back(logicalTensor1); // 输出到Mix子图C1
    nonMixFunc2Ptr->inCasts_.push_back(nonMixInput2);   
    nonMixFunc2Ptr->outCasts_.push_back(logicalTensor2); // 输出到Mix子图C4

    // 创建Mix子图内部tensors
    // Scope C1的tensor
    auto c1_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c1_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V2的tensor
    auto v2_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v2_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // Scope C3的tensor
    auto c3_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c3_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope C4的tensor
    auto c4_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto c4_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V5的tensor
    auto v5_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v5_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    
    // Scope V6的tensor
    auto v6_tensor1 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto v6_tensor2 = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 全局输出tensor
    auto globalOutputTensor = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);

    // 设置incast/outcast
    mixFunc3Ptr->inCasts_.push_back(logicalTensor1);  // 来自非Mix子图1
    mixFunc3Ptr->inCasts_.push_back(logicalTensor2);  // 来自非Mix子图2
    mixFunc3Ptr->outCasts_.push_back(globalOutputTensor);  // 全局输出

    // 创建OpImmediate
    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;

    // 构建非Mix子图1的内部结构（纯CUBE）
    auto& copyin_nonmix1 = nonMixFunc1Ptr->AddRawOperation(Opcode::OP_COPY_IN, {nonMixInput1}, {logicalTensor1});
    copyin_nonmix1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_nonmix1.SetIOpAttrOffset(0, 0);
    copyin_nonmix1.SetAttr(OpAttributeKey::isCube, true);

    // 构建非Mix子图2的内部结构（纯VECTOR）
    auto& copyin_nonmix2 = nonMixFunc2Ptr->AddRawOperation(Opcode::OP_COPY_IN, {nonMixInput2}, {logicalTensor2});
    copyin_nonmix2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_nonmix2.SetIOpAttrOffset(0, 0);

    // ==================== 构建Mix子图内部结构 ====================
    // 路径1: C1 -> V2 -> C3 -> V6
    // Scope C1: 处理来自非Mix子图1的输入
    auto& copyin_c1 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_IN, {logicalTensor1}, {c1_tensor1});
    copyin_c1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_c1.SetIOpAttrOffset(0, 0);
    copyin_c1.UpdateInternalSubgraphID(0);
    copyin_c1.SetAttr(OpAttributeKey::isCube, true);

    auto& sqrt_c1= mixFunc3Ptr->AddRawOperation(Opcode::OP_SQRT, {c1_tensor1}, {c1_tensor2});
    sqrt_c1.UpdateInternalSubgraphID(0);
    sqrt_c1.SetAttr(OpAttributeKey::isCube, true);

    // Scope V2: 处理C1的输出
    auto& abs_v2 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {c1_tensor2}, {v2_tensor1});
    abs_v2.UpdateInternalSubgraphID(1);  // V2 scope
    abs_v2.SetAIVCore(AIVCore::AIV0);

    auto& abs1_v2 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {v2_tensor1}, {v2_tensor2});
    abs1_v2.UpdateInternalSubgraphID(1);
    abs1_v2.SetAIVCore(AIVCore::AIV0);

    // Scope C3: 处理V2的输出和V5的输出
    // 创建一个辅助tensor用于融合两个输入
    auto c3_aux_tensor = std::make_shared<LogicalTensor>(*mixFunc3Ptr, DT_FP32, tensorShape);
    auto& add_c3_inputs = mixFunc3Ptr->AddRawOperation(Opcode::OP_ADD, {v2_tensor2, v5_tensor2}, {c3_aux_tensor});
    add_c3_inputs.UpdateInternalSubgraphID(2);  // C3 scope
    add_c3_inputs.SetAttr(OpAttributeKey::isCube, true);

    auto& exp_c3 = mixFunc3Ptr->AddRawOperation(Opcode::OP_EXP, {c3_aux_tensor}, {c3_tensor1});
    exp_c3.UpdateInternalSubgraphID(2);
    exp_c3.SetAttr(OpAttributeKey::isCube, true);

    auto& log_c3 = mixFunc3Ptr->AddRawOperation(Opcode::OP_LN, {c3_tensor1}, {c3_tensor2});
    log_c3.UpdateInternalSubgraphID(2);
    log_c3.SetAttr(OpAttributeKey::isCube, true);

    // 路径2: C4 -> V5 -> C3 -> V6
    // Scope C4: 处理来自非Mix子图2的输入
    auto& copyin_c4 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_IN, {logicalTensor2}, {c4_tensor1});
    copyin_c4.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    copyin_c4.SetIOpAttrOffset(0, 0);
    copyin_c4.UpdateInternalSubgraphID(3);  // C4 scope
    copyin_c4.SetAttr(OpAttributeKey::isCube, true);

    auto& exp_c4 = mixFunc3Ptr->AddRawOperation(Opcode::OP_EXP, {c4_tensor1}, {c4_tensor2});
    exp_c4.UpdateInternalSubgraphID(3);
    exp_c4.SetAttr(OpAttributeKey::isCube, true);

    // Scope V5: 处理C4的输出
    auto& neg_v5 = mixFunc3Ptr->AddRawOperation(Opcode::OP_NEG, {c4_tensor2}, {v5_tensor1});
    neg_v5.UpdateInternalSubgraphID(4);  // V5 scope
    neg_v5.SetAIVCore(AIVCore::AIV1);

    auto& abs_v5 = mixFunc3Ptr->AddRawOperation(Opcode::OP_ABS, {v5_tensor1}, {v5_tensor2});
    abs_v5.UpdateInternalSubgraphID(4);
    abs_v5.SetAIVCore(AIVCore::AIV1);

    // Scope V6: 处理C3的输出，生成全局输出
    auto& sqrt_v6 = mixFunc3Ptr->AddRawOperation(Opcode::OP_SQRT, {c3_tensor2}, {v6_tensor1});
    sqrt_v6.UpdateInternalSubgraphID(5);
    sqrt_v6.SetAIVCore(AIVCore::AIV0);

    auto& copyout_v6 = mixFunc3Ptr->AddRawOperation(Opcode::OP_COPY_OUT, {v6_tensor1}, {globalOutputTensor});
    copyout_v6.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
    copyout_v6.SetOOpAttrOffset(0, 0);
    copyout_v6.UpdateInternalSubgraphID(5);
    copyout_v6.SetAIVCore(AIVCore::AIV0);

    // ==================== 第二部分：在root function中创建CallOp ====================
    // 创建一个函数来构建线性参数列表
    auto createLinearArgListForTensor = [&](const std::shared_ptr<LogicalTensor>& tensor) {
        (void)tensor;
        std::vector<SymbolicScalar> args;
        args.push_back(SymbolicScalar(1)); 
        // 第一维
        args.push_back(SymbolicScalar(0));  
        args.push_back(SymbolicScalar(1));   
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(0));  
        // 第二维
        args.push_back(SymbolicScalar(0));  
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(16)); 
        args.push_back(SymbolicScalar(0)); 
        return args;
    };

    nonMixFunc1Ptr->ComputeHash();
    FunctionHash nonMixHash1 = nonMixFunc1Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(nonMixHash1, *nonMixFunc1Ptr);
    nonMixFunc2Ptr->ComputeHash();
    FunctionHash nonMixHash2 = nonMixFunc2Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(nonMixHash2, *nonMixFunc2Ptr);   
    mixFunc3Ptr->ComputeHash();
    FunctionHash mixFuncHash = mixFunc3Ptr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *mixFunc3Ptr);
    // CallOp1: 指向非Mix子图1
    auto& callOp1 = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr1 = std::make_shared<CallOpAttribute>();
    auto invokeInfo1 = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo1->UpdateProgramSubgraphId(nonMixProgramId1);
    callAttr1->SetCalleeHash(nonMixHash1); 
    callAttr1->invokeInfo_ = invokeInfo1;
    callOp1.SetOpAttribute(callAttr1);
    // CallOp2: 指向非Mix子图2
    auto& callOp2 = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr2 = std::make_shared<CallOpAttribute>();
    auto invokeInfo2 = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo2->UpdateProgramSubgraphId(nonMixProgramId2);
    callAttr2->SetCalleeHash(nonMixHash2);
    callAttr2->invokeInfo_ = invokeInfo2;
    callOp2.SetOpAttribute(callAttr2);
    // 创建指向Mix子图3的CallOp
    auto& callOp = rootFuncPtr->AddRawOperation(Opcode::OP_CALL, {}, {});
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(mixProgramId);    
    // 设置linearArgList_（Mix子图有2个输入，1个输出）
    std::vector<SymbolicScalar> linearArgs;
    // 输入1
    auto inputArgs1_mix = createLinearArgListForTensor(logicalTensor1);
    linearArgs.insert(linearArgs.end(), inputArgs1_mix.begin(), inputArgs1_mix.end());
    // 输入2
    auto inputArgs2_mix = createLinearArgListForTensor(logicalTensor2);
    linearArgs.insert(linearArgs.end(), inputArgs2_mix.begin(), inputArgs2_mix.end());
    // 输出
    auto outputArgs_mix = createLinearArgListForTensor(globalOutputTensor);
    linearArgs.insert(linearArgs.end(), outputArgs_mix.begin(), outputArgs_mix.end());
    callAttr->linearArgList_ = linearArgs;
    callAttr->SetCalleeHash(mixFuncHash);
    callAttr->invokeInfo_ = invokeInfo;
    callOp.SetOpAttribute(callAttr);

    // ==================== 第三部分：执行MixSubgraphSplit ====================
    MixSubgraphSplit splitter;

    // 执行拆分
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";

    // ==================== 第四部分：验证依赖重建结果 ====================
    auto& programs = rootFuncPtr->programs_;
    // 1. 验证Mix子图已被删除
    bool mixFuncStillExists = false;
    for (const auto& [progId, func] : programs) {
        if (func == mixFunc3Ptr.get()) {
            mixFuncStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(mixFuncStillExists) << "Original mix function should be deleted";

    // 2. 验证总Function数量
    EXPECT_EQ(programs.size(), 2 + 6) << "Should have 2 non-mix + 6 mix leaf functions";

    std::vector<Function*> mixSplitFunctions;
    for (const auto& [progId, func] : programs) {
        auto leafAttr = func->GetLeafFuncAttribute();
        if (!leafAttr) continue;
        
        // 如果设置了mixId，说明是Mix子图拆分出来的
        if (leafAttr->mixId != -1) {
            mixSplitFunctions.push_back(func);
        }
    }
    
    EXPECT_EQ(mixSplitFunctions.size(), 6) << "Should have 6 functions from mix subgraph split";
    // 从programs中查找各个LeafFunction
    Function* c1Func = programs[2];
    Function* v2Func = programs[3];
    Function* c3Func = programs[4];
    Function* c4Func = programs[5];
    Function* v5Func = programs[6];
    Function* v6Func = programs[7];
    ASSERT_NE(c1Func, nullptr) << "C1 function not found at programId=2";
    ASSERT_NE(v2Func, nullptr) << "V2 function not found at programId=3";
    ASSERT_NE(c3Func, nullptr) << "C3 function not found at programId=4";
    ASSERT_NE(c4Func, nullptr) << "C4 function not found at programId=5";
    ASSERT_NE(v5Func, nullptr) << "V5 function not found at programId=6";
    ASSERT_NE(v6Func, nullptr) << "V6 function not found at programId=7";

    // 3. 验证外部依赖正确传播
    // 检查C1 scope应该接收到logicalTensor1（来自非Mix子图1）
    bool hasLogicalTensor1 = false;
    for (const auto& tensor : c1Func->GetIncast()) {
        if (tensor == logicalTensor1) {
            hasLogicalTensor1 = true;
            break;
        }
    }
    EXPECT_TRUE(hasLogicalTensor1) << "C1 should have logicalTensor1 as incast";

    // 检查V2组件也应该接收到logicalTensor1（通过C1传播）
    bool v2HasLogicalTensor1 = false;
    for (const auto& tensor : v2Func->GetIncast()) {
        if (tensor == logicalTensor1) {
            v2HasLogicalTensor1 = true;
            break;
        }
    }
    EXPECT_TRUE(v2HasLogicalTensor1) << "V2 should have logicalTensor1 as incast (propagated from C1)";
    // 检查C4 scope应该接收到logicalTensor2（来自非Mix子图2）
    bool hasLogicalTensor2 = false;
    for (const auto& tensor : c4Func->GetIncast()) {
        if (tensor == logicalTensor2) {
            hasLogicalTensor2 = true;
            break;
        }
    }
    EXPECT_TRUE(hasLogicalTensor2) << "C4 should have logicalTensor2 as incast";

    // 检查V5组件也应该接收到logicalTensor2（通过C4传播）
    bool v5HasLogicalTensor2 = false;
    for (const auto& tensor : v5Func->GetIncast()) {
        if (tensor == logicalTensor2) {
            v5HasLogicalTensor2 = true;
            break;
        }
    }
    EXPECT_TRUE(v5HasLogicalTensor2) << "V5 should have logicalTensor2 as incast (propagated from C4)";

    // 检查V6组件应该输出globalOutputTensor
    bool hasGlobalOutput = false;
    for (const auto& tensor : v6Func->GetOutcast()) {
        if (tensor == globalOutputTensor) {
            hasGlobalOutput = true;
            break;
        }
    }
    EXPECT_TRUE(hasGlobalOutput) << "V6 should have globalOutputTensor as outcast";

    // 验证冗余依赖消除：检查C3和V6不应该直接接收logicalTensor1或logicalTensor2（应通过V2和V5间接获取）
    bool c3HasDirectTensor1 = false;
    bool c3HasDirectTensor2 = false;
    for (const auto& tensor : c3Func->GetIncast()) {
        if (tensor == logicalTensor1) c3HasDirectTensor1 = true;
        if (tensor == logicalTensor2) c3HasDirectTensor2 = true;
    }
    
    EXPECT_FALSE(c3HasDirectTensor1) << "C3 should not have direct incast of logicalTensor1 (should get via V2)";
    EXPECT_FALSE(c3HasDirectTensor2) << "C3 should not have direct incast of logicalTensor2 (should get via V5)";

    bool v6HasDirectTensor1 = false, v6HasDirectTensor2 = false;
    for (const auto& tensor : v6Func->GetIncast()) {
        if (tensor == logicalTensor1) v6HasDirectTensor1 = true;
        if (tensor == logicalTensor2) v6HasDirectTensor2 = true;
    }
    EXPECT_FALSE(v6HasDirectTensor1) << "V6 should not have direct incast of logicalTensor1 (redundant)";
    EXPECT_FALSE(v6HasDirectTensor2) << "V6 should not have direct incast of logicalTensor2 (redundant)";

    // 获取所有CallOps并构建Function到CallOp的映射
    auto newCallOps = rootFuncPtr->GetCallopList();
    std::unordered_map<Function*, Operation*> funcToCallOp;
    for (auto* callOpPtr : newCallOps) {
        auto loopCallAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
        if (loopCallAttr && loopCallAttr->invokeInfo_) {
            uint64_t progId = loopCallAttr->invokeInfo_->GetProgramId();
            auto it = programs.find(progId);
            if (it != programs.end()) {
                funcToCallOp[it->second] = callOpPtr;
            }
        }
    }
    Operation* c1CallOp = funcToCallOp[c1Func];
    Operation* v2CallOp = funcToCallOp[v2Func];
    Operation* c3CallOp = funcToCallOp[c3Func];
    Operation* c4CallOp = funcToCallOp[c4Func];
    Operation* v5CallOp = funcToCallOp[v5Func];
    Operation* v6CallOp = funcToCallOp[v6Func];
    ASSERT_NE(c1CallOp, nullptr) << "C1 callOp not found";
    ASSERT_NE(v2CallOp, nullptr) << "V2 callOp not found";
    ASSERT_NE(c3CallOp, nullptr) << "C3 callOp not found";
    ASSERT_NE(c4CallOp, nullptr) << "C4 callOp not found";
    ASSERT_NE(v5CallOp, nullptr) << "V5 callOp not found";
    ASSERT_NE(v6CallOp, nullptr) << "V6 callOp not found";
    // 验证内部依赖Dummy Tensors
    auto checkDependency = [](Operation* consumer, Operation* producer) -> bool {
        auto dependOperands = consumer->GetDependOperands();
        for (const auto& tensor : dependOperands) {
            if (!tensor) continue;
            bool hasProducer = false, hasConsumer = false;
            for (auto* p : tensor->GetProducers()) if (p == producer) hasProducer = true;
            for (auto* c : tensor->GetConsumers()) if (c == consumer) hasConsumer = true;
            if (hasProducer && hasConsumer) return true;
        }
        return false;
    };
    // C-C依赖：C1 -> C3, C4 -> C3
    EXPECT_TRUE(checkDependency(c3CallOp, c1CallOp)) << "Should have dependency C1 -> C3 (C-C dependency)";
    EXPECT_TRUE(checkDependency(c3CallOp, c4CallOp)) << "Should have dependency C4 -> C3 (C-C dependency)";

    // V-V依赖：V2 -> V6, V5 -> V6
    EXPECT_TRUE(checkDependency(v6CallOp, v2CallOp)) << "Should have dependency V2 -> V6 (V-V dependency)";
    EXPECT_TRUE(checkDependency(v6CallOp, v5CallOp)) << "Should have dependency V5 -> V6 (V-V dependency)";

    // 验证不应该有跨类型的依赖
    EXPECT_FALSE(checkDependency(v2CallOp, c1CallOp)) << "Should NOT have dependency C1 -> V2 (cross-type)";
    EXPECT_FALSE(checkDependency(v5CallOp, c4CallOp)) << "Should NOT have dependency C4 -> V5 (cross-type)";
    EXPECT_FALSE(checkDependency(c3CallOp, v2CallOp)) << "Should NOT have dependency V2 -> C3 (cross-type)";
    EXPECT_FALSE(checkDependency(c3CallOp, v5CallOp)) << "Should NOT have dependency V5 -> C3 (cross-type)";
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