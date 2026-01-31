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
 * \file test_mix_call_operation_builder.cpp
 * \brief Unit test for MixCallOperationBuilder
 * */
#include <gtest/gtest.h>
#include "passes/block_graph_pass/mix_subgraph_split/mix_call_operation_builder.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

constexpr int MS_NUM1 = 1;
constexpr int MS_NUM2 = 2;
constexpr int MS_NUM4 = 4;
constexpr int MS_NUM10 = 10;
constexpr int MS_NUM16 = 16;
constexpr uint64_t TEST_PROGRAM_ID = 100;
constexpr int32_t OP_MAGIC_BASE = 10000;
constexpr int TENSOR_DIMENSIONS = 2;
constexpr int TENSOR_ARGS_COUNT = 9;

constexpr int OFFSET_INPUT1 = 100;
constexpr int OFFSET_INPUT2 = 101;
constexpr int OFFSET_ADD_INPUT1 = 200;
constexpr int OFFSET_ADD_INPUT2 = 201;
constexpr int OFFSET_ADD_OUTPUT = 300;
constexpr int OFFSET_OUTPUT = 400;

constexpr int COMPONENT_ID_0 = 0;
constexpr int COMPONENT_ID_1 = 1;

constexpr int TENSOR_INDEX_0 = 0;
constexpr int TENSOR_INDEX_1 = 1;
constexpr int TENSOR_INDEX_2 = 2;

constexpr int DEFAULT_NUM_ORIGINAL_CALLOPS = 2;
constexpr int DEFAULT_NUM_COMPONENTS = 2;
constexpr int DEFAULT_NUM_TENSORS = 3;

class MixCallOperationBuilderTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig("KEY_ENABLE_COST_MODEL", false);
        
        // 创建root function
        rootFunc = std::make_shared<Function>(
            Program::GetInstance(), "test_root", "test_root", nullptr);
        rootFunc->rootFunc_ = rootFunc.get();
        
        builder = std::make_unique<MixCallOperationBuilder>();
    }

    void TearDown() override
    {
        builder.reset();
    }

protected:
    // 测试场景结构体
    struct TestScenario {
        std::vector<int64_t> shape;
        std::shared_ptr<LogicalTensor> inputTensor1;
        std::shared_ptr<LogicalTensor> inputTensor2;
        std::shared_ptr<LogicalTensor> outputTensor;
        std::shared_ptr<Function> originalMixFunc;
        Operation* originalCallOp = nullptr;
        std::shared_ptr<CallOpAttribute> originalCallAttr;
    };

    // 传播依赖张量结构体
    struct PropagatedTensors {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
    };
    
    // 创建简单的Operation
    Operation* createSimpleCallOp(Function& func)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto input1 = std::make_shared<LogicalTensor>(func, DT_FP32, shape);
        auto input2 = std::make_shared<LogicalTensor>(func, DT_FP32, shape);
        auto output1 = std::make_shared<LogicalTensor>(func, DT_FP32, shape);
        
        auto& callOp = func.AddRawOperation(
            Opcode::OP_CALL,
            {input1, input2},
            {output1});
        
        auto callAttr = std::make_shared<CallOpAttribute>();
        callOp.SetOpAttribute(callAttr);
        
        return &callOp;
    }
    
    // 创建简单的Function
    std::shared_ptr<Function> createSimpleFunction(const std::string& name)
    {
        auto func = std::make_shared<Function>(
            Program::GetInstance(), name, name, rootFunc.get());
        func->SetGraphType(GraphType::BLOCK_GRAPH);
        func->SetFunctionType(FunctionType::STATIC);
        return func;
    }
    
    // 创建同时包含C_SCOPE和V_SCOPE的components
    std::vector<InternalComponentInfo> createMixedComponents()
    {
        std::vector<InternalComponentInfo> components;
        components.emplace_back(COMPONENT_ID_0, "comp_c_scope", AIVCore::AIV0, ComponentType::C_SCOPE);
        components.emplace_back(COMPONENT_ID_1, "comp_v_scope", AIVCore::AIV1, ComponentType::V_SCOPE);
        return components;
    }

    // 创建SubfuncInvokeInfoTy
    SubfuncInvokeInfoTy createInvokeInfoWithTensorParams(
        uint64_t programId,
        const std::shared_ptr<LogicalTensor>& input1,
        const std::shared_ptr<LogicalTensor>& input2,
        const std::shared_ptr<LogicalTensor>& output)
    {
        SubfuncInvokeInfoTy invokeInfo;
        invokeInfo.UpdateProgramSubgraphId(programId);
        
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        const std::vector<int64_t> offset = {0, 0};
        const std::vector<int64_t> rawShape = {MS_NUM16, MS_NUM16};
        
        // 记录连接信息
        invokeInfo.RecordConnection(-1, programId, TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
            input1, OP_MAGIC_BASE + MS_NUM1);
        invokeInfo.RecordConnection(-1, programId, TENSOR_INDEX_1, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
            input2, OP_MAGIC_BASE + MS_NUM2);
        
        SubfuncInvokeInfoTy::SuccessorIncastInfoTy emptySuccessorInfo;
        invokeInfo.RecordOutcast(programId, TENSOR_INDEX_0, TENSOR_INDEX_1, TENSOR_INDEX_0, emptySuccessorInfo, offset,
            shape, rawShape, DT_FP32, output, OP_MAGIC_BASE + MS_NUM4);
        
        // 记录张量参数
        invokeInfo.RecordTensorArg(TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
            false, input1, OP_MAGIC_BASE + MS_NUM1);
        invokeInfo.RecordTensorArg(TENSOR_INDEX_1, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
            false, input2, OP_MAGIC_BASE + MS_NUM2);
        invokeInfo.RecordTensorArg(TENSOR_INDEX_0, TENSOR_INDEX_0, offset, shape, rawShape, DT_FP32,
            true, output, OP_MAGIC_BASE + MS_NUM4);
        
        invokeInfo.DoFinishRecord();
        
        return invokeInfo;
    }

    // 创建SubgraphToFunction，确保与components数量匹配
    SubgraphToFunction createSubgraphToFunctionForComponents(
        size_t componentCount,
        uint64_t baseProgramId = TEST_PROGRAM_ID)
    {
        SubgraphToFunction subgraphToFunction;
        for (size_t i = 0; i < componentCount; ++i) {
            auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
            invokeInfo->UpdateProgramSubgraphId(baseProgramId + i);
            subgraphToFunction.subFuncInvokeInfos.push_back(*invokeInfo);
        }
        return subgraphToFunction;
    }

    // 创建基本测试场景
    TestScenario createBasicTestScenario()
    {
        TestScenario scenario;
        
        scenario.shape = {MS_NUM16, MS_NUM16};
        scenario.inputTensor1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, scenario.shape);
        scenario.inputTensor2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, scenario.shape);
        scenario.outputTensor = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, scenario.shape);
        
        return scenario;
    }
    
    // 构建OriginalMixFunc和CallOp
    void buildOriginalMixFuncAndCallOp(
        const std::string& funcName,
        TestScenario& scenario,
        uint64_t programId = TEST_PROGRAM_ID)
    {
        scenario.originalMixFunc = createFunctionWithRealOffsetOps(
            funcName, COMPONENT_ID_0, scenario.inputTensor1, scenario.inputTensor2, scenario.outputTensor);
        
        scenario.originalCallOp = createCallOpWithArgList(
            programId, scenario.inputTensor1, scenario.inputTensor2, scenario.outputTensor);
        scenario.originalCallAttr = std::dynamic_pointer_cast<CallOpAttribute>(
            scenario.originalCallOp->GetOpAttribute());
    }
    
    // 验证wrapId设置
    void verifyWrapIdSet(CallOpAttribute* callAttr)
    {
        EXPECT_NE(callAttr->wrapId, static_cast<uint64_t>(-1))
            << "wrapId should be set (not -1)";
    }
    
    // 验证CallOp的输入输出数量
    void verifyCallOpIOCount(Operation* callOp, size_t expectedInputs, size_t expectedOutputs)
    {
        auto iOperands = callOp->GetIOperands();
        auto oOperands = callOp->GetOOperands();
        EXPECT_EQ(iOperands.size(), expectedInputs) << "Input count should match";
        EXPECT_EQ(oOperands.size(), expectedOutputs) << "Output count should match";
    }

    // 创建传播依赖张量
    PropagatedTensors createPropagatedTensors()
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        PropagatedTensors tensors;
        tensors.input = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
        tensors.output = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
        return tensors;
    }
    
    // 为函数添加传播依赖的 incast/outcast
    void addPropagatedIncastOutcast(
        const std::shared_ptr<Function>& func,
        const std::shared_ptr<LogicalTensor>& propagatedInput,
        const std::shared_ptr<LogicalTensor>& propagatedOutput)
    {
        func->inCasts_.push_back(propagatedInput);
        func->outCasts_.push_back(propagatedOutput);
    }
    
    // 创建带有传播依赖的叶子函数
    std::shared_ptr<Function> createLeafFuncWithPropagatedTensors(
        const std::string& name,
        int componentId,
        const TestScenario& scenario,
        const PropagatedTensors& propagatedTensors)
    {
        auto leafFunc = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        
        // 添加常规 incasts
        leafFunc->inCasts_.push_back(scenario.inputTensor1);
        leafFunc->inCasts_.push_back(scenario.inputTensor2);
        
        // 添加常规 outcasts
        leafFunc->outCasts_.push_back(scenario.outputTensor);
        
        // 添加传播依赖的 incast/outcast
        addPropagatedIncastOutcast(leafFunc, propagatedTensors.input, propagatedTensors.output);
        
        // 为常规张量创建操作
        createOperationsWithOffsets(leafFunc, componentId, scenario.inputTensor1,
                                    scenario.inputTensor2, scenario.outputTensor, shape);
        
        // 为传播依赖张量创建操作
        addPropagatedTensorOperations(leafFunc, componentId,
                                      propagatedTensors.input, propagatedTensors.output);
        
        return leafFunc;
    }
    
    // 创建包含传播依赖的 OriginalCallOp
    Operation* createOriginalCallOpWithPropagatedTensors(
        const TestScenario& scenario,
        const PropagatedTensors& propagatedTensors)
    {
        // 创建包含所有输入输出的 CallOp
        auto& originalCallOp = rootFunc->AddRawOperation(
            Opcode::OP_CALL,
            {scenario.inputTensor1, scenario.inputTensor2, propagatedTensors.input},
            {scenario.outputTensor, propagatedTensors.output});
        
        auto originalCallAttr = std::make_shared<CallOpAttribute>();
        originalCallOp.SetOpAttribute(originalCallAttr);
        
        return &originalCallOp;
    }
    
    // 验证包含传播依赖的 CallOp
    void verifyCallOpWithPropagatedDependencies(Operation* callOp, Operation* originalCallOp)
    {
        if (callOp == originalCallOp) {
            return;  // 跳过原始 CallOp
        }
        
        // 验证输入输出数量：3个输入（2个常规+1个传播），2个输出（1个常规+1个传播）
        auto iOperands = callOp->GetIOperands();
        auto oOperands = callOp->GetOOperands();
        
        EXPECT_EQ(iOperands.size(), 3U)
            << "New call op should have 3 inputs (2 regular + 1 propagated)";
        EXPECT_EQ(oOperands.size(), 2U)
            << "New call op should have 2 outputs (1 regular + 1 propagated)";
        
        // 验证 wrapId 设置
        auto newCallAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
        ASSERT_NE(newCallAttr, nullptr) << "CallOp should have CallOpAttribute";
        EXPECT_NE(newCallAttr->wrapId, static_cast<uint64_t>(-1))
            << "wrapId should be set (not -1)";
    }

    // 创建叶子函数和对应的Function指针向量
    void createLeafFunctionsAndPointers(
        int count,
        const std::string& baseName,
        std::vector<std::shared_ptr<Function>>& leafFuncs,
        std::vector<Function*>& newFunctions)
    {
        leafFuncs.clear();
        newFunctions.clear();
        
        for (int i = 0; i < count; ++i) {
            auto leafFunc = createSimpleFunction(baseName + std::to_string(i));
            leafFuncs.push_back(leafFunc);
            newFunctions.push_back(leafFunc.get());
        }
    }
    
    // 创建程序ID向量
    std::vector<uint64_t> createProgramIds(size_t count, uint64_t baseProgramId = TEST_PROGRAM_ID)
    {
        std::vector<uint64_t> programIds;
        for (size_t i = 0; i < count; ++i) {
            programIds.push_back(baseProgramId + i);
        }
        return programIds;
    }
    
    // 创建组件和对应的子图函数信息
    void createComponentsAndSubgraphInfo(
        const std::vector<ComponentType>& componentTypes,
        const TestScenario& scenario,
        std::vector<InternalComponentInfo>& components,
        std::vector<std::shared_ptr<Function>>& leafFuncs,
        std::vector<Function*>& newFunctions,
        SubgraphToFunction& subgraphToFunction,
        std::vector<uint64_t>& newProgramIDs)
    {
        components.clear();
        leafFuncs.clear();
        newFunctions.clear();
        subgraphToFunction.subFuncInvokeInfos.clear();
        newProgramIDs.clear();
        
        for (size_t i = 0; i < componentTypes.size(); ++i) {
            int componentId = static_cast<int>(i);
            components.emplace_back(componentId, "comp_" + std::to_string(i), AIVCore::UNSPECIFIED, componentTypes[i]);
            
            auto leafFunc = createFunctionWithRealOffsetOps(
                "leaf_" + std::to_string(i), componentId,
                scenario.inputTensor1, scenario.inputTensor2, scenario.outputTensor);
            leafFuncs.push_back(leafFunc);
            newFunctions.push_back(leafFunc.get());
            
            auto invokeInfo = createInvokeInfoWithTensorParams(
                TEST_PROGRAM_ID + i, scenario.inputTensor1, scenario.inputTensor2, scenario.outputTensor);
            subgraphToFunction.subFuncInvokeInfos.push_back(invokeInfo);
            
            newProgramIDs.push_back(TEST_PROGRAM_ID + i);
        }
    }

protected:
    std::shared_ptr<Function> rootFunc;
    std::unique_ptr<MixCallOperationBuilder> builder;

private:
    // 创建带有偏移量的操作
    std::shared_ptr<Function> createOperationsWithOffsets(
        const std::shared_ptr<Function>& func,
        int internalSubgraphId,
        const std::shared_ptr<LogicalTensor>& input1,
        const std::shared_ptr<LogicalTensor>& input2,
        const std::shared_ptr<LogicalTensor>& output,
        const std::vector<int64_t>& shape)
    {
        auto internal1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        auto internal2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        auto internal3 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        
        createCopyInOperation(func, internalSubgraphId, input1, internal1, OFFSET_INPUT1);
        createCopyInOperation(func, internalSubgraphId, input2, internal2, OFFSET_INPUT2);
        createAddOperation(func, internalSubgraphId, internal1, internal2, internal3);
        createCopyOutOperation(func, internalSubgraphId, internal3, output, OFFSET_OUTPUT);
        
        return func;
    }
    
    // 创建CopyIn操作
    void createCopyInOperation(const std::shared_ptr<Function>& func, int internalSubgraphId,
        const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output, int offset)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;
        
        auto& copyIn = func->AddRawOperation(Opcode::OP_COPY_IN, {input}, {output});
        copyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
        copyIn.SetIOpAttrOffset(TENSOR_INDEX_0, offset);
        copyIn.UpdateInternalSubgraphID(internalSubgraphId);
        copyIn.SetAttr(OpAttributeKey::isCube, true);
    }
    
    // 创建Add操作
    void createAddOperation(
        const std::shared_ptr<Function>& func,
        int internalSubgraphId,
        const std::shared_ptr<LogicalTensor>& input1,
        const std::shared_ptr<LogicalTensor>& input2,
        const std::shared_ptr<LogicalTensor>& output)
    {
        auto& addOp = func->AddRawOperation(Opcode::OP_ADD, {input1, input2}, {output});
        addOp.SetIOpAttrOffset(TENSOR_INDEX_0, OFFSET_ADD_INPUT1);
        addOp.SetIOpAttrOffset(TENSOR_INDEX_1, OFFSET_ADD_INPUT2);
        addOp.SetOOpAttrOffset(TENSOR_INDEX_0, OFFSET_ADD_OUTPUT);
        addOp.UpdateInternalSubgraphID(internalSubgraphId);
        addOp.SetAttr(OpAttributeKey::isCube, true);
    }
    
    // 创建CopyOut操作
    void createCopyOutOperation(const std::shared_ptr<Function>& func, int internalSubgraphId,
        const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output, int offset)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;
        
        auto& copyOut = func->AddRawOperation(Opcode::OP_COPY_OUT, {input}, {output});
        copyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
        copyOut.SetOOpAttrOffset(TENSOR_INDEX_0, offset);
        copyOut.UpdateInternalSubgraphID(internalSubgraphId);
        copyOut.SetAttr(OpAttributeKey::isCube, true);
    }
    
    // 为2维张量创建参数
    std::vector<SymbolicScalar> createTensorArgsFor2D(int tensorIndex)
    {
        std::vector<SymbolicScalar> tensorArgs;
        tensorArgs.push_back(SymbolicScalar(tensorIndex));
        
        // offset, shape, rawshape, validshape (每个2个维度)
        for (int i = 0; i < MS_NUM4; ++i) {
            for (int d = 0; d < TENSOR_DIMENSIONS; ++d) {
                tensorArgs.push_back(SymbolicScalar(i == 0 ? 0 : MS_NUM16));
            }
        }
        return tensorArgs;
    }
    
    // 创建带argList的CallOp
    Operation* createCallOpWithArgList(
        uint64_t programId,
        const std::shared_ptr<LogicalTensor>& input1,
        const std::shared_ptr<LogicalTensor>& input2,
        const std::shared_ptr<LogicalTensor>& output)
    {
        auto& callOp = rootFunc->AddRawOperation(
            Opcode::OP_CALL,
            {input1, input2},
            {output});
        
        std::vector<std::vector<SymbolicScalar>> argList;
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_0));
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_1));
        argList.push_back(createTensorArgsFor2D(TENSOR_INDEX_2));
        
        auto callAttr = std::make_shared<CallOpAttribute>(
            FunctionHash(programId),
            argList,
            "",  // calleMagicName
            std::map<int, SymbolicScalar>(),  // outIndexToExpr
            std::vector<SymbolicScalar>()     // linearArgList
        );
        
        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programId);
        callAttr->invokeInfo_ = invokeInfo;
        
        callOp.SetOpAttribute(callAttr);
        callOp.UpdateSubgraphID(programId);
        
        return &callOp;
    }

    // 创建带有实际offset的Function
    std::shared_ptr<Function> createFunctionWithRealOffsetOps(
        const std::string& name,
        int internalSubgraphId,
        const std::shared_ptr<LogicalTensor>& input1,
        const std::shared_ptr<LogicalTensor>& input2,
        const std::shared_ptr<LogicalTensor>& output)
    {
        auto func = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        
        func->inCasts_.push_back(input1);
        func->inCasts_.push_back(input2);
        func->outCasts_.push_back(output);
        
        return createOperationsWithOffsets(func, internalSubgraphId, input1, input2, output, shape);
    }

    // 为函数添加传播依赖张量的操作
    void addPropagatedTensorOperations(
        const std::shared_ptr<Function>& func,
        int componentId,
        const std::shared_ptr<LogicalTensor>& propagatedInput,
        const std::shared_ptr<LogicalTensor>& propagatedOutput)
    {
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        
        // 为传播依赖的输入张量创建 CopyIn 操作
        auto internalPropagatedInput = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        createCopyInOperation(func, componentId, propagatedInput, internalPropagatedInput,
            OFFSET_INPUT1 + MS_NUM10);  // 不同的偏移量
        
        // 为传播依赖的输出张量创建 CopyOut 操作
        auto internalPropagatedOutput = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
        createCopyOutOperation(func, componentId, internalPropagatedOutput, propagatedOutput,
            OFFSET_OUTPUT + MS_NUM10);  // 不同的偏移量
    }
    
    // 创建带有传播依赖的原始混合函数
    std::shared_ptr<Function> createOriginalMixFuncWithPropagatedTensors(
        const std::string& name,
        const TestScenario& scenario,
        const PropagatedTensors& propagatedTensors)
    {
        auto originalMixFunc = createSimpleFunction(name);
        const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
        
        // 添加常规 incasts
        originalMixFunc->inCasts_.push_back(scenario.inputTensor1);
        originalMixFunc->inCasts_.push_back(scenario.inputTensor2);
        
        // 添加常规 outcasts
        originalMixFunc->outCasts_.push_back(scenario.outputTensor);
        
        // 添加传播依赖的 incast/outcast
        addPropagatedIncastOutcast(originalMixFunc, propagatedTensors.input, propagatedTensors.output);
        
        // 为常规张量创建操作
        createOperationsWithOffsets(originalMixFunc, COMPONENT_ID_0, scenario.inputTensor1,
                                    scenario.inputTensor2, scenario.outputTensor, shape);
        
        // 为传播依赖张量创建操作
        addPropagatedTensorOperations(originalMixFunc, COMPONENT_ID_0,
                                      propagatedTensors.input, propagatedTensors.output);
        
        return originalMixFunc;
    }

    void verifyCallOpOffsets(Operation* callOp,
                            const std::vector<int>& expectedInputOffsets,
                            const std::vector<int>& expectedOutputOffsets)
    {
        ASSERT_NE(callOp, nullptr) << "CallOp should not be null";
        
        // 验证输入偏移
        if (!expectedInputOffsets.empty()) {
            for (size_t i = 0; i < expectedInputOffsets.size(); ++i) {
                int actualOffset = callOp->GetIOpAttrOffset(i);
                EXPECT_EQ(actualOffset, expectedInputOffsets[i])
                    << "Input offset at index " << i << " should match expected value";
            }
        }
        
        // 验证输出偏移
        if (!expectedOutputOffsets.empty()) {
            for (size_t i = 0; i < expectedOutputOffsets.size(); ++i) {
                int actualOffset = callOp->GetOOpAttrOffset(i);
                EXPECT_EQ(actualOffset, expectedOutputOffsets[i])
                    << "Output offset at index " << i << " should match expected value";
            }
        }
    }
};

// 测试同一个originalCallOp对应的不同组件有相同wrapId
TEST_F(MixCallOperationBuilderTest, TestSameWrapIdForSameOriginalCallOp)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    
    // 创建一个originalCallOp
    auto originalCallOp = createSimpleCallOp(*rootFunc);
    
    auto components = createMixedComponents();  // 2个组件
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());
    
    // 使用辅助函数创建叶子函数和指针
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctions);
    
    // 使用辅助函数创建程序ID
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());
    
    std::vector<InternalDependencyInfo> emptyDeps;
    
    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
        newProgramIDs, subgraphToFunction, newFunctions, emptyDeps);
    
    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed";
    
    // 获取所有新创建的CallOp
    auto newCallOps = rootFunc->GetCallopList();
    Operation* newCallOp1 = nullptr;
    Operation* newCallOp2 = nullptr;
    
    for (auto* op : newCallOps) {
        if (op != originalCallOp) {
            if (newCallOp1 == nullptr) {
                newCallOp1 = op;
            } else {
                newCallOp2 = op;
                break;
            }
        }
    }
    
    ASSERT_NE(newCallOp1, nullptr) << "Should have created first new CallOp";
    ASSERT_NE(newCallOp2, nullptr) << "Should have created second new CallOp";
    
    auto callAttr1 = dynamic_cast<CallOpAttribute*>(newCallOp1->GetOpAttribute().get());
    auto callAttr2 = dynamic_cast<CallOpAttribute*>(newCallOp2->GetOpAttribute().get());
    
    ASSERT_NE(callAttr1, nullptr) << "CallOp1 should have CallOpAttribute";
    ASSERT_NE(callAttr2, nullptr) << "CallOp2 should have CallOpAttribute";
    
    // 验证同一个originalCallOp对应的不同组件有相同的wrapId
    EXPECT_EQ(callAttr1->wrapId, callAttr2->wrapId)
        << "Same originalCallOp should have same wrapId for different components";
}

// 测试不同的originalCallOp有不同的wrapId
TEST_F(MixCallOperationBuilderTest, TestDifferentWrapIdForDifferentOriginalCallOps)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    
    // 创建两个不同的originalCallOps
    auto originalCallOp1 = createSimpleCallOp(*rootFunc);
    auto originalCallOp2 = createSimpleCallOp(*rootFunc);
    
    auto components = createMixedComponents();  // 2个组件
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());
    
    // 使用辅助函数创建叶子函数和指针
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctions);
    
    // 使用辅助函数创建程序ID
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());
    
    std::vector<InternalDependencyInfo> emptyDeps;
    
    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp1, originalCallOp2}, originalMixFunc.get(),
        components, newProgramIDs, subgraphToFunction, newFunctions, emptyDeps);
    
    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed";
    
    // 获取所有新创建的CallOp
    auto newCallOps = rootFunc->GetCallopList();
    
    // 收集所有wrapId
    std::set<uint64_t> wrapIds;
    for (auto* op : newCallOps) {
        if (op != originalCallOp1 && op != originalCallOp2) {
            auto callAttr = dynamic_cast<CallOpAttribute*>(op->GetOpAttribute().get());
            if (callAttr) {
                wrapIds.insert(callAttr->wrapId);
            }
        }
    }
    
    // 应该有4个新CallOp (2个originalCallOps × 2个组件)
    // 每个originalCallOp的wrapId应该不同
    EXPECT_GE(wrapIds.size(), 2U) << "Different originalCallOps should have different wrapIds";
}

// 测试Global Tensor的处理
TEST_F(MixCallOperationBuilderTest, TestGlobalTensorHandling)
{
    // 创建测试场景
    auto scenario = createBasicTestScenario();
    buildOriginalMixFuncAndCallOp("original_mix", scenario);
    
    // 创建多个组件
    std::vector<InternalComponentInfo> components = {
        {COMPONENT_ID_0, "comp_cube", AIVCore::UNSPECIFIED, ComponentType::C_SCOPE},
        {COMPONENT_ID_1, "comp_vector", AIVCore::AIV0, ComponentType::V_SCOPE}
    };
    
    // 创建叶子函数
    auto leafFuncCube = createFunctionWithRealOffsetOps("leaf_cube", COMPONENT_ID_0, scenario.inputTensor1,
        scenario.inputTensor2, scenario.outputTensor);
    auto leafFuncVector = createFunctionWithRealOffsetOps("leaf_vector", COMPONENT_ID_1, scenario.inputTensor1,
        scenario.inputTensor2, scenario.outputTensor);
    std::vector<Function*> newFunctions = {leafFuncCube.get(), leafFuncVector.get()};
    
    // 创建完整的invokeInfo，包含global tensor信息
    SubgraphToFunction subgraphToFunction;
    for (int i = 0; i < MS_NUM2; ++i) {
        auto leafInvokeInfo = createInvokeInfoWithTensorParams(TEST_PROGRAM_ID + i, scenario.inputTensor1,
            scenario.inputTensor2, scenario.outputTensor);
        subgraphToFunction.subFuncInvokeInfos.push_back(leafInvokeInfo);
    }
    
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());
    std::vector<InternalDependencyInfo> emptyDeps;
    
    Status status = builder->CreateCallOps(*rootFunc, {scenario.originalCallOp}, scenario.originalMixFunc.get(),
        components, newProgramIDs, subgraphToFunction, newFunctions, emptyDeps);
    
    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed with global tensors";
    
    // 验证新创建的CallOps
    auto newCallOps = rootFunc->GetCallopList();
    EXPECT_GE(newCallOps.size(), 2U) << "Should have created at least 2 new CallOps";
    
    for (auto* callOp : newCallOps) {
        if (callOp != scenario.originalCallOp) {
            // 验证输入输出数量
            verifyCallOpIOCount(callOp, 2U, 1U);  // 2个输入，1个输出
            
            auto callAttr = dynamic_cast<CallOpAttribute*>(callOp->GetOpAttribute().get());
            ASSERT_NE(callAttr, nullptr) << "CallOp should have CallOpAttribute";
            
            // 验证argList
            const auto& argList = callAttr->GetArgList();
            EXPECT_FALSE(argList.empty()) << "argList should not be empty";
            
            // 验证wrapId
            verifyWrapIdSet(callAttr);
            
            // 验证programId
            if (callAttr->invokeInfo_) {
                uint64_t progId = callAttr->invokeInfo_->GetProgramId();
                EXPECT_TRUE(progId == TEST_PROGRAM_ID || progId == TEST_PROGRAM_ID + 1)
                    << "programId should be in expected range";
            }
        }
    }
}

// 测试带内部依赖的情况
TEST_F(MixCallOperationBuilderTest, TestInternalDependencies)
{
    auto originalMixFunc = createSimpleFunction("original_mix");
    auto originalCallOp = createSimpleCallOp(*rootFunc);
    
    auto components = createMixedComponents();
    auto subgraphToFunction = createSubgraphToFunctionForComponents(components.size());
    
    // 使用辅助函数创建叶子函数和指针
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    createLeafFunctionsAndPointers(MS_NUM2, "leaf", leafFuncs, newFunctions);
    
    std::vector<InternalDependencyInfo> internalDeps = {
        {COMPONENT_ID_0, COMPONENT_ID_1, ComponentType::C_SCOPE}
    };
    
    // 使用辅助函数创建程序ID
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());
    
    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp}, originalMixFunc.get(), components,
        newProgramIDs, subgraphToFunction, newFunctions, internalDeps);
    
    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed with internal dependencies";
}

TEST_F(MixCallOperationBuilderTest, TestOffsets)
{
    auto scenario = createBasicTestScenario();
    buildOriginalMixFuncAndCallOp("test_component_types", scenario);
    
    // 只使用存在的ComponentType枚举值
    const std::vector<ComponentType> componentTypes = {
        ComponentType::C_SCOPE,
        ComponentType::V_SCOPE
    };
    
    // 使用辅助函数创建组件和子图信息
    std::vector<InternalComponentInfo> components;
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    SubgraphToFunction subgraphToFunction;
    std::vector<uint64_t> newProgramIDs;
    
    createComponentsAndSubgraphInfo(componentTypes, scenario, components, leafFuncs, newFunctions, subgraphToFunction,
        newProgramIDs);
    
    std::vector<InternalDependencyInfo> emptyDeps;
    
    Status status = builder->CreateCallOps(*rootFunc, {scenario.originalCallOp},
        scenario.originalMixFunc.get(), components, newProgramIDs,
        subgraphToFunction, newFunctions, emptyDeps);
    
    EXPECT_EQ(status, SUCCESS) << "CreateCallOps should succeed for different component types";
    
    auto newCallOps = rootFunc->GetCallopList();
    size_t newCallOpCount = 0;
    
    for (auto* callOp : newCallOps) {
        if (callOp != scenario.originalCallOp) {
            newCallOpCount++;
            verifyCallOpOffsets(callOp, {OFFSET_INPUT1, OFFSET_INPUT2}, {OFFSET_OUTPUT});
        }
    }
    
    EXPECT_EQ(newCallOpCount, componentTypes.size())
        << "Should create one CallOp per component";
}

// 重构后的测试传播依赖的Incast/Outcast处理
TEST_F(MixCallOperationBuilderTest, TestPropagatedIncastOutcast)
{
    // 1. 创建基本测试场景
    auto scenario = createBasicTestScenario();
    
    // 2. 创建传播依赖的张量
    auto propagatedTensors = createPropagatedTensors();
    
    // 3. 创建带有传播依赖的原始混合函数
    auto originalMixFunc = createOriginalMixFuncWithPropagatedTensors(
        "original_mix", scenario, propagatedTensors);
    
    // 4. 创建包含传播依赖的 OriginalCallOp
    auto originalCallOp = createOriginalCallOpWithPropagatedTensors(scenario, propagatedTensors);
    
    // 5. 创建两个组件
    std::vector<InternalComponentInfo> components = {
        {COMPONENT_ID_0, "comp_c_scope", AIVCore::UNSPECIFIED, ComponentType::C_SCOPE},
        {COMPONENT_ID_1, "comp_v_scope", AIVCore::AIV0, ComponentType::V_SCOPE}
    };
    
    // 6. 创建带有传播依赖的叶子函数
    std::vector<std::shared_ptr<Function>> leafFuncs;
    std::vector<Function*> newFunctions;
    
    for (size_t i = 0; i < components.size(); ++i) {
        std::string leafName = "leaf" + std::to_string(i);
        auto leafFunc = createLeafFuncWithPropagatedTensors(
            leafName, static_cast<int>(i), scenario, propagatedTensors);
        leafFuncs.push_back(leafFunc);
        newFunctions.push_back(leafFunc.get());
    }
    
    // 7. 创建 subgraphToFunction
    SubgraphToFunction subgraphToFunction;
    for (size_t i = 0; i < components.size(); ++i) {
        auto invokeInfo = createInvokeInfoWithTensorParams(
            TEST_PROGRAM_ID + i, scenario.inputTensor1, scenario.inputTensor2, scenario.outputTensor);
        subgraphToFunction.subFuncInvokeInfos.push_back(invokeInfo);
    }
    
    // 8. 准备调用参数
    std::vector<uint64_t> newProgramIDs = createProgramIds(components.size());
    
    std::vector<InternalDependencyInfo> emptyDeps;
    
    // 9. 调用 CreateCallOps
    Status status = builder->CreateCallOps(*rootFunc, {originalCallOp},
        originalMixFunc.get(), components, newProgramIDs, subgraphToFunction,
        newFunctions, emptyDeps);
    
    // 10. 验证结果
    EXPECT_EQ(status, SUCCESS)
        << "CreateCallOps should succeed with propagated incast/outcast for multiple components";
    
    // 11. 验证新创建的 CallOps
    auto newCallOps = rootFunc->GetCallopList();
    
    size_t newCallOpCount = 0;
    for (auto* callOpPtr : newCallOps) {
        verifyCallOpWithPropagatedDependencies(callOpPtr, originalCallOp);
        if (callOpPtr != originalCallOp) {
            newCallOpCount++;
        }
    }
    
    EXPECT_EQ(newCallOpCount, components.size())
        << "Should create one CallOp per component";
}

} // namespace tile_fwk
} // namespace npu