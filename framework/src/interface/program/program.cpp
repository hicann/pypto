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
 * \file program.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "interface/utils/log.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/serialization.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/function/function.h"
#include "interface/interpreter/flow_verifier.h"
#include "interface/machine/host/host_machine.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk {
const std::string PROGRAM_ENTRY_FUNCTION_NAME = "PROGRAM_ENTRY";
void GetEnv(const char * const envName, std::string &envValue)
{
    const size_t envValueMaxLen = 1024UL * 1024UL;
    const char * const envTemp = std::getenv(envName);
    if ((envTemp == nullptr) || (strnlen(envTemp, envValueMaxLen) >= envValueMaxLen)) {
        ALOG_INFO_F("Env[%s] not found.\n", envName);
        return;
    }
    envValue = envTemp;
}

// Program Definitions
Program::Program() : currentFunctionPtr_(nullptr) {
    CreateInitFunction();

    HostMachine::GetInstance().Init(HostMachineMode::SERVER);
    std::string envLogLevel;
    GetEnv("GLOBAL_LOG_LEVEL", envLogLevel);
    if (envLogLevel.empty()) {
        return;
    }
    int32_t logLevel = 0;
    try {
        logLevel = std::stoi(envLogLevel);
    } catch (...) {
        return;
    }
    if (logLevel < 0 || logLevel > static_cast<int32_t>(LoggerLevel::NONE)) {
        printf("Log level %d is not valid.\n", logLevel);
        return;
    }
    LoggerManager::GetManager().ResetLevel(static_cast<LoggerLevel>(logLevel));
    printf("Set global log level as %d\n", logLevel);
}

Program::~Program() {
    HostMachine::GetInstance().Destroy();
}

Program &Program::GetInstance() {
    static Program sProgram;
    return sProgram;
}

void Program::Reset() {
    name_.clear();
    functionmap_.clear();
    functionMagicNameStack_.clear();
    currentFunctionMagicName_ = PROGRAM_ENTRY_FUNCTION_NAME;
    config::Reset();
    IdGen<IdType::LOGICAL_TENSOR>::Inst().Reset();
    aliveTensors_.clear();
    functionCache_.Reset();
    functionSequence_.clear();
    CreateInitFunction();
    tensorSlotManager_ = nullptr;
    currentFunctionPtr_ = functionmap_[currentFunctionMagicName_].get();
}

Function *Program::GetFunctionByRawName(const std::string &rawName) const {
    for (auto &ele : functionmap_) {
        if (ele.second->GetRawName() == rawName) {
            return ele.second.get();
        }
    }
    return nullptr;
}

void Program::SetCurrentFunction(Function *function) {
    if (function == nullptr) {
        return;
    }
    currentFunctionPtr_ = function;
    currentFunctionMagicName_ = function->GetMagicName();
}

void Program::CreateInitFunction() {
    currentFunctionMagicName_ = PROGRAM_ENTRY_FUNCTION_NAME;
    auto newFunc =
        std::make_shared<Function>(*this, currentFunctionMagicName_, currentFunctionMagicName_, nullptr);
    newFunc->SetFunctionType(FunctionType::EAGER);
    currentFunctionPtr_ = newFunc.get();
    functionmap_.emplace(currentFunctionMagicName_, std::move(newFunc));
}

void Program::CreateCallerCalleeLink(Function *caller, Function *callee) {
    ASSERT(caller->IsGraphType(GraphType::TENSOR_GRAPH) && callee->IsGraphType(GraphType::TENSOR_GRAPH));
    // add callop
    for (auto &outcast : callee->outCasts_) {
        auto newOutcast = outcast->Clone(*caller, true);
        caller->outCasts_.push_back(newOutcast);
        caller->GetTensorMap().Insert(newOutcast);
    }
    for (auto &incast : callee->inCasts_) {
        auto newIncast = incast->Clone(*caller, true);
        caller->inCasts_.push_back(newIncast);
        caller->GetTensorMap().Insert(newIncast);
    }

    FunctionCallArgs args = {
        .iOperands = caller->inCasts_,
        .oOperands = caller->outCasts_,
        .iOpAttrOffset = {},
        .oOpAttrOffset = {},
        .outIndexToExpr = {},
        .argList = {},
    };
    currentFunctionPtr_ = callee;
    ConnectCallerGusket(*caller, args);

    caller->ComputeHash();
    auto cacheValue = TryHitCahce(caller->GetFunctionHash());
    if (cacheValue == std::nullopt) {
        functionCache_.Insert(caller->GetFunctionHash(), *caller);
        caller->AppendCalleeMagicName(callee->GetMagicName());
    }
}

void Program::RefillCompileQueue(Function* func) {
    functionSequence_.emplace_back(func);
}

void Program::UpdateCompileTask() {
    for (auto func : functionSequence_) {
        HostMachine::GetInstance().StashTask(func);
    }
    HostMachine::GetInstance().SubAllStashedTask();
}

void SetParamConfig(Function* currentFunctionPtr_) {
    std::shared_ptr<ConfigScope> currentScope = ConfigManagerNg::GetInstance().CurrentScope();
    currentFunctionPtr_->paramConfigs_.l1ReuseNum = currentScope->GetPassConfig<int>(CUBE_L1_REUSE_MODE);
    currentFunctionPtr_->paramConfigs_.cubeNBufferMode = currentScope->GetPassConfig<int>(CUBE_NBUFFER_MODE);
    currentFunctionPtr_->paramConfigs_.sgPgUpperBound = currentScope->GetPassConfig<int>(SG_PG_UPPER_BOUND);
    currentFunctionPtr_->paramConfigs_.sgPgLowerBound = currentScope->GetPassConfig<int>(SG_PG_LOWER_BOUND);
    currentFunctionPtr_->paramConfigs_.sgParallelNum = currentScope->GetPassConfig<int>(SG_PARALLEL_NUM);
    currentFunctionPtr_->paramConfigs_.sgMgCopyInUpperBound = currentScope->GetPassConfig<int>(MG_COPYIN_UPPER_BOUND);
    currentFunctionPtr_->paramConfigs_.machineConfig_ = currentScope->GetRuntimeConfig<uint8_t>(DEVICE_SCHED_MODE);
    currentFunctionPtr_->paramConfigs_.stitchFunctionNumInitial_ = currentScope->GetRuntimeConfig<uint16_t>(STITCH_FUNCTION_NUM_INITIAL);
    currentFunctionPtr_->paramConfigs_.stitchFunctionNumStep_ = currentScope->GetRuntimeConfig<uint16_t>(STITCH_FUNCTION_NUM_STEP);
    currentFunctionPtr_->paramConfigs_.cubeL1ReuseSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_L1_REUSE_SETTING);
    currentFunctionPtr_->paramConfigs_.cubeNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_NBUFFER_SETTING);
    currentFunctionPtr_->paramConfigs_.vecNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(VEC_NBUFFER_SETTING);
    currentFunctionPtr_->paramConfigs_.vecNBuffermode = currentScope->GetPassConfig<int>(VEC_NBUFFER_MODE);
    currentFunctionPtr_->paramConfigs_.sgCubeParallelNum = currentScope->GetPassConfig<int>(SG_CUBE_PARALLEL_NUM);
    currentFunctionPtr_->paramConfigs_.mgVecParallelLb = currentScope->GetPassConfig<int>(MG_VEC_PARALLEL_LB);
    currentFunctionPtr_->paramConfigs_.pgSkipPartition = currentScope->GetPassConfig<bool>(PG_SKIP_PARTITION);
    currentFunctionPtr_->paramConfigs_.copyOutResolveCoalescing = currentScope->GetPassConfig<int>(COPYOUT_RESOLVE_COALESCING);
}

#if ENABLE_HIDDENLOOP
void Program::BeginHiddenLoop(Function *func, const FunctionType &funcType, const std::string funcName) {
    if (func->GetGraphType() == GraphType::TENSOR_GRAPH 
        && func->GetFunctionType() == funcType
        && !func->IsHiddenFunction()) {
        BeginFunction(funcName, FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH, {}, true);
    }
}

void Program::EndHiddenLoop(Function *func, bool generateCall) {
    if (func->GetGraphType() == GraphType::TENSOR_GRAPH 
        && func->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH 
        && func->IsHiddenFunction() 
        && !func->Parent().IsHiddenFunction()) {
        func->Parent().SetHiddenFunction(true);
        EndFunction(func->GetRawName(), generateCall);
        func->Parent().SetHiddenFunction(false);
    }
}
#endif

// Start a new function and push it to the functions vector
bool Program::BeginFunction(const std::string &funcName,
    const FunctionType funcType,
    const GraphType graphType,
    const std::vector<std::reference_wrapper<const Tensor>>& explicitOpArgs,
    bool isHiddenFunction) {
    if (currentFunctionPtr_->IsFlattening() && (funcType == FunctionType::STATIC && (graphType == GraphType::TENSOR_GRAPH || graphType == GraphType::TILE_GRAPH))) {
        // Static function's subfunction should be ignored
        ASSERT(funcName != currentFunctionPtr_->GetRawName());
        return false;
    }

#if ENABLE_HIDDENLOOP
    // End previous hidden loop if exists
    EndHiddenLoop(currentFunctionPtr_, true);
#endif

    // Push the current function index to the stack
    functionMagicNameStack_.push_back(currentFunctionMagicName_);

    auto funcMagicName = funcName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    if (functionmap_.find(funcMagicName) == functionmap_.end()) { // new function
        auto newFunc =
            std::make_unique<Function>(*this, funcMagicName, funcName, currentFunctionPtr_);
        newFunc->SetFunctionType(funcType);
        newFunc->SetGraphType(graphType);
        newFunc->SetHiddenFunction(isHiddenFunction);
        newFunc->BeginFunction(explicitOpArgs);

        currentFunctionPtr_ = newFunc.get();
        ASSERT(functionmap_.count(funcMagicName) == 0);
        functionmap_.emplace(funcMagicName, std::move(newFunc));
        currentFunctionMagicName_ = funcMagicName;
    } else {
        ALOG_DEBUG("funcMagicName[", funcMagicName, "] is already in the function map");
        currentFunctionMagicName_ = funcMagicName;
        currentFunctionPtr_ = functionmap_[funcMagicName].get();
    }
    if (currentFunctionPtr_->GetGraphType() != GraphType::BLOCK_GRAPH &&
        currentFunctionPtr_->GetGraphType() != GraphType::EXECUTE_GRAPH) {
        GetTensorSlotManager()->BeginScope(currentFunctionPtr_);
    }
    SetParamConfig(currentFunctionPtr_);

#if ENABLE_HIDDENLOOP
    // Begin new hidden loop for the new function
    BeginHiddenLoop(currentFunctionPtr_, FunctionType::DYNAMIC_LOOP_PATH,
        currentFunctionPtr_->GetRawName() + "_hiddenfunc" + std::to_string(currentFunctionPtr_->GetCallopList().size()));
#endif
    return true;
}

Operation &Program::ConnectCallerGusket(Function &caller, FunctionCallArgs &args) const {
    // callFunc is used for:
    //  1. Submit to machine
    //  2. Draw graph
    auto &callFunc = caller.AddRawOperation(Opcode::OP_CALL, args.iOperands, args.oOperands, false);
    callFunc.SetOpAttribute(currentFunctionPtr_->CreateCallOpAttribute(args.argList, args.outIndexToExpr));
    callFunc.SetOpOffset(args.iOpAttrOffset, args.oOpAttrOffset);
    return callFunc;
}

Operation *Program::FinishCurrentFunction(const std::shared_ptr<TensorSlotScope> &scope, bool generateCall) {
    ASSERT(functionMagicNameStack_.size() != 0);
    auto funcMagicName = currentFunctionPtr_->GetRawName() + "_" + std::to_string(currentFunctionPtr_->GetFuncMagic());
    ASSERT(currentFunctionPtr_->GetMagicName() == funcMagicName);

    ALOG_DEBUG("func.end.finish: name=", funcMagicName);

    auto funcArgs = currentFunctionPtr_->EndFunction(scope);

    currentFunctionPtr_->ComputeHash();
    ALOG_DEBUG(currentFunctionPtr_->ComputeHash());
    if (!generateCall) {
        return nullptr;
    }
    ASSERT(currentFunctionPtr_->HasParent());
    if (scope) {
        GetTensorSlotManager()->ConnectSlot(scope);
    }
    return &ConnectCallerGusket(currentFunctionPtr_->Parent(), funcArgs);
}

// Helper function: Dump tensor graph if needed
void Program::DumpTensorGraphIfNeeded(Function *result) {
    if (config::GetPassDefaultConfig("print_graph", false) &&
        result->IsGraphType(GraphType::TENSOR_GRAPH)) {
        result->DumpJsonFile(config::LogTensorGraphFolder() + "/" + result->GetRawName() + ".json");
        result->DumpFile(config::LogTensorGraphFolder() + "/" + result->GetRawName() + ".tifwkgr");
    }
}

// Helper function: Handle task submission
void Program::HandleTaskSubmission(Function *result) {
    if (result->IsGraphType(GraphType::TENSOR_GRAPH) || result->IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        if (result->IsUnderDynamicFunction() || currentDynamicFunctionPtr_ != nullptr) {
            if (!result->IsHiddenFunction() || result->Operations().size() > 0) {
                HostMachine::GetInstance().StashTask(result);
            } else {
                ALOG_INFO("Empty function: ", result->GetRawName(), ", skip stashing and removed");
                functionmap_.erase(result->GetMagicName());
                auto &scopes = GetTensorSlotManager()->scopeList;
                scopes.erase(std::remove_if(scopes.begin(), scopes.end(),
                                 [result](const std::shared_ptr<TensorSlotScope> &scope) {
                                     return scope->tensorFunc == result;
                                 }),
                    scopes.end());
            }
        } else if (!config::GetPlatformConfig(KEY_ONLY_TENSOR_GRAPH, false)) {
            HostMachine::GetInstance().SubTask(result);
            HostMachine::GetInstance().WaitTaskFinish();
        }
    }
}

// End the current function and pop the function index from the stack
std::tuple<Function*, Operation *, bool> Program::EndFunction(const std::string &funcName,
                                                                          bool generateCall) {
#if ENABLE_HIDDENLOOP
    // End child hidden loop
    EndHiddenLoop(currentFunctionPtr_, generateCall);
#endif

    currentFunctionPtr_->paramConfigs_.dynamicAlignedOps = config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_ALIGNED);
    std::shared_ptr<TensorSlotScope> scope = nullptr;
    // root & leaf do not need scope, use tensor/tile graph's
    if (currentFunctionPtr_->GetGraphType() != GraphType::BLOCK_GRAPH &&
        currentFunctionPtr_->GetGraphType() != GraphType::EXECUTE_GRAPH) {
        scope = GetTensorSlotManager()->EndScope();
    }
    currentFunctionPtr_->SetUnderDynamicFunction(Program::GetInstance().GetCurrentDynamicFunction() != nullptr);
    if (currentFunctionPtr_->IsStatic() && funcName != currentFunctionPtr_->GetRawName()) {
        ALOG_ERROR("Function name not match current: ", currentFunctionPtr_->GetRawName(), " != ", funcName);
        return std::make_tuple(nullptr, nullptr, false);
    }

    if (currentFunctionPtr_->IsHiddenFunction() && currentFunctionPtr_->Operations(false).size() <= 0) {
        generateCall = false;
    }
    Operation *callop = FinishCurrentFunction(scope, generateCall);
    bool hit = QueryAndUpdateCurrentFunction();
    auto result = currentFunctionPtr_;

    DumpTensorGraphIfNeeded(result);
    PopStackAndUpdateCurrent();
    HandleTaskSubmission(result);

#if ENABLE_HIDDENLOOP
    // Begin new hidden loop for parent function
    BeginHiddenLoop(result, FunctionType::DYNAMIC_LOOP,
        currentFunctionPtr_->GetRawName() + "_hiddenfunc" + std::to_string(currentFunctionPtr_->GetCallopList().size()));
#endif

    return std::make_tuple(result, callop, hit);
}

void Program::PopStackAndUpdateCurrent() {
    if (!functionMagicNameStack_.empty()) {
        currentFunctionMagicName_ = functionMagicNameStack_.back();
        currentFunctionPtr_ = functionmap_[currentFunctionMagicName_].get();
        functionMagicNameStack_.pop_back();
    } else {
        currentFunctionMagicName_ = ""; // If the stack is empty, no function is active
        currentFunctionPtr_ = nullptr;
    }
}

// Add an operation to the current function and insert operands into the TensorMap
Operation &Program::AddOperation(const std::string &opName,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand) {
    return AddOperation(FindOpcode(opName), iOperand, oOperand);
}

Operation &Program::AddOperation(const Opcode opCode,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand) {
    // Add the operation to the current function
    if (currentFunctionMagicName_ == PROGRAM_ENTRY_FUNCTION_NAME) {
        ALOG_FATAL("Error: No active function to add operation.");
        ASSERT(false);
    }
    return currentFunctionPtr_->AddOperation(opCode, iOperand, oOperand);
}

std::string Program::DumpStack(const std::string &funcName) const {
    std::ostringstream oss;
    oss << "dump: " << funcName << "\n";
    for (size_t i = 0; i < functionMagicNameStack_.size(); i++) {
        auto func = functionmap_.find(functionMagicNameStack_[i])->second;
        if (func) {
            oss << "stack-" << i << ": " << functionMagicNameStack_[i] << " " << GetFunctionTypeNameDict().Find(func->GetFunctionType()) << "\n";
        } else {
            oss << "stack-:" << i << ": " << functionMagicNameStack_[i] << " is nullptr\n";
        };
    }
    if (currentFunctionPtr_) {
        oss << "current: " << currentFunctionMagicName_ << " " << GetFunctionTypeNameDict().Find(currentFunctionPtr_->GetFunctionType()) << "\n";
    } else {
        oss << "current: " << currentFunctionMagicName_ << " is nullptr\n";
    }
    return oss.str();
}

void Program::UpdateAliveTensorsParent(int outcastRawMagic, Function &parent) {
    for (auto *tensor : aliveTensors_) {
        if (tensor->GetStorage() == nullptr) {
            continue;
        }
        if (tensor->GetStorage()->tensor->rawmagic == outcastRawMagic) {
            tensor->GetStorage(false)->UpdateBelongFunction(parent);
            tensor->GetStorage()->magic = IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId();
        }
    }
}

void TraverAndDumpParent(Function *func, Json &progDump) {
    if (func != nullptr) {
        TraverAndDumpParent(&func->Parent(), progDump);
        progDump["functions"].emplace_back(func->DumpJson());
    }
    return;
}

Json Program::DumpJson(Function *mainFunc) const {
    Json progDump;
    progDump["version"] = T_VERSION;
    progDump["pass_thread_num"] = config::GetPassGlobalConfig("pass_thread_num", 1);
    progDump["enable_cvfuse"] = config::GetPassGlobalConfig("enable_cv_fuse", false);
    if (mainFunc == nullptr) {
        std::shared_ptr<npu::tile_fwk::Function> dyndevFunc = nullptr;
        std::vector<std::shared_ptr<npu::tile_fwk::Function>> rootFuncs;
        std::vector<std::shared_ptr<npu::tile_fwk::Function>> tileGraphFuncs;
        std::shared_ptr<npu::tile_fwk::Function> tensorGraphFunc = nullptr;
        /* 先Dump Leaf、Tensor、PROGRAM_ENTRY */
        for (const auto &func : functionmap_) {
            if (func.second->GetGraphType() == GraphType::EXECUTE_GRAPH) {
                rootFuncs.emplace_back(func.second);
                continue;
            }

            if (func.second->GetGraphType() == GraphType::TILE_GRAPH) {
                tileGraphFuncs.emplace_back(func.second);
                continue;
            }

            if (func.second->GetFunctionType() == FunctionType::DYNAMIC) {
                dyndevFunc = func.second;
            }
            if (func.second->IsGraphType(GraphType::TENSOR_GRAPH)) {
                tensorGraphFunc = func.second;
            }
            progDump["functions"].emplace_back(func.second->DumpJson());
        }
        /* Dump RootFunction， rootFunction中涉及Leaf的索引，在LoadJson时需要先LoadLeaf再LoadRoot */
        if (!rootFuncs.empty()) {
            for (auto &rootFunc : rootFuncs) {
                progDump["functions"].emplace_back(rootFunc->DumpJson());
            }
        }
        /* Dump TileGraph, TileGraph Function中涉及 Root的指针，在LoadJson时需要先LoadRoot再LoadTile */
        if (!tileGraphFuncs.empty()) {
            for (auto &tileGraphFunc : tileGraphFuncs) {
                progDump["functions"].emplace_back(tileGraphFunc->DumpJson());
            }
        }

        if (dyndevFunc != nullptr) {
            progDump["entryhash"] = dyndevFunc->GetFunctionHash().c_str();
            progDump["curr_funcmagic"] = dyndevFunc->GetFuncMagic();
        } else {
            // 纯静态场景
            if (!rootFuncs.empty()) {
                progDump["entryhash"] = rootFuncs[0]->GetFunctionHash().c_str();
                if (!tileGraphFuncs.empty()) {
                    progDump["curr_funcmagic"] = tileGraphFuncs[0]->GetFuncMagic();
                } else if (tensorGraphFunc != nullptr) {
                    progDump["curr_funcmagic"] = tensorGraphFunc->GetFuncMagic();
                } else {
                    ASSERT(false) << "cannot find current function magic";
                }
            } else if (!tileGraphFuncs.empty()) {
                progDump["entryhash"] = tileGraphFuncs[0]->GetFunctionHash().c_str();
                progDump["curr_funcmagic"] = tileGraphFuncs[0]->GetFuncMagic();
            } else if (tensorGraphFunc != nullptr) {
                progDump["entryhash"] = tensorGraphFunc->GetFunctionHash().c_str();
                progDump["curr_funcmagic"] = tensorGraphFunc->GetFuncMagic();
            } else {
                ALOG_ERROR_F("Failed to find main function.");
            }
        }
    } else {
        progDump["curr_funcmagic"] = mainFunc->GetFuncMagic();
        progDump["entryhash"] = mainFunc->GetFunctionHash().c_str();

        if (mainFunc->rootFunc_ != nullptr) {
            progDump["functions"].emplace_back(mainFunc->rootFunc_->DumpJson());
            progDump["entryhash"] = mainFunc->rootFunc_->GetFunctionHash().c_str();
            for (auto &leaf : mainFunc->rootFunc_->programs_) {
                progDump["functions"].emplace_back(leaf.second->DumpJson());
            }
        }
        progDump["functions"].emplace_back(mainFunc->DumpJson());
    }
    return progDump;
}

std::shared_ptr<Function> Program::GetFunctionByMagic(int funcMagic)
{
    for (auto &func : functionmap_) {
        if (func.second->GetFuncMagic() == funcMagic) {
            return func.second;
        }
    }
    ALOG_ERROR("Cannot find function iter by magic ", funcMagic);
    return nullptr;
}

Function* Program::GetFunctionByMagicName(const std::string &magicName) const {
    auto it = functionmap_.find(magicName);
    if (it != functionmap_.end()) {
        return it->second.get();
    } else {
        return nullptr;
    }
}

void Program::LoadJson(Json &programJson) {
    int currFuncMagicJson = programJson["curr_funcmagic"].get<int>();
    functionmap_.clear();
    std::shared_ptr<Function> tensorGraph = nullptr;
    std::shared_ptr<Function> tileGraph = nullptr;
    std::unordered_map<int, std::shared_ptr<Function>> loadedFunctions;
    size_t index = 0;
    for (auto &functionJson : programJson["functions"]) {
        auto functionPtr = Function::LoadJson(*this, functionJson);
        loadedFunctions[index++] = functionPtr;
        if (functionPtr != nullptr) {
            functionmap_[functionPtr->GetMagicName()] = functionPtr;
            if (functionPtr->GetGraphType() == GraphType::TILE_GRAPH) {
                tileGraph = functionPtr;
            }
            if (functionPtr->IsGraphType(GraphType::TENSOR_GRAPH)) {
                tensorGraph = functionPtr;
            }

            if (functionPtr->GetFuncMagic() == currFuncMagicJson) {
                currentFunctionPtr_ = functionPtr.get();
            }
        }
    }

    if (tileGraph != nullptr) {
        currentFunctionPtr_ = tileGraph.get();
    } else if (tensorGraph != nullptr) {
        currentFunctionPtr_ = tensorGraph.get();
    }
    // 更新Parent指针
    index = 0;
    for (auto &functionJson : programJson["functions"]) {
        if (functionJson.count("parent_funcmagic") == 0) {
            ++index;
            continue;
        }
        int parentMagic = functionJson["parent_funcmagic"].get<int>();
        auto &functionPtr = loadedFunctions[index++];
        if (functionPtr != nullptr) {
            auto parent = GetFunctionByMagic(parentMagic).get();
            functionPtr->SetParent(parent);
            continue;
        }
    }
    ASSERT(currentFunctionPtr_ != nullptr);
}

void Program::DumpJsonFile(const std::string &fileName, Function *mainFunc) {
    auto filePath = name_ + ".json";
    if (!fileName.empty()) {
        filePath = fileName;
    }

    std::ofstream file(filePath);
    file << DumpJson(mainFunc).dump(1) << std::endl;
    file.close();
}

// Serialize Program briefly
std::string Program::Dump() const {
    std::stringstream ss;
    ss << "Program Begin\n";

    for (const auto &func : functionmap_) {
        ss << func.second->Dump();
        ss << "\n";
    }

    ss << "Program End\n";
    return ss.str();
}

void Program::GraphCheck() const {
    for (const auto &[tmpName, functionPtr] : functionmap_) {
        (void)tmpName;
        auto &function = *functionPtr;
        auto opsView = function.Operations();
        int opMagic = -1000000;
        for (auto &op : opsView) {
            opMagic = std::max(opMagic, op.GetOpMagic());
        }
        ASSERT(opMagic + 1 == function.opSeed_);
        std::unordered_set<const Operation *> opMap;
        for (auto &op : opsView) {
            opMap.emplace(&op);
        }

        for (auto &op : opsView) {
            if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_ASSEMBLE ||
                op.GetOpcode() == Opcode::OP_CALL) {
                ASSERT(op.GetOpAttribute() != nullptr);
            } else {
                ASSERT(op.GetOpAttribute() == nullptr);
            }

            ASSERT(op.opmagic >= 0 && op.opmagic < function.opSeed_)
                << "function opSeed_ is: " << function.opSeed_ << ", opmagic is: " << op.opmagic;
            if (!op.IsCall()) { // call 允许多输出，其余操作目前不允许
                ASSERT(op.oOperand.size() == 1) << "size: " << op.oOperand.size();
            } else {
                // Call Op
                std::size_t bracketPos = op.GetCalleeBracketName().find('[');
                std::string calleeName = op.GetCalleeBracketName().substr(0, bracketPos);
                auto it = functionmap_.find(calleeName);
                ASSERT(it != functionmap_.end());
                ASSERT(op.iOperand.size() == it->second->inCasts_.size())
                    << "operation \"" << op.GetOpcodeStr() << "\" iOperand size: " << op.iOperand.size()
                    << ", function \"" << it->second->GetMagicName() << "\" inCasts_ size: " << it->second->inCasts_.size();
                ASSERT(op.oOperand.size() == it->second->outCasts_.size())
                    << "operation \"" << op.GetOpcodeStr() << "\" oOperand size: " << op.oOperand.size()
                    << ", function \"" << it->second->GetMagicName()
                    << "\" outCasts_ size: " << it->second->outCasts_.size();
            }
            for (auto &oOperand : op.oOperand) {
                if (!oOperand->HasProducer(op)) {
                    std::cout << "ASSERT FAILED: " << oOperand->HasProducer(op)
                              << "  opmagic: " << op.opmagic << " oOperand:" << oOperand->magic << std::endl;
                }
            }
            for (auto &iOperand : op.iOperand) {
                for (auto producer : iOperand->GetProducers()) {
                    ASSERT(producer->BelongTo() == &function);
                    ASSERT(producer->GetOpMagic() >= 0 && producer->GetOpMagic() < function.opSeed_)
                        << "function opSeed_ is: " << function.opSeed_ << ", producer in tensor(" << iOperand->magic
                        << "," << iOperand->tensor->rawmagic << ") is: " << producer;
                    if (opMap.find(producer) == opMap.end()) {
                        ASSERT(opMap.find(producer) != opMap.end());
                    }
                }
            }
        }

        for (auto &[magic, tensor] : function.GetTensorMap().inverseMap_) {
            (void)magic;
            for (auto producer : tensor->GetProducers()) {
                ASSERT(producer->BelongTo() == &function);
                ASSERT(opMap.find(producer) != opMap.end());
            }
        }
    }
}

bool Program::QueryAndUpdateCurrentFunction() {
    auto cacheValue = TryHitCahce(currentFunctionPtr_->GetFunctionHash());
    if (cacheValue == std::nullopt) {
        functionCache_.Insert(currentFunctionPtr_->GetFunctionHash(), *currentFunctionPtr_);
        if (currentFunctionPtr_->HasParent()) {
            auto &parent = currentFunctionPtr_->Parent();
            parent.AppendCalleeMagicName(currentFunctionMagicName_);
        }
        return false;
    } else {
        ASSERT(currentFunctionPtr_->IsGraphType(GraphType::BLOCK_GRAPH));
        auto cacheFunc = cacheValue->cacheFunction;
        functionmap_.erase(currentFunctionPtr_->GetMagicName());
        currentFunctionPtr_ = cacheFunc;
        currentFunctionMagicName_ = currentFunctionPtr_->GetMagicName();
        return true;
    }
}

/* submit to hostmachine , prepare compile */
int Program::EndFunction(const bool isWaitTaskFinished)
{
    ASSERT(currentFunctionPtr_ != nullptr);
    auto scope = GetTensorSlotManager()->EndScope();
    auto funcArgs = currentFunctionPtr_->EndFunction(scope);
    if (currentFunctionPtr_->HasParent()) {
        auto &callop =
            currentFunctionPtr_->Parent().AddOperation(Opcode::OP_CALL, funcArgs.iOperands, funcArgs.oOperands, false);
        callop.SetOpAttribute(currentFunctionPtr_->CreateCallOpAttribute(funcArgs.argList, funcArgs.outIndexToExpr));
        callop.SetOpOffset(funcArgs.iOpAttrOffset, funcArgs.oOpAttrOffset);
    }
    HostMachine::GetInstance().SubTask(currentFunctionPtr_);
    if (isWaitTaskFinished) {
        HostMachine::GetInstance().WaitTaskFinish();
    }
    ALOG_INFO("End func: name = ", currentFunctionPtr_->GetRawName());
    return 0;
}

void Program::VerifyTensorGraph() {
    Function *func = GetLastFunction();

    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> outputDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> goldenDataViewList;
    ProgramData::GetInstance().CopyToInputDataViewList(inputDataViewList);
    ProgramData::GetInstance().CopyToOutputDataViewList(outputDataViewList);
    ProgramData::GetInstance().CopyToGoldenDataViewList(goldenDataViewList);

    auto &flowVerifier = FlowVerifier::GetInstance();
    flowVerifier.VerifyTensorGraph(func, inputDataViewList, outputDataViewList, goldenDataViewList, GetTensorSlotManager());
}

void Program::VerifyPass(Function *func, int passIndex, const std::string &passIdentifier) {
    // SubgraphToFunction阶段还未进行validShape推导，会导致非尾块的计算会按照尾块大小进行计算，导致部分数据的拷贝或者计算丢失，
    // 该Pass需要与InferParamIndexPass进行“合并”后才会完成VaildShape推导，才可以完成完整功能；
    if (passIdentifier == "SubgraphToFunction") {
        ALOG_INFO("Skip verify pass [SubgraphToFunction] for interpreter!");
        return;
    }
    auto &flowVerifier = FlowVerifier::GetInstance();
    flowVerifier.VerifyPass(func, passIndex, passIdentifier);
}

std::shared_ptr<Function> Program::GetFunctionSharedPtr(Function* rawPtr) {
    for (const auto& pair : functionmap_) {
        auto sharedPtr = pair.second;
        if (sharedPtr.get() == rawPtr) {
            return sharedPtr;
        }
    }
    ALOG_WARN("not find function ptr in function map");
    return nullptr;
}

void static MergeAllFuncDupIocast(Function* func) {
    if(func == nullptr) {
        auto rootFunc = Program::GetInstance().GetFunctionByMagicName(PROGRAM_ENTRY_FUNCTION_NAME);
        if (rootFunc != nullptr) {
            auto calleeLists = rootFunc->GetCalleeFunctionList();
            for (auto callee : calleeLists) {
                MergeAllFuncDupIocast(callee);
            }
        }
        return;
    }

    ALOG_INFO("Merge Duplicated Iocast for function :", func->GetMagicName());
    auto calleeLists = func->GetCalleeFunctionList();
    // leaf function has no duplicated tensor
    if (calleeLists.size() == 0) {
        return;
    }

    // 1.merge duplicated incast/outcast
    func->MergeFunctionDupIocast();
    // 2. remove useless view assemble op
    func->RemoveCallOpViewAssemble();
    if (config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false) &&
        func->IsGraphType(GraphType::TENSOR_GRAPH)) {
        func->DumpJsonFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + "_remove_dup.json");
        func->DumpFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + "_remove_dup.tifwkgr");
    }

    for (auto callee : calleeLists) {
        MergeAllFuncDupIocast(callee);
    }
}

void RecordFunc::RecordDynFuncInner(const std::vector<std::reference_wrapper<const Tensor>> &startArgsInputTensorList,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsOutputTensorList,
    const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>> &inplaceArgs) {
        ASSERT(config::GetFunctionType() == FunctionType::DYNAMIC);

#if ENABLE_HIDDENLOOP
        recordLoopFunc_ = std::make_unique<RecordLoopFunc>(
            funcName + "_loop",
            FunctionType::DYNAMIC_LOOP,
            funcName +"_unused_hidden_record_func_loop_idx",
            LoopRange(1)
        );
#endif

        Program::GetInstance().BeginFunction(funcName, config::GetFunctionType());

        std::shared_ptr<TensorSlotManager> manager = Program::GetInstance().GetTensorSlotManager();
        for (auto &param : startArgsInputTensorList) {
            manager->MarkInput(param.get());
        }
        for (auto &param : startArgsOutputTensorList) {
            manager->MarkOutput(param.get());
        }
        for (auto &param : inplaceArgs) {
            manager->MarkInplace(param.first.get(), param.second.get());
        }

        dynFunc_ = Program::GetInstance().GetCurrentFunction();
        dynFunc_->SetUnderDynamicFunction(true);
        dynFunc_->SetSourceLocation(SourceLocation::GetLocation());

        std::shared_ptr<DyndevFunctionAttribute> attr = std::make_shared<DyndevFunctionAttribute>();
        attr->startArgsInputTensorList = startArgsInputTensorList;
        attr->startArgsOutputTensorList = startArgsOutputTensorList;

        attr->startArgsInputLogicalTensorList.resize(startArgsInputTensorList.size());
        attr->startArgsOutputLogicalTensorList.resize(startArgsOutputTensorList.size());
        for (size_t k = 0; k < startArgsInputTensorList.size(); k++) {
            attr->startArgsInputLogicalTensorList[k] = startArgsInputTensorList[k].get().GetStorage(false);
        }
        for (size_t k = 0; k < startArgsOutputTensorList.size(); k++) {
            attr->startArgsOutputLogicalTensorList[k] = startArgsOutputTensorList[k].get().GetStorage(false);
        }

        dynFunc_->SetDyndevAttribute(attr);
        Program::GetInstance().SetCurrentDynamicFunction(dynFunc_);
}

RecordFunc::RecordFunc(const std::string &name) : funcName(FUNCTION_PREFIX + name) {
    ConfigManager::Instance().ResetLog();
    Program::GetInstance().BeginFunction(funcName, config::GetFunctionType());
}

RecordFunc::RecordFunc(const std::string &name,
    const std::vector<std::reference_wrapper<const Tensor>> &explicitOpArgs)
    : funcName(FUNCTION_PREFIX + name) {
    // RecordFunc start with TENSOR_GRAPH
    ConfigManager::Instance().ResetLog();
    if (config::GetFunctionType() == FunctionType::DYNAMIC) {
        RecordDynFuncInner(explicitOpArgs, {}, {});
    } else {
        Program::GetInstance().BeginFunction(funcName, config::GetFunctionType(), GraphType::TENSOR_GRAPH, explicitOpArgs);
    }
}

RecordFunc::RecordFunc(const std::string &name,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsInputTensorList,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsOutputTensorList,
    const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>> &inplaceArgs)
    : funcName(FUNCTION_PREFIX + name) {
    ConfigManager::Instance().ResetLog();
    RecordDynFuncInner(startArgsInputTensorList, startArgsOutputTensorList, inplaceArgs);
}

inline bool IsVerifyEnable() {
    return config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY);
}

void RecordFunc::EndFunction() {
    if (recordLoopFunc_) {
        recordLoopFunc_.reset();
    }

    if (IsVerifyEnable()) {
        config::SetRunDataOption(KEY_FLOW_VERIFY_PATH, config::GetAbsoluteTopFolder() + "/verify");
    }

    // might raise exception in EndFunction, force isEnd_ is always set
    Defer clean([this](){ isEnd_ = true;});
    (void)Program::GetInstance().EndFunction(funcName);
    if (dynFunc_) {
        Program::GetInstance().SetLastFunction(dynFunc_);
        if (dynFunc_->IsDyndev()) {
            dynFunc_->CleanRedundantOutCast();
            // Destructor GetTensorData small Tensor
            auto attr = dynFunc_->GetDyndevAttribute();
            attr->getTensorDataDescDict.clear();

            dynFunc_->ApplyLoopCallOrderGroup();
            if (config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY)) {
                Program::GetInstance().VerifyTensorGraph();
            }
            MergeAllFuncDupIocast(nullptr);
            PassManager::Instance().RunPass(Program::GetInstance(),
                *Program::GetInstance().GetFunctionByMagicName(PROGRAM_ENTRY_FUNCTION_NAME), "FunctionUnroll");
            if (!config::GetPlatformConfig(npu::tile_fwk::KEY_ONLY_TENSOR_GRAPH, false)) {
                Program::GetInstance().UpdateCompileTask();
            }
        }
        Program::GetInstance().SetCurrentDynamicFunction(nullptr);
        dynFunc_->SetUnderDynamicFunction(false);
    }
}

RecordFunc::Iterator RecordFunc::begin() {
    if (recordLoopFunc_) {
        return Iterator(*this, recordLoopFunc_->begin());
    }
    return Iterator(*this);
}

RecordFunc::IteratorEnd RecordFunc::end() {
    if (recordLoopFunc_) {
        return IteratorEnd(*this, recordLoopFunc_->end());
    }
    return IteratorEnd(*this);
}

RecordFunc::Iterator RecordFunc::Iterator::operator++() {
    if (!wrappedIter_.has_value()) {
        cur_ = 1;
        return *this;
    }
    ++(*wrappedIter_);
    return *this;
}

bool RecordFunc::Iterator::operator!=(const IteratorEnd &rhs) {
    if (!wrappedIter_.has_value()) {
        return cur_ != 1;
    }
    ASSERT(rhs.wrappedEnd.has_value());
    bool result = *wrappedIter_ != *rhs.wrappedEnd;
    return result;
}

RecordLoopFunc::RecordLoopFunc(const std::string &name, FunctionType funcType, const std::string &iterName,
    const LoopRange &range, const std::set<int> &unrollList, bool submitBeforeLoop)
    : name_(FUNCTION_PREFIX + name),
        iterName_(iterName),
        loopRange_(std::make_shared<LoopRange>(range)),
        submitBeforeLoop_(submitBeforeLoop),
        funcType_(funcType) {
    ASSERT(funcType == FunctionType::STATIC || funcType == FunctionType::DYNAMIC_LOOP);
    Program::GetInstance().GetLoopStack().emplace_back(*this);

    GenDefaultUnrollTimes(unrollList);
    location_ = SourceLocation::GetLocation();
}

RecordLoopFunc::~RecordLoopFunc() {
    Program::GetInstance().GetLoopStack().pop_back();
}

bool RecordLoopFunc::IterationEnd() {
    auto result = Program::GetInstance().EndFunction(curPathFuncName_);
    auto pathFunc = std::get<0>(result);
    pathFunc->ApplyLoopCallOrderGroup();
    Program::GetInstance().GetTensorSlotManager()->Restore();
    auto isEnd = GetLoopAttr()->IterationEnd(CurUnrollTimes(), pathFunc, std::get<1>(result));
    if (isEnd) {
        endCount_ = 0;
    }
    return isEnd;
}

void RecordLoopFunc::BeginLoopFunction() {
    auto loopFuncName = name_ + "_Unroll" + std::to_string(CurUnrollTimes());
    Program::GetInstance().BeginFunction(loopFuncName, FunctionType::DYNAMIC_LOOP);
    currentLoopFunc_ = Program::GetInstance().GetCurrentFunction();
    ASSERT(currentLoopFunc_->InsertLoopIdxNameList(iterName_)) << "Forbid duplicate name of loop idx. It names " << iterName_;
    auto currentStep = CurUnrollTimes() == 1 ? loopRange_->Step() : loopRange_->Step() * CurUnrollTimes();
    if (rangeOfEaceUnroll_.empty()) {
        std::shared_ptr<LoopRange> newRange = std::make_shared<LoopRange>(loopRange_->Begin(), loopRange_->End() / currentStep * currentStep, currentStep);
        rangeOfEaceUnroll_.push_back(newRange);
    } else {
        auto prevRange = rangeOfEaceUnroll_.back();
        std::shared_ptr<LoopRange> newRange = std::make_shared<LoopRange>(prevRange->End(), prevRange->End() + (loopRange_->End() - prevRange->End()) / currentStep * currentStep, currentStep);
        rangeOfEaceUnroll_.push_back(newRange);
    }
    auto range = rangeOfEaceUnroll_.back();
    range->End().AsIntermediateVariable();
    auto attr = std::make_shared<DynloopFunctionAttribute>(iterName_, *range, *loopRange_, submitBeforeLoop_);
    currentLoopFunc_->SetDynloopAttribute(attr);
    currentLoopFunc_->SetSourceLocation(location_);
}

void RecordLoopFunc::EndLoopFunction() {
    auto loopFuncName = name_ + "_Unroll" + std::to_string(CurUnrollTimes());
    Program::GetInstance().EndFunction(loopFuncName);
    currentLoopFunc_ = nullptr;
}

bool RecordLoopFunc::MatchUnrollTimes(int unrollTimes) {
    ASSERT(unrollTimes > 0) << "unrollTimes must larger than zero!";
    auto &curRlf = Program::GetInstance().GetLoopStack().back().get();
    curRlf.customUnrollTimes_.emplace(unrollTimes);
    if (!curRlf.hasManualUnroll_) {
        curRlf.hasManualUnroll_ = true;
        curRlf.dryRun_ = true;
    }
    if (curRlf.dryRun_) {
        return false;
    }
    if (!curRlf.VisitedUnroll(unrollTimes)) {
        curRlf.VisitUnroll(unrollTimes);
    }
    ASSERT(curRlf.StillHaveUnrollTimes());
    if (curRlf.CurUnrollTimes() == unrollTimes) {
        return true;
    }

    // if cur unrollTimes is not user defined, use unrollTimes = 1 to auto merge loop
    if (curRlf.CurUnrollTimes() > 1 && !curRlf.CustomUnrollTimesMatched() && unrollTimes == 1) {
        return true;
    }
    return false;
}

RecordLoopFunc::Iterator RecordLoopFunc::Iterator::operator++() {
    if (rlf_.dryRun_) {
        return *this;
    }
    if (!rlf_.CustomUnrollTimesMatched()) {
        scalar_ = scalar_ + rlf_.LoopStep();
        cur_++;
    } else {
        ASSERT(cur_ == 0);
        scalar_ = scalar_ + rlf_.LoopStep() * rlf_.CurUnrollTimes();
        cur_ += rlf_.CurUnrollTimes();
    }

    rlf_.IterationNext();
    return *this;
}

bool RecordLoopFunc::Iterator::operator!=(const IteratorEnd &rhs) {
    (void)rhs;
    if (rlf_.dryRun_) {
        rlf_.dryRun_ = false;
        ASSERT(cur_ == 0);
        if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes())) {
            scalar_.AsLoopEnd(true);
        }
        return true;
    }
    ASSERT(rlf_.StillHaveUnrollTimes());
    if (cur_ < rlf_.CurUnrollTimes()) {
        if (cur_ == 0) {
            scalar_.AsLoopBegin(true);
            rlf_.IterationBegin();
            if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes())) {
                scalar_.AsLoopEnd(true);
            }
        }
        if (cur_ + 1 == rlf_.CurUnrollTimes()) {
            scalar_.AsLoopEnd(true);
        }
        return true;
    }
    ASSERT(cur_ == rlf_.CurUnrollTimes());
    if (rlf_.IterationEnd()) {
        rlf_.EndLoopFunction();
        rlf_.NextUnrollTimes();
        if (!rlf_.StillHaveUnrollTimes()) {
            return false;
        }
    }
    ASSERT(rlf_.StillHaveUnrollTimes());
    cur_ = 0;
    scalar_ = originalScalar_;
    if (rlf_.LoopBegin().IsImmediate()) {
        auto beginValue = std::static_pointer_cast<RawSymbolicImmediate>(rlf_.LoopBegin().Raw())->Immediate();
        if (rlf_.LoopStep().IsImmediate() && rlf_.LoopEnd().IsImmediate()) {
            auto endValue = std::static_pointer_cast<RawSymbolicImmediate>(rlf_.LoopEnd().Raw())->Immediate();
            scalar_.Raw()->ResetValueGuesser(
                ValueGuesser(NotLessThan(beginValue), NotGreaterThan(endValue - 1)));
        } else {
            scalar_.Raw()->ResetValueGuesser(ValueGuesser(NotLessThan(beginValue)));
        }
    } else {
        scalar_.Raw()->ResetValueGuesser(ValueGuesser::Any());
    }

    scalar_.AsLoopBegin(true);
    rlf_.IterationBegin();
    if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes()) || cur_ + 1 == rlf_.CurUnrollTimes()) {
        scalar_.AsLoopEnd(true);
    }
    return true;
}

RecordLoopFunc::Iterator RecordLoopFunc::begin() {
    if (loopRange_->Begin().ConcreteValid()) {
        return {*this, SymbolicScalar(iterName_, NotLessThan(loopRange_->Begin().Concrete()))};
    }
    return {*this, SymbolicScalar(iterName_)};
}

RecordLoopFunc::IteratorEnd RecordLoopFunc::end() {
    if (loopRange_->End().ConcreteValid()) {
        if (funcType_ == FunctionType::STATIC) {
            /* Static loop, expand all */
            return {*this, SymbolicScalar(iterName_, NotGreaterThan(loopRange_->End().Concrete()))};
        } else {
            /* Runtime: Run only once */
            return {*this, SymbolicScalar(iterName_, NotGreaterThan(loopRange_->End().Concrete()))};
        }
    }
    return {*this, SymbolicScalar(iterName_)};
}

void RecordLoopFunc::IterationBegin() {
    if (currentLoopFunc_ == nullptr) {
        BeginLoopFunction();
    }
    curPathFuncName_ = name_ + "_Unroll" + std::to_string(CurUnrollTimes()) + GetLoopSuffix(endCount_++);
    Program::GetInstance().GetTensorSlotManager()->Checkpoint();
    Program::GetInstance().BeginFunction(curPathFuncName_, FunctionType::DYNAMIC_LOOP_PATH);
    auto loopPathFunc = Program::GetInstance().GetCurrentFunction();
    loopPathFunc->SetSourceLocation(location_);
    GetLoopAttr()->IterationBegin();
}

void RecordLoopFunc::IterationNext() {
    ASSERT(customUnrollTimes_.empty() || customUnrollTimes_.count(1) > 0) << "Must have unroll 1 if user defined custom unroll times.";
}

bool RecordLoopFunc::Condition(const SymbolicScalar &cond, const std::string &file, int line) {
    bool result = GetLoopAttr()->AppendCond(cond, file, line);
    ALOG_INFO(file, ":", line, "]: ", result);
    return result;
}

void RecordLoopFunc::GenDefaultUnrollTimes(const std::set<int> &unrollList) {
    unrollTimes_.clear();
    visited_.clear();
    if (config::GetPlatformConfig("ONLY_MANUAL_UNROLL", false) || unrollList.empty()) {
        unrollTimes_.emplace(1);
        visited_.emplace(1);
    } else {
        for (auto n : unrollList) {
            unrollTimes_.emplace(n);
            visited_.emplace(n);
        }
    }
}

void RecordLoopFunc::VisitUnroll(int unrollTimes) {
    ASSERT(visited_.count(unrollTimes) == 0);
    ASSERT(unrollTimes_.count(unrollTimes) == 0);
    visited_.emplace(unrollTimes);
    unrollTimes_.emplace(unrollTimes);
}

int RecordLoopFunc::CurUnrollTimes() const {
    ASSERT(StillHaveUnrollTimes());
    return *unrollTimes_.begin();
}

void RecordLoopFunc::NextUnrollTimes() {
    ASSERT(StillHaveUnrollTimes());
    unrollTimes_.erase(unrollTimes_.begin());
}

std::shared_ptr<DynloopFunctionAttribute> RecordLoopFunc::GetLoopAttr() {
    return currentLoopFunc_->GetDynloopAttribute();
}

const SymbolicScalar &RecordLoopFunc::LoopBegin() const { return loopRange_->Begin(); }
const SymbolicScalar &RecordLoopFunc::LoopStep() const { return loopRange_->Step(); }
const SymbolicScalar &RecordLoopFunc::LoopEnd() const { return loopRange_->End(); }

RecordIfBranch::operator bool() const {
    bool cond = Program::GetInstance().GetLoopStack().back().get().Condition(cond_, file_, line_);
    return cond;
}
} // namespace npu::tile_fwk
