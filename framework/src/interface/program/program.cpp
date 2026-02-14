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

#include <sstream>
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
#include "interface/program/program.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk {
const std::string PROGRAM_ENTRY_FUNCTION_NAME = "PROGRAM_ENTRY";
void GetEnv(const char * const envName, std::string &envValue) {
    const size_t envValueMaxLen = 1024UL * 1024UL;
    const char * const envTemp = std::getenv(envName);
    if ((envTemp == nullptr) || (strnlen(envTemp, envValueMaxLen) >= envValueMaxLen)) {
        FUNCTION_LOGI("Env[%s] not found.\n", envName);
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
    if (function != nullptr) {
        currentFunctionPtr_ = function;
        currentFunctionMagicName_ = function->GetMagicName();
        FUNCTION_LOGD("Set current function successfully.");
    }
    FUNCTION_LOGW("Failed to set current function.");
}

void Program::CreateInitFunction() {
    currentFunctionMagicName_ = PROGRAM_ENTRY_FUNCTION_NAME;
    auto newFunc = std::make_shared<Function>(*this, currentFunctionMagicName_, currentFunctionMagicName_, nullptr);
    newFunc->SetFunctionType(FunctionType::EAGER);
    currentFunctionPtr_ = newFunc.get();
    functionmap_.emplace(currentFunctionMagicName_, std::move(newFunc));
}

void Program::CreateCallerCalleeLink(Function *caller, Function *callee) {
    ASSERT(caller->IsGraphType(GraphType::TENSOR_GRAPH) &&
        callee->IsGraphType(GraphType::TENSOR_GRAPH))
        << "caller graphType: " << GetGraphTypeNameDict().Find(caller->GetGraphType())
        << ", callee graphType: " << GetGraphTypeNameDict().Find(callee->GetGraphType());
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

void Program::ClearEmptyHiddenFunction() {
    std::vector<std::string> funcNames;
    for (auto &[name, func] : functionmap_) {
        if (func->IsHiddenFunction() && func->Operations(false).IsEmpty()) {
            funcNames.push_back(name);
        }
    }
    for (auto &name : funcNames) {
        functionmap_.erase(name);
    }
}

void SetParamConfig(Function* currentFuncPtr) {
    std::shared_ptr<ConfigScope> currentScope = ConfigManagerNg::GetInstance().CurrentScope();
    currentFuncPtr->paramConfigs_.L1ReuseMode = currentScope->GetPassConfig<int>(CUBE_L1_REUSE_MODE);
    currentFuncPtr->paramConfigs_.cubeNBufferMode = currentScope->GetPassConfig<int>(CUBE_NBUFFER_MODE);
    currentFuncPtr->paramConfigs_.sgPgUpperBound = currentScope->GetPassConfig<int>(SG_PG_UPPER_BOUND);
    currentFuncPtr->paramConfigs_.sgPgLowerBound = currentScope->GetPassConfig<int>(SG_PG_LOWER_BOUND);
    currentFuncPtr->paramConfigs_.sgParallelNum = currentScope->GetPassConfig<int>(SG_PARALLEL_NUM);
    currentFuncPtr->paramConfigs_.sgMgCopyInUpperBound = currentScope->GetPassConfig<int>(MG_COPYIN_UPPER_BOUND);
    currentFuncPtr->paramConfigs_.machineConfig_ = currentScope->GetRuntimeConfig<uint8_t>(DEVICE_SCHED_MODE);
    currentFuncPtr->paramConfigs_.stitchFunctionNumInitial_ = currentScope->GetRuntimeConfig<uint16_t>(STITCH_FUNCTION_NUM_INITIAL);
    currentFuncPtr->paramConfigs_.stitchFunctionNumStep_ = currentScope->GetRuntimeConfig<uint16_t>(STITCH_FUNCTION_NUM_STEP);
    currentFuncPtr->paramConfigs_.cubeL1ReuseSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_L1_REUSE_SETTING);
    currentFuncPtr->paramConfigs_.cubeNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(CUBE_NBUFFER_SETTING);
    currentFuncPtr->paramConfigs_.vecNBufferSetting = currentScope->GetPassConfig<std::map<int64_t, int64_t>>(VEC_NBUFFER_SETTING);
    currentFuncPtr->paramConfigs_.vecNBuffermode = currentScope->GetPassConfig<int>(VEC_NBUFFER_MODE);
    currentFuncPtr->paramConfigs_.mgVecParallelLb = currentScope->GetPassConfig<int>(MG_VEC_PARALLEL_LB);
    currentFuncPtr->paramConfigs_.pgSkipPartition = currentScope->GetPassConfig<bool>(PG_SKIP_PARTITION);
    currentFuncPtr->paramConfigs_.copyOutResolveCoalescing = currentScope->GetPassConfig<int>(COPYOUT_RESOLVE_COALESCING);
    currentFuncPtr->paramConfigs_.combineAxis = currentScope->GetOperationConfig<bool>(KEY_COMBINE_AXIS);
    currentFuncPtr->paramConfigs_.forceCombineAxis = currentScope->GetOperationConfig<bool>(KEY_FORCE_COMBINE_AXIS);
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
        CHECK(funcName != currentFunctionPtr_->GetRawName())
            << "funcName: " << funcName << ", currentFuncRawName: " << currentFunctionPtr_->GetRawName();
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
        FUNCTION_LOGD("Create a new function[%s].", funcMagicName.c_str());
        auto newFunc = std::make_unique<Function>(*this, funcMagicName, funcName, currentFunctionPtr_);
        newFunc->SetFunctionType(funcType);
        newFunc->SetGraphType(graphType);
        newFunc->SetHiddenFunction(isHiddenFunction);
        newFunc->BeginFunction(explicitOpArgs);

        currentFunctionPtr_ = newFunc.get();
        ASSERT(functionmap_.count(funcMagicName) == 0) << funcMagicName << " already exists in funcmap.";
        functionmap_.emplace(funcMagicName, std::move(newFunc));
        currentFunctionMagicName_ = funcMagicName;
    } else {
        FUNCTION_LOGE("funcMagicName[%s] is already in the function map", funcMagicName.c_str());
        currentFunctionMagicName_ = funcMagicName;
        currentFunctionPtr_ = functionmap_[funcMagicName].get();
    }
    if (currentFunctionPtr_->GetGraphType() != GraphType::BLOCK_GRAPH &&
        currentFunctionPtr_->GetGraphType() != GraphType::EXECUTE_GRAPH) {
        GetTensorSlotManager()->BeginScope(currentFunctionPtr_);
    }


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
    ASSERT(functionMagicNameStack_.size() != 0) << "The stack of functionMagicName is null.";
    auto funcMagicName = currentFunctionPtr_->GetRawName() + "_" + std::to_string(currentFunctionPtr_->GetFuncMagic());
    ASSERT(currentFunctionPtr_->GetMagicName() == funcMagicName)
        << "currentFunc magicName: " << currentFunctionPtr_->GetMagicName()
        << ", rawName: " << currentFunctionPtr_->GetRawName() << "funcMagic: " << currentFunctionPtr_->GetFuncMagic();

    FUNCTION_LOGD("func.end.finish: name=%s", funcMagicName.c_str());

    auto funcArgs = currentFunctionPtr_->EndFunction(scope);

    currentFunctionPtr_->ComputeHash();
    FUNCTION_LOGD("The hash of current func is %lu", currentFunctionPtr_->ComputeHash().GetHash());
    if (!generateCall) {
        return nullptr;
    }
    ASSERT(currentFunctionPtr_->HasParent()) << "CurrentFunction doesn't have a parent function.";
    if (scope) {
        GetTensorSlotManager()->ConnectSlot(scope);
    }
    return &ConnectCallerGusket(currentFunctionPtr_->Parent(), funcArgs);
}

// Helper function: Dump tensor graph if needed
void Program::DumpTensorGraphIfNeeded(Function *result) {
    if (config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false) &&
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
                FUNCTION_LOGI("Empty function: %s, skip stashing and removed", result->GetRawName().c_str());
                auto &scopes = GetTensorSlotManager()->scopeList;
                scopes.erase(std::remove_if(scopes.begin(), scopes.end(),
                                 [result](const std::shared_ptr<TensorSlotScope> &scope) {
                                     return scope->tensorFunc == result;
                                 }),
                    scopes.end());
            }
        } else {
            HostMachine::GetInstance().SubTask(result);
            HostMachine::GetInstance().WaitTaskFinish();
        }
    }
}

// End the current function and pop the function index from the stack
std::tuple<Function*, Operation *, bool> Program::EndFunction(const std::string &funcName,
                                                                          bool generateCall) {
    FUNCTION_LOGD("EndFunction start.");
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
        FUNCTION_LOGE(
            "Function name not match current: %s != %s", currentFunctionPtr_->GetRawName().c_str(), funcName.c_str());
        return std::make_tuple(nullptr, nullptr, false);
    }

    if (currentFunctionPtr_->IsHiddenFunction() && currentFunctionPtr_->Operations(false).size() <= 0) {
        generateCall = false;
    }
    Operation *callop = FinishCurrentFunction(scope, generateCall);
    bool hit = QueryAndUpdateCurrentFunction();
    auto result = currentFunctionPtr_;

    SetParamConfig(currentFunctionPtr_);

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
        FUNCTION_LOGE("Error: No active function to add operation.");
        ASSERT(false) << "No active function to add operation.";
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

void TraverAndDumpParent(Function *func, Json &programDump) {
    if (func != nullptr) {
        TraverAndDumpParent(&func->Parent(), programDump);
        programDump["functions"].emplace_back(func->DumpJson());
    }
    return;
}

Json Program::DumpJson(Function *mainFunc) const {
    Json programDump;
    programDump["version"] = T_VERSION;
    programDump["pass_thread_num"] = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
    programDump["enable_cvfuse"] = config::GetPassGlobalConfig(KEY_ENABLE_CV_FUSE, false);
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
            programDump["functions"].emplace_back(func.second->DumpJson());
        }
        /* Dump RootFunction， rootFunction中涉及Leaf的索引，在LoadJson时需要先LoadLeaf再LoadRoot */
        if (!rootFuncs.empty()) {
            for (auto &rootFunc : rootFuncs) {
                programDump["functions"].emplace_back(rootFunc->DumpJson());
            }
        }
        /* Dump TileGraph, TileGraph Function中涉及 Root的指针，在LoadJson时需要先LoadRoot再LoadTile */
        if (!tileGraphFuncs.empty()) {
            for (auto &tileGraphFunc : tileGraphFuncs) {
                programDump["functions"].emplace_back(tileGraphFunc->DumpJson());
            }
        }

        if (dyndevFunc != nullptr) {
            programDump["entryhash"] = dyndevFunc->GetFunctionHash().c_str();
            programDump["curr_funcmagic"] = dyndevFunc->GetFuncMagic();
        } else {
            // 纯静态场景
            if (!rootFuncs.empty()) {
                programDump["entryhash"] = rootFuncs[0]->GetFunctionHash().c_str();
                if (!tileGraphFuncs.empty()) {
                    programDump["curr_funcmagic"] = tileGraphFuncs[0]->GetFuncMagic();
                } else if (tensorGraphFunc != nullptr) {
                    programDump["curr_funcmagic"] = tensorGraphFunc->GetFuncMagic();
                } else {
                    ASSERT(false) << "cannot find current function magic";
                }
            } else if (!tileGraphFuncs.empty()) {
                programDump["entryhash"] = tileGraphFuncs[0]->GetFunctionHash().c_str();
                programDump["curr_funcmagic"] = tileGraphFuncs[0]->GetFuncMagic();
            } else if (tensorGraphFunc != nullptr) {
                programDump["entryhash"] = tensorGraphFunc->GetFunctionHash().c_str();
                programDump["curr_funcmagic"] = tensorGraphFunc->GetFuncMagic();
            } else {
                FUNCTION_LOGE("Failed to find main function.");
            }
        }
    } else {
        programDump["curr_funcmagic"] = mainFunc->GetFuncMagic();
        programDump["entryhash"] = mainFunc->GetFunctionHash().c_str();

        if (mainFunc->rootFunc_ != nullptr) {
            programDump["functions"].emplace_back(mainFunc->rootFunc_->DumpJson());
            programDump["entryhash"] = mainFunc->rootFunc_->GetFunctionHash().c_str();
            for (auto &leaf : mainFunc->rootFunc_->programs_) {
                programDump["functions"].emplace_back(leaf.second->DumpJson());
            }
        }
        programDump["functions"].emplace_back(mainFunc->DumpJson());
    }
    return programDump;
}

std::shared_ptr<Function> Program::GetFunctionByMagic(int funcMagic)
{
    for (auto &func : functionmap_) {
        if (func.second->GetFuncMagic() == funcMagic) {
            return func.second;
        }
    }
    FUNCTION_LOGE("Cannot find function iter by magic %d", funcMagic);
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
    ASSERT(currentFunctionPtr_ != nullptr) << "loss of pointer.";
}

void Program::DumpJsonFile(const std::string &fileName, Function *mainFunc) {
    auto filePath = name_ + ".json";
    if (!fileName.empty()) {
        filePath = fileName;
    }
    FUNCTION_LOGD("Program dump json to %s.", filePath.c_str());
    std::ofstream file(filePath);
    ASSERT(file.is_open()) << "Failed to open file: " << filePath;
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
        ASSERT(currentFunctionPtr_->IsGraphType(GraphType::BLOCK_GRAPH))
            << "currentFunction graphType: "
            << GetGraphTypeNameDict().Find(currentFunctionPtr_->GetGraphType());
        auto cacheFunc = cacheValue->GetFunction();
        functionmap_.erase(currentFunctionPtr_->GetMagicName());
        currentFunctionPtr_ = cacheFunc;
        currentFunctionMagicName_ = currentFunctionPtr_->GetMagicName();
        return true;
    }
}

void Program::VerifyTensorGraph() {
    FUNCTION_LOGI("VerifyTensorGraph start.");
    Function *func = GetLastFunction();

    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> outputDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> goldenDataViewList;
    ProgramData::GetInstance().CopyToInputDataViewList(inputDataViewList);
    ProgramData::GetInstance().CopyToOutputDataViewList(outputDataViewList);
    ProgramData::GetInstance().CopyToGoldenDataViewList(goldenDataViewList);

    auto &flowVerifier = FlowVerifier::GetInstance();
    flowVerifier.VerifyTensorGraph(func, inputDataViewList, outputDataViewList, goldenDataViewList, GetTensorSlotManager());
    FUNCTION_LOGI("VerifyTensorGraph end.");
}

void Program::VerifyPass(Function *func, int passIndex, const std::string &passIdentifier) {
    // SubgraphToFunction阶段还未进行validShape推导，会导致非尾块的计算会按照尾块大小进行计算，导致部分数据的拷贝或者计算丢失，
    // 该Pass需要与InferParamIndexPass进行“合并”后才会完成VaildShape推导，才可以完成完整功能；
    FUNCTION_LOGI("VerifyPass start.");
    if (passIdentifier == "SubgraphToFunction") {
        FUNCTION_LOGI("Skip verify pass [SubgraphToFunction] for interpreter!");
        return;
    }
    auto &flowVerifier = FlowVerifier::GetInstance();
    flowVerifier.VerifyPass(func, passIndex, passIdentifier);
    FUNCTION_LOGI("VerifyPass end.");
}

std::shared_ptr<Function> Program::GetFunctionSharedPtr(Function* rawPtr) {
    for (const auto& pair : functionmap_) {
        auto sharedPtr = pair.second;
        if (sharedPtr.get() == rawPtr) {
            return sharedPtr;
        }
    }
    FUNCTION_LOGW("not find function ptr in function map");
    return nullptr;
}
} // namespace npu::tile_fwk
