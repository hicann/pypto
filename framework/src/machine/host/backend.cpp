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
 * \file backend.cpp
 * \brief
 */

#include "machine/host/backend.h"
#include "tilefwk/tilefwk.h"
#include "codegen/codegen.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/compile/machine_compiler.h"
#include "machine/cache_manager/cache_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/host/device_agent_task.h"
#include "machine/compile/aicore_compiler.h"
#include "machine/compile/compile_control_bin.h"
#include "tilefwk/comm_group_recorder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "tilefwk/op_registry.h"
#include "main_block.h"
#include <dlfcn.h>

using namespace npu::tile_fwk::dynamic;
namespace npu::tile_fwk {

void ForceLinkLibraryCompiler() {}

static constexpr size_t TABSIZE = 2;
constexpr int ALIGN_SIZE_8 = 8;
constexpr uint32_t STITCH_FUNCTION_MAX_SIZE = 65535;
extern "C" int32_t Initialize() {
    CacheManager::Instance().Initialize();
    return 0;
}

extern "C" bool MatchCache(const std::string &cacheKey) {
    return CacheManager::Instance().MatchBinCache(cacheKey);
}

static void InitSocVersion(std::string &socVersion) {
    socVersion = "UnknownVersion";
#ifdef BUILD_WITH_CANN
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
        return;
    }
    static constexpr uint32_t kMaxVersionLengh = 50;
    char version[kMaxVersionLengh] = {0};
    auto rtGetSocVersionFunc = (int (*)(char* version, const uint32_t maxlen))dlsym(nullptr, "rtGetSocVersion");
    auto ret = rtGetSocVersionFunc(version, kMaxVersionLengh);
    if (ret == 0) {
        socVersion = std::string(version);
    }
#endif
    ALOG_WARN_F("InitSocVersion requires BUILD_WITH_CANN.");
}

extern "C" std::string GetPlatformFile(const std::string &socVersion) {
    #ifdef PROCESSOR_SUBPATH
        const char *configSubpath = PROCESSOR_SUBPATH;
    #else
        const char *configSubpath = "";
    #endif
    const char *configRelativePath = "data/platform_config/";

    ALOG_INFO_F("Get Soc version [%s].", socVersion.c_str());
    if (socVersion.empty()) {
        return "";
    }
    // get platform file path
    const char *envPath = std::getenv("ASCEND_HOME_PATH");
    if (envPath == nullptr) {
        ALOG_WARN_F("Env[ASCEND_HOME_PATH] is not existed or empty.");
        return "";
    }
    ALOG_INFO_F("Get Env[ASCEND_HOME_PATH] is [%s].", std::string(envPath).c_str());
    std::string platformConfDir = std::string(envPath) + "/" + std::string(configSubpath) + "/" + configRelativePath;
    if (RealPath(platformConfDir).empty()) {
        platformConfDir = std::string(envPath) + "/" + configRelativePath;
    }
    ALOG_INFO_F("Get platformConfDir [%s].", platformConfDir.c_str());
    std::string platformFile = platformConfDir + socVersion + ".ini";
    ALOG_INFO_F("Get platformFile [%s].", platformFile.c_str());
    if (RealPath(platformFile).empty()) {
        return "";
    }
    return platformFile;
}

extern "C" std::string GetPlatformInfo() {
    std::string socVersion;
    ALOG_DEBUG_F("Start InitSocVersion.");
    InitSocVersion(socVersion);
#ifdef BUILD_WITH_CANN
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM) {
        ALOG_WARN("GetPlatformInfo: run in SIM mode, platform info not available.");
        return "";
    }
    ALOG_DEBUG_F("GetPlatformFile by %s.", socVersion.c_str());
    return GetPlatformFile(socVersion);
#else
    ALOG_WARN_F("GetPlatformInfo requires BUILD_WITH_CANN.");
    return "";
#endif // BUILD_WITH_CANN
}

extern "C" int32_t Execute(MachineTask *task, FunctionCache &cache) {
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) >= CS_TENSOR_GRAPH &&
        config::GetHostOption<int64_t>(COMPILE_STAGE) <= CS_EXECUTE_GRAPH) {
        ALOG_INFO("Compile stage terminates after execution graph generation.");
        return 0;
    }
    auto deviceMachineTask = std::make_shared<MachineTask>(task->GetTaskId(), task->GetFunction());
    deviceMachineTask->SetCacheReuseType(task->GetCacheReuseType());
    deviceMachineTask->SetCacheKey(task->GetCacheKey());
    auto deviceAgentTask = std::make_shared<DeviceAgentTask>(deviceMachineTask);
    auto function = deviceAgentTask->compileTask->GetFunction();
    deviceAgentTask->SetAsync(false);
    deviceAgentTask->SetOpOriginArgsInfo(function->GetOpOriginArgsInfo());
    deviceAgentTask->compileInfo.commGroups = npu::tile_fwk::Distributed::CommGroupRecorder::GetInstance().Output();
    std::string kernelPath;
    // recover task info and bin
    if (task->GetCacheReuseType() == CacheReuseType::Bin) {
        if (!CacheManager::Instance().RecoverTask(task->GetCacheKey(), deviceAgentTask.get())) {
            ALOG_WARN_F("Fail to recover task from cache[%s].", task->GetCacheKey().c_str());
            return 0;
        }
    } else {
        if (function->IsFunctionType(FunctionType::STATIC) && function->GetRootFunction()) {
            /* calc workspace size and every sub function invoke entry para offset */
            CalcFunctionInvokeWorkespace(nullptr, function, deviceAgentTask->compileInfo);
        }

        deviceAgentTask->compileInfo.workSpaceStackSize = function->GetStackWorkespaceSize();

        if (function->IsFunctionType(
                {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH})) {
            if (function->GetGraphType() == GraphType::TILE_GRAPH) {
                // When expression fusion, don't need tile graph codegen.
                return 0;
            }
        }
        (void)GenCode(deviceAgentTask->compileTask.get(), deviceAgentTask->compileInfo.invokeParaOffset, cache, kernelPath);
        function = deviceAgentTask->compileTask->GetFunction();
        /* finish compile add function cache */
        cache.Insert(function->GetFunctionHash(), *function);
        deviceAgentTask->SetFunctionCache(cache.Get(function->GetFunctionHash()));
        if (function->IsFunctionType(FunctionType::STATIC)) {
            deviceAgentTask->Validate();
            deviceAgentTask->UpdateCompileInfo();
        }
        // save compile result on disk
        CacheManager::Instance().SaveTaskFile(deviceAgentTask.get());
    }
    if (function->IsFunctionType(FunctionType::STATIC)) {
        gDeviceAgentTaskPtr = deviceAgentTask;
    }
    return 0;
}

static std::vector<Function *> GetCalleeList(FunctionCache &cache, Function *func) {
    std::vector<Function *> calleeList;

    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList = func->GetCallopAttrList();
    for (auto &callopAttr : callopAttrList) {
        auto hash = callopAttr->GetCalleeHash();
        Function *cacheFunction = cache.GetCacheFunction(hash);
        if (cacheFunction != nullptr) {
            calleeList.push_back(cacheFunction);
        } else {
            ALOG_ERROR_F("Cannot find cache %lu", hash.GetHash());
        }
    }
    return calleeList;
}

static void HandleExecuteGraph(FunctionCache &cache, Linker &linker, Function *func);
static void FindAllExpression(FunctionCache &cache, Linker &linker, Function *func) {
    if (func->IsDynloop()) {
        auto dynloopAttr = func->GetDynloopAttribute();
        auto ss = SymbolicScalar(dynloopAttr->iterSymbolName);
        linker.AddSymbol(ss);
    }
    if (func->IsFunctionTypeAndGraphType({FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        ALOG_INFO("Compile control:", func->Dump());
        for (auto &callee : GetCalleeList(cache, func)) {
            FindAllExpression(cache, linker, callee);
        }
        if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
            auto attr = func->GetDynloopAttribute();
            linker.AddPrimaryExpressionForLoopBes(func, attr->Begin());
            linker.AddPrimaryExpressionForLoopBes(func, attr->End());
            linker.AddPrimaryExpressionForLoopBes(func, attr->Step());

            for (const DynloopFunctionPath &path : attr->GetPathList()) {
                Function *loopPath = path.GetRoot();
                for (auto &cond : path.GetPathCondList()) {
                    linker.AddPrimaryExpressionForLoopPathCond(loopPath, cond.GetCond());
                }
            }
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        ALOG_INFO("Compile tile:", func->Dump());
        Function *root = func->GetRootFunction();
        FindAllExpression(cache, linker, root);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        HandleExecuteGraph(cache, linker, func);
    } else if (func->GetGraphType() == GraphType::BLOCK_GRAPH) {
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VEC_DUP) {
                if (op.HasAttr(OpAttributeKey::dynScalar)) {
                    auto dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                    linker.AddPrimaryExpressionForDevLeafOp(func, &op, dynScalar);
                }
            }
        }
    } else {
        ASSERT(false) << "Impossible function type: " << GetFunctionTypeNameDict().Find(func->GetFunctionType());
    }
}

static void HandleExecuteGraph(FunctionCache &cache, Linker &linker, Function *func)
{
    ALOG_INFO("Compile root:", func->Dump());
    MainBlockCondBulider builder;
    builder.CollectCallopMainBlockConds(func);
    for (auto &callopAttr : func->GetCallopAttrList()) {
        for (auto &arg : callopAttr->GetLinearArgList()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
        auto hash = callopAttr->GetCalleeHash();
        Function *leafFunc = cache.GetCacheFunction(hash);
        if (leafFunc == nullptr) {
            continue;
        }
        builder.CollectCoaMainBlockConds(callopAttr->GetArgList());
        FindAllExpression(cache, linker, leafFunc);
    }
    for (auto &incast : func->inCasts_) {
        for (auto  &arg : incast->GetRawTensor()->GetDynRawShape()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
    }
    for (auto &outcast : func->outCasts_) {
        for (auto  &arg : outcast->GetRawTensor()->GetDynRawShape()) {
            linker.AddPrimaryExpressionForDevRootCoa(func, arg);
        }
    }
    SymbolicScalar cond = builder.BuildMainBlockExpression();
    linker.SetMainBlockExpressionForDevRootCoa(func, cond);
}

static void AlignUpTo(std::vector<uint8_t> &code, int align, uint8_t padding) {
    while (code.size() % align != 0) {
        code.push_back(padding);
    }
}

static void ReplaceSlotIndex(DyndevFunctionAttribute *attr, std::vector<bool>& slotUsed,
                             std::unordered_map<int, int>& slotIdxMapping) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
    for (int i = 0; i < inoutLink.totalSlot; i++) {
        if (slotUsed[i] && !slotIdxMapping.count(i)) {
            slotIdxMapping.emplace(i, slotIdxMapping.size());
        }
    }

    auto replaceSlotIdx = [&slotIdxMapping](std::vector<int> &slots) {
        for (int &slot : slots) {
            slot = slotIdxMapping.count(slot) ? slotIdxMapping[slot] : -1;
        }
        slots.erase(std::remove(slots.begin(), slots.end(), -1), slots.end());
    };

    inoutLink.totalSlot = slotIdxMapping.size();
    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile))<<"Function pointer "<<devTile->GetMagicName()<<" not found in ioslotDict";
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];

        for (auto &incastSlots : ioslot.incastSlot) {
            replaceSlotIdx(incastSlots);
        }

        for (auto &outcastSlots : ioslot.outcastSlot) {
            replaceSlotIdx(outcastSlots);
        }
    }

    replaceSlotIdx(inoutLink.inputSlotIndexList);
    replaceSlotIdx(inoutLink.outputSlotIndexList);
    replaceSlotIdx(inoutLink.assembleSlotIndexList);
    replaceSlotIdx(inoutLink.shmemTensorSlotIndexList);
    replaceSlotIdx(inoutLink.partialUpdateSlotIdexList);
    for (auto &slot : inoutLink.inplaceSlotIndexList) {
        if (slot != -1)
            slot = slotIdxMapping[slot];
    }

    auto replaceSlotIdxForFunc = [&slotIdxMapping, replaceSlotIdx](Function *func) {
        std::shared_ptr<TensorSlotScope> scope = func->GetSlotScope();
        if (scope) {
            replaceSlotIdx(scope->constructAssembleSlotList);
        }
    };
    for (auto loopPathFunc : attr->funcGroup.loopPathList) {
        replaceSlotIdxForFunc(loopPathFunc);
    }

    inoutLink.UpdateRuntimeSlotKindSetList();
}

static void MarkUsedSlotsFromInoutLink(const IncastOutcastLink &inoutLink, std::vector<bool> &slotUsed) {
    for (int slotIdx : inoutLink.inputSlotIndexList) {
        slotUsed[slotIdx] = true;
    }
    for (int slotIdx : inoutLink.outputSlotIndexList) {
        slotUsed[slotIdx] = true;
    }
    for (int slotIdx : inoutLink.shmemTensorSlotIndexList) {
        slotUsed[slotIdx] = true;
    }
    for (int slotIdx : inoutLink.assembleSlotIndexList) {
        slotUsed[slotIdx] = true;
    }
    // partialUpdateSlotIdexList的数据有问题
}

static void SimplifySlots(DyndevFunctionAttribute *attr, std::unordered_map<int, int>& slotIdxMapping) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
    std::vector<bool> slotUsed(inoutLink.totalSlot);

    MarkUsedSlotsFromInoutLink(inoutLink, slotUsed);
    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile))<<"Function pointer "<<devTile->GetMagicName()<<" not found in ioslotDict";
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];

        for (auto &incastSlots : ioslot.incastSlot) {
            if (incastSlots.empty()) {
                ALOG_WARN("devTile: " + devTile->GetMagicName());
                continue;
            }
            int32_t simplifiedIncastSlot = -1;
            for (auto &incastSlot : incastSlots) {
                if (slotUsed[incastSlot]) {
                    simplifiedIncastSlot = incastSlot;
                    break;
                }
            }
            if (simplifiedIncastSlot != -1) {
                incastSlots.front() = simplifiedIncastSlot;
            }
            incastSlots.resize(1); // meaningless to maintain multi incast slots
            slotUsed[incastSlots.front()] = true;
        }
    }

    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile))<<"Function pointer "<<devTile->GetMagicName()<<" not found in ioslotDict";
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];
        for (auto &outcastSlots : ioslot.outcastSlot) {
            ASSERT(!outcastSlots.empty()) << "devTile: " << devTile->GetMagicName();
            bool outcastSlotFound = false;
            for (auto &outcastSlot : outcastSlots) {
                outcastSlotFound = outcastSlotFound || slotUsed[outcastSlot];
            }
            if (!outcastSlotFound) {
                slotUsed[outcastSlots.front()] = true;
            }
        }
    }

    ReplaceSlotIndex(attr, slotUsed, slotIdxMapping);
}

static void BuildSlotRootIncastOutcastDict(DyndevFunctionAttribute *attr) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
    for (size_t idx = 0; idx < attr->funcGroup.devRootList.size(); idx++) {
        Function *devRoot = attr->funcGroup.devRootList[idx];
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile))<<"Function pointer "<<devTile->GetMagicName()<<" not found in ioslotDict";
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];
        for (size_t incastIndex = 0; incastIndex < ioslot.incastSlot.size(); incastIndex++) {
            for (auto &slotIndex : ioslot.incastSlot[incastIndex]) {
                attr->slotRootIncastDict[slotIndex][devRoot] = incastIndex;
            }
        }
        for (size_t outcastIndex = 0; outcastIndex < ioslot.outcastSlot.size(); outcastIndex++) {
            for (auto &slotIndex : ioslot.outcastSlot[outcastIndex]) {
                attr->slotRootOutcastDict[slotIndex][devRoot] = outcastIndex;
            }
        }
    }
}

static void BuildRootFuncKeyDict(DyndevFunctionAttribute *attr) {
    for (size_t idx = 0; idx < attr->funcGroup.devRootList.size(); idx++) {
        int funcKey = (int)idx;
        Function *devRoot = attr->funcGroup.devRootList[idx];
        attr->rootFuncKeyDict[devRoot] = funcKey;
    }
}

static std::string BuildControlFlowCallee(Function *func, int ident) {
    std::ostringstream oss;
    auto loc = func->GetSourceLocation();
    if (loc) {
        oss << std::string(ident, ' ') << "// " << loc->ToString() << "\n";
    }
    oss << std::string(ident, ' ') << "// " << "#name: " << func->GetRawName() << " #hash: " << func->GetFunctionHash()
        << " #magic: " << func->GetFuncMagic() << "\n";
    return oss.str();
}

static void BuildControlFlow(FunctionCache &cache, Linker &linker, const std::string &sectionName,
    Function *func,
    std::unordered_map<int, int> &slotIdxMapping,
    DyndevFunctionAttribute::FunctionGroup &group,
    std::unordered_map<Function *, Function *> &rootTileDict,
    std::ostringstream &controlFlowOss,
    std::ostringstream &expressionOss,
    int indent, const std::string &expName) {
    auto funcType = func->GetFunctionType();
        if (funcType == FunctionType::DYNAMIC) {
        controlFlowOss
            << "#define __TILE_FWK_AICPU__ 1\n"
            << "#include <stdint.h>\n"
            << "#include \"" << expName << "\"\n"
            << "#include \"tilefwk/aikernel_data.h\"\n"
            << "#include \"tilefwk/aicpu_runtime.h\"\n"
            << "#include \"tilefwk/aicpu_distributed.h\"\n";
        expressionOss
            << "\n/* Symbol table list */\n"
            << linker.GetSymbolTable()->BuildSymbolList();
        const std::vector<std::string> &inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
        const std::vector<std::string> &outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();

        expressionOss << "\n/* Input tensor list */\n";
        for (size_t idx = 0; idx < inputNameList.size(); idx++) {
            expressionOss << "#define " << AddArgPrefix(inputNameList[idx]) << " " << idx << "\n";
        }

        expressionOss << "\n/* Output tensor list */\n";
        for (size_t idx = 0; idx < outputNameList.size(); idx++) {
            expressionOss << "#define " << AddArgPrefix(outputNameList[idx]) << " " << idx + inputNameList.size() << "\n";
        }

        controlFlowOss << "#define LOOP(idx, b, e, s) for (int64_t idx = (b), idxEnd = (e), idxStep = (s); idx < idxEnd; idx += idxStep)\n"
            << "namespace npu::tile_fwk {\n"
            << BuildControlFlowCallee(func, 0)
            << "__attribute__((section(\"" << sectionName
            << "\")))\n"
            << "uint64_t ControlFlowEntry(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs) {\n";
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, slotIdxMapping, group, rootTileDict, controlFlowOss, expressionOss, indent + 1, expName);
        }
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_FINISH); // Notify finish \n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
        controlFlowOss << "}\n";
        controlFlowOss << "} // namespace npu::tile_fwk\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
        std::function<void(const std::shared_ptr<DynloopFunctionPathNode> &, int)> condBuilder =
            [&cache, &linker, &sectionName, &slotIdxMapping, &group, &rootTileDict, &controlFlowOss, &expressionOss, &condBuilder,
             &expName] (const std::shared_ptr<DynloopFunctionPathNode> &node, int condIndent) {
                if (!node->cond.IsValid()) {
                    BuildControlFlow(cache, linker, sectionName, node->root, slotIdxMapping, group, rootTileDict, controlFlowOss, expressionOss, condIndent, expName);
                } else {
                    std::string cond = SymbolicExpressionTable::BuildExpression(node->cond);
                    if (node->branchNodeList[1] != nullptr) {
                        if (node->branchNodeList[0] != nullptr) {
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "if (" << cond << ") {" << "\n";
                            condBuilder(node->branchNodeList[1], condIndent + 1);
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "} else {" << "\n";
                            condBuilder(node->branchNodeList[0], condIndent + 1);
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "}" << "\n";
                        } else {
                            condBuilder(node->branchNodeList[1], condIndent);
                        }
                    } else {
                        if (node->branchNodeList[0] != nullptr) {
                            condBuilder(node->branchNodeList[0], condIndent);
                        } else {
                            ASSERT(false) << "Both conds is nullptr!";
                        }
                    }
                }
            };
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << func->GetFunctionHash() << "\n";
        auto attr = func->GetDynloopAttribute();
        ASSERT(attr != nullptr)<<"attr is nullptr!";
        if (attr->submitBeforeLoop) {
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_LOOP_BARRIER); // force submit before LOOP \n";
        }

        auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
        if (currDynFuncAttr->valueDependDescDict.count(func)) {
            auto valueDependDesc = currDynFuncAttr->valueDependDescDict[func];
            if (valueDependDesc.getInputDataCount + valueDependDesc.getTensorDataCount != 0) {
                controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_CACHESTOP); // force stop cache due to value depend in control\n";
            }
        }

        std::string iterBegin = SymbolicExpressionTable::BuildExpression(attr->Begin());
        std::string iterEnd = SymbolicExpressionTable::BuildExpression(attr->End());
        std::string iterStep = SymbolicExpressionTable::BuildExpression(attr->Step());
        std::string iterVar = "VAR_" + attr->iterSymbolName;
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "LOOP(" << iterVar << ", " << iterBegin << ", " << iterEnd << ", " << iterStep << ") {\n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << attr->iterSymbolName << " = " << iterVar << ";\n";

        auto pathNode = attr->BuildPathNode();
        ALOG_INFO("Paths: \n", pathNode->Dump());
        std::vector<Function *> calleeList = GetCalleeList(cache, func);
        std::sort(calleeList.begin(), calleeList.end());

        std::vector<Function *> pathRootList;
        for (size_t i = 0; i < attr->pathList.size(); i++) {
            pathRootList.push_back(attr->pathList[i].root);
        }
        std::sort(pathRootList.begin(), pathRootList.end());
        ASSERT(calleeList == pathRootList)<<"calleeList size:"<<calleeList.size()<<" pathRootList size:"<<pathRootList.size();
        condBuilder(pathNode, indent + 1);
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "}\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH)) {
        controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
        auto scope = func->GetSlotScope();
        for (auto slot : scope->constructAssembleSlotList) {
            if (!slotIdxMapping.count(slot)) {
                slotIdxMapping.emplace(slot, slotIdxMapping.size());
            }
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_SlotMarkNeedAlloc(" << slotIdxMapping.at(slot) << ");\n";
        }
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, slotIdxMapping, group, rootTileDict, controlFlowOss, expressionOss, indent + 1, expName);
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
        Function *root = func->GetRootFunction();
        rootTileDict[root] = func;
        BuildControlFlow(cache, linker, sectionName, root, slotIdxMapping, group, rootTileDict, controlFlowOss, expressionOss, indent, expName);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        if (group.devRootList.count(func) <= 0) {
            return;
        }

        auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
        ASSERT(rootTileDict.count(func))<<"Function not found in rootTileDict";
        Function *tile = rootTileDict[func];
        if (currDynFuncAttr->valueDependDescDict.count(tile)) {
            auto valueDependDesc = currDynFuncAttr->valueDependDescDict[tile];
            if (valueDependDesc.getInputDataCount + valueDependDesc.getTensorDataCount != 0) {
                controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_CACHESTOP); // force stop cache due to value depend in data\n";
            }
        }

        int devRootKey = group.devRootList.GetIndex(func);
        controlFlowOss << BuildControlFlowCallee(func, indent * TABSIZE);
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "uint64_t *exprList" << devRootKey << " = (uint64_t *)RUNTIME_RootAlloc(" << devRootKey << "ULL);\n";

        SymbolicExpressionTable *exprTable = linker.LookupDevRootCoa(func);
        if (exprTable != nullptr) {
            for (auto &expr : exprTable->GetPrimaryExpressionSet()) {
                auto index = exprTable->GetPrimaryExpressionSet().GetIndex(expr);
                auto exprStr = exprTable->BuildExpression(expr);
                controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_SetExpr(exprList" << devRootKey << ", " << index << ", " << exprStr << ");\n";
            }
        }
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootStitch(" << devRootKey << "ULL);\n";
    } else {
        ASSERT(false) << "Impossible function type: " << GetFunctionTypeNameDict().Find(funcType);
    }
}

static std::string Arm64TargetTool(const std::string &bin) {
    const char *homePath = std::getenv("ASCEND_HOME_PATH");
    if (homePath == nullptr) {
        return "";
    }
    // use toolchain from CANN for better compatibility as the controlflow will run on aicpu
    return std::string(homePath) + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-" + bin;
}

static void FillL2PrefetchInfo(std::shared_ptr<DyndevFunctionAttribute> attr) {
    uint64_t idx = 0;
    for (auto &param : attr->startArgsInputTensorList) {
        const auto &tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor == nullptr) {
          idx++;
          attr->disableL2List.emplace_back(0);
          continue;
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::PREFETCH)) {
          attr->l2InfoList.emplace_back(L2Info(tensor.GetStorage()->MemorySize(), idx));
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
          attr->disableL2List.emplace_back(1);
        } else {
          attr->disableL2List.emplace_back(0);
        }
        idx++;
    }
    for (auto &param : attr->startArgsOutputTensorList) {
        const auto &tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor == nullptr) {
          idx++;
          attr->disableL2List.emplace_back(0);
          continue;
        }
        if (tensor.GetStorage()->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
          attr->disableL2List.emplace_back(1);
        } else {
          attr->disableL2List.emplace_back(0);
        }
        idx++;
    }
    ALOG_INFO_F("Need prefetch tensor size is:%zu\n", attr->l2InfoList.size());
    return;
}

static void SetDyndevProgBinary(Function *function) {
    if (function == nullptr || function->GetDyndevAttribute() == nullptr) {
        return;
    }
    std::shared_ptr<DyndevFunctionAttribute> dynAttrPtr = function->GetDyndevAttribute();
    uint64_t size = 0;
    dynamic::EncodeDevAscendProgram(function, size, nullptr);
    dynAttrPtr->devProgBinary.resize(size);

    dynamic::DevAscendProgram *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(&dynAttrPtr->devProgBinary[0]);
    dynamic::EncodeDevAscendProgram(function, size, devProg);

    if (config::GetPassDefaultConfig(npu::tile_fwk::KEY_PRINT_PROGRAM, false)) {
        devProg->DumpFile(config::LogTopFolder() + "/program.tifwkbintxt");
        std::string loopDirPath = config::LogTopFolder() + "/loop";
        CreateMultiLevelDir(loopDirPath);
        for (size_t index = 0; index < dynAttrPtr->funcGroup.loopList.size(); index++) {
            Function *func = dynAttrPtr->funcGroup.loopList[index];
            func->DumpFile(loopDirPath + "/" + func->GetMagicName() + ".tifwkgr");
        }
    }
    devProg->RelocProgram(reinterpret_cast<int64_t>(devProg), 0);
    if (config::GetPassDefaultConfig(npu::tile_fwk::KEY_PRINT_PROGRAM, false)) {
        SaveFile(config::LogTopFolder() + "/program.tifwkbin", dynAttrPtr->devProgBinary);
    }
    ALOG_INFO_F("Dev prog binary size is:%zu\n", dynAttrPtr->devProgBinary.size());
}

std::vector<SymbolicExpressionTable *> GetAllExpressionTable(DyndevFunctionAttribute::ExpressionTableDictGroup &exprTableGroup) {
    std::vector<SymbolicExpressionTable *> exprTableList;
    for (auto &[func, exprTable] : exprTableGroup.loopBesDict)  {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto &[func, ifDict] : exprTableGroup.loopPathCondDict) {
        (void)func;
        for (auto &[expr, exprTable] : ifDict) {
            (void)expr;
            exprTableList.push_back(&exprTable);
        }
    }
    for (auto &[func, exprTable] : exprTableGroup.devRootCoaDict) {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto &[func, opDict] : exprTableGroup.devLeafOpDict) {
        (void)func;
        for (auto &[op, exprTable] : opDict) {
            (void)op;
            exprTableList.push_back(&exprTable);
        }
    }
    return exprTableList;
}

static void ConstructCodeInfo(struct EncodeDevAscendFunctionParam &encodeDevAscendFunctionParam,
    std::map<uint64_t, Function *> &leafDict,
     std::shared_ptr<DyndevFunctionAttribute> attr) {
    attr->cceCodeInfo.resize(leafDict.size() + 1);
    /* cceIdx 0 for dummy callop */
    attr->cceCodeInfo[0].coreType = static_cast<uint32_t>(CoreType::HUB);
    attr->cceCodeInfo[0].psgId = 0;
    attr->cceCodeInfo[0].funcHash = 0;
    encodeDevAscendFunctionParam.calleeHashIndexDict[0] = 0;

    int leafIndex = 1;
    for (auto &[hash, leaf] : leafDict) {
      auto leafFuncAttr = leaf->GetLeafFuncAttribute();
      ASSERT(leafFuncAttr != nullptr)<<"leafFuncAttr is null\n";
      encodeDevAscendFunctionParam.calleeHashIndexDict[hash] = leafIndex;
      attr->devLeafIndex2Hash[leafIndex] = hash;
      ALOG_INFO("Dyndev.codegen: [", leafIndex, "] hash=", hash, " binpath=", leafFuncAttr->binPath);
      attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(leafFuncAttr->coreType);
      if (leaf->IsDummyFunction())
        attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(CoreType::HUB);
      attr->cceCodeInfo[leafIndex].psgId = leaf->GetProgramId();
      attr->cceCodeInfo[leafIndex].funcHash = hash;
      attr->cceCodeInfo[leafIndex].aicpuLeafCode = leafFuncAttr->aicpuLeafCode;
      attr->cceCodeInfo[leafIndex].wrapVecId = static_cast<int32_t>(leafFuncAttr->aivCore);
      attr->cceCodeInfo[leafIndex].mixResourceType = static_cast<uint32_t>(leafFuncAttr->mixResourceType);
      leafIndex++;
    }
    encodeDevAscendFunctionParam.cceCodeInfoList = attr->cceCodeInfo;
    return;
}

static void EncodeOutcastProperty(
        EncodeDevAscendFunctionParam &encodeDevAscendFunctionParam,
        const IncastOutcastLink *inoutLink,
        const IncastOutcastSlot *slot) {
    encodeDevAscendFunctionParam.outcastDescList.clear();
    encodeDevAscendFunctionParam.assembleSlotList.clear();
    Function *devRoot = encodeDevAscendFunctionParam.devRoot;

    std::unordered_map<std::shared_ptr<RawTensor>, int> incastDict;
    for (size_t incastIndex = 0; incastIndex < devRoot->GetIncast().size(); incastIndex++) {
        incastDict[devRoot->GetIncast()[incastIndex]->GetRawTensor()] = incastIndex;
    }

    std::vector<RuntimeSlotKindSet> outcastSlotKindSetList(slot->outcastSlot.size());
    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        for (auto &slotIndex : slot->outcastSlot[outcastIndex]) {
            outcastSlotKindSetList[outcastIndex] = outcastSlotKindSetList[outcastIndex] | inoutLink->runtimeSlotKindSetList[slotIndex];
        }
    }
    encodeDevAscendFunctionParam.outcastDescList.resize(slot->outcastSlot.size());
    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        RuntimeSlotDesc &desc = encodeDevAscendFunctionParam.outcastDescList[outcastIndex];
        if (outcastSlotKindSetList[outcastIndex].Count(RuntimeSlotKind::INPUT)) {
            desc.kind = RuntimeSlotKind::INPUT;
        } else if (outcastSlotKindSetList[outcastIndex].Count(RuntimeSlotKind::OUTPUT)) {
            desc.kind = RuntimeSlotKind::OUTPUT;
        } else if (outcastSlotKindSetList[outcastIndex].Count(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            desc.kind = RuntimeSlotKind::ASSEMBLE_OUTCAST;
        } else {
            int incastIndex = -1;
            auto outcastRawTensor = devRoot->GetOutcast()[outcastIndex]->GetRawTensor();
            if (devRoot->outIncastLinkMap.count(outcastRawTensor)) {
                auto incastRawTensor = devRoot->outIncastLinkMap[outcastRawTensor];
                incastIndex = incastDict[incastRawTensor];
            }
            if (incastIndex != -1) {
                desc.kind = RuntimeSlotKind::INPLACE_INCAST;
                desc.inplaceIncastIndex = incastIndex;
            } else {
                desc.kind = RuntimeSlotKind::EXCLUSIVE_OUTCAST;
            }
        }
    }

    for (size_t outcastIndex = 0; outcastIndex < slot->outcastSlot.size(); outcastIndex++) {
        if (encodeDevAscendFunctionParam.outcastDescList[outcastIndex].kind == RuntimeSlotKind::ASSEMBLE_OUTCAST) {
            for (auto &slotIndex : slot->outcastSlot[outcastIndex]) {
                encodeDevAscendFunctionParam.assembleSlotList.push_back(slotIndex);
            }
        }
    }
}

static bool IsNeedDumpAicpuKernel(const std::string &inputFile) {
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_FORCE_OVERWRITE, true)) {
        // force dump, default is true
        return true;
    }
    // not force dump
    if (npu::tile_fwk::FileExist(inputFile)) {
        return false;
    }
    return true;
}
static void OverCallOpMaxNum(Function *devRoot, DevAscendFunction *funcBin){
    uint32_t CallOpSize = funcBin->GetOperationSize();
    uint32_t CallOpmaxSize = config::GetRuntimeOption<uint32_t>(STITCH_FUNCTION_SIZE);
    auto funcMagicName = devRoot->GetRawName() + "_" + std::to_string(devRoot->GetFuncMagic());
    ALOG_ERROR_F("the loop function operation: %s size is %u hitting the maxinum single-loop-operation limit:%u.\n",
    funcMagicName.c_str(), CallOpSize, CallOpmaxSize);
    ASSERT(CallOpSize <= CallOpmaxSize) << " loopFunction: " << funcMagicName << " CallOpSize: " << CallOpSize
    << " CallOpmaxSize: " << CallOpmaxSize;
}

static void CompileControlFlow(const std::string &aicpuDirPath,
                               const std::string &funcName, const std::string &constrolFlow, std::string express) {
    std::string controlFlowCompilepath = aicpuDirPath + "/" + funcName + "/aicpu";
    ALOG_DEBUG_F("Dumpath is %s, functionName %s, path is %s",
                 aicpuDirPath.c_str(), funcName.c_str(), controlFlowCompilepath.c_str());
    if (!CreateMultiLevelDir(controlFlowCompilepath)) {
        ALOG_ERROR_F("Creat AicpuCompile dir not success\n");
        return;
    }
    std::string controlFlowFileName = controlFlowCompilepath + "/controlFlow_dev" + funcName + ".h";
    std::string expressFileName = controlFlowCompilepath + "/expression_0.h";
    if (!DumpFile(constrolFlow, controlFlowFileName) || !DumpFile(express, expressFileName)) {
        ALOG_DEBUG_F("Dump controlFlow and express files failed\n");
        return;
    }
#ifdef BUILD_WITH_CANN
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM) {
        if (std::getenv("ASCEND_HOME_PATH") != nullptr) {
            ASSERT(TileFwkAiCpuCompile(funcName, aicpuDirPath)) << ": PyPto Control Flow compile failed";
        }
    }
#endif
}

static void CompileDyndevFunction(Function *function, FunctionCache &cache, [[maybe_unused]] const std::string &ccePath,
                                  std::string &kernelPath) {
    ASSERT((PassManager::Instance().RunPass(Program::GetInstance(), *function, "ExecuteGraph") == SUCCESS));

    std::shared_ptr<DyndevFunctionAttribute> attr = function->GetDyndevAttribute();
    ASSERT(attr != nullptr)<<"DyndevFunctionAttribute is nullptr\n";
    Linker linker(attr->symbolTable, attr->funcGroup, attr->exprTableDictGroup);
    FindAllExpression(cache, linker, function);

    FillL2PrefetchInfo(attr);
    attr->commGroupNames = npu::tile_fwk::Distributed::CommGroupRecorder::GetInstance().Output();
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    attr->inoutLink = slotManager->BuildIncastOutcastLink(function->GetRawName());

    int idx = 0;
    for (auto name : slotManager->GetInputNameList()) {
        attr->inputSymbolDict[AddArgPrefix(name)] = idx++;
    }
    for (auto name : slotManager->GetOutputNameList()) {
        attr->inputSymbolDict[AddArgPrefix(name)] = idx++;
    }

    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;

    expressionOss << "#ifndef TILE_FWK_EXPRESSION_H" << "\n"
                  << "#define TILE_FWK_EXPRESSION_H" << "\n";
    auto &exprTableGroup = linker.GetExpressionTableDictGroup();
    std::vector<SymbolicExpressionTable *> exprTableList = GetAllExpressionTable(exprTableGroup);
    linker.GetSymbolTable()->NormalizeForSymbol();
    for (auto exprTable : exprTableList) {
        exprTable->NormalizeForSymbolTable(*linker.GetSymbolTable());
        expressionOss << exprTable->BuildExpressionList();
    }
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    const std::string expName = "expression_" + std::to_string(tilingKey) + ".h";
    std::unordered_map<int, int> slotIdxMapping;
    BuildControlFlow(cache, linker, "ast2", function, slotIdxMapping, attr->funcGroup, attr->rootTileDict, controlFlowOss,
                     expressionOss, 0, expName);
    expressionOss << "#endif/*TILE_FWK_EXPRESSION_H*/" << "\n";
    std::string controlFlowSource = controlFlowOss.str();
    std::string expressionSource = expressionOss.str();
    SimplifySlots(attr.get(), slotIdxMapping);
    BuildSlotRootIncastOutcastDict(attr.get());
    BuildRootFuncKeyDict(attr.get());

#ifdef __x86_64__
    std::string cflags = "-mno-sse2 -mno-sse";
#else
    std::string cflags = "-mgeneral-regs-only";
#endif

    std::string aicpuDirPath = GetEmitPath("kernel_aicpu");
    npu::tile_fwk::CreateMultiLevelDir(aicpuDirPath);

    std::string expressionFilePath = aicpuDirPath + "/" + expName;
    if (IsNeedDumpAicpuKernel(expressionFilePath)) {
        DumpFile(expressionSource, expressionFilePath);
    }

    std::string funcHash = function->GetFunctionHash().Data();
    std::string controlFlowHostFilePath = aicpuDirPath + "/controlFlow_host_" + funcHash + ".cpp";
    attr->hostControlFlowBinary = CompileAndLoadSection(controlFlowSource, controlFlowHostFilePath,
        "g++", "objcopy", "ast2", IsNeedDumpAicpuKernel(controlFlowHostFilePath), cflags);
    AlignUpTo(attr->hostControlFlowBinary, 0x8, 0);
    std::string funcName = function->GetMagicName() + function->GetFunctionHash().Data();
    CompileControlFlow(aicpuDirPath, funcName, controlFlowSource, expressionSource);
    std::string arm64TargetToolPath = Arm64TargetTool("g++");
    if (FileExist(arm64TargetToolPath)) {
        std::string controlFlowDevFilePath = aicpuDirPath + "/controlFlow_dev_" + funcHash + ".cpp";
        ALOG_INFO_F("Compile control flow src file[%s] with arm64 target tool[%s].",
                    controlFlowDevFilePath.c_str(), arm64TargetToolPath.c_str());
        attr->devControlFlowBinary = CompileAndLoadSection(
            controlFlowSource, controlFlowDevFilePath,
            arm64TargetToolPath, Arm64TargetTool("objcopy"), "ast2", IsNeedDumpAicpuKernel(controlFlowDevFilePath));
    } else {
        // brk #0
        ALOG_WARN_F("Arm64 target tool is not found.");
        attr->devControlFlowBinary = std::vector<uint8_t>{0xd4, 0x20, 0x00, 0x00};
    }
    AlignUpTo(attr->devControlFlowBinary, 0x8, 0);
    std::map<uint64_t, Function *> leafDict;
    for (auto &devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];
        config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, devTile->paramConfigs_.dynamicAlignedOps);
        npu::tile_fwk::CodeGenCtx codeGenCtx("", GetEmitPath("kernel_aicore"));
        npu::tile_fwk::CodeGen codeGen(codeGenCtx);
        codeGen.GenCode(*devTile, {});
        MainBlockCondBulider builder;
        builder.Gencode(devTile, {});
        for (auto &[psgId, leaf] : devRoot->programs_) {
            (void)psgId;
            auto hash = leaf->GetFunctionHash().GetHash();
            if (!leafDict.count(hash)) {
                leafDict[hash] = leaf;
                ALOG_INFO("Dyndev.codegen: ", leaf->GetRawName());
            } else {
                ALOG_ERROR(" Duplicate func hash ", hash, " name ", leaf->GetRawName());
            }
        }
    }

    struct EncodeDevAscendFunctionParam encodeDevAscendFunctionParam = {};
    ConstructCodeInfo(encodeDevAscendFunctionParam, leafDict, attr);

    encodeDevAscendFunctionParam.inoutLink = &attr->inoutLink;

#ifdef BUILD_WITH_CANN
    if (config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) != CFG_RUN_MODE_SIM &&
        config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_CODEGEN_INSTRUCTION) {

        int ret = CompileAICoreKernel(leafDict, encodeDevAscendFunctionParam,
                                    ccePath, function->GetFunctionHash().Data(), kernelPath);
        if (ret != 0) {
            ALOG_ERROR_F("Compile dynamic aicore.o failed.");
            return;
        }
    }
#endif

    attr->kernelBinary = LoadFile(kernelPath);
    ALOG_DEBUG_F("KernelBinary size[%zu].", attr->kernelBinary.size());

    attr->devEncodeList.resize(attr->funcGroup.devRootList.size());
    for (auto &devRoot : attr->funcGroup.devRootList) {
        int devRootKey = attr->funcGroup.devRootList.GetIndex(devRoot);
        ALOG_INFO("Dyndev.encode: ", devRoot->GetRawName());
        ASSERT(attr->rootTileDict.count(devRoot))<<"devRoot not found in rootTileDict";
        Function *devTile = attr->rootTileDict[devRoot];
        ASSERT(attr->inoutLink.ioslotDict.count(devTile))<<"devTile not found in rootTileDict";
        IncastOutcastSlot *slot = &attr->inoutLink.ioslotDict[devTile];
        encodeDevAscendFunctionParam.symbolTable = linker.GetSymbolTable();
        if (linker.GetExpressionTableDictGroup().devRootCoaDict.count(devRoot) != 0) {
            encodeDevAscendFunctionParam.expressionTable = &linker.GetExpressionTableDictGroup().devRootCoaDict.find(devRoot)->second;
        }
        encodeDevAscendFunctionParam.devRoot = devRoot;
        encodeDevAscendFunctionParam.slot = slot;
        EncodeOutcastProperty(encodeDevAscendFunctionParam, &attr->inoutLink, slot);
        uint64_t size = 0;
        EncodeDevAscendFunction(function, encodeDevAscendFunctionParam, size, nullptr);
        attr->devEncodeList[devRootKey].resize(size);
        DevAscendFunction *funcBin = reinterpret_cast<DevAscendFunction *>(&attr->devEncodeList[devRootKey][0]);
        funcBin->rootHash = devRoot->GetFunctionHash().GetHash();
        funcBin->funcKey = devRootKey;
        funcBin->stackWorkSpaceSize = devTile->GetStackWorkespaceSize();
        funcBin->getInputDataCount = 0;
        funcBin->getTensorDataCount = 0;
        EncodeDevAscendFunction(function, encodeDevAscendFunctionParam, size, funcBin);
        funcBin->Reloc(-reinterpret_cast<int64_t>(funcBin), true);
        uint32_t CallOpmaxSize = config::GetRuntimeOption<uint32_t>(STITCH_FUNCTION_SIZE);
        ASSERT(CallOpmaxSize <= STITCH_FUNCTION_MAX_SIZE) << " CallOpmaxSize set: "<< CallOpmaxSize
        << "exceeds the maximum allowed value of 65535.";
        if (funcBin->GetOperationSize() > CallOpmaxSize) {
            OverCallOpMaxNum(devRoot,funcBin);
        }
    }

    for (size_t index = 0; index < attr->symbolTable.GetSymbolTable().size(); index++) {
        std::string name = attr->symbolTable.GetSymbolTable()[index];
        if (symbolHandlerIndexDict.count(name)) {
            attr->startArgsSymbolHandlerList.emplace_back(symbolHandlerIndexDict.find(name)->second, index);
        }
    }
    // save dev prog binary
    SetDyndevProgBinary(function);
}

MachineTask *GenCode(
    MachineTask *task, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset, FunctionCache &cache,
    std::string &kernelPath) {
    npu::tile_fwk::CodeGenCtx codeGenCtx("", GetEmitPath("kernel_aicore"));
    npu::tile_fwk::CreateMultiLevelDir(codeGenCtx.cceDir);
    npu::tile_fwk::CodeGen codeGen(codeGenCtx);
    auto function = task->GetFunction();
    /* each leafFunction inside is compiled to a standalone object file.
     * the filepath of the object file is updated to the binPath_ member.
     */
    if (function->GetGraphType() == GraphType::TILE_GRAPH) {
        codeGen.GenCode(*function, invokeParaOffset);
        MainBlockCondBulider builder;
        builder.Gencode(function, invokeParaOffset);
    } else {
        if (function->IsFunctionType(FunctionType::DYNAMIC)) {
            std::string cce_path = RealPath(codeGenCtx.cceDir) + "/";
            CompileDyndevFunction(function, cache, cce_path, kernelPath);
        }
    }

    return task;
}
} // namespace npu::tile_fwk
