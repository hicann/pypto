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
 * \file pass_manager.cpp
 * \brief
 */

#include "pass_manager.h"

#include <cstdlib>
#include <unistd.h>
#include "interface/configs/config_manager.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_interface/pass_type.h"
#include "pass_registry.h"
#include "interface/tensor/expected_value.h"
#include "tilefwk/error.h"
#include "tilefwk/platform.h"
#include "pass_dependency.h"
// tensor graph pass
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"
#include "passes/tensor_graph_pass/auto_cast.h"
#include "passes/tensor_graph_pass/infer_memory_conflict.h"
#include "passes/tensor_graph_pass/remove_undriven_view.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "passes/tensor_graph_pass/loop_unroll.h"
//  tile graph pass
#include "passes/tile_graph_pass/graph_partition/graph_partition.h"
#include "passes/tile_graph_pass/graph_optimization/graph_optimization.h"
#include "passes/tile_graph_pass/graph_constraint/graph_constraint.h"
#include "passes/tile_graph_pass/data_path/data_path.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
// execute graph pass
#include "passes/block_graph_pass/memory_reuse/memory_reuse.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/block_graph_pass/schedule_ooo/schedule.h"
#include "passes/block_graph_pass/codegen_preproc.h"
#include "passes/block_graph_pass/infer_param_index.h"
#include "passes/block_graph_pass/copy_out_resolve.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "passes/block_graph_pass/mix_subgraph_split.h"

namespace npu::tile_fwk {
PassManager &PassManager::Instance() {
    static PassManager instance;
    return instance;
}

void RegPass() {
    REG_PASS(GlobalMemoryReuse);
    REG_PASS(SubgraphToFunction);
    REG_PASS(GraphPartition);
    REG_PASS(ReduceCopyMerge);
    REG_PASS(InsertSync);
    REG_PASS(OoOSchedule);
    REG_PASS(RemoveUndrivenView);
    REG_PASS(ExpandFunction);
    REG_PASS(CommonOperationEliminate);
    REG_PASS(GenerateMoveOp);
    REG_PASS(AssignMemoryType);
    REG_PASS(RemoveRedundantReshape);
    REG_PASS(AutoCast);
    REG_PASS(InferMemoryConflict);
    REG_PASS(NBufferMerge);
    REG_PASS(L1CopyInReuseMerge);
    REG_PASS(MergeViewAssemble);
    REG_PASS(IntraSubgraphAdapter);
    REG_PASS(PadLocalBuffer);
    REG_PASS(ReplaceTensor);
    REG_PASS(PreGraphProcess);
    REG_PASS(RemoveRedundantOp);
    REG_PASS(SplitRawTensor);
    REG_PASS(SplitReshape);
    REG_PASS(RemoveUnalignedReshape);
    REG_PASS(CodegenPreproc);
    REG_PASS(SplitLargeFanoutTensor);
    REG_PASS(SplitK);
    REG_PASS(InferDynShape);
    REG_PASS(InferParamIndex);
    REG_PASS(AddAlloc);
    REG_PASS(RemoveAlloc);
    REG_PASS(CopyOutResolve);
    REG_PASS(SrcDstBufferMerge);
    REG_PASS(LoopUnroll);
    REG_PASS(DynAttrToStatic);
    REG_PASS(InferDiscontinuousInput);
    REG_PASS(MixSubgraphSplit);
    REG_PASS(DuplicateOp);
    REG_PASS(AxisCombine);
}

void PassManager::RegDefaultStrategy() {
    RegisterStrategy(
        "PVC2_OOO", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
            {                 "AutoCast",                 "AutoCast"},
            {      "InferMemoryConflict",      "InferMemoryConflict"},
            {       "RemoveUndrivenView",       "RemoveUndrivenView"},
            {           "ExpandFunction",           "ExpandFunction"},
            {        "MergeViewAssemble",        "MergeViewAssemble"},
            {             "SplitReshape",             "SplitReshape"},
            {           "SplitRawTensor",           "SplitRawTensor"},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
            {              "DuplicateOp",              "DuplicateOp"},
            {         "AssignMemoryType",         "AssignMemoryType"},
            {  "InferDiscontinuousInput",  "InferDiscontinuousInput"},
            {        "RemoveRedundantOp",        "RemoveRedundantOp"},
            {                   "SplitK",                   "SplitK"},
            {           "GraphPartition",           "GraphPartition"},
            {          "ReduceCopyMerge",          "ReduceCopyMerge"},
            {             "NBufferMerge",             "NBufferMerge"},
            {       "L1CopyInReuseMerge",       "L1CopyInReuseMerge"},
            {     "IntraSubgraphAdapter",     "IntraSubgraphAdapter"},
            {           "GenerateMoveOp",           "GenerateMoveOp"},
            { "CommonOperationEliminate", "CommonOperationEliminate"},
            {              "AxisCombine",              "AxisCombine"},
            {           "PadLocalBuffer",           "PadLocalBuffer"},
            {   "RemoveUnalignedReshape",   "RemoveUnalignedReshape"},
            {          "ReplaceTensor",              "ReplaceTensor"},
            {          "PreGraphProcess",          "PreGraphProcess"},
            {            "InferDynShape",            "InferDynShape"},
            {       "SubgraphToFunction",       "SubgraphToFunction"},
            {          "InferParamIndex",          "InferParamIndex"},
            {        "SrcDstBufferMerge",        "SrcDstBufferMerge"},
            {                 "AddAlloc",                 "AddAlloc"},
            {              "OoOSchedule",              "OoOSchedule"},
            {        "GlobalMemoryReuse",        "GlobalMemoryReuse"},
            {              "RemoveAlloc",              "RemoveAlloc"},
            {           "CopyOutResolve",           "CopyOutResolve"},
            {               "InsertSync",               "InsertSync"},
            {         "MixSubgraphSplit",         "MixSubgraphSplit"},
            {           "CodegenPreproc",           "CodegenPreproc"},
    });
    RegisterStrategy(
        "FunctionUnroll", {
            {               "LoopUnroll",               "LoopUnroll"}
    });
    RegisterStrategy(
        "ExecuteGraph", {
            {          "DynAttrToStatic",          "DynAttrToStatic"},
    });
}

PassManager::PassManager() {
    RegPass();
    // Register strategies
    RegDefaultStrategy();
}

void PassManager::RegisterStrategy(const std::string &strategy, const std::vector<PassEntry> &passEntries) {
    // check pass dependency
    std::vector<std::string> passes;
    for (const auto &passEntry : passEntries){
        passes.emplace_back(passEntry.passName);
    }
    PassDependency::Instance().CheckStrategyDependency(strategy, passes);

    // check identifiers duplication
    std::vector<PassEntry> newPassEntries;
    std::set<std::string> identifiers;
    for (auto &pass : passEntries) {
        if (!(identifiers.insert(pass.identifier).second)) {
            ALOG_WARN_F("Duplicated identifier: %s.", pass.identifier.c_str());
            continue;
        }
        newPassEntries.push_back(pass);
    }
    auto strategyPasses = strategies_.find(strategy);
    if (strategyPasses == strategies_.end()) {
        strategies_.emplace(strategy, newPassEntries);
        return;
    }
    strategyPasses->second = newPassEntries;
    ALOG_WARN_F("Strategy %s has been changed.", strategy.c_str());
}

std::vector<PassManager::PassEntry> PassManager::GetStrategyPasses(const std::string &strategy) const {
    auto it = strategies_.find(strategy);
    if (it == strategies_.end()) {
        ALOG_WARN_F("Strategy %s does not exist.", strategy.c_str());
        auto emptyPass = std::vector<PassManager::PassEntry>();
        return emptyPass;
    }
    NPUArch currArch = Platform::Instance().GetSoc().GetNPUArch();
 	auto selectedPass = std::vector<PassManager::PassEntry>();
 	for (auto &currPassEntry : it->second) {
 	    const auto &passName = currPassEntry.passName;
 	    auto pass = PassRegistry::GetInstance().CreatePass(passName);
 	    if (pass == nullptr) {
            ALOG_WARN_F("Pass %s does not exist.", passName.c_str());
 	        continue;
 	    }
 	    std::vector<NPUArch> &arches = pass->GetSupportedArches();
 	    if ((!arches.empty()) && (std::find(arches.begin(), arches.end(), currArch) == arches.end())) {
 	        continue;
 	    }
 	    selectedPass.push_back(currPassEntry);
 	}
 	return selectedPass;
}

std::string PassManager::GetResumePath(const std::string &strategy) {
    auto strategyPasses = GetStrategyPasses(strategy);
    for (size_t i = 0; i < strategyPasses.size(); i++) {
        const auto &identifier = strategyPasses[i].identifier;
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        if (passDfxCfg.resumePath != "") {
            if (access(passDfxCfg.resumePath.c_str(), F_OK) == 0) {
                startIdx = i;
            }
            return passDfxCfg.resumePath;
        }
    }
    startIdx = static_cast<size_t>(0);
    return "";
}

Status PassManager::RunPass(Program &program, Function &function, const std::string &strategy) const {
    Platform::Instance().ObtainPlatformInfo();
    auto strategyPasses = GetStrategyPasses(strategy);
    std::vector<std::string> identifiers;
    std::transform(strategyPasses.begin(), strategyPasses.end(), std::back_inserter(identifiers),
        [](const PassEntry &elem) { return elem.identifier; });
    ConfigManager::Instance().PassConfigsDebugInfo(strategy, identifiers);
    for (size_t i = startIdx; i < strategyPasses.size(); i++) {
        const auto &identifier = strategyPasses[i].identifier;
        const auto &passName = strategyPasses[i].passName;
        auto pass = PassRegistry::GetInstance().CreatePass(passName);
        if (pass == nullptr) {
            ALOG_ERROR_F("Pass [%s] does not exist.", passName.c_str());
            return FAILED;
        }
        std::string originLogOutPath = config::LogFile();
        std::string logFolder = pass->LogFolder(config::LogTopFolder(), i);
        std::string logfilePath = logFolder + "/" + (pass->GetName() + function.GetMagicName() + ".log");
        LoggerManager::FileLoggerReplace(originLogOutPath, logfilePath, true);
        Defer rollback([logfilePath, originLogOutPath]() {
            LoggerManager::FileLoggerReplace(logfilePath, originLogOutPath, true);
        });
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        if (config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE) == CFG_DEBUG_ALL) {
            passDfxCfg.printGraph = true;
            passDfxCfg.dumpGraph = true;
        }
        pass->SetPassConfigs(passDfxCfg);
        ALOG_INFO_F("[PassManager] Apply pass <%s> on function: %s.", identifier.c_str(), function.GetMagicName().c_str());
        auto start = std::chrono::high_resolution_clock::now();
        if (pass->Run(function, strategy, identifier, i) != SUCCESS) {
            ALOG_ERROR_F("Run pass <%s> failed.", identifier.c_str());
            return FAILED;
        }
        if (passDfxCfg.dumpPassTimeCost) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            ALOG_INFO_F("Runtime of pass %s for program %s function %s is %ld us.", identifier.c_str(), program.Name().c_str(),
                function.GetMagicName().c_str(), duration.count());
        }
        if (config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY)) {
            Program::GetInstance().VerifyPass(&function, i, identifier);
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
