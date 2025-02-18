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
 * \file pass_dependency.cpp
 * \brief
 */

#include "pass_dependency.h"
#include "pass_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "PassDependency"

namespace npu::tile_fwk {
PassDependency &PassDependency::Instance(){
    static PassDependency instance;
    return instance;
}

PassDependency::PassDependency() {
    auto registerDependency = [this](const std::string &name, const std::vector<std::string> &dependencies) {
        passDependencies_[name] = std::move(dependencies);
    };

    registerDependency(PassNameStr(PassName::SUBGRAPH_TO_FUNCTION), {PassNameStr(PassName::GRAPH_PARTITION),
        PassNameStr(PassName::PRE_GRAPH_PROCESS), PassNameStr(PassName::INPLACE_PROCESS), PassNameStr(PassName::INFER_DYN_SHAPE)});
    APASS_LOG_DEBUG_F(Elements::Manager, "Strategy dependency checker initialized.");
}

Status PassDependency::CheckStrategyDependency(const std::string &strategyName, const std::vector<std::string> &passes) {
    APASS_LOG_DEBUG_F(Elements::Manager, "Start dependency check for strategy %s.", strategyName.c_str());
    bool needWarn = false;
    std::unordered_set<std::string> processedPasses;
    std::unordered_set<std::string> duplicates;
    std::string prePass;

    for (auto &pName : passes) {
        if (!prePass.empty() && prePass == pName) {
            needWarn = true;
            duplicates.emplace(pName);
            prePass = pName;
            continue;
        }
        prePass = pName;

        processedPasses.emplace(pName);
        auto it = passDependencies_.find(pName);
        if (it == passDependencies_.end()) {
            continue;
        }
        std::vector<std::string> missingDeps;
        for (const auto &dependentPass : it->second) {
            if (processedPasses.find(dependentPass) == processedPasses.end()) {
                missingDeps.push_back(dependentPass);
            }
        }
        if (missingDeps.empty()) {
            continue;
        }
        needWarn = true;
        APASS_LOG_WARN_F(Elements::Manager, "In strategy %s, %s is missing dependencies, %s are required; Please insert %s before %s.",
            strategyName.c_str(), pName.c_str(), CommonUtils::ContainerToStr(it->second).c_str(),
            CommonUtils::ContainerToStr(missingDeps).c_str(), pName.c_str());
    }
    if (duplicates.size() != 0) {
        APASS_LOG_WARN_F(Elements::Manager, "In strategy %s, %s are each arranged at least twice in a row; Please make sure all are needed.",
            strategyName.c_str(), CommonUtils::ContainerToStr(duplicates).c_str());
    }
    APASS_LOG_DEBUG_F(Elements::Manager, "Finish dependency check for strategy %s.", strategyName.c_str());
    return needWarn ? WARNING: SUCCESS;
}
} //namespace npu::tile_fwk