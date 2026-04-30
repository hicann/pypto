/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \main_block.h
 * \brief
 */

#pragma once

#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
#include "../include/tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {
constexpr const uint8_t MAIN_BLOCK_SIZE = 2;
constexpr const size_t MAX_RUNTIME_AND_NESTING_DEPTH = 7000;

inline std::string GetEmitPath(const std::string& name)
{
    std::string dirPath;
    if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false)) {
        std::vector<std::string> groupNames = npu::tile_fwk::Distributed::CommGroupRecorder::GetInstance().Output();
        if (groupNames.size() == 0) {
            dirPath = name;
        } else {
            const char* rankId = std::getenv("TILE_FWK_DEVICE_ID");
            dirPath = std::string(rankId) + "/" + name;
        }
    } else {
        dirPath = config::LogTopFolder() + "/" + name;
    }
    return dirPath;
}

class MainBlockCondBulider {
public:
    MainBlockCondBulider();
    void CollectCallopMainBlockConds(Function* func);
    void CollectCoaMainBlockConds(const std::vector<std::vector<SymbolicScalar>>& argList);
    SymbolicScalar BuildMainBlockExpression();
    static void Gencode(Function* function);
    const std::vector<SymbolicScalar>& GetCondGroup() const;
    const std::unordered_set<std::string>& GetCondStrSet() const;

private:
    void AddUniqueCondition(const SymbolicScalar& newCond);
    bool CheckShapeEquality(const Shape& shape, const std::vector<SymbolicScalar>& dynShape);

    bool GetValidShapeFromCoa(
        const std::vector<SymbolicScalar>& argList, Shape& shape, std::vector<SymbolicScalar>& dynValidShape);

private:
    std::vector<SymbolicScalar> mainBlockCondGroup_;
    std::unordered_set<std::string> mainBlockStrSet_;
};
} // namespace npu::tile_fwk
