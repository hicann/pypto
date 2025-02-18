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
 * \file pass_manager.h
 * \brief
 */

#ifndef PASSES_PASS_MGR_H_
#define PASSES_PASS_MGR_H_
#include <vector>
#include <string>
#include <unordered_map>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

class PassManager {
public:
    static PassManager &Instance();

    PassManager(const PassManager &) = delete;
    void operator=(const PassManager &) = delete;
    Status RunPass(Program &program, Function &function, const std::string &strategy) const;
    std::string GetResumePath(const std::string &strategy);

    struct PassEntry {
        std::string identifier;
        std::string passName;
        PassEntry(std::string id, std::string name) : identifier(id), passName(name) {}
    };

    void RegisterStrategy(const std::string &strategy, const std::vector<PassEntry> &passEntries);

private:
    PassManager();
    ~PassManager() = default;
    void RegDefaultStrategy();

    std::vector<PassEntry> GetStrategyPasses(const std::string &strategy) const;

    std::unordered_map<std::string, std::vector<PassEntry>> strategies_;

    size_t startIdx{0};
};
} // namespace npu::tile_fwk
#endif // PASSES_PASS_MGR_H_