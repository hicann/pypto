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
 * \file PipeSimulator.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include "cost_model/simulation/arch/PipeMachineImpl.h"
#include "cost_model/simulation_ca/A2A3/SimulatorA2A3.h"

namespace CostModel
{
    template <typename Simulator>
    class PipeSimulator : public PipeMachineImpl
    {
    public:
        uint64_t Simulate(const TileOpPtr& tileOp) override;
        uint64_t PostSimulate(const TileOpPtr &tileOp) override;
    private:
        std::unordered_map<std::string, uint64_t> tileopLatencyCacheMp;
    };

    namespace PipeSimulatorUtils {
        std::string ReplaceGMStr(const std::string &str) { 
            std::regex pattern(R"(\(\(__gm__ GMTensorInfo\*\)\(param\) \+ \d+\)->Addr)");  // 正则表达式匹配目标格式
            std::string result = std::regex_replace(str, pattern, "charArray1");
            result = std::regex_replace(result, std::regex("GMStackBase"), "charArray1");
            std::regex getParamPattern(R"(GET_PARAM_ADDR\(param, \d+, \d+\))");
            result = std::regex_replace(result, getParamPattern, "charArray2");
            std::regex oriAddrPattern(R"(\(\(__gm__ GMTensorInfo\*\)\(oriAddrParam\) \+ \d+\)->Addr)");
            result = std::regex_replace(result, oriAddrPattern, "charArray3");
            std::regex runtimeCoaPattern(R"(RUNTIME_COA_GET_PARAM_OFFSET\(\d+,\d+,\d+\))");
            result = std::regex_replace(result, runtimeCoaPattern, "0");
            std::regex runtimeCoaSpacePattern(R"(RUNTIME_COA_GET_PARAM_OFFSET\(\d+, \d+, \d+\))");
            result = std::regex_replace(result, runtimeCoaSpacePattern, "0");
            std::regex runtimeCoaParamPattern(R"(RUNTIME_COA_GET_PARAM\(\d+\))");
            result = std::regex_replace(result, runtimeCoaParamPattern, "0");
            return result;
        }
    }

    extern "C" UnifiedPipeMachinePtr CreatePipeSimulatorSimulatorA2A3();
} // namespace CostModel
