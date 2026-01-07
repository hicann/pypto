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
            std::regex pattern(R"(\(\(__gm__ GMTensorInfo\*\)\(param\) \+ \d+\)->Addr)");
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
            std::regex runtimeRawShapePattern1(R"(GET_PARAM_RAWSHAPE_1\(param, \d+, \d+\))");
            result = std::regex_replace(result, runtimeRawShapePattern1, "1");
            std::regex runtimeRawShapePattern2(R"(GET_PARAM_RAWSHAPE_2\(param, \d+, \d+\))");
            result = std::regex_replace(result, runtimeRawShapePattern2, "1, 1");
            std::regex runtimeRawShapePattern3(R"(GET_PARAM_RAWSHAPE_3\(param, \d+, \d+\))");
            result = std::regex_replace(result, runtimeRawShapePattern3, "1, 1, 1");
            std::regex runtimeRawShapePattern4(R"(GET_PARAM_RAWSHAPE_4\(param, \d+, \d+\))");
            result = std::regex_replace(result, runtimeRawShapePattern4, "1, 1, 1, 1");
            std::regex getParamOffsetPattern1(R"(GET_PARAM_OFFSET_1\(param, \d+, \d+\))");
            result = std::regex_replace(result, getParamOffsetPattern1, "0");
            std::regex getParamOffsetPattern2(R"(GET_PARAM_OFFSET_2\(param, \d+, \d+\))");
            result = std::regex_replace(result, getParamOffsetPattern2, "0, 0");
            std::regex getParamOffsetPattern3(R"(GET_PARAM_OFFSET_3\(param, \d+, \d+\))");
            result = std::regex_replace(result, getParamOffsetPattern3, "0, 0");
            std::regex getParamOffsetPattern4(R"(GET_PARAM_OFFSET_4\(param, \d+, \d+\))");
            result = std::regex_replace(result, getParamOffsetPattern4, "0, 0, 0, 0");
            std::regex getParamShapePattern(R"(GET_PARAM_RAWSHAPE_BY_IDX\(param, \d+, \d+, \d+, \d+\))");
            result = std::regex_replace(result, getParamShapePattern, "0");
            std::regex sysdimPattern(R"(sym_\d+_dim_\d+)");
            result = std::regex_replace(result, sysdimPattern, "1");
            std::regex maybeConst(R"(RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST\(-?\d+, -?\d+, -?\d+, -?\d+, -?\d+\))");
            result = std::regex_replace(result, maybeConst, "0");
            std::regex maybeConst_0(R"(RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST_0\(-?\d+, -?\d+, -?\d+, -?\d+\))");
            result = std::regex_replace(result, maybeConst_0, "0");
            return result;
        }
    }

    extern "C" UnifiedPipeMachinePtr CreatePipeSimulatorSimulatorA2A3();
} // namespace CostModel
