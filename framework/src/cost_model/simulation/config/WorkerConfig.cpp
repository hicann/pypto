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
 * \file WorkerConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/WorkerConfig.h"

using namespace std;

namespace CostModel {
WorkerConfig::WorkerConfig()
{
    Config::prefix = "Worker";
    Config::dispatcher = {
        {"sessionNum", [&](string v){ sessionNum = ParseInteger(v); }},
        {"moeNum", [&](string v){ moeNum = ParseInteger(v); }},
        {"layerNum", [&](string v){ layerNum = ParseInteger(v); }},
        {"expertNum", [&](string v){ expertNum = ParseInteger(v); }},
        {"attnBatch", [&](string v){ attnBatch = ParseInteger(v); }},
        {"ffnBatch", [&](string v){ ffnBatch = ParseInteger(v); }},
        {"attnLatencyAvg", [&](string v){ attnLatencyAvg = ParseInteger(v); }},
        {"attnLatencySdv", [&](string v){ attnLatencySdv = ParseInteger(v); }},
        {"attnLatencyMax", [&](string v){ attnLatencyMax = ParseInteger(v); }},
        {"attnLatencyMin", [&](string v){ attnLatencyMin = ParseInteger(v); }},
        {"ffnLatencyAvg", [&](string v){ ffnLatencyAvg = ParseInteger(v); }},
        {"ffnLatencySdv", [&](string v){ ffnLatencySdv = ParseInteger(v); }},
        {"commLatencyAvg", [&](string v){ commLatencyAvg = ParseInteger(v); }},
        {"commLatencySdv", [&](string v){ commLatencySdv = ParseInteger(v); }},
        {"layerWaitTime", [&](string v){ layerWaitTime = ParseInteger(v); }},
        {"isBsp", [&](string v){ isBsp = ParseBoolean(v); }},
        {"attnPolling", [&](string v){ attnPolling = ParseBoolean(v); }},
        {"ffnPolling", [&](string v){ ffnPolling = ParseBoolean(v); }},
        {"layerStartVariation", [&](string v){ layerStartVariation = ParseInteger(v); }},
        {"useFixedRandomSeed", [&](string v){ useFixedRandomSeed = ParseBoolean(v); }},
        {"randomSeed", [&](string v){ randomSeed = ParseInteger(v); }},
    };

    Config::recorder = {
        {"sessionNum", [&](){ return "sessionNum = " + ParameterToStr(sessionNum); }},
        {"moeNum", [&](){ return "moeNum = " + ParameterToStr(moeNum); }},
        {"layerNum", [&](){ return "layerNum = " + ParameterToStr(layerNum); }},
        {"expertNum", [&](){ return "expertNum = " + ParameterToStr(expertNum); }},
        {"attnBatch", [&](){ return "attnBatch = " + ParameterToStr(attnBatch); }},
        {"ffnBatch", [&](){ return "ffnBatch = " + ParameterToStr(ffnBatch); }},
        {"attnLatencyAvg", [&](){ return "attnLatencyAvg = " + ParameterToStr(attnLatencyAvg); }},
        {"attnLatencySdv", [&](){ return "attnLatencySdv = " + ParameterToStr(attnLatencySdv); }},
        {"attnLatencyMax", [&](){ return "attnLatencyMax = " + ParameterToStr(attnLatencyMax); }},
        {"attnLatencyMin", [&](){ return "attnLatencyMin = " + ParameterToStr(attnLatencyMin); }},
        {"ffnLatencyAvg", [&](){ return "ffnLatencyAvg = " + ParameterToStr(ffnLatencyAvg); }},
        {"ffnLatencySdv", [&](){ return "ffnLatencySdv = " + ParameterToStr(ffnLatencySdv); }},
        {"commLatencyAvg", [&](){ return "commLatencyAvg = " + ParameterToStr(commLatencyAvg); }},
        {"commLatencySdv", [&](){ return "commLatencySdv = " + ParameterToStr(commLatencySdv); }},
        {"layerWaitTime", [&](){ return "layerWaitTime = " + ParameterToStr(layerWaitTime); }},
        {"isBsp", [&](){ return "isBsp = " + ParameterToStr(isBsp); }},
        {"attnPolling", [&](){ return "attnPolling = " + ParameterToStr(attnPolling); }},
        {"ffnPolling", [&](){ return "ffnPolling = " + ParameterToStr(ffnPolling); }},
        {"layerStartVariation", [&](){ return "layerStartVariation = " + ParameterToStr(layerStartVariation); }},
        {"useFixedRandomSeed", [&](){ return "useFixedRandomSeed = " + ParameterToStr(useFixedRandomSeed); }},
        {"randomSeed", [&](){ return "randomSeed = " + ParameterToStr(randomSeed); }},
    };
}
}