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
 * \file machine_compiler.h
 * \brief
 */

#pragma once

#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
#include "tilefwk/comm_group_recorder.h"
#include "tilefwk/tilefwk_log.h"

namespace npu::tile_fwk {
constexpr int64_t AICORE_NUM = 75;

/* 每次编译生成的信息，offset信息后面再放cache里,每个AscendFunction一个 */
struct MachineCompileInfo {
    uint32_t aicoreCnt{AICORE_NUM};// 从全局配置获取
    uint64_t programFunctionCnt; // 同构后的funciton 个数
    uint64_t coreFunctionCnt;
    uint64_t workSpaceStackSize{0}; // ooo 调度use stack workspace
    uint64_t invokeParaWorkSpaceSize{0};
    size_t invokeOffsetSize{0};
    std::vector<std::string> commGroups;
    std::map<uint64_t, std::list<InvokeParaOffset>> invokeParaOffset; // map esgid to all para list
    std::map<uint64_t, uint64_t> coreFunctionIdToProgramId; // 对应graph 里 esgid map psgid
    std::vector<CoreFunctionReadyState> coreFunctionReadyState;
    std::vector<uint64_t> readyAicIdVec;
    std::vector<uint64_t> readyAivIdVec;
    std::vector<uint64_t> readyAicpuIdVec;
    std::vector<uint64_t> coreFuncBinOffset;
    std::vector<std::vector<uint64_t>> invokeArgsOffset; // esgid to all para offset
    std::vector<std::vector<int64_t>> invokeTensorsIdx; // esgid to all para tensorIdx: input0 input1 ... output0 output1, -1 means workspace
    std::vector<uint64_t> coreFunctionInvokeEntryOffset;
    std::vector<TensorInfo> coreTensorInfoVec;
    std::vector<uint64_t> coreFunctionTensorInfoOffset;
    std::vector<uint64_t> coreTensorNum;
    void Print() {
        MACHINE_LOGD(
            "programFunctionCnt =  %lu, coreFunctionCnt = %lu, workSpaceStackSize = %lu, invokeParaWorkSpaceSize = %lu",
            programFunctionCnt, coreFunctionCnt, workSpaceStackSize, invokeParaWorkSpaceSize);
    }
};

void CalcFunctionInvokeWorkespace(Function* cacheFunction, Function* function, MachineCompileInfo& compileInfo);
}
