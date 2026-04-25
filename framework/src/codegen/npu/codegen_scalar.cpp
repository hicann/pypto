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
 * \file codegen_scalar.cpp
 * \brief
 */

#include "interface/tensor/logical_tensor.h"
#include "codegen_op_npu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
std::string CodeGenOpNPU::GenBarrier() const
{
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "pipe_barrier(%s);\n", pipeId1.c_str());
    if (ret < 0) {
        CODEGEN_LOGI("genBarrier snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOpNPU::GenSyncSetOp() const
{
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(
        buffer, sizeof(buffer), sizeof(buffer) - 1, "set_flag(%s, %s, EVENT_ID%d);\n", pipeId1.c_str(), pipeId2.c_str(),
        syncQueue.eventId_);
    if (ret < 0) {
        CODEGEN_LOGI("genSyncSetOp snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOpNPU::GenSyncWaitOp() const
{
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(
        buffer, sizeof(buffer), sizeof(buffer) - 1, "wait_flag(%s, %s, EVENT_ID%d);\n", pipeId1.c_str(),
        pipeId2.c_str(), syncQueue.eventId_);
    if (ret < 0) {
        CODEGEN_LOGI("genSyncWaitOp snprintf_s failed %d", ret);
    }
    return buffer;
}

void InsertSetSysCnt(std::ostringstream& oss){
    oss << "#ifdef OPEN_MIX_PERF\n";
    oss << "taskStat->setEventCycle[taskStat->setEventIdx++] = get_sys_cnt();\n";
    oss << "#endif\n";
}

std::string CodeGenOpNPU::GenCVSyncSetOp() const
{
    auto pipeId = GetPipeId(syncQueue.pipeId_);
    std::ostringstream oss;
    InsertSetSysCnt(oss);
    oss << "set_intra_block(" << pipeId << ", " << std::to_string(syncQueue.eventId_) << ");\n";
    return oss.str();
}

void InsertWaitSysCnt(std::ostringstream& oss){
    oss << "#ifdef OPEN_MIX_PERF\n";
    oss << "taskStat->waitEventCycle[taskStat->waitEventIdx++] = get_sys_cnt();\n";
    oss << "#endif\n";
}

std::string CodeGenOpNPU::GenCVSyncWaitOp() const
{
    auto pipeId = GetPipeId(syncQueue.trigPipeId_);
    std::ostringstream oss;
    InsertWaitSysCnt(oss);
    oss << "wait_intra_block(" << pipeId << ", " << std::to_string(syncQueue.eventId_) << ");\n";
    return oss.str();
}

static const std::unordered_map<int, std::string> aicpuCallNumDict = {
    {AICPU_CALL_NUM_COPYOUT_RESOLVE, "AICPU_CALL_NUM_COPYOUT_RESOLVE"},
};

std::string CodeGenOpNPU::GenAicpuCallOp() const
{
    ASSERT(OperErr::ATTRIBUTE_INVALID, opAttrs.count(OpAttributeKey::aicpuCall))
        << "OpAttributeKey::aicpuCall not found";
    uint32_t call = static_cast<uint32_t>(AnyCast<int64_t>(opAttrs.find(OpAttributeKey::aicpuCall)->second));
    uint16_t callNum = call >> AICPU_CALL_ARG_BIT;
    uint16_t callArg = call & ((1 << AICPU_CALL_ARG_BIT) - 1);

    std::ostringstream oss;
    std::string callNumName = std::to_string(callNum);
    if (aicpuCallNumDict.count(callNum)) {
        callNumName = aicpuCallNumDict.find(callNum)->second;
    }
    oss << tileOpName << "<" << callNumName << "," << callArg << ">(GET_CURRENT_TASKID());\n";
    return oss.str();
}

} // namespace npu::tile_fwk
