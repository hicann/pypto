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
 * \file device_task.h
 * \brief
 */

#pragma once

#include "vector.h"
#include "allocator/allocators.h"
#include "tilefwk/core_func_data.h"
#ifndef __DEVICE__
#include "interface/configs/config_manager.h"
#endif

namespace npu::tile_fwk::dynamic {
class DeviceWorkspaceAllocator;
struct DynDeviceTask : DynDeviceTaskBase {
    Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator> stitchedList;
    WsAllocation selfAlloc;
    WsSlabStageAllocMem taskStageAllocMem;

    static uint32_t GetReadyQueueIndexByCoreType(CoreType coreType) {
        if (coreType == CoreType::AICPU) {
            return static_cast<uint32_t>(READY_QUEUE_SIZE) - 1;
        }
        return static_cast<uint32_t>(coreType);
    }

    DynDeviceTask(DeviceWorkspaceAllocator &allocator) {
        memset_s(&devTask, sizeof(devTask), 0, sizeof(devTask));
        stitchedList.InitAllocator(allocator);
    }

    predcount_t &GetOperationCurrPredCount(uint32_t id) {
        return stitchedList[FuncID(id)].GetOperationCurrPredCount(TaskID(id));
    }

    int GetOperationCoreType(uint32_t id) {
        auto callee = stitchedList[FuncID(id)].GetSource()->GetOperationAttrCalleeIndex(TaskID(id));
        return cceBinary[callee].coreType;
    }

    std::string DumpTaskData(uint32_t id) {
        auto &funcDup = stitchedList[FuncID(id)];
        return funcDup.DumpDyn(FuncID(id), TaskID(id), cceBinary);
    }

    void DumpTopo() {
        auto header = GetDynFuncDataList();
#ifdef __DEVICE__
        std::string path = "./output/dyn_topo.txt";
#else
        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
#endif
        static std::string lastPath;
        static std::ofstream of;
        if (path != lastPath) {
            if (of.is_open()) {
                of.flush();
                of.close();
            }
            lastPath = path;
            of.open(path);
        }
        if (of.tellp() == 0) {
            of << "seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,psgId,successors\n";
        }
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTopo(of, header->seqNo, funcIdx, cceBinary);
        }
        of.flush();
    }

    void DumpLeafs() {
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            auto lines = stitchedList[funcIdx].DumpLeafs(GetDynFuncDataList()->seqNo, funcIdx);
            for (auto &&line : lines) {
                DEV_ERROR("[DumpLeafs] %s", line.c_str());
            }
        }
    }

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(uintdevptr_t dumpTensorWsAddr, uint64_t dumpTensorWsSize) {
        UNUSED(dumpTensorWsAddr);
        UNUSED(dumpTensorWsSize);
        std::stringstream oss;
        std::vector<std::string> infos;
        for (uint32_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTensorAddrInfo(infos, GetDynFuncDataList()->seqNo, funcIdx);
        }
        auto str = std::move(oss).str();
        DEV_ERROR("[DumpTensor] seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)");
        DEV_ERROR("[DumpTensor] >>>");
        for (auto &info : infos) {
            DEV_ERROR("[DumpTensor] %s", info.c_str());
        }
        DEV_ERROR("[DumpTensor] <<<");
    }
#endif
};

#define DYN_DEVICE_TASK_EXT_SIZE 0x300
static_assert(sizeof(DynDeviceTask) < sizeof(DynDeviceTaskBase) + DYN_DEVICE_TASK_EXT_SIZE, "Invalid dyn device task extension");
}
