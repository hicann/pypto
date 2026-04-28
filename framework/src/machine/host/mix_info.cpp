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
 * \file mix_info.cpp
 * \brief
 */

#include <fstream>
#include <nlohmann/json.hpp>
#include "mix_info.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
using json = nlohmann::json;
namespace npu {
namespace tile_fwk {
namespace mix_info {
constexpr uint32_t MAX_SYNC_EVENT_NUM = 48; // the max set/wait insts in a mix subgraph leaffunction is 48, can set larger manually
void GetExecuteFunc(Function* func, std::map<int, std::set<Function*>>& leafFunctions)
{
    auto funcType = func->GetGraphType();
    if (func->IsFunctionTypeAndGraphType(
            {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH},
            GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            GetExecuteFunc(callFunc, leafFunctions);
        }
        return;
    } else if (funcType == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            auto wrapId = callopAttr->wrapId;
            if (wrapId == -1) {
                continue;
            }
            auto callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
            if (callFunc == nullptr) {
                continue;
            }
            leafFunctions[wrapId].insert(callFunc);
        }
        return;
    } else if (funcType == GraphType::TILE_GRAPH) {
        GetExecuteFunc(func->GetRootFunction(), leafFunctions);
    }
    return;
}

void DumpMixInfoToJson(const std::map<uint64_t, std::map<int, WrapInfo>>& wrapInfos) {
    std::vector<MixInfo> wrapinfoList;
    for (auto& [mixId, rootWrapinfo] : wrapInfos) {
        MixInfo mixInfo;
        mixInfo.mixId = mixId;
        for (auto& [wrapId, wrapInfo] : rootWrapinfo) {
            (void)wrapId;
            mixInfo.wrapInfos.push_back(wrapInfo);
        }
        wrapinfoList.push_back(mixInfo);
    }
json j = wrapinfoList;
    std::string path = npu::tile_fwk::config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::ofstream of(path);
    if (of.is_open()) {
        of << j.dump(4);
        of.close();
    }
}

int DumpMixInfo(Function* topFunc)
{
    if (topFunc == nullptr) {
        return 0;
    }
    std::map<int, std::set<Function*>> leafFunctions;
    GetExecuteFunc(topFunc, leafFunctions);
    std::map<uint64_t, std::map<int, WrapInfo>> wrapInfos;
    for (auto& [wrapID, leafFuncs] : leafFunctions) {
        for (auto& leafFunc : leafFuncs) {
            auto leafAttr = leafFunc->GetLeafFuncAttribute();
            if (leafAttr == nullptr) {
                continue;
            }
            auto mixId = leafAttr->mixId;
            if (wrapInfos.find(mixId) == wrapInfos.end() ||
                wrapInfos[mixId].find(wrapID) == wrapInfos[mixId].end()) {
                WrapInfo info;
                info.wrapID = wrapID;
                wrapInfos[mixId][wrapID] = info;
            }
            CoreTask leafFuncSyncInfo;
            leafFuncSyncInfo.hashValue = leafFunc->GetFunctionHash().GetHash();
            for (auto& op : leafFunc->Operations(false).DuplicatedOpList()) {
                SyncInfo syncInfo;
                if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
                    syncInfo.isSet = true;
                } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
                    syncInfo.isSet = false;
                } else {
                    continue;
                }
                syncInfo.eventID = op->GetSyncQueue().eventId_;
                leafFuncSyncInfo.syncMsg.push_back(syncInfo);
            }
            ASSERT(DevCommonErr::PARAM_CHECK_FAILED, leafFuncSyncInfo.syncMsg.size() <= MAX_SYNC_EVENT_NUM)
                << "leaffunction's syncEvent's size(" << leafFuncSyncInfo.syncMsg.size() << "is lager than MAX_SYNC_EVENT_NUM"
                << MAX_SYNC_EVENT_NUM << "need to set MAX_SYNC_EVENT_NUM larger manually";
            wrapInfos[mixId][wrapID].coreTask.push_back(leafFuncSyncInfo);
        }
    }
    DumpMixInfoToJson(wrapInfos);
    return 0;
}
} // namespace mix_info
} // namespace tile_fwk
} // namespace npu