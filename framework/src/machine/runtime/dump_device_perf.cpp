/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_device_perf.cpp
 * \brief
 */
#include "dump_device_perf.h"
#ifdef BUILD_WITH_CANN

#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/utils/log.h"
#include "runtime/mem.h"
#include "interface/inner/config.h"
#include "interface/utils/file_utils.h"
#include "machine/device/dynamic/device_utils.h"
#include "interface/configs/config_manager.h"
namespace npu::tile_fwk::dynamic {
constexpr int DUMP_LEVEL_FOUR = 4;

void ContructTaskInfo(const uint32_t &blockNum, json &rootTaskStats,
                     const DeviceArgs &args, const std::vector<void *> &perfData) {
    for (uint32_t i = 0; i < blockNum; i++) {
        void* devPtr = perfData[i];
        size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
        std::vector<uint8_t> hostBuffer(dataSize);
        rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
        Metrics *aicpuMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
        if (aicpuMetric->taskCount > MAX_DFX_TASK_NUM_PER_CORE) {
            aicpuMetric->taskCount = MAX_DFX_TASK_NUM_PER_CORE;
        }
        TaskStat* taskStats = aicpuMetric->tasks;
        size_t numTasks = aicpuMetric->taskCount;
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        json coreObj;
        coreObj["blockIdx"] = i;
        coreObj["coreType"] = coreType;
        json tasksArr = json::array();
        for (size_t j = 0; j < numTasks; ++j) {
            if (taskStats[j].execEnd != 0) {
                json taskObj;
                taskObj["seqNo"] = taskStats[j].seqNo;
                taskObj["subGraphId"] = taskStats[j].subGraphId;
                taskObj["taskId"] = taskStats[j].taskId;
                taskObj["execStart"] = taskStats[j].execStart;
                taskObj["execEnd"] = taskStats[j].execEnd;
                tasksArr.push_back(taskObj);
            }
        }
        coreObj["tasks"] = tasksArr;
        if (!tasksArr.empty()) {
            rootTaskStats.push_back(coreObj);
        }
    }
}

void DumpAicoreTaskExectInfo(DeviceArgs &args, const std::vector<void *> &perfData) {
    json rootTaskStatus = json::array();
    auto blockNum = args.GetBlockNum();
    ALOG_INFO("GetBlockNum : %lu",  blockNum);
    ContructTaskInfo(blockNum, rootTaskStatus, args, perfData);
    std::string jsonFilePath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    if (!DumpFile(rootTaskStatus.dump(DUMP_LEVEL_FOUR), jsonFilePath)) {
        ALOG_WARN_F("Contrust custom op json failed");
        return;
    }
    ALOG_DEBUG_F("tilefwk_L1_prof_data have saved in: %s",  jsonFilePath.c_str());
    std::string topo_txt_path = npu::tile_fwk::config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = npu::tile_fwk::config::LogTopFolder() + "/program.json";
    std::string draw_swim_lane_py_path = GetCurrentSharedLibPath() + "/scripts/draw_swim_lane.py";
    npu::tile_fwk::config::SetRunDataOption(KEY_SWIM_GRAPH_PATH,
                npu::tile_fwk::config::GetAbsoluteTopFolder() + "/merged_swimlane.json");
    uint64_t freq = (args.archInfo == ArchInfo::DAV_2201) ? FREQ_DAV_2201 : FREQ_DAV_3510;

    if (FileExist(program_json_path) && FileExist(topo_txt_path)) {
        ALOG_INFO("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 "+ draw_swim_lane_py_path + " \""
                                + jsonFilePath + "\" \""
                                + topo_txt_path + "\" \""
                                + program_json_path + "\" --label_type=1 --time_convert_denominator="
                                + std::to_string(freq);
        if (system(command.c_str()) != 0) {
           ALOG_WARN("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        ALOG_WARN("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
}

} // namespce
#endif
