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
#include "interface/utils/file_utils.h"
#include "machine/device/dynamic/device_utils.h"
#include "interface/configs/config_manager.h"
#include "machine/device/distributed/common.h"
#include "tilefwk/pypto_fwk_log.h"
namespace npu::tile_fwk::dynamic {
constexpr int DUMP_LEVEL_FOUR = 4;

void ConstructTaskInfo(const uint32_t &index, json &rootTaskStats, const std::vector<void *> &perfData,
    const std::string& coreType) {
    void* devPtr = perfData[index];
    size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
    std::vector<uint8_t> hostBuffer(dataSize);
    rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
    Metrics *aicpuMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
    if (aicpuMetric->taskCount > MAX_DFX_TASK_NUM_PER_CORE) {
        aicpuMetric->taskCount = MAX_DFX_TASK_NUM_PER_CORE;
    }
    TaskStat* taskStats = aicpuMetric->tasks;
    size_t numTasks = aicpuMetric->taskCount;
    json coreObj;
    coreObj["blockIdx"] = index;
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

void DumpAicoreTaskExectInfo(DeviceArgs &args, const std::vector<void *> &perfData) {
    json rootTaskStatus = json::array();
    auto blockNum = args.GetBlockNum();
    MACHINE_LOGI("GetBlockNum : %lu",  blockNum);
    for (uint32_t i = 0; i < blockNum; i++) {
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        ConstructTaskInfo(i, rootTaskStatus, perfData, coreType);
    }
    uint32_t aicoreBlockNum = args.nrAic + args.nrAiv;
    for (uint32_t i = aicoreBlockNum; i < aicoreBlockNum + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        ConstructTaskInfo(i, rootTaskStatus, perfData, "AI-CPU");
    }
    std::string jsonFilePath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    if (!DumpFile(rootTaskStatus.dump(DUMP_LEVEL_FOUR), jsonFilePath)) {
        MACHINE_LOGW("Contrust custom op json failed");
        return;
    }
    MACHINE_LOGD("tilefwk_L1_prof_data have saved in: %s",  jsonFilePath.c_str());
    std::string topo_txt_path = npu::tile_fwk::config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = npu::tile_fwk::config::LogTopFolder() + "/program.json";
    std::string draw_swim_lane_py_path = GetCurrentSharedLibPath() + "/scripts/draw_swim_lane.py";
    npu::tile_fwk::config::SetRunDataOption(KEY_SWIM_GRAPH_PATH,
                npu::tile_fwk::config::GetAbsoluteTopFolder() + "/merged_swimlane.json");
    uint64_t freq = (args.archInfo == ArchInfo::DAV_2201) ? FREQ_DAV_2201 : FREQ_DAV_3510;

    if (FileExist(program_json_path) && FileExist(topo_txt_path)) {
        MACHINE_LOGI("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 "+ draw_swim_lane_py_path + " \""
                                + jsonFilePath + "\" \""
                                + topo_txt_path + "\" \""
                                + program_json_path + "\" --label_type=1 --time_convert_denominator="
                                + std::to_string(freq);
        if (system(command.c_str()) != 0) {
           MACHINE_LOGW("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        MACHINE_LOGW("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
    DumpAicpuPerfInfo(args, perfData, freq);
}

inline void DevTaskPerfFormat(uint32_t tid, uint32_t type, json &devTaskJson, const MetricPerf *aicpuPer) {
    json per_dev_task;
    for (uint32_t i = 0; i < aicpuPer->perfAicpuTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)]; i++) {
        std::string name = PerfTraceName[type];
        if (type != PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK) {
            name = name + "(" + std::to_string(i) + ")";
        }
        per_dev_task["name"] = name;
        per_dev_task["end"] = aicpuPer->perfAicpuTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][i];
        devTaskJson.push_back(per_dev_task);
    }
}

inline void SparateCore(int total, int idx, int part, const int &offset, std::vector<int> &coreArray) {
    int perCpu = total / part;
    int remain = total % part;
    int start = idx * perCpu + ((idx < remain) ? idx : remain);
    int end = start + perCpu + ((idx < remain) ? 1 : 0);
    for (int i = start; i < end; i++) {
        coreArray[i + offset] = idx + 1;
    }
}

inline void DumpAicoreDevTask(DeviceArgs &args, json &aicpuPrefArray,
                              const std::vector<void *> &perfData, const uint32_t &freq) {
    std::vector<int> coreArray;
    coreArray.resize(args.GetBlockNum());
    for(uint32_t i = 0; i < args.scheCpuNum; i++) {
        SparateCore(args.nrValidAic, i, args.scheCpuNum, 0, coreArray);
        SparateCore(args.nrValidAic * AIV_NUM_PER_AI_CORE, i, args.scheCpuNum, args.nrValidAic, coreArray);
    }
    for (uint32_t i = 0; i < args.GetBlockNum(); i++) {
        void* devPtr = perfData[i];
        size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
        std::vector<uint8_t> hostBuffer(dataSize);
        rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
        Metrics *aicpuMetric = reinterpret_cast<Metrics*>(hostBuffer.data());
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        json aicoreTask;
        aicoreTask["blockIdx"] = i;
        aicoreTask["coreType"] = "SCHED" + std::to_string(coreArray[i]) + "-" + coreType;
        aicoreTask["freq"] = freq;
        json tasksArr = json::array();
        uint64_t curCycle = 0;
        for (uint32_t type = 0; type < PERF_TRACE_CORE_MAX; type++) {
            for (uint32_t cnt = 0; cnt < aicpuMetric->perfTraceCnt[type]; cnt++) {
                json aicoreTaskType;
                curCycle = aicpuMetric->perfTrace[type][cnt];
                if (curCycle == 0) {
                    break;
                }
                std::string name = AicorePerfTraceName[type];
                if (aicpuMetric->perfTraceDevTaskId[type][cnt] != INVALID_DEV_TASK_ID) {
                    name = name + "(" + std::to_string(aicpuMetric->perfTraceDevTaskId[type][cnt]) + ")";
                }
                aicoreTaskType["name"] = name;
                aicoreTaskType["end"] = curCycle;
                tasksArr.push_back(aicoreTaskType);
            }
            aicpuMetric->perfTraceCnt[type] = 0;
        }
        aicoreTask["tasks"] = tasksArr;
        aicpuPrefArray.push_back(aicoreTask);
    }
}

inline void DumpAicpuDevTask(const DeviceArgs &args, json &aicpuPrefArray, const uint32_t &freq) {
    auto aicpuPer = (ValueToPtr(args.aicpuPerfAddr));
    if (aicpuPer == nullptr) {
        MACHINE_LOGW("Aicpu per ptr is null");
        return;
    }
    MetricPerf aicpuMetric;
    auto ret = rtMemcpy(PtrToPtr<MetricPerf, void>(&aicpuMetric), sizeof(MetricPerf), aicpuPer,
                        sizeof(MetricPerf), RT_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        MACHINE_LOGW("aicpu meter copy failed ret: %d", ret);
        return;
    }
    for (uint32_t i = 0; i < args.nrAicpu - 1; i++) {
        json aicpu;
        std::string coreType = "AICPU";
        if (i == 0) {
            coreType = "AICPU-CTRL";
        } else if (i <= args.scheCpuNum){
            coreType = "AICPU-SCHED";
        }
        aicpu["blockIdx"] = i;
        aicpu["coreType"] = coreType;
        aicpu["freq"] = freq;
        json aicpuDevTasks = json::array();
        for (uint32_t type = 0; type < PERF_TRACE_MAX; type++) {
            if (PerfTraceIsDevTask[type]) {
                DevTaskPerfFormat(i, type, aicpuDevTasks, &aicpuMetric);
                continue;
            }
            if (aicpuMetric.perfAicpuTrace[i][type] == 0) {
                continue;
            }
            json schCtrAicpu;
            schCtrAicpu["name"] = PerfTraceName[type];
            schCtrAicpu["end"] = aicpuMetric.perfAicpuTrace[i][type];
            aicpuDevTasks.push_back(schCtrAicpu);
        }
        aicpu["tasks"] = aicpuDevTasks;
        aicpuPrefArray.push_back(aicpu);
    }
}


void DumpAicpuPerfInfo(DeviceArgs &args, const std::vector<void *> &perfData, const uint32_t &freq) {
    json aicpuPrefArray = json::array();
    DumpAicpuDevTask(args, aicpuPrefArray, freq);
    DumpAicoreDevTask(args, aicpuPrefArray, perfData, freq);
    std::string aicpuPerfilePath = npu::tile_fwk::config::LogTopFolder() + "/aicpu_dev_pref.json";
    if (!DumpFile(aicpuPrefArray.dump(DUMP_LEVEL_FOUR), aicpuPerfilePath)) {
        MACHINE_LOGW("Contrust custom op json failed");
        return;
    }

    // toolkit drawm aicpu preffto
    std::string scriptPath = GetCurrentSharedLibPath() + "/scripts/machine_perf_trace.py";
    std::string cmd = "python3 " + scriptPath + " gen_perfetto " + aicpuPerfilePath + " "
                        + npu::tile_fwk::config::LogTopFolder() + "/machine_runtime_operator_trace.json";
    if (system(cmd.c_str()) != 0) {
        MACHINE_LOGW("Failed to execute machine_perf_trace.py, cannot get aicpu perfetto.json.");
    }
    npu::tile_fwk::config::SetRunDataOption(KEY_AICPU_PERF_GRAPH_PATH,
            npu::tile_fwk::config::GetAbsoluteTopFolder() + "/machine_runtime_operator_trace.json");
}
} // namespce
#endif
