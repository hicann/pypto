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
 * \file device_runner.cpp
 * \brief
 */

#ifdef BUILD_WITH_CANN
#include "machine/runtime/device_runner.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <limits.h>
#include "securec.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/load_aicpu_op.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "runtime/mem.h"
#include "machine/utils/device_switch.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "toolchain/prof_api.h"
#include "prof_common.h"
#include "load_aicpu_op.h"
#include "tilefwk/platform.h"
#include "machine/platform/platform_manager.h"
#include "machine/runtime/device_error_tracking.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
extern char _binary_kernel_o_start[];
extern char _binary_kernel_o_end[];

constexpr int32_t AICORE_ADDR_TYPE = 2; // nocache Addr type for aicore/aicpu map
constexpr int32_t PMU_ADDR_TYPE = 3;    // nGnRnE Addr type for Geting pmuInfo
// pmu event type
constexpr int32_t ARITHMETIC_UTILIZATION = 1;
constexpr int32_t PIPE_UTILIZATION = 2;
constexpr int32_t MEMORY = 4;
constexpr int32_t MEMORY_L0 = 5;
constexpr int32_t RESOURCE_CONFLICT_RATION = 6;
constexpr int32_t MEMORY_UB = 7;
constexpr int32_t L2_CACHE = 8;
constexpr int32_t PATH_LENGTH = 64;
constexpr uint32_t LOG_BUF_SIZE = 64 * 1024;
bool g_IsFirstInit = false;
bool g_IsNullLaunched = false;
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;

constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;

extern "C" __attribute__((weak)) int AdxDataDumpServerUnInit();
namespace npu::tile_fwk {
DeviceRunner &DeviceRunner::Get() {
    static DeviceRunner runner;
    std::call_once(runner.once_, [&]() { runner.Init(); });
    return runner;
}

HostProf& DeviceRunner::GetHostProfInstance() {
    return hostProf_;
}

void DeviceRunner::GetHostProfTypeSwtich() {
    auto profType = hostProf_.GetProfType();
    auto profSwitch = hostProf_.GetProfSwitch();
    if (profType == PROF_COMMANDHANDLE_TYPE_START) {
        isOpenHostProf_ = true;
    }
    if ((profSwitch & PROF_TASK_TIME_L1_MASK) != 0) {
        isHostProfL1_ = true;
    }
    ALOG_DEBUG_F("isOpenHostProf %d, l1 = %d.", isOpenHostProf_, isHostProfL1_);
}

void *DeviceRunner::DevAlloc(int size) {
    uint8_t *devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, size);
    int rc = rtMemset(devPtr, size, 0, size);
    if (rc != 0) {
        machine::GetRA()->FreeTensor(devPtr);
        ALOG_ERROR_F("rtMemset failed size=%d rc=%d\n", size, rc);
        return nullptr;
    }
    return devPtr;
}

void DeviceRunner::SetPmuEventType(int32_t &profPmuType) {
    // 按照环境变量设置的数值，获取pmu事件类型
    switch (profPmuType) {
        case  ARITHMETIC_UTILIZATION:
            pmuEvtType_ = {0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x0};
            break;
        case PIPE_UTILIZATION:
            pmuEvtType_ = {0x08, 0x0a, 0x09, 0x0b, 0x0c, 0x0d, 0x55, 0x54};
            break;
        case MEMORY:
            pmuEvtType_ = {0x15, 0x16, 0x31, 0x32, 0x0f, 0x10, 0x12, 0x13};
            break;
        case MEMORY_L0:
            pmuEvtType_ = {0x1b, 0x1c, 0x21, 0x22, 0x27, 0x28, 0x0, 0x0};
            break;
        case RESOURCE_CONFLICT_RATION:
            pmuEvtType_ = {0x64, 0x65, 0x66, 0x0, 0x0, 0x0, 0x0, 0x0};
            break;
        case MEMORY_UB:
            pmuEvtType_ = {0x3d, 0x10, 0x13, 0x3e, 0x43, 0x44, 0x37, 0x38};
            break;
        case L2_CACHE:
            pmuEvtType_ = {0x500, 0x502, 0x504, 0x506, 0x508, 0x50a, 0x0, 0x0};
            break;
        default:
            ALOG_WARN_F("Invalid profPmuType %d, only support [1,2,4,5,6,7,8].\n", profPmuType);
    }
}

void DeviceRunner::GetPmuEventType() {
    // 获取pmu事件类型环境变量获取方式
    std::string eventTypeStr = GetEnvVar("PROF_PMU_EVENT_TYPE");
    if (eventTypeStr.empty()) {
        ALOG_WARN_F("Dont support PROF_PMU_EVENT_TYPE env, use default pmu event type PIPE_UTILIZATION.\n");
        eventTypeStr = "2";
    }
    int32_t profPmuType = std::stoi(eventTypeStr);
    SetPmuEventType(profPmuType);
}

void DeviceRunner::InitDynamicArgs(DeviceArgs &args) {
    devArgs_ = reinterpret_cast<DeviceArgs *>(DevAlloc(sizeof(DeviceArgs)));
    rtMemcpy(reinterpret_cast<void *>(devArgs_), sizeof(DeviceArgs), &args, sizeof(DeviceArgs),
        RT_MEMCPY_HOST_TO_DEVICE);

    for (uint64_t i = 0; i < args.nrAic + args.nrAiv; i++) {
        perfData_.push_back(DevAlloc(MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics)));
    }
}

void DeviceRunner::ResetPerData() {
    auto size = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
    for (uint64_t i = 0; i < args_.nrAic + args_.nrAiv; i++) {
        int rc = rtMemset(perfData_[i], size, 0, size);
        if (rc != 0) {
            ALOG_WARN_F("CoreId %lu, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

int DeviceRunner::InitDeviceArgs(DeviceArgs &args) {
    addressMappingTable_[ArchInfo::DAV_2201] = [&args](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (machine::GetRA()->GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return -1;
        }
        if (machine::GetRA()->GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
            return 0;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());
        regsPmu.insert(regsPmu.end(), aicPmu.begin(), aicPmu.end());
        regsPmu.insert(regsPmu.end(), aivPmu.begin(), aivPmu.end());
        return 0;
    };

    addressMappingTable_[ArchInfo::DAV_3510] = [&args](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        return machine::GetRA()->GetAicoreRegInfoForDAV3510(regs, regsPmu);
    };
    
    hostProf_.RegHostProf();
    aicpuNum_ = aicpuNum_ < static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum() - 1) ? aicpuNum_ : static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum() - 1);
    GetHostProfTypeSwtich();

    memset_s(&args, sizeof(args), 0, sizeof(args));
    std::vector<int64_t> regs;
    std::vector<int64_t> regsPmu;

    args.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    auto it = addressMappingTable_.find(args.archInfo);
    if (it != addressMappingTable_.end()){
        if (it->second(regs, regsPmu) != 0) {
            return -1;
        }
    }

    uint32_t totalCoreCount = regs.size();
    uint32_t aicCount = totalCoreCount / SUB_CORE;
    uint32_t aivCount = aicCount * AIV_PER_AICORE;
    args.nrAic = aicCount;
    args.nrAiv = aivCount;
    blockDim_ = dynamic::GetCfgBlockdim();
    args.nrValidAic = blockDim_;
    args.nrAicpu = aicpuNum_;
    int nrCore = regs.size();
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.coreRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    args.taskWastTime = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(sizeof(uint64_t))));
    size_t shmSize = dynamic::DEVICE_SHM_SIZE + dynamic::DEVICE_TASK_QUEUE_SIZE * aicpuNum_;
    uint64_t shmAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(shmSize)));
    args.startArgsAddr = shmAddr;
    args.taskCtrl = shmAddr + dynamic::DEV_ARGS_SIZE;
    args.taskQueue = shmAddr + dynamic::DEV_ARGS_SIZE + dynamic::DEVICE_TASK_CTRL_SIZE;
    pmuEvtType_.resize(PMU_EVENT_TYPE_MAX, 0x0);
    args.pmuEventAddr = reinterpret_cast<uint64_t>(DevAlloc(pmuEvtType_.size() * sizeof(int64_t)));

    if (args.sharedBuffer == 0 || args.coreRegAddr == 0 || args.corePmuAddr == 0 || args.corePmuRegAddr == 0) {
        return -1;
    }
    GetPmuEventType();
    size_t size = nrCore * sizeof(uint64_t);
    rtMemcpy(reinterpret_cast<void *>(args.coreRegAddr), size, regs.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    rtMemcpy(reinterpret_cast<void *>(args.corePmuRegAddr), size, regsPmu.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    size = pmuEvtType_.size() * sizeof(int64_t);
    rtMemcpy(reinterpret_cast<void *>(args.pmuEventAddr), size, pmuEvtType_.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
    ALOG_INFO_F("aic %u aiv %u  blockDim_ %d sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx\n", args.nrAic,
        args.nrAiv, blockDim_, args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);
    InitDynamicArgs(args);
    return 0;
}

uint64_t DeviceRunner::GetTasksTime() const {
    uint64_t buffer;
    int rc = rtMemcpy(reinterpret_cast<void *>(&buffer), sizeof(uint64_t),
                      reinterpret_cast<void *>(static_cast<uintptr_t>(args_.taskWastTime)),
                      sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST);
    (void)rc;
    return buffer;
}

int DeviceRunner::Run(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType) {
    int rc;

    rc = RunAsync(aicpuStream, aicoreStream, taskId, taskData, taskType);
    if (rc < 0) {
        return rc;
    }
    rc = rtStreamSynchronize(aicoreStream);
    if (rc != 0) {
        ALOG_INFO_F("aicore stream sync failed");
    }
    rc = rtStreamSynchronize(aicpuStream);
    if (rc != 0) {
        ALOG_INFO_F("aicpu stream sync failed");
    }
    ASSERT(rc == 0);
    if (IsAstDataDumpEnabled()) {
        ALOG_DEBUG_F("DataDumpServerInit is called \n");
        rc = AdxDataDumpServerUnInit();
        if (rc != 0) {
            ALOG_ERROR_F("AdxDataDumpServerUnInit is failed %d \n", rc);
        }
    }
    uint64_t taskWastTime = GetTasksTime();
    ALOG_INFO_F("task wast time %lu\n", taskWastTime);
    return rc;
}

int DeviceRunner::LaunchAiCore(rtStream_t aicoreStream, int taskType) {
    struct Args {
        int64_t *syncAddr = nullptr;
        int64_t *inputs = nullptr;
        int64_t *outputs = nullptr;
        int64_t *workspace = nullptr;
        int64_t *tilingData = nullptr;
        DeviceArgs *cfgdata = nullptr;
        int64_t *logBuf = nullptr;
    };

    auto localArgs = args_;
    auto size = sizeof(args_);
    localArgs.taskType = taskType;
    if (devArgs_ == nullptr) {
        ALOG_ERROR_F("devArgs_ is null..");
        return -1;
    }
    int rc = rtMemcpy(devArgs_, size, &localArgs, size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        ALOG_ERROR_F("rtmemcpy failed %p rc %d\n", devArgs_, rc);
        return rc;
    }
    Args args{nullptr, nullptr, nullptr, nullptr, nullptr, devArgs_};
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    return rtKernelLaunchWithHandleV2(binHdl_, 0, blockDim_, &rtArgs, nullptr, aicoreStream, nullptr);
}

int DeviceRunner::LaunchAiCpu(
    const rtStream_t aicpuStream, const uint64_t taskId, const uint64_t taskData, int taskType) const {
    struct Args {
        DeviceArgs devArgs;
        const char kernelName[32] = {"StaticTileFwkKernelServer"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.devArgs = args_;
    args.devArgs.taskType = taskType;
    args.devArgs.taskId = taskId;
    args.devArgs.taskData = taskData;

    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_AICPU", aicpuNum_, &rtArgs, nullptr, aicpuStream, 0);
}

void DeviceRunner::AllocDfxMetricMemory() {
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv; i++) {
        KernelArgs kernelArgs;
        memset_s(&kernelArgs, sizeof(kernelArgs), 0, sizeof(kernelArgs));
        kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] =
            reinterpret_cast<int64_t>(DevAlloc(MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics)));
        rtMemcpy((reinterpret_cast<uint8_t *>(args_.sharedBuffer)) + i * SHARED_BUFFER_SIZE, sizeof(kernelArgs),
            reinterpret_cast<uint8_t *>(&kernelArgs), sizeof(kernelArgs), RT_MEMCPY_HOST_TO_DEVICE);
        ALOG_INFO_F("aicore %u , dfxaddr 0x%ld \n", i, kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }
}

int DeviceRunner::RunAsync(rtStream_t aicpuStream, rtStream_t aicoreStream, int64_t taskId, uint64_t taskData, int taskType) {
    int rc;

    aclrtEvent event;
    rc = aclrtCreateEvent(&event);
    if (rc < 0) {
        ALOG_INFO_F("aclrtCreateEvent failed %d\n", rc);
    }

    rc = aclrtRecordEvent(event, aicpuStream);
    if (rc < 0) {
        ALOG_INFO_F("aclrtRecordEvent failed %d\n", rc);
    }

    rc = aclrtStreamWaitEvent(aicoreStream, event);
    if (rc < 0) {
        ALOG_INFO_F("aclrtStreamWaitEvent failed %d\n", rc);
    }

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
    AllocDfxMetricMemory();
#else
    int size = (args_.nrAic + args_.nrAiv) * SHARED_BUFFER_SIZE;
    rtMemset(reinterpret_cast<void *>(args_.sharedBuffer), size, 0, size);
#endif

    std::lock_guard<FileLock> lock(lock_);
    rc = LaunchAiCore(aicoreStream, DEVICE_TASK_TYPE_STATIC);
    if (rc < 0) {
        ALOG_INFO_F("launch aicpu failed %d\n", rc);
        return rc;
    }

    if (!g_IsFirstInit) {
        InitAiCpuSoBin();
    }
    g_IsFirstInit = true;

    rc = LaunchAiCpu(aicpuStream, taskId, taskData, taskType);
    if (rc < 0) {
        ALOG_INFO_F("launch aicpu failed %d\n", rc);
        return rc;
    }

    return 0;
}

void DeviceRunner::Dump() {
    ALOG_INFO_F("======== aicore status ========");

    int coreNum = args_.nrAic + args_.nrAiv;
    uint64_t size = coreNum * SHARED_BUFFER_SIZE;
    std::vector<uint64_t> buffer(size / sizeof(uint64_t));
    int rc =
        rtMemcpy(buffer.data(), size, reinterpret_cast<void *>(args_.sharedBuffer), size, RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        ALOG_INFO_F("rtmemcpy failed");
        return;
    }

    uint64_t buffAddr = reinterpret_cast<uint64_t>(buffer.data());
    for (int i = 0; i < coreNum; i++) {
        KernelArgs *arg = reinterpret_cast<KernelArgs *>(buffAddr + i * SHARED_BUFFER_SIZE);
        ALOG_INFO_F("aicore %d hello status %ld", i, arg->shakeBuffer[0]);
        ALOG_INFO_F("last_taskId %ld", arg->shakeBuffer[1]);
        ALOG_INFO_F("task status %ld", arg->shakeBuffer[2]);

        for (int k = 0; k < static_cast<int>(sizeof(arg->taskStat) / sizeof(TaskStat)); k++) {
            ALOG_INFO_F("task rsp index %d: taskId %d, subGraphID %d execStart %ld execEnd %ld\n", k,
                arg->taskStat[k].taskId, arg->taskStat[k].subGraphId, arg->taskStat[k].execStart,
                arg->taskStat[k].execEnd);
        }
    }
}

/**************************** DynamicFunction *****************************/
void DeviceRunner::DumpAiCoreExecutionTimeData() {
    json root_taskStats = json::array();
    uint32_t block_num_ = args_.GetBlockNum();
    ALOG_INFO("GetBlockNum : %d",  block_num_);
    for (uint32_t i = 0; i < block_num_; i++) {
        void* devPtr = perfData_[i];
        size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
        std::vector<uint8_t> hostBuffer(dataSize);
        rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
        Metrics *metric = reinterpret_cast<Metrics*>(hostBuffer.data());
        if (metric->taskCount > MAX_DFX_TASK_NUM_PER_CORE) {metric->taskCount = MAX_DFX_TASK_NUM_PER_CORE;} // Limit to the maximum value 
        TaskStat* taskStats = metric->tasks;
        size_t numTasks = metric->taskCount;
        std::string coreType = (i < args_.nrValidAic) ? "AIC" : "AIV";
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
            root_taskStats.push_back(coreObj);
        }
    }
    std::string jsonFilePath = config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    std::ofstream jsonFile(jsonFilePath);
    jsonFile << root_taskStats << std::endl;
    jsonFile.close();
    ALOG_INFO("tilefwk_L1_prof_data have saved in: %s",  jsonFilePath);
    std::string topo_txt_path = config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = config::LogTopFolder() + "/program.json";
    std::string draw_swim_lane_py_path = GetCurrentSharedLibPath() + "/scripts/draw_swim_lane.py";
    config::SetRunDataOption(KEY_SWIM_GRAPH_PATH, config::GetAbsoluteTopFolder() + "/merged_swimlane.json");        

    if (FileExist(program_json_path) && FileExist(topo_txt_path)) {
        ALOG_INFO("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 "+ draw_swim_lane_py_path + " \""
                                + jsonFilePath + "\" \""
                                + topo_txt_path + "\" \""
                                + program_json_path + "\" --label_type=1 --time_convert_denominator=50";
        if (system(command.c_str()) != 0) {
           ALOG_WARN("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        ALOG_WARN("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
}

void DeviceRunner::DumpAiCorePmuData() {
    ALOG_INFO_F("TODO: DumpAiCorePmuData");
}

void DeviceRunner::SynchronizeDeviceToHostProfData() {
    if (lastLaunchToSubMachineConfig_.profConfig.Contains(ProfConfig::AICORE_TIME)) {
        DumpAiCoreExecutionTimeData();
    }
}

int DeviceRunner::DynamicLaunchSynchronize(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream) {
    int rcAicore = rtStreamSynchronize(aicoreStream);
    int rcAicpu = rtStreamSynchronize(aicpuStream);
    int rcCtrl = 0;
    if (ctrlStream != nullptr) {
        rcCtrl = rtStreamSynchronize(aicpuStream);
    }
    if (IsAstDataDumpEnabled()) {
        ALOG_DEBUG_F("DataDumpServerInit is called \n");
        AdxDataDumpServerUnInit();
    }
    if (rcAicore != 0 || rcAicpu != 0 || rcCtrl != 0) {
        ALOG_WARN_F("sync stream failed aicpu:%d aicore:%d ctrl cpu:%d", rcAicpu, rcAicore, rcCtrl);
    }
    return rcAicore + rcAicpu + rcCtrl;
}

int DeviceRunner::launchDynamicAiCore(rtStream_t aicoreStream, AstKernelArgs *kernelArgs) {
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void *> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    return rtKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim_, &rtArgs, nullptr, aicoreStream, &cfg);
}

int DeviceRunner::launchDynamicAiCpu(rtStream_t aicpuStream, AstKernelArgs *kArgs) {
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, aicpuNum_, "PyptoRun");
#endif
    struct Args {
        AstKernelArgs kArgs;
        const char kernelName[32] = {"DynTileFwkKernelServer"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;
    args.kArgs = *kArgs;
    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum_, &rtArgs, nullptr, aicpuStream, 0);
}

void DeviceRunner::InitAiCpuSoBin() {
    std::vector<char> buffer;
    std::string fileName = GetCurrentSharedLibPath() + "/libtilefwk_backend_server.so";
    if (!ReadBytesFromFile(fileName, buffer)) {
        ALOG_ERROR_F("Read bin form tilefwk_backend_server.so failed, please check the so[%s]", fileName.c_str());
        return;
    }
    size_t aicpuDataLength = buffer.size();
    auto dAicpuData = DevAlloc(aicpuDataLength);
    rtMemcpy(dAicpuData, aicpuDataLength, reinterpret_cast<void *>(buffer.data()),
             aicpuDataLength, RT_MEMCPY_HOST_TO_DEVICE);
    args_.aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    args_.aicpuSoLen = buffer.size();
    args_.deviceId = GetLogDeviceId();
}

int DeviceRunner::launchDynamicAiCpuInit(rtStream_t aicpuStream, AstKernelArgs *kArgs) {
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, 1, "PyptoInit");
#endif
    struct Args {
        AstKernelArgs kArgs;
        const char kernelName[32] = {"DynTileFwkKernelServerInit"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs = *kArgs;

    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rtArgs, nullptr, aicpuStream, 0);
}

int DeviceRunner::RunPrepare() {
   for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv; i++) {
        rtMemcpy((reinterpret_cast<uint8_t *>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) + i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t),
            reinterpret_cast<uint8_t *>(&perfData_[i]),
            sizeof(uint64_t),
            RT_MEMCPY_HOST_TO_DEVICE);
    }
    if (isCapture_) {
        aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        aclmdlRICaptureThreadExchangeMode(&mode);
        ALOG_INFO_F("captureMode is: %d", mode);
    }
    return 0;
}

int DeviceRunner::RunPreSync(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    aclrtEvent event;
    int rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    if (rc < 0) {
        ALOG_ERROR_F("aclrtCreateEvent failed %d\n", rc);
        return rc;
    }

    rc = aclrtRecordEvent(event, aicoreStream);
    if (rc < 0) {
        ALOG_ERROR_F("aclrtRecordEvent failed %d\n", rc);
        return rc;
    }

    rc = aclrtStreamWaitEvent(aicpuStream, event);
    if (rc < 0) {
        ALOG_ERROR_F("aclrtStreamWaitEvent failed %d\n", rc);
        return rc;
    }
    return 0;
}

int DeviceRunner::RunPost(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    int rc;

    aclrtEvent event;
    rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    if (rc < 0) {
        ALOG_INFO_F("aclrtCreateEvent failed %d\n", rc);
    }

    rc = aclrtRecordEvent(event, aicpuStream);
    if (rc < 0) {
        ALOG_INFO_F("aclrtRecordEvent failed %d\n", rc);
    }

    rc = aclrtStreamWaitEvent(aicoreStream, event);
    if (rc < 0) {
        ALOG_INFO_F("aclrtStreamWaitEvent failed %d\n", rc);
    }

    return 0;
}

int DeviceRunner::DynamicKernelLaunch(rtStream_t aicpuStream, rtStream_t aicoreStream, AstKernelArgs *kernelArgs, int blockdim) {
    uint64_t startTime = MsprofSysCycleTime();
    int rc = launchDynamicAiCpuInit(aicpuStream, kernelArgs);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicpu init failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, 1, MSPROF_GE_TASK_TYPE_AI_CPU);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, aicpuNum_, MSPROF_GE_TASK_TYPE_AI_CPU);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);
    return rc;
}

int DeviceRunner::DynamicSeparateLaunch(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream,
    AstKernelArgs *kernelArgs, int blockdim) {
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
    std::string initKernel =  OpInfoManager::GetInstance().GetOpFuncName() + "Init";
    std::string mainKernel =  OpInfoManager::GetInstance().GetOpFuncName() + "Run";
    uint64_t startTime = MsprofSysCycleTime();
    int rc = LoadAicpuOp::GetInstance().LaunchCustomOp(ctrlStream, kernelArgs, initKernel);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_AI_CPU, true);

    rc = RunPreSync(ctrlStream, aicoreStream);
    if (rc < 0) {
        ALOG_ERROR_F("prepare failed %d\n", rc);
        return rc;
    }

    startTime = MsprofSysCycleTime();
    rc = LoadAicpuOp::GetInstance().LaunchCustomOp(ctrlStream, kernelArgs, mainKernel);
    if (rc < 0) {
        ALOG_ERROR_F("launch custom aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_AI_CPU, true);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCpu(aicpuStream, kernelArgs);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, aicpuNum_, MSPROF_GE_TASK_TYPE_AI_CPU);

    startTime = MsprofSysCycleTime();
    rc = launchDynamicAiCore(aicoreStream, kernelArgs);
    if (rc < 0) {
        ALOG_ERROR_F("launch aicore failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(startTime, blockdim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);

    rc = RunPost(ctrlStream, aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream, int64_t taskId,
    AstKernelArgs *kernelArgs, int blockdim, int launchAicpuNum) {
    InitializeErrorCallback();
    if (!g_IsFirstInit) {
        InitAiCpuSoBin();
    }
    g_IsFirstInit = true;
    #ifdef BUILD_WITH_NEW_CANN
    if (!g_IsNullLaunched) {
        auto ret = LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, 1, "PyptoNull");
        if (ret != 0) {
            ALOG_ERROR_F("launch built null failed");
            return ret;
        }
    }
    g_IsNullLaunched = true;
    #endif
    auto localArgs = args_;
    localArgs.taskId = taskId;
    localArgs.taskType = DEVICE_TASK_TYPE_DYN;
    if (kernelArgs == nullptr) {
        return -1;
    }
    lastLaunchToSubMachineConfig_ = kernelArgs->toSubMachineConfig;
    localArgs.machineConfig = kernelArgs->machineConfig;
    localArgs.toSubMachineConfig = kernelArgs->toSubMachineConfig;
    localArgs.nrValidAic = blockdim;
    localArgs.nrAicpu = launchAicpuNum;
    blockDim_ = blockdim;
    aicpuNum_ = launchAicpuNum;
    localArgs.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(blockdim, aicpuNum_);
    localArgs.enableCtrl = ctrlStream == nullptr ? 1 : 0; // need set 0 if use custom cpu launch ctrl cpu
    localArgs.validGetPgMask = machine::GetRA()->GetValidGetPgMask();
    localArgs.disableSync = config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_NO_DEVICE_TENSOR_DEPEND ? 1 : 0;
    localArgs.generalAddr = kernelArgs->opMetaAddrs.generalAddr;
    localArgs.stitchPoolAddr = kernelArgs->opMetaAddrs.stitchPoolAddr;
    localArgs.isGETensorList = kernelArgs->toSubMachineConfig.isGETensorList;
    int rc = rtMemcpy(kernelArgs->cfgdata, sizeof(localArgs), &localArgs, sizeof(localArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        ALOG_ERROR_F("Copy args failed %p rc %d\n", kernelArgs->cfgdata, rc);
        return rc;
    }
    if (RunPrepare() < 0) {
        ALOG_ERROR_F("Prepare failed %d\n", rc);
        return rc;
    }
    if (ctrlStream == nullptr) {
        return DynamicKernelLaunch(aicpuStream, aicoreStream, kernelArgs, blockDim_);
    } else {
        return DynamicSeparateLaunch(aicpuStream, ctrlStream, aicoreStream, kernelArgs, blockDim_);
    }
}

void DeviceRunner::ReportHostProfInfo(uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore) {
    if (isOpenHostProf_) {
        uint64_t endTime = MsprofSysCycleTime();
        if (isCore) {
            uint32_t mixBlockDim = MIX_BLOCK_DIM;
            blockDim = (mixBlockDim << HIGHT_BIT) | blockDim;
            hostProf_.HostProfReportContextInfo(endTime);
        }
        if (isHostProfL1_) {
            hostProf_.HostProfReportNodeInfo(endTime, blockDim, taskType);
        }
        endTime = MsprofSysCycleTime();
        hostProf_.HostProfReportApi(startTime, endTime);
    }
}

int DeviceRunner::DynamicRun(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream, int64_t taskId, AstKernelArgs *kernelArgs, int blockdim, int launchAicpuNum) {
    int rc = DynamicLaunch(aicpuStream, ctrlStream, aicoreStream, taskId, kernelArgs, blockdim, launchAicpuNum);
    if (rc < 0) {
        return rc;
    }
    if (isCapture_) {
        return 0;
    }
    return DynamicLaunchSynchronize(aicpuStream, ctrlStream, aicoreStream);
}

/**************************** DynamicFunction *****************************/
std::vector<uint8_t> g_binBuf;

void DeviceRunner::SetBinData(const std::vector<uint8_t> &binBuf) {
  g_binBuf = binBuf;
  ALOG_DEBUG_F("Set kernel size:%zu", g_binBuf.size());
  return;
}

int DeviceRunner::RegisterKernelBin(void **hdl) {
    if (*hdl) {
        binHdl_ = *hdl;
        ALOG_DEBUG_F("RegisterKernelBin reuse cache.");
        return 0;
    }
    void *bin = nullptr;
    size_t binSize = 0;
    if (g_binBuf.size() != 0) {
        bin = g_binBuf.data();
        binSize = g_binBuf.size();
        ALOG_DEBUG_F("Reg dynamic bin size %zu.", binSize);
    } else {
        bin = _binary_kernel_o_start;
        binSize = _binary_kernel_o_end - _binary_kernel_o_start;
        ALOG_DEBUG_F("Reg static bin size %zu.", binSize);
    }
    rtDevBinary_t binary{.magic = RT_DEV_BINARY_MAGIC_ELF, .version = 0, .data = bin, .length = binSize};
    int rc = rtRegisterAllKernel(&binary, hdl);
    if (rc != 0) {
        ALOG_ERROR("RegisterKernelBin failed\n");
    }
    binHdl_ = *hdl;
    ALOG_DEBUG_F("finish RegisterKernelBin.");
    return rc;
}

int DeviceRunner::Init(void) {
    char path[PATH_LENGTH];
    sprintf_s(path, PATH_LENGTH, "/tmp/aicpu%d.lock", devId_);
    lock_.Init(path);
    std::string builtInOpPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(builtInOpPath);
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo(builtInOpPath);
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        ALOG_ERROR("Get builtInOp Funchandle failed\n");
        return -1;
    }
    if (InitDeviceArgs(args_) != 0) {
        ALOG_ERROR("prepareArgs failed\n");
        return -1;
    }
    if (RegisterKernelBin(&binHdl_) != 0) {
        ALOG_ERROR("RegisterKernelBin failed\n");
        return -1;
    }
    return 0;
}
} // namespace npu::tile_fwk
#endif // BUILD_WITH_CANN
