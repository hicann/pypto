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
 * \file aicore_hal.h
 * \brief
 */

#pragma once

#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/aicore_constants.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/costmodel_utils.h"

namespace npu::tile_fwk::dynamic {
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_THIRTY_TWO = 32;
constexpr uint32_t SHIFT_NUM_FORTYEIGHT = 48;

const int32_t CORE_QUEUE_MODE_NUM_8 = 8;
const int32_t CORE_QUEUE_MODE_NUM_7 = 7;
const int32_t CORE_QUEUE_MODE_NUM_6 = 6;
const int32_t CORE_QUEUE_MODE_NUM_5 = 5;
const int32_t CORE_QUEUE_MODE_NUM_4 = 4;
const int32_t CORE_QUEUE_MODE_NUM_3 = 3;
const int32_t CORE_QUEUE_MODE_NUM_2 = 2;
const int32_t CORE_QUEUE_MODE_NUM_1 = 1;

const uint32_t REG_SPR_MAGIC = 0x78;
constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
constexpr int32_t AICORE_BLOCKID_MASK = 0x0FFF;

namespace DAV_2201 {
    const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0;
    const uint32_t REG_SPR_COND = 0x4C8;
}

namespace DAV_3510 {
    const uint32_t REG_SPR_DATA_MAIN_BASE = 0xD0;
    const uint32_t REG_SPR_COND = 0x5108;
}

class AicoreHAL {
public:
    inline void Init(DeviceArgs *deviceArgs, AiCoreProf *aicoreProf) {
        aicoreProf_ = aicoreProf;
        sharedBuffer_ = deviceArgs->sharedBuffer;
        regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
        regNum_ = deviceArgs->nrAic + deviceArgs->nrAiv;
        freq_ = GetFreq() / (NSEC_PER_SEC / NSEC_PER_USEC);
        readyRegQueues_.fill(nullptr);
        finishRegQueues_.fill(nullptr);
        blockIdToPhyCoreId_.fill(-1);
        args_.fill(nullptr);
        if (deviceArgs->archInfo == ArchInfo::DAV_3510) {
            regSprDataMainBase_ = DAV_3510::REG_SPR_DATA_MAIN_BASE;
            regSprCond_ = DAV_3510::REG_SPR_COND;
            isNeedWriteRegForFastPath_ = false;
        }
    }

    inline uint32_t GetRegSprDataMainBase() {
        return regSprDataMainBase_;
    }

    inline void SetMngCoreBlockId(int aicStart, int aicEnd, int aivStart, int aivEnd) {
        aicStart_ = aicStart;
        aicEnd_ = aicEnd;
        aivStart_ = aivStart;
        aivEnd_ = aivEnd;
    }

    inline void SetModel(uint64_t costModel) {
        costModel_ = reinterpret_cast<CostModel::AiCoreModel*>(costModel);
    }

    int64_t *GetRegAddrs() const { return regAddrs_; }
    uint32_t GetregNum() const { return regNum_; }

    inline uint32_t ReadReg32(int coreIdx, int offset) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          return *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset));
        }
        return 0;
    }

    inline uint32_t ReadPathReg(int coreIdx) {
        if (!isNeedWriteRegForFastPath_) {
            return 0;
        }

        return ReadReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE);
    }

    inline void WriteReg32(int coreIdx, int offset, uint32_t val) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset)) = val;
        }
        return;
    }

    inline void WriteReg32All(int aicNum, int aivNum, int offset, uint32_t val) {
        for (int i = 0; i < aicNum + aivNum; ++i) {
            if (regAddrs_[i] != 0) {
                *(reinterpret_cast<volatile uint32_t *>(regAddrs_[i] + offset)) = val;
            }
        }
    }

    inline bool IsSpecialTask(uint32_t taskId) {
        return taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP || (taskId & 0xFFFFFFFF) == AICORE_FUNC_STOP;
    }

    inline void SetReadyQueue(int coreIdx, uint64_t value) {
        if constexpr (IsDeviceMode()) {
            *readyRegQueues_[GetPhyIdByBlockId(coreIdx)] = value;
        } else {
            DEV_VERBOSE_DEBUG("set coreidx %d value %lx.", coreIdx, value);
            auto taskId = value - 1;
            if (value == 0 || taskId == AICORE_TASK_STOP || (taskId & 0xFFFFFFFF) == AICORE_FUNC_STOP) return;
            CostModelSendTask(coreIdx, taskId);
        }
    }

    inline void SetReadyQueue(int coreStart, int coreEnd, uint32_t val) {
        if constexpr (IsDeviceMode()) {
            int i, idx = coreStart;
            int n = coreEnd - coreStart;
            for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                default:
                    break;
            }
        }
    }

    inline void SetReadyQueue(const uint32_t *coreIdx, const uint32_t *vals, int n) {
        if constexpr (IsDeviceMode()) {
            for (int i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                default:
                    break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                auto taskId = vals[i] - 1;
                if (IsSpecialTask(taskId))
                    continue;
                CostModelSendTask(coreIdx[i], taskId);
            }
        }
    }

    inline void GetFinishQueue(const uint32_t *coreIdx, uint32_t *vals, int n) {
        if constexpr (IsDeviceMode()) {
            for (int i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                default:
                    break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                vals[i] = CostModelGetTask(coreIdx[i]);
            }
        }
    }

    inline void WaitFinQueue(int coreStart, int coreEnd, uint64_t val) {
        for (int idx = coreStart; idx < coreEnd; idx++) {
            uint64_t startCycle = GetCycles();
            while (*finishRegQueues_[GetPhyIdByBlockId(idx)] != val) {
                if (GetCycles() - startCycle > TIMEOUT_CYCLES) {
                    DEV_ERROR("CoreId: %d cannot get finish Flag", idx);
                    return;
                }
            }
        }
    }

    inline uint64_t GetFinishedTask(int coreIdx) {
        if constexpr (IsDeviceMode()) {
            return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]);
        } else {
            return CostModelGetTask(coreIdx);
        }
    }

    void SetTaskTimeCost(std::function<uint64_t(uint64_t, uint64_t, uint64_t)> func) { getTaskTimeCost = func; }

    uint64_t CostModelGetTask(int coreIdx) {
        auto currentTime = GetCycles();
        DEV_DEBUG("CostModel AICore polling aicore %d, time %lu.", coreIdx, currentTime);
        if (taskIds[coreIdx].empty()) return AICORE_FUNC_STOP | AICORE_FIN_MASK;
        uint64_t taskId = 0;
        while (!taskIds[coreIdx].empty() && currentTime >= taskTimes[coreIdx].front()) {
            taskId = taskIds[coreIdx].front();
            taskTimes[coreIdx].pop_front();
            taskIds[coreIdx].pop_front();
        }
        if (taskIds[coreIdx].empty()) {
            DEV_DEBUG("CostModel AICore %d finish task 0x%lx, current time %lu.", coreIdx, taskId, currentTime);
            return taskId | AICORE_FIN_MASK;
        }
        DEV_DEBUG("CostModel AICore %d running task 0x%lx, current time %lu, finish time %lu.",
                  coreIdx, taskIds[coreIdx].front(), currentTime, taskTimes[coreIdx].front());
        return taskIds[coreIdx].front();
    }

    void CostModelSendTask(int coreIdx, uint64_t taskId) {
        uint64_t time = taskIds[coreIdx].empty() ? GetCycles() : taskTimes[coreIdx].back();
        uint64_t timeCost = getTaskTimeCost == nullptr ? 0 : getTaskTimeCost(coreIdx, taskId, time);
        taskTimes[coreIdx].push_back(time + timeCost);
        taskIds[coreIdx].push_back(taskId);
        if (costModel_) {
            costModel_->SendTask(coreIdx, taskId);
        }
        DEV_DEBUG("CostModel AICore %d add task 0x%lx, new queue size %lu, finish time %lu.",
                  coreIdx, taskId, taskIds[coreIdx].size(), time + timeCost);
    }

    int64_t GetSharedBuffer() { return sharedBuffer_; }

    inline void MapRegistersForAllCores(int aicNum) {
        for (uint32_t idx = 0; idx < static_cast<u_int32_t>(aicNum * CORE_NUM_PER_AI_CORE); idx++) {
            void *addr = reinterpret_cast<void *>(regAddrs_[idx]);
            if (addr == nullptr) {
                continue;
            }
            DEV_VERBOSE_DEBUG("phy core %u Addr is %p.", idx, addr);
            volatile uint64_t *reqQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + regSprDataMainBase_);
            readyRegQueues_[idx] = reqQueueReg;
            volatile uint64_t *finishQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + regSprCond_);
            finishRegQueues_[idx] = finishQueueReg;
        }
    }

    inline int &GetPhyIdByBlockId(int coreIdx) {
        return blockIdToPhyCoreId_[coreIdx];
    }

    Metrics* GetMetrics(int coreIdx) {
        volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        volatile Metrics*  metric = reinterpret_cast<Metrics *>(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        DEV_INFO("aicore %d host alloc metric memory :%p.", coreIdx, metric);
        if (metric == nullptr) {
            DEV_ERROR("aicore %d Null metric.", coreIdx);
           return nullptr;
        }

        uint64_t cycles_start = GetCycles();
        while (metric->isMetricStop != 1) {
            if (GetCycles() - cycles_start > PROF_DUMP_TIMEOUT_CYCLES) {
                DEV_ERROR("wait metrics done timeout !!!.");
                return nullptr;
            }
        }; // wait aicore dcci metric data finish

        return reinterpret_cast<Metrics *>(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }

    int DumpTaskProf(int coreIdx) {
        Metrics* metric =  GetMetrics(coreIdx);
        if (metric == nullptr) {
            return DEVICE_MACHINE_ERROR;
        }

        DEV_VERBOSE_DEBUG("Dump core %d prof data , task cnt %ld, metric:%p.", coreIdx, metric->taskCount, metric);
        for (int i = 0; i < metric->taskCount; i++) {
            volatile TaskStat *stat = &metric->tasks[i];
            aicoreProf_->ProfGet(coreIdx, stat->subGraphId, stat->taskId, &(metric)->tasks[i]);
            DEV_VERBOSE_DEBUG("  Dump prof for task %d, execstart: %ld execend :%ld.",
                     stat->taskId, stat->execStart, stat->execEnd);
        }
        return 0;
    }

    int DumpAicorePerfTrace(int aicpuIdx, int coreIdx, CoreType coretype, std::ostringstream& oss) {
        (void)coreIdx;
        (void)coretype;
        (void)oss;
        (void)aicpuIdx;
#if ENABLE_PERF_TRACE
        Metrics* metric =  GetMetrics(coreIdx);
        if (metric == nullptr) {
            return DEVICE_MACHINE_ERROR;
        }

        oss << "{\"blockIdx\":" << coreIdx << ",\"coreType\":\"SCHED" << aicpuIdx << "-"
            << (coretype == CoreType::AIC ? "AIC" : "AIV") << "\",\"freq\":" << freq_ << ",\"tasks\":[";

        uint64_t curCycle = 0;
        for (uint32_t type = 0; type < PERF_TRACE_CORE_MAX; type++) {
            for (uint32_t cnt = 0; cnt < metric->perfTraceCnt[type]; cnt++) {
                curCycle = metric->perfTrace[type][cnt];
                if (curCycle == 0) {
                    break;
                }

                oss << "{\"name\":\"" << AicorePerfTraceName[type];
                if (metric->perfTraceDevTaskId[type][cnt] != INVALID_DEV_TASK_ID) {
                    oss << "(" << metric->perfTraceDevTaskId[type][cnt] << ")";
                }

                oss << "\",\"end\":" << curCycle << "}"
                    << (((type == PERF_TRACE_CORE_MAX - 1) && (cnt ==  metric->perfTraceCnt[type] - 1)) ? "" : ",");
            }
        }
        oss << "]}";
        if (!aicoreProf_->ProfIsEnable()) {
            memset_s(metric, sizeof(Metrics), 0, sizeof(Metrics));
        }
#endif
        return DEVICE_MACHINE_OK;
    }

    void DumpAicoreStatus(int coreIdx) const {
        volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        DEV_VERBOSE_DEBUG("!!***********************aicore %d last status **************************!!", coreIdx);
        DEV_VERBOSE_DEBUG("hello status %ld.", arg->shakeBuffer[0]);
        DEV_VERBOSE_DEBUG("last_taskId %ld task status [%ld, %ld, %ld, %ld].", arg->shakeBuffer[NUM_ONE],
            arg->shakeBuffer[NUM_TWO], arg->shakeBuffer[NUM_THREE], arg->shakeBuffer[NUM_FOUR], arg->shakeBuffer[NUM_FIVE]);

        for (size_t i = 0; i < sizeof(arg->taskStat) / sizeof(TaskStat); i++) {
            DEV_VERBOSE_DEBUG("task rsp index %lu: taskId %d, subGraphID %d execStart %ld execEnd %ld.", i,
                arg->taskStat[i].taskId, arg->taskStat[i].subGraphId,
                arg->taskStat[i].execStart, arg->taskStat[i].execEnd);
        }
    }

    uint64_t GetAicoreStatus(int coreIdx) const {
        int aicoreStatusIndex = 2;
        volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        return arg->shakeBuffer[aicoreStatusIndex];
    }

    inline void InitTaskData(int coreIdx, int64_t funcdata, int64_t buffer) {
        (void)buffer;
        if constexpr (IsDeviceMode()) {
            if (args_[coreIdx] == nullptr) {
                args_[coreIdx] = reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
            }
            volatile KernelArgs *arg = args_[coreIdx];
#if ENABLE_AICORE_PRINT
            arg->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX] = buffer;
            __sync_synchronize();
#endif
            arg->shakeBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX] = funcdata;
            arg->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = 0;
        } else {
            if (costModel_) {
                costModel_->InitData(coreIdx, funcdata);
            }
        }
    }

    int HandShakeByGm(int coreIdx, int64_t dotStatus) {
        auto args =
            reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
        args->taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
        volatile int64_t *shakeBuffer = args->shakeBuffer;
        uint64_t cycles_start = GetCycles();
        while ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            if (GetCycles() - cycles_start > HAND_SHAKE_TIMEOUT) {
                DEV_ERROR("hand shake %d timeout.\n", coreIdx);
                return -1;
            }
        }
        args_[coreIdx] = args;
        GetPhyIdByBlockId(coreIdx) = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        return DEVICE_MACHINE_OK;
    }

    bool TryHandShakeByGm(int coreIdx, int64_t dotStatus) {
        auto args =
            reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
        volatile int64_t *shakeBuffer = args->shakeBuffer;
        if ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            return false;
        }

        args_[coreIdx] = args;
        args->taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
        GetPhyIdByBlockId(coreIdx) = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        if (isNeedWriteRegForFastPath_) {
            WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
        }
        SetReadyQueue(coreIdx, (uint64_t)0);
        DEV_VERBOSE_DEBUG("hand shake success coreidex:%d", coreIdx);
        return true;
    }

    // We must makesure close 0x18 before aicore exit.
    void ResetShakeBuf(int coreIdx) {
        if (isNeedWriteRegForFastPath_) {
            WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            __sync_synchronize();
        }
        args_[coreIdx]->shakeBuffer[0] = 0;
        args_[coreIdx]->shakeBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
        args_[coreIdx]->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = AICORE_SAY_GOODBYE;
        return;
    }

    volatile TaskStat *GetTaskStat(int coreIdx, int pos) {
        volatile TaskStat *stat = &args_[coreIdx]->taskStat[pos];
        return stat;
    }

private:
    int64_t sharedBuffer_;
    int64_t* regAddrs_{nullptr};
    int aicStart_{0};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint32_t regNum_{0};
    uint64_t freq_{50};

    std::array<volatile KernelArgs*, MAX_AICORE_NUM> args_;

    std::array<volatile uint64_t*, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t*, MAX_AICORE_NUM> finishRegQueues_;

    // cost model aicore
    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> getTaskTimeCost{nullptr};
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskIds;
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskTimes;

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;

    uint32_t regSprDataMainBase_{DAV_2201::REG_SPR_DATA_MAIN_BASE};
    uint32_t regSprCond_{DAV_2201::REG_SPR_COND};

    bool isNeedWriteRegForFastPath_{true};
    AiCoreProf *aicoreProf_{nullptr};
    CostModel::AiCoreModel *costModel_{nullptr};
};
}
