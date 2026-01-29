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
 * \file aicore_prof.cpp
 * \brief
 */

#include "aicore_prof.h"
#include "aicore_manager.h"
#include "machine/device/tilefwk/aicpu_common.h"
namespace {
constexpr int AICPUNUM = 6;
constexpr int64_t HIG_32BIT = 32;
} // namespace

namespace npu::tile_fwk::dynamic {
AiCoreProfLevel CreateProfLevel(ProfConfig profConfig) {
    if (profConfig.Contains(ProfConfig::AICORE_PMU)) {
        return PROF_LEVEL_FUNC_LOG_PMU;
    } else if (profConfig.Contains(ProfConfig::AICORE_TIME) ) {
        return PROF_LEVEL_FUNC_LOG;
    } else if (profConfig.Contains(ProfConfig::AICPU_FUNC)) {
        return PROF_LEVEL_FUNC;
    }
    return PROF_LEVEL_OFF;
}

bool ProfCheckLevel(uint64_t feature) {
    if (AdprofCheckFeatureIsOn == nullptr) {
        return false;
    }
    return AdprofCheckFeatureIsOn(feature) > 0;
}


void AiCoreProf::ProInitHandShake() {
    handkShakeMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    handkShakeHeadSize_ = sizeof(MsprofAicpuHandShakeHead);
    handShakeDataSize_ = sizeof(AiCpuHandShakeSta);
    HandShakeMsg_.resize(AICPUNUM);
    HandShakeHead_.resize(AICPUNUM, nullptr);
    handShakeData_.resize(AICPUNUM, nullptr);
    for (int32_t i = 0; i < AICPUNUM; i++) {
        HandShakeHead_[i] = reinterpret_cast<MsprofAicpuHandShakeHead *>(&HandShakeMsg_[i].data);
        handShakeData_[i] =
            reinterpret_cast<AiCpuHandShakeSta *>(reinterpret_cast<uintptr_t>(HandShakeHead_[i]) + handkShakeHeadSize_);
        HandShakeHead_[i]->cnt = 0;

        HandShakeMsg_[i].magicNumber = 0x5A5AU;
        HandShakeMsg_[i].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        HandShakeMsg_[i].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        HandShakeMsg_[i].threadId = hostAicoreMng_.aicpuIdx_;
        HandShakeMsg_[i].dataLen = handkShakeHeadSize_;

        HandShakeHead_[i]->magicNumber = 0x6BD3U;
        HandShakeHead_[i]->coreId = i;
        HandShakeHead_[i]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(i));
        HandShakeHead_[i]->dataType = PROF_DATATYPE_HAND_SHAKE;
        HandShakeHead_[i]->taskId = 0;
        HandShakeHead_[i]->streamId = 0;
    }
    DEV_INFO("ProfInitHandShake finish.");
    sleep(1);
}

void AiCoreProf::ProInitAiCpuTaskStat() {
    aiCpuStatMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    aiCpuStatHeadSize_ = sizeof(MsProfAiCpuTaskStatHead);
    aiCpuStatDataSize_ = sizeof(AiCpuTaskStat);
    aiCpuStatMsg_.resize(AICPUNUM);
    aiCpuStatHead_.resize(AICPUNUM, nullptr);
    aiCpuStatData_.resize(AICPUNUM, nullptr);
    for (int32_t i = 0; i < AICPUNUM; i++) {
        aiCpuStatHead_[i] = reinterpret_cast<MsProfAiCpuTaskStatHead *>(&aiCpuStatMsg_[i].data);
        aiCpuStatData_[i] =
            reinterpret_cast<AiCpuTaskStat *>(reinterpret_cast<uintptr_t>(aiCpuStatHead_[i]) + aiCpuStatHeadSize_);
        aiCpuStatHead_[i]->cnt = 0;

        aiCpuStatMsg_[i].magicNumber = 0x5A5AU;
        aiCpuStatMsg_[i].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        aiCpuStatMsg_[i].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        aiCpuStatMsg_[i].threadId = hostAicoreMng_.aicpuIdx_;
        aiCpuStatMsg_[i].dataLen = aiCpuStatHeadSize_;

        aiCpuStatHead_[i]->magicNumber = 0x6BD3U;
        aiCpuStatHead_[i]->coreId = i;
        aiCpuStatHead_[i]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(i));
        aiCpuStatHead_[i]->dataType = PROF_DATATYPE_EXE;
        aiCpuStatHead_[i]->taskId = 0;
        aiCpuStatHead_[i]->streamId = 0;
    }
    DEV_INFO("ProfInitAicpuStat finish.");
    sleep(1);
}

void AiCoreProf::ProfInit([[maybe_unused]]int64_t *regAddrs, [[maybe_unused]]int64_t *pmuEventAddrs,
    ProfConfig profConfig, ArchInfo archInfo) {
    DEV_DEBUG("Begin Prof init");
    profLevel_ = CreateProfLevel(profConfig);
    coreNum_ = hostAicoreMng_.GetAllAiCoreNum();
    if (archInfo == ArchInfo::DAV_2201) {
        profReportAdditionalInfoFunc_ = AdprofReportAdditionalInfo;
    } else {
        profReportAdditionalInfoFunc_ = MsprofReportAdditionalInfo;
    }
    DEV_DEBUG("Pypto config prof level is %d", profLevel_);
    archInfo_ = archInfo;
    if ((ProfCheckLevel(PROF_TASK_TIME_L2) == true) || (profLevel_ == PROF_LEVEL_FUNC_LOG) || (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU)) {
        profLevel_ = PROF_LEVEL_FUNC_LOG;
        ProfInitLog();
        #if PMU_COLLECT
            ProfInitPmu(regAddrs, pmuEventAddrs);
            profLevel_ = PROF_LEVEL_FUNC_LOG_PMU;
        #endif
    } else {
        profLevel_ = PROF_LEVEL_OFF;
        DEV_INFO("aicore profiling is closed..");
        return;
    }
    hostAicoreMng_.SetDotStatus(static_cast<int64_t>(profLevel_));
    DEV_INFO("aicore profiling is opened, level is %d.", profLevel_);
}

void AiCoreProf::ProfStart() {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }

    DEV_INFO("aicore profiling start.");

    if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        ProfStartPmu();
    }
}

void AiCoreProf::ProGetHandShake(int &threadIdx, const struct AiCpuHandShakeSta *handShakeStat) {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }
    MsprofAicpuHandShakeHead *handShakeHead = HandShakeHead_[threadIdx];
    PyPtoMsprofAdditionalInfo &handShakeMsg = HandShakeMsg_[threadIdx];
    DEV_DEBUG("aicore profiling gen handShake mesg, coreId: %d thread id: %d, shakeHand used %lu.",
        handShakeStat->coreId, threadIdx, (handShakeStat->shakeEnd - handShakeStat->shakeStart));
    if (handShakeHead->cnt < handkShakeMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(
                     reinterpret_cast<uintptr_t>(handShakeData_[threadIdx]) + handShakeDataSize_ * handShakeHead->cnt),
            handShakeDataSize_, handShakeStat, handShakeDataSize_);
        handShakeMsg.dataLen += handShakeDataSize_;
        handShakeHead->cnt++;
    } else if (handShakeHead->cnt == logDataMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(
                     reinterpret_cast<uintptr_t>(handShakeData_[threadIdx]) + handShakeDataSize_ * handShakeHead->cnt),
            handShakeDataSize_, handShakeStat, handShakeDataSize_);
        handShakeHead->cnt++;
        handShakeMsg.dataLen += logDataSize_;
        int32_t ret = profReportAdditionalInfoFunc_(1, &handShakeMsg, sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG(
            "aicore profiling send log mesg, core id: %d, task num: %d, ret: %d.", threadIdx, handShakeHead->cnt, ret);
        // reset
        (void)(ret);
        handShakeHead->cnt = 0;
        handShakeMsg.dataLen = handkShakeHeadSize_;
    }
}

void AiCoreProf::ProfGet(int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, const struct TaskStat *taskStat) {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }

    taskCnt_++;
    if (profLevel_ == PROF_LEVEL_FUNC_LOG) {
        ProfGetLog(coreIdx, taskStat);
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        ProfGetLog(coreIdx, taskStat);
        ProfGetPmu(coreIdx, subGraphId, taskId, taskStat);
    }
}

void AiCoreProf::ProfGetSwitch(int64_t &flag) const {
    if (profLevel_ == PROF_LEVEL_FUNC_LOG) {
        flag |= 0x1;
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        flag |= 0x3;
    }
}

void AiCoreProf::ProfStop() {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG) {
        ProfStopLog();
    } else if (profLevel_ == PROF_LEVEL_FUNC_LOG_PMU) {
        ProfStopPmu();
        ProfStopLog();
    }
    DEV_INFO("aicore profiling stop, total run task num: %lu.", taskCnt_);
}

inline void AiCoreProf::ProfInitLog() {
    logMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    logHeadSize_ = sizeof(MsprofAicpuPyPtoLogHead);
    logDataSize_ = sizeof(MsprofAicpuPyPtoLogData);
    logMsg_.resize(coreNum_);
    logHead_.resize(coreNum_, nullptr);
    logData_.resize(coreNum_, nullptr);
    for (int32_t i = 0; i < coreNum_; i++) {
        logHead_[i] = reinterpret_cast<MsprofAicpuPyPtoLogHead *>(&logMsg_[i].data);
        logData_[i] =
            reinterpret_cast<MsprofAicpuPyPtoLogData *>(reinterpret_cast<uintptr_t>(logHead_[i]) + logHeadSize_);
        logHead_[i]->cnt = 0;

        logMsg_[i].magicNumber = 0x5A5AU;
        logMsg_[i].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        logMsg_[i].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        logMsg_[i].threadId = hostAicoreMng_.aicpuIdx_;
        logMsg_[i].dataLen = logHeadSize_;
        logHead_[i]->magicNumber = 0x6BD3U;
        logHead_[i]->coreId = i;
        logHead_[i]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(i));
        logHead_[i]->dataType = PROF_DATATYPE_LOG;
        logHead_[i]->taskId = 0;
        logHead_[i]->streamId = 0;
    }
    DEV_INFO("ProfInitLog finish.");
}

inline void AiCoreProf::ProfStopLog() {
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        if (logHead_[coreIdx]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &logMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG("aicore profiling send log mesg, core id: %d, task num: %d, ret: %d.", coreIdx,
                logHead_[coreIdx]->cnt, ret);
            (void)(ret);
            memset_s(&logMsg_[coreIdx], logMsgSize_, 0, logMsgSize_);
        }
    });
}

inline void AiCoreProf::ProfGetLog(int32_t coreIdx, const struct TaskStat *taskStat) {
    MsprofAicpuPyPtoLogHead *logHead = logHead_[coreIdx];
    PyPtoMsprofAdditionalInfo &logMsg = logMsg_[coreIdx];
    if (logHead->cnt < logDataMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(logData_[coreIdx]) + logDataSize_ * logHead->cnt),
            logDataSize_, taskStat, logDataSize_);
        logMsg.dataLen += logDataSize_;
        logHead->cnt++;
        DEV_DEBUG("aicore profiling gen log mesg, taskid: %d core id: %d, task start: %ld, end: %ld.",
            taskStat->taskId, coreIdx, taskStat->execStart, taskStat->execEnd);
    } else if (logHead->cnt == logDataMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(logData_[coreIdx]) + logDataSize_ * logHead->cnt),
            logDataSize_, taskStat, logDataSize_);
        logHead->cnt++;
        logMsg.dataLen += logDataSize_;
        int32_t ret = profReportAdditionalInfoFunc_(1, &logMsg, sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG("aicore profiling send log mesg, core id: %d, task num: %d, ret: %d.", coreIdx, logHead->cnt, ret);
        // reset
        (void)(ret);
        logHead->cnt = 0;
        logMsg.dataLen = logHeadSize_;
    }
}

void AiCoreProf::ProfInitPmu(int64_t *regAddrs, int64_t *pmuEventAddrs) {
    pmuMsgSize_ = sizeof(PyPtoMsprofAdditionalInfo);
    pmuHeadSize_ = sizeof(MsprofAicpuPyPtoPmuHead);
    pmuDataSize_ = sizeof(MsprofAicpuPyPtoPmuData);
    pmuMsg_.resize(coreNum_);
    pmuHead_.resize(coreNum_, nullptr);
    pmuData_.resize(coreNum_, nullptr);
    for (int32_t i = 0; i < coreNum_; i++) {
        pmuHead_[i] = reinterpret_cast<MsprofAicpuPyPtoPmuHead *>(&pmuMsg_[i].data);
        pmuData_[i] =
            reinterpret_cast<MsprofAicpuPyPtoPmuData *>(reinterpret_cast<uintptr_t>(pmuHead_[i]) + pmuHeadSize_);
        pmuHead_[i]->cnt = 0;
    }

    pmuCnt0Plain_.resize(coreNum_, nullptr);
    pmuCnt1Plain_.resize(coreNum_, nullptr);
    pmuCnt2Plain_.resize(coreNum_, nullptr);
    pmuCnt3Plain_.resize(coreNum_, nullptr);
    pmuCnt4Plain_.resize(coreNum_, nullptr);
    pmuCnt5Plain_.resize(coreNum_, nullptr);
    pmuCnt6Plain_.resize(coreNum_, nullptr);
    pmuCnt7Plain_.resize(coreNum_, nullptr);
    pmuCntTotal0Plain_.resize(coreNum_, nullptr);
    pmuCntTotal1Plain_.resize(coreNum_, nullptr);

    pmuCnt8Plain_.resize(coreNum_, nullptr);
    pmuCnt9Plain_.resize(coreNum_, nullptr);
    regAddrs_ = regAddrs;
    pmuEventAddrs_ = pmuEventAddrs;

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it != kArchPmuConfigs.end()) {
        size_t pmuCntSize = it->second.pmuCntIdxOffsets.size();
        if (pmuCntSize == MAX_PMU_CNT) {
            DEV_INFO("0: %x, 1: %x, 2: %x, 3: %x, 4: %x, 5: %x, 6: %x, 7: %x.",
                (uint32_t)pmuEventAddrs_[0], (uint32_t)pmuEventAddrs_[1], (uint32_t)pmuEventAddrs_[2],
                (uint32_t)pmuEventAddrs_[3], (uint32_t)pmuEventAddrs_[4], (uint32_t)pmuEventAddrs_[5],
                (uint32_t)pmuEventAddrs_[6], (uint32_t)pmuEventAddrs_[7]);
        } else if (pmuCntSize == MAX_PMU_CNT_3510) {
            DEV_INFO("0: %x, 1: %x, 2: %x, 3: %x, 4: %x, 5: %x, 6: %x, 7: %x, 8: %x, 9: %x.",
                (uint32_t)pmuEventAddrs_[0], (uint32_t)pmuEventAddrs_[1], (uint32_t)pmuEventAddrs_[2],
                (uint32_t)pmuEventAddrs_[3], (uint32_t)pmuEventAddrs_[4], (uint32_t)pmuEventAddrs_[5],
                (uint32_t)pmuEventAddrs_[6], (uint32_t)pmuEventAddrs_[7], (uint32_t)pmuEventAddrs_[8],
                (uint32_t)pmuEventAddrs_[9]);
        }
    }
}

void AiCoreProf::ReadPmuCounters(const int32_t coreIdx) const {
    volatile uint32_t dummy_read = 0;
    auto read_reg = [&dummy_read](volatile uint32_t *reg) {
        dummy_read = *reg; // 通过volatile访问确保实际读取操作
    };

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto &cfg = it->second;

    std::vector<const std::vector<volatile uint32_t *> *> pmuCntPlains = {
        &pmuCnt0Plain_, &pmuCnt1Plain_, &pmuCnt2Plain_, &pmuCnt3Plain_,
        &pmuCnt4Plain_, &pmuCnt5Plain_, &pmuCnt6Plain_, &pmuCnt7Plain_,
        &pmuCnt8Plain_, &pmuCnt9Plain_
    };

    for (size_t i = 0; i < cfg.pmuCntOffsets.size(); ++i) {
        read_reg((*pmuCntPlains[i])[coreIdx]);
    }

    read_reg(pmuCntTotal0Plain_[coreIdx]);
    read_reg(pmuCntTotal1Plain_[coreIdx]);

    (void)dummy_read; // 抑制未使用变量警告
}

void AiCoreProf::SetPmuEvents(void *mapBase, const int32_t coreIdx) const {
    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto &cfg = it->second;
    for (size_t i = 0; i < cfg.pmuCntIdxOffsets.size(); ++i) {
        uint32_t *cntIdxAddr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.pmuCntIdxOffsets[i]);
        *cntIdxAddr = pmuEventAddrs_[i];
    }
    (void)coreIdx;
}

AiCoreProf::PmuCtrlAddrs AiCoreProf::InitPmuRegAddrsForCore(void *addr, void *mapBase, int coreIdx) {
    PmuCtrlAddrs addrs;
    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return addrs;
    }
    const auto &cfg = it->second;

    std::vector<std::vector<volatile uint32_t *> *> pmuCntPlains = {
        &pmuCnt0Plain_, &pmuCnt1Plain_, &pmuCnt2Plain_, &pmuCnt3Plain_,
        &pmuCnt4Plain_, &pmuCnt5Plain_, &pmuCnt6Plain_, &pmuCnt7Plain_,
        &pmuCnt8Plain_, &pmuCnt9Plain_
    };

    for (size_t i = 0; i < cfg.pmuCntOffsets.size(); ++i) {
        (*pmuCntPlains[i])[coreIdx] = reinterpret_cast<volatile uint32_t *>(reinterpret_cast<uint8_t *>(addr) + cfg.pmuCntOffsets[i]);
    }

    pmuCntTotal0Plain_[coreIdx] = reinterpret_cast<volatile uint32_t *>(reinterpret_cast<uint8_t *>(addr) + cfg.pmuCntTotal0Offset);
    pmuCntTotal1Plain_[coreIdx] = reinterpret_cast<volatile uint32_t *>(reinterpret_cast<uint8_t *>(addr) + cfg.pmuCntTotal1Offset);

    addrs.ctrl0Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.ctrl0Offset);
    if (cfg.ctrl1Offset != 0) {
        addrs.ctrl1Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.ctrl1Offset);
    }
    addrs.startCntCyc0Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.startCntCyc0Offset);
    addrs.startCntCyc1Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.startCntCyc1Offset);
    addrs.stopCntCyc0Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.stopCntCyc0Offset);
    addrs.stopCntCyc1Addr = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(mapBase) + cfg.stopCntCyc1Offset);

    return addrs;
}

void AiCoreProf::ProgramPmuStartForCore(void *mapBase, int coreIdx, const PmuCtrlAddrs &addrs) {
    // 在enable前先读取一次寄存器,将cnt清0
    ReadPmuCounters(coreIdx);

    // 设置PMU寄存器记录类型事件
    SetPmuEvents(mapBase, coreIdx);

    auto it = kArchPmuConfigs.find(archInfo_);
    if (it == kArchPmuConfigs.end()) {
        return;
    }
    const auto &cfg = it->second;

    *addrs.startCntCyc0Addr = 0x0;
    *addrs.startCntCyc1Addr = 0x0;
    *addrs.stopCntCyc0Addr = 0xFFFFFFFF;
    *addrs.stopCntCyc1Addr = 0xFFFFFFFF;
    
    *addrs.ctrl0Addr = cfg.ctrl0Val;
    if (cfg.ctrl1Offset != 0 && addrs.ctrl1Addr != nullptr) {
        *addrs.ctrl1Addr = cfg.ctrl1Val;
    }
}

void AiCoreProf::ProfStartPmu() {
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        void *addr = reinterpret_cast<void *>(regAddrs_[hostAicoreMng_.GetPhyIdByBlockId(coreIdx)]);
        uint32_t pageSize = static_cast<uint32_t>(sysconf(_SC_PAGESIZE));
        void *mapBase =
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(addr) & ~(static_cast<uint64_t>(pageSize) - 1));
        PmuCtrlAddrs addrs = InitPmuRegAddrsForCore(addr, mapBase, coreIdx);
        ProgramPmuStartForCore(mapBase, coreIdx, addrs);
    });
}

void AiCoreProf::ProfStopPmu() {
    hostAicoreMng_.ForEachManageAicore([&](int coreIdx) {
        if (pmuHead_[coreIdx]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &pmuMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG("aicore profiling send pmu mesg, core id: %d, task num: %d, ret: %d.", coreIdx,
                pmuHead_[coreIdx]->cnt, ret);
            (void)(ret);
            memset_s(&pmuMsg_[coreIdx], pmuMsgSize_, 0, pmuMsgSize_);
        }
    });
}

void AiCoreProf::ProfStopHandShake() {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }
    for (int i = 0; i < AICPUNUM; i++) {
        if (HandShakeHead_[i]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &HandShakeMsg_[i], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG(
                "aicore profiling send pmu mesg, core id: %d, task num: %d, ret: %d.", i, HandShakeHead_[i]->cnt, ret);
            memset_s(&HandShakeMsg_[i], handkShakeMsgSize_, 0, handkShakeMsgSize_);
            (void)(ret);
        }
    }
}

void AiCoreProf::ProfStopAiCpuTaskStat() {
    for (int i = 0; i < AICPUNUM; i++) {
        if (aiCpuStatHead_[i]->cnt != 0) {
            int32_t ret = profReportAdditionalInfoFunc_(1, &aiCpuStatMsg_[i], sizeof(PyPtoMsprofAdditionalInfo));
            DEV_DEBUG("aicore profiling send aicpu stat mesg, aicpu id: %d, task num: %d, ret: %d.", i,
                aiCpuStatHead_[i]->cnt, ret);
            memset_s(&aiCpuStatMsg_[i], aiCpuStatMsgSize_, 0, aiCpuStatMsgSize_);
            (void)(ret);
        }
    }
}

void AiCoreProf::SetAiCpuTaskStat(const uint32_t &taskId, struct AiCpuTaskStat &aiCpuTaskStat) {
    aiCpuStatMap_[taskId] = aiCpuTaskStat;
}

struct AiCpuTaskStat AiCoreProf::GetAiCpuTaskStat(const uint32_t &taskId) {
    if (aiCpuStatMap_.find(taskId) != aiCpuStatMap_.end()) {
        return aiCpuStatMap_[taskId];
    }
    return AiCpuTaskStat();
}

void AiCoreProf::ProfGetAiCpuTaskStat(int &threadIdx, struct AiCpuTaskStat *aiCpuStat) {
    if (profLevel_ == PROF_LEVEL_OFF) {
        return;
    }
    MsProfAiCpuTaskStatHead *aiCpuStatHead = aiCpuStatHead_[threadIdx];
    PyPtoMsprofAdditionalInfo &aiCpuStatMsg = aiCpuStatMsg_[threadIdx];
    if (aiCpuStatHead->cnt < aiCpuStatMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(
                   reinterpret_cast<uintptr_t>(aiCpuStatData_[threadIdx]) + aiCpuStatDataSize_ * aiCpuStatHead->cnt),
            aiCpuStatDataSize_, aiCpuStat, aiCpuStatDataSize_);
        aiCpuStatMsg.dataLen += aiCpuStatDataSize_;
        aiCpuStatHead->cnt++;
        DEV_DEBUG("aicore profiling gen aiCpuStat mesg, coreId: %d thread id: %d, startExeTask: %lu shakeHand start: "
                  "%lu, shakeHandend: %lu.",
            aiCpuStat->coreId, threadIdx, aiCpuStat->taskGetStart, aiCpuStat->execStart, aiCpuStat->execEnd);
    } else if (aiCpuStatHead->cnt == aiCpuStatMaxNum_ - 1) {
        memcpy_s(reinterpret_cast<void *>(
                     reinterpret_cast<uintptr_t>(aiCpuStatData_[threadIdx]) + aiCpuStatDataSize_ * aiCpuStatHead->cnt),
            aiCpuStatDataSize_, aiCpuStat, aiCpuStatDataSize_);
        aiCpuStatHead->cnt++;
        aiCpuStatMsg.dataLen += logDataSize_;
        int32_t ret = profReportAdditionalInfoFunc_(1, &aiCpuStatMsg, sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG("aicore profiling send aiCpuStat mesg, core id: %d, task num: %d, ret: %d.", threadIdx,
            aiCpuStatHead->cnt, ret);
        // reset
        (void)(ret);
        aiCpuStatHead->cnt = 0;
        aiCpuStatMsg.dataLen = handkShakeHeadSize_;
    }
}

void AiCoreProf::FillPmuData(MsprofAicpuPyPtoPmuData &data, int32_t &coreIdx, uint32_t &subGraphId, uint32_t &taskId,
    const struct TaskStat *taskStat) const {
    data.seqNo = taskStat->seqNo;
    data.taskId = taskId;
    data.totalCyc = *(pmuCntTotal0Plain_[coreIdx]) + (static_cast<uint64_t>(*(pmuCntTotal1Plain_[coreIdx])) << HIG_32BIT);
    data.pmuCnt0 = *(pmuCnt0Plain_[coreIdx]);
    data.pmuCnt1 = *(pmuCnt1Plain_[coreIdx]);
    data.pmuCnt2 = *(pmuCnt2Plain_[coreIdx]);
    data.pmuCnt3 = *(pmuCnt3Plain_[coreIdx]);
    data.pmuCnt4 = *(pmuCnt4Plain_[coreIdx]);
    data.pmuCnt5 = *(pmuCnt5Plain_[coreIdx]);
    data.pmuCnt6 = *(pmuCnt6Plain_[coreIdx]);
    data.pmuCnt7 = *(pmuCnt7Plain_[coreIdx]);
    if (archInfo_ == ArchInfo::DAV_3510) {
        data.pmuCnt8 = *(pmuCnt8Plain_[coreIdx]);
        data.pmuCnt9 = *(pmuCnt9Plain_[coreIdx]);
    } 
    (void)subGraphId;
}

void AiCoreProf::ProfGetPmu(
    int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, const struct TaskStat *taskStat) {
    MsprofAicpuPyPtoPmuData data = {0};
    FillPmuData(data, coreIdx, subGraphId, taskId, taskStat);
    DEV_DEBUG("aicore profiling pmu info, core id: %d: (%u, %u | %lu | %p=%u, %p=%u, %p=%u, %p=%u, "
            "%p=%u, %p=%u, %p=%u, %p=%u, %p=%u, %p=%u).",
    coreIdx, data.seqNo, data.taskId, data.totalCyc, pmuCnt0Plain_[coreIdx], data.pmuCnt0,
    pmuCnt1Plain_[coreIdx], data.pmuCnt1, pmuCnt2Plain_[coreIdx], data.pmuCnt2, pmuCnt3Plain_[coreIdx],
    data.pmuCnt3, pmuCnt4Plain_[coreIdx], data.pmuCnt4, pmuCnt5Plain_[coreIdx], data.pmuCnt5,
    pmuCnt6Plain_[coreIdx], data.pmuCnt6, pmuCnt7Plain_[coreIdx], data.pmuCnt7, pmuCnt8Plain_[coreIdx],
    data.pmuCnt8, pmuCnt9Plain_[coreIdx], data.pmuCnt9);

    if (pmuHead_[coreIdx]->cnt == 0) {
        pmuMsg_[coreIdx].magicNumber = 0x5A5AU;
        pmuMsg_[coreIdx].level = PYPTO_MSPROF_REPORT_AICPU_LEVEL;
        pmuMsg_[coreIdx].type = PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE;
        pmuMsg_[coreIdx].threadId = syscall(SYS_gettid);
        pmuHead_[coreIdx]->magicNumber = 0x6BD3U;
        pmuHead_[coreIdx]->coreId = coreIdx;
        pmuHead_[coreIdx]->coreType = static_cast<uint16_t>(hostAicoreMng_.AicoreType(coreIdx));
        pmuHead_[coreIdx]->dataType = PROF_DATATYPE_PMU;
        pmuHead_[coreIdx]->taskId = 0;
        pmuHead_[coreIdx]->streamId = 0;
        memcpy_s(pmuData_[coreIdx], pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen = pmuHeadSize_ + pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
    } else if (pmuHead_[coreIdx]->cnt == pmuDataMaxNum_ - 1) {
        pmuMsg_[coreIdx].timeStamp = ProfGetCurCpuTimestamp();
        memcpy_s(reinterpret_cast<void *>(
                     (reinterpret_cast<uintptr_t>(pmuData_[coreIdx]) + pmuDataSize_ * pmuHead_[coreIdx]->cnt)),
            pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen += pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
        int32_t ret = profReportAdditionalInfoFunc_(1, &pmuMsg_[coreIdx], sizeof(PyPtoMsprofAdditionalInfo));
        DEV_DEBUG("aicore profiling send pmu mesg, core id: %d, task num: %d, ret: %d.", coreIdx,
            pmuHead_[coreIdx]->cnt, ret);
        (void)(ret);
        memset_s(&pmuMsg_[coreIdx], pmuMsgSize_, 0, pmuMsgSize_);
    } else {
        memcpy_s(reinterpret_cast<void *>(
                     (reinterpret_cast<uintptr_t>(pmuData_[coreIdx]) + pmuDataSize_ * pmuHead_[coreIdx]->cnt)),
            pmuDataSize_, &data, pmuDataSize_);
        pmuMsg_[coreIdx].dataLen += pmuDataSize_;
        pmuHead_[coreIdx]->cnt++;
    }
}

void AiCoreProf::AsmCntvc(uint64_t &cntvct) const {
#if defined __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
#else
    cntvct = 0;
#endif
}

uint64_t AiCoreProf::ProfGetCurCpuTimestamp() {
    uint64_t cntvct;
    AsmCntvc(cntvct);
    return cntvct;
}

} // namespace npu::tile_fwk
