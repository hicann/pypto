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
 * \file aicore_prof.h
 * \brief
 */

#ifndef AICORE_PROF_H
#define AICORE_PROF_H

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include "tilefwk/aicpu_common.h"

extern "C" {
__attribute__((weak)) int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length);
__attribute__((weak)) int32_t AdprofCheckFeatureIsOn(uint64_t feature);
};

namespace npu::tile_fwk::dynamic {
class AiCoreManager;
struct PyPtoMsprofAdditionalInfo { // for MsprofReportAdditionalInfo buffer data
    uint16_t magicNumber = 0x5A5AU;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t data[232];
};

constexpr uint32_t PYPTO_MSPROF_REPORT_AICPU_LEVEL = 6000U;
constexpr uint32_t PYPTO_MSPROF_REPORT_AICPU_NODE_TYPE = 10U; /* type info: DATA_PREPROCESS.AICPU */

constexpr uint64_t PROF_TASK_TIME_L0 = 0x00000008ULL;
constexpr uint64_t PROF_TASK_TIME_L1 = 0x00000010ULL;
constexpr uint64_t PROF_TASK_TIME_L2 = 0x00000020ULL;

constexpr bool GLB_PMU_EN = true;
constexpr bool USER_PMU_MODE_EN = (GLB_PMU_EN && true);
constexpr bool SAMPLE_PMU_MODE_EN = (GLB_PMU_EN && USER_PMU_MODE_EN);
constexpr bool DUAL_PAGE_EN = false; // 双页表是否使能，如何感知？？？通过runtime？
constexpr uint32_t MAX_PMU_CNT = 8;

constexpr int32_t PMU_CYCLE = 80; // 记录按照了20MHZ的时钟周期，单位归一按照1600MHZ的时钟周期进行统一，所以80
typedef enum AiCoreProfLevel {
    PROF_LEVEL_OFF = 0,
    PROF_LEVEL_FUNC = 1,

    PROF_LEVEL_FUNC_LOG = 2,
    PROF_LEVEL_FUNC_LOG_PMU = 3,
    PROF_LEVEL_MAX = 3
} AiCoreProfLevel;

typedef enum AiCoreProfDataType {
    PROF_DATATYPE_UNKNOWN = 0,
    PROF_DATATYPE_FUNC = 1,
    PROF_DATATYPE_LOG = 2,
    PROF_DATATYPE_PMU = 3,
    PROF_DATATYPE_HAND_SHAKE = 4,
    PROF_DATATYPE_EXE = 5
} AiCoreProfDataType;

typedef enum AiCoreRegister {
    PMU_CTRL_0 = 0x200,
    PMU_CNT0 = 0x210,
    PMU_CNT1 = 0x218,
    PMU_CNT2 = 0x220,
    PMU_CNT3 = 0x228,
    PMU_CNT4 = 0x230,
    PMU_CNT5 = 0x238,
    PMU_CNT6 = 0x240,
    PMU_CNT7 = 0x248,
    PMU_CNT8 = 0x250,
    PMU_CNT9 = 0x254,
    PMU_CNT0_IDX = 0x1280,
    PMU_CNT1_IDX = 0x1284,
    PMU_CNT2_IDX = 0x1288,
    PMU_CNT3_IDX = 0x128C,
    PMU_CNT4_IDX = 0x1290,
    PMU_CNT5_IDX = 0x1294,
    PMU_CNT6_IDX = 0x1298,
    PMU_CNT7_IDX = 0x129C,
    PMU_START_CNT_CYC_0 = 0x2A0,
    PMU_START_CNT_CYC_1 = 0x2A4,
    PMU_STOP_CNT_CYC_0 = 0x2A8,
    PMU_STOP_CNT_CYC_1 = 0x2AC,
} AiCoreRegister;

typedef enum AiCorePmuEvent {
    VEC_BUSY_CYCLE = 0x8,
    SU_BUSY_CYCLE = 0x9,
    CUBE_BUSY_CYCLE = 0xA,
    MTE2_BUSY_CYCLE = 0xC,
    MTE3_BUSY_CYCLE = 0xD,

    ICACHE_REQ_CNT = 0x54,
    ICACHE_MISS_CNT = 0x55,
    DCACHE_HIT_CNT = 0xA6,
    DCACHE_MISS_CNT = 0xA7,

    FIXP_L1_WR_CYCLE = 0x206,
    FIXP_L1_RD_CYCLE = 0x209,
    FIXP_L0C_RD_CYCLE = 0x20C,
    FIXP_BUSY_CYCLE = 0x20D,
    FIXP_ACTIVE_CYCLE = 0x303, // FIXP busy minus hset/hwait time

    L2_WR_HIT_CNT = 0x500,
    L2_WR_MISS_CNT = 0x501, // L2BUF_HIT, ineffective
    L2_WR_MISS_ALLOC_CNT = 0x502,
    L2_WR_MISS_NON_ALLOC_CNT = 0x503,
    L2_R0_HIT_CNT = 0x504,
    L2_R0_MISS_CNT = 0x505,
    L2_R0_MISS_ALLOC_CNT = 0x506,
    L2_R0_MISS_NON_ALLOC_CNT = 0x507,
    L2_R1_HIT_CNT = 0x508,
    L2_R1_MISS_CNT = 0x509,
    L2_R1_MISS_ALLOC_CNT = 0x50A,
    L2_R1_MISS_NON_ALLOC_CNT = 0x50B,
} AiCorePmuEvent;

const uint16_t AIV_EVENT_LIST[MAX_PMU_CNT] = {
    MTE3_BUSY_CYCLE,
    L2_WR_HIT_CNT,
    L2_WR_MISS_ALLOC_CNT,
    MTE2_BUSY_CYCLE,
    L2_R0_HIT_CNT,
    L2_R0_MISS_ALLOC_CNT,
    L2_R1_HIT_CNT,
    L2_R0_MISS_ALLOC_CNT,
};

const uint16_t AIC_EVENT_LIST[MAX_PMU_CNT] = {
    FIXP_BUSY_CYCLE,
    L2_WR_HIT_CNT,
    L2_WR_MISS_ALLOC_CNT,
    MTE2_BUSY_CYCLE,
    L2_R0_HIT_CNT,
    L2_R0_MISS_ALLOC_CNT,
    L2_R1_HIT_CNT,
    L2_R0_MISS_ALLOC_CNT,
};

struct MsprofAicpuPyPtoPmuData {
    uint32_t seqNo{0};
    uint32_t taskId{0};
    uint64_t totalCyc{0};
    uint32_t pmuCnt0{0}; // 单个task不能超过3s, 按50MHZ计算
    uint32_t pmuCnt1{0};
    uint32_t pmuCnt2{0};
    uint32_t pmuCnt3{0};
    uint32_t pmuCnt4{0};
    uint32_t pmuCnt5{0};
    uint32_t pmuCnt6{0};
    uint32_t pmuCnt7{0};
};

// !!注意和 TaskStat 前面的数据区保持一致
struct MsprofAicpuPyPtoLogData {
    int16_t seqNo;
    int16_t subGraphId;
    int32_t taskId{0};
    int64_t execStart{0};
    int64_t execEnd{0};
};

struct MsprofAicpuPyPtoHead {
    uint16_t magicNumber = 0x6BD3U;
    uint16_t coreId   : 7;
    uint16_t coreType : 3;
    uint16_t dataType : 6;
    uint32_t taskId;
    uint16_t streamId;
    uint8_t cnt;
    uint8_t rsv[29];
};

using MsprofAicpuPyPtoPmuHead = struct MsprofAicpuPyPtoHead;
using MsprofAicpuPyPtoLogHead = struct MsprofAicpuPyPtoHead;
using MsprofAicpuHandShakeHead = struct MsprofAicpuPyPtoHead;
using MsProfAiCpuTaskStatHead = struct MsprofAicpuPyPtoHead;

struct AiCpuHandShakeSta {
    int32_t threadId{0};
    int32_t coreId{0};
    uint64_t shakeStart{0};
    uint64_t shakeEnd{0};
};

struct AiCpuTaskStat {
    int32_t coreId;
    int32_t taskId;
    uint64_t taskGetStart;
    uint64_t execStart;
    uint64_t execEnd;
};

bool ProfCheckLevel(uint64_t feature);
AiCoreProfLevel CreateProfLevel(ProfConfig profConfig);

class AiCoreProf {
public:
    explicit AiCoreProf(AiCoreManager &aicoreMng) : hostAicoreMng_(aicoreMng) {}
    ~AiCoreProf() {}

    void ProfInit([[maybe_unused]]int64_t *regAddrs, [[maybe_unused]]int64_t *pmuEventAddrs, ProfConfig profConfig);
    void ProfStart();
    void ProfGet(int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, const struct TaskStat *taskStat);
    void ProfGetSwitch(int64_t &flag) const;
    void ProfStop();
    void ProfStopHandShake();
    void ProGetHandShake(int &threadIdx, const struct AiCpuHandShakeSta *handShakeStat);
    void AsmCntvc(uint64_t &cntvct) const;
    void SetAiCpuTaskStat(const uint32_t &taskId, struct AiCpuTaskStat &aiCpuTaskStat);
    struct AiCpuTaskStat GetAiCpuTaskStat(const uint32_t &taskId);
    void ProfGetAiCpuTaskStat(int &threadIdx, struct AiCpuTaskStat *aiCpuStat);
    void ProfStopAiCpuTaskStat();
    void ProInitAiCpuTaskStat();
    void ProInitHandShake();
    bool ProfIsEnable() { return profLevel_ != PROF_LEVEL_OFF; }
    
private:
    inline void ProfInitLog();
    inline void ProfStopLog();
    inline void ProfGetLog(int32_t coreIdx, const struct TaskStat *taskStat);
    inline void ProfInitPmu(int64_t *regAddrs, int64_t *pmuEventAddrs);
    inline void ReadPmuCounters(const int32_t coreIdx) const;
    inline void SetPmuEvents(void *mapBase, const int32_t coreIdx) const;
    inline void ProfStartPmu();
    inline void ProfStopPmu();
    void FillPmuData(MsprofAicpuPyPtoPmuData &data, int32_t &coreIdx, uint32_t &subGraphId, uint32_t &taskId,
        const struct TaskStat *taskStat) const;
    inline void ProfGetPmu(int32_t coreIdx, uint32_t subGraphId, uint32_t taskId, const struct TaskStat *taskStat);
    inline uint64_t ProfGetCurCpuTimestamp();

private:
    int32_t coreNum_ = 0;
    AiCoreProfLevel profLevel_ = PROF_LEVEL_OFF;
    uint64_t taskCnt_ = 0;
    int64_t *regAddrs_{nullptr};
    int64_t *pmuEventAddrs_{nullptr};

    // PMU_CNT0 ~ PMU_CNT7 共计8个cnt寄存器,32位寄存器,用来获取对应读数,单位为cycle
    std::vector<volatile uint32_t *> pmuCnt0Plain_;
    std::vector<volatile uint32_t *> pmuCnt1Plain_;
    std::vector<volatile uint32_t *> pmuCnt2Plain_;
    std::vector<volatile uint32_t *> pmuCnt3Plain_;
    std::vector<volatile uint32_t *> pmuCnt4Plain_;
    std::vector<volatile uint32_t *> pmuCnt5Plain_;
    std::vector<volatile uint32_t *> pmuCnt6Plain_;
    std::vector<volatile uint32_t *> pmuCnt7Plain_;
    std::vector<volatile uint32_t *> pmuCnt8Plain_;
    std::vector<volatile uint32_t *> pmuCnt9Plain_;

    // pmu data
    uint32_t pmuDataMaxNum_ = 4;
    uint32_t pmuMsgSize_ = 0;
    uint32_t pmuDataSize_ = 0;
    uint32_t pmuHeadSize_ = 0;
    std::vector<PyPtoMsprofAdditionalInfo> pmuMsg_;
    std::vector<MsprofAicpuPyPtoPmuHead *> pmuHead_;
    std::vector<MsprofAicpuPyPtoPmuData *> pmuData_;

    // log data
    uint32_t logDataMaxNum_ = 8;
    uint32_t logMsgSize_ = 0;
    uint32_t logDataSize_ = 0;
    uint32_t logHeadSize_ = 0;
    std::vector<PyPtoMsprofAdditionalInfo> logMsg_;
    std::vector<MsprofAicpuPyPtoLogHead *> logHead_;
    std::vector<MsprofAicpuPyPtoLogData *> logData_;

    // handshake data
    const uint32_t handkShakeMaxNum_ = 8;
    uint32_t handkShakeMsgSize_{0};
    uint32_t handShakeDataSize_{0};
    uint32_t handkShakeHeadSize_{0};
    std::vector<PyPtoMsprofAdditionalInfo> HandShakeMsg_;
    std::vector<MsprofAicpuHandShakeHead *> HandShakeHead_;
    std::vector<AiCpuHandShakeSta *> handShakeData_;
    // AiCputStat data
    const uint32_t aiCpuStatMaxNum_ = 6;
    uint32_t aiCpuStatMsgSize_{0};
    uint32_t aiCpuStatDataSize_{0};
    uint32_t aiCpuStatHeadSize_{0};
    std::vector<PyPtoMsprofAdditionalInfo> aiCpuStatMsg_;
    std::vector<MsprofAicpuHandShakeHead *> aiCpuStatHead_;
    std::vector<AiCpuTaskStat *> aiCpuStatData_;

    std::map<uint32_t, AiCpuTaskStat> aiCpuStatMap_;

    AiCoreManager &hostAicoreMng_;
};
}
#endif
