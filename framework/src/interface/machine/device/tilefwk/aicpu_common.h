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
 * \file aicpu_common.h
 * \brief
 */

#ifndef RUNTIME_COMMON_DEF_H
#define RUNTIME_COMMON_DEF_H

#include <cstdint>

const uint64_t AICORE_TASK_INIT = 0xFFFFFFFF;
const uint64_t AICORE_TASK_STOP = 0x7FFFFFF0;
const uint64_t AICORE_FUNC_STOP = 0x7FFFFFE0;
const uint64_t AICORE_FIN_MASK = 0x80000000;
const uint64_t AICORE_TASK_MAX = 0x70000000;

const uint64_t AICORE_SAY_HELLO = 0x80000000;
const uint64_t AICORE_SAY_ACK = 0x80000000;
const int64_t PRO_LEVEL1 = 2;
const int64_t PRO_LEVEL2 = 3;

constexpr int REG_LOW_TASK_PING = 0;
constexpr int REG_LOW_TASK_PONG = 1;
constexpr int MAX_DFX_TASK_NUM_PER_CORE = 10000;

constexpr int SHAK_BUF_PRINT_BUFFER_INDEX = 5;
constexpr int SHAK_BUF_COREFUNC_DATA_INDEX = 6;
constexpr int SHAK_BUF_DFX_DATA_INDEX = 7;

constexpr int CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX = 0;

constexpr int FUNC_ID_BATCH = 0x7FF;

const uint64_t SHARED_BUFFER_SIZE = 512;
const uint64_t PMU_BUFFER_SIZE = 4096;
const uint64_t DEVICE_QUEUE_SIZE = 512;
const uint64_t PRINT_BUFFER_SIZE = 16384;

constexpr const int DEV_SHAPE_DIM_NUM_2 = 2;
constexpr const int DEV_SHAPE_DIM_NUM_3 = 3;
constexpr const int DEV_SHAPE_DIM_NUM_4 = 4;
constexpr const int DEV_SHAPE_DIM_NUM_5 = 5;

enum class ArchInfo {
    DAV_1001 = 1001,
    DAV_2201 = 2201,
    DAV_3510 = 3510,
    DAV_UNKNOWN
};

#define DEVICE_TASK_STOP 0x7FFFFFFE

#define DEVICE_TASK_TYPE_STATIC  0
#define DEVICE_TASK_TYPE_DYN     1
#define DEVICE_TASK_TYPE_INVALID 0xf

template <typename DerivedType, typename UnderlyingType>
class BitmaskBase {
public:
    using underlying_type = UnderlyingType;
    underlying_type value{0};
    constexpr BitmaskBase(underlying_type v = 0) : value(v) {}
    constexpr bool Empty() const {
        return value == 0;
    }
    constexpr bool Contains(underlying_type mask) const {
        return (value & mask) == mask;
    }
    constexpr bool Overlaps(underlying_type mask) const {
        return (value & mask) != 0;
    }
    constexpr void Add(underlying_type mask) {
        value |= mask;
    }
    constexpr void Remove(underlying_type mask) {
        value &= ~mask;
    }
    friend constexpr DerivedType operator|(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value | rhs.value);
    }
    friend constexpr DerivedType operator&(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value & rhs.value);
    }
    friend constexpr DerivedType operator^(DerivedType lhs, DerivedType rhs) {
        return DerivedType(lhs.value ^ rhs.value);
    }
    friend constexpr DerivedType operator~(DerivedType lhs) {
        return DerivedType(~lhs.value);
    }
    constexpr operator underlying_type() const {
        return value;
    }
};

struct ProfConfig : public BitmaskBase<ProfConfig, uint32_t> {
    using BitmaskBase::BitmaskBase;
    enum : underlying_type {
        OFF = 0x0,
        AICPU_FUNC = 0x1 << 0,
        AICORE_TIME = 0x1 << 1,
        AICORE_PMU = 0x1 << 2,
    };
};

struct ToSubMachineConfig {
    ProfConfig profConfig{ProfConfig::OFF};
    uint64_t isGETensorList{0};
};

struct OpMetaAddrs {
    uint64_t generalAddr{0};     // aicpu meta addr
    uint64_t stitchPoolAddr{0};  // aicpu meta addr
};

struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};       // store device global data, must be init with zero
    uint64_t devQueueAddr;    // pcie/XLink mem, used between host and device, `DEVICE_QUEUE_SIZE`
    uint64_t sharedBuffer;    // SHARED_BUFFER_SIZE per core, aics first
    uint64_t coreRegAddr;     // core reg addr, uint64_t per core, aic first
    uint64_t corePmuRegAddr;  // pmu reg addr, uint64_t per core, aic first
    uint64_t corePmuAddr;     // pmu data addr, PAGE_SIZE per core, aic first
    uint64_t pmuEventAddr;    // pmu event addr
    uint64_t taskType : 4;    // initial task type
    uint64_t machineConfig : 8; // machine config
    uint64_t taskId   : 52;   // initial task id
    uint64_t taskData;        // initial task data
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};    // server so Bin
    uint64_t aicpuSoLen{0};    // server so len
    uint64_t deviceId{0};      // for device copy fileName
    uint64_t startArgsAddr{0}; // DevStartArgs addr
    uint64_t taskQueue{0};     // task queue between ctrl and sche
    uint64_t taskCtrl{0};      // task ctrl between ctrl and sche
    uint32_t scheCpuNum{0};    // sche cpu num calc by host
    uint32_t enableCtrl : 2;    // if enable builtin ctrl
    uint32_t validGetPgMask : 2; // mark pgmask is invalid
    uint32_t disableSync : 2;    // close ctrl and sche soft sync
    uint32_t isGETensorList : 26;    // GE graph is tensor list
    uint64_t generalAddr{0};     // aicpu meta addr
    uint64_t stitchPoolAddr{0};  // aicpu meta addr
    uint64_t GetBlockNum() { return nrValidAic * (nrAiv / nrAic + 1); }
    ArchInfo archInfo{ArchInfo::DAV_2201};
    ToSubMachineConfig toSubMachineConfig;
};
#endif