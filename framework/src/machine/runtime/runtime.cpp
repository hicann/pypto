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
 * \file runtime.cpp
 * \brief
 */

#ifdef BUILD_WITH_CANN

#include "machine/runtime/runtime.h"
namespace {
const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint8_t AICORE_MAP_BUFF_LEN = 2;
} // namespace
namespace npu::tile_fwk {

static bool GetPgMask(uint64_t &valid, int32_t &deviceId) {
    deviceId = GetLogDeviceId();
    uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
    auto halFuncDevInfo = (int (*)(uint32_t deviceId, int32_t moduleType, int32_t infoType,
                           void* buf, int32_t *size))dlsym(nullptr, "halGetDeviceInfoByBuff");
    if (halFuncDevInfo == nullptr) {
        ALOG_WARN_F("Hal function not found.");
        return false;
    }
    auto ret = halFuncDevInfo(static_cast<uint32_t>(deviceId), MODULE_TYPE_AI_CORE, INFO_TYPE_OCCUPY,
                              reinterpret_cast<void *>(&aicore_bitmap[0]), &size_n);
    if (ret != 0) {
        return false;
    }
    valid = aicore_bitmap[0];
    return true;
}

constexpr uint32_t SUB_CORE_PER_AICORE = 3;

namespace DAV_2201 {
    constexpr uint32_t MAX_CORE = 25;
}

namespace DAV_3510 {
    constexpr uint32_t MAX_CORE = 36;
}

int RuntimeAgentMemory::GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType) {
    int32_t deviceId = 0;
    uint64_t valid = 0;
    if (!GetPgMask(valid, deviceId)) {
        ALOG_WARN_F("Get Device Info failed or no valid core exists.");
        valid = 0xFFFFFFFF;
        validGetPgMask = false;
    }
    ASLOGI("The valid cores are: %ld.", valid);
    uint64_t coreStride = 8 * 1024 * 1024; // 8M
    uint64_t subCoreStride = 0x100000ULL;

    auto isValid = [&valid](int id) {
        const uint64_t mask = (1ULL << 25) - 1;
        return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
    };
    auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
        size_t *outSizeRet))dlsym(nullptr, "halMemCtl");
    if (halFunc == nullptr) {
        ALOG_ERROR_F("Hal function not found.");
        return -1;
    }
    struct AddrMapInPara inMapPara;
    struct AddrMapOutPara outMapPara;
    inMapPara.devid = deviceId;
    inMapPara.addr_type = addrType;
    auto ret = halFunc(0, reinterpret_cast<void *>(&inMapPara), sizeof(struct AddrMapInPara),
        reinterpret_cast<void *>(&outMapPara), nullptr);
    if (ret != 0) {
        ALOG_ERROR_F("Map reg addr fail, maybe others are using current device. (ret=%d).", ret);
        return ret;
    }
    for (uint32_t i = 0; i < DAV_2201::MAX_CORE; i++) {
        for (uint32_t j = 0; j < SUB_CORE_PER_AICORE; j++) {
            uint64_t vaddr = 0UL;
            if (isValid(i)) {
                vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
            }
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }
    return 0;
}

int RuntimeAgentMemory::GetAicoreRegInfoForDAV3510(std::vector<int64_t> &regs, std::vector<int64_t> &regsPmu) {
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return 0;
    }
    constexpr uint32_t AICORE_PER_DIE = 18;
    constexpr uint32_t AIV_BASE_OFFSET = 18;
    constexpr uint32_t SUB_CORE_PER_DIE = AICORE_PER_DIE * SUB_CORE_PER_AICORE;

    constexpr unsigned long SUB_CORE_STRIDE = 0x100000ULL;
    constexpr unsigned long AIV_STRIDE = SUB_CORE_STRIDE;
    constexpr unsigned long AIV_SECOND_STRIDE = 2 * SUB_CORE_STRIDE;
    constexpr size_t MAX_INDEX = DAV_3510::MAX_CORE * SUB_CORE_PER_AICORE;

    auto halFunc = (int (*)(unsigned int devId, struct res_map_info *res_info, unsigned long *va,
        unsigned int *len))dlsym(nullptr, "halResMap");
    unsigned int devId = GetLogDeviceId();

    struct res_map_info mapInfo;
    mapInfo.target_proc_type = tagProcType::PROCESS_CP1;
    mapInfo.res_type = res_map_type::RES_AICORE;
    mapInfo.flag = 0;
    mapInfo.rsv[0] = 0;

    regs.resize(MAX_INDEX);
    regsPmu.resize(MAX_INDEX);
    for (uint32_t coreIndex = 0; coreIndex < DAV_3510::MAX_CORE; coreIndex++) {
        mapInfo.res_id = coreIndex;
        unsigned long mapAddr;
        unsigned int len = 0x300000;
        halFunc(devId, &mapInfo, &mapAddr, &len);
        uint32_t dieIdx = coreIndex / AICORE_PER_DIE;
        uint32_t localIdx = coreIndex % AICORE_PER_DIE;
        uint32_t dieBase = dieIdx * SUB_CORE_PER_DIE;

        uint32_t aicoreIndex = dieBase + localIdx;
        uint32_t aivFirstIndex = dieBase + AIV_BASE_OFFSET + localIdx * 2;
        uint32_t aivSecondIndex = aivFirstIndex + 1;
        //aic
        regs[aicoreIndex] = mapAddr;
        regsPmu[aicoreIndex] = mapAddr;
        // first aiv
        regs[aivFirstIndex] = mapAddr + AIV_STRIDE;
        regsPmu[aivFirstIndex] = mapAddr + AIV_STRIDE;
        // second aiv
        regs[aivSecondIndex] = mapAddr + AIV_SECOND_STRIDE;
        regsPmu[aivSecondIndex] = mapAddr + AIV_SECOND_STRIDE;
    }
    return 0;
}

void *RuntimeAgentMemory::MapAiCoreReg() {
    std::vector<int64_t> aiv;
    std::vector<int64_t> aic;

    if (GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
        return nullptr;
    }

    std::vector<int64_t> regAddr;
    regAddr.insert(regAddr.end(), aic.begin(), aic.end());
    regAddr.insert(regAddr.end(), aiv.begin(), aiv.end());
    void *devAddr = nullptr;
    size_t regAddrSize = sizeof(void *) * regAddr.size();
    int rc = rtMalloc(&devAddr, regAddrSize, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        ASLOGE("rtMalloc failed. size: %zu", regAddrSize);
        return nullptr;
    }

    rc = rtMemcpy(devAddr, regAddrSize, regAddr.data(), regAddrSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        ASLOGE("rtMemcpy failed. size: %zu", regAddrSize);
        return nullptr;
    }

    ASLOGI("All AiCore Reg mapped: %p. size: %zu", devAddr, regAddrSize);
    allocatedDevAddr.emplace_back((uint8_t *)devAddr);
    return devAddr;
}

} // namespace npu::tile_fwk

#endif // BUILD_WITH_CANN
