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
 * \file platform_manager.h
 * \brief
 */

#pragma once

#include <string>
#include <cstdint>
#include <array>
#include <map>
#include <vector>
#include <functional>

namespace npu::tile_fwk {
#define GENERATE_GET_INTPARAM_FUNC(FUNC_NAME, PARAM_ITEM)                  \
int64_t Get##FUNC_NAME() const {                                       \
  return pmIntItemArray_[static_cast<size_t>(PmIntItem::PARAM_ITEM)];  \
}

#define GENERATE_GET_STRPARAM_FUNC(FUNC_NAME, PARAM_ITEM)                  \
const std::string& Get##FUNC_NAME() const {                            \
  return pmStrItemArray_[static_cast<size_t>(PmStrItem::PARAM_ITEM)];  \
}

class PlatformManager {
public:
    PlatformManager(const PlatformManager &) = delete;
    PlatformManager &operator=(const PlatformManager &) = delete;

    static PlatformManager &Instance();

    bool Initialize(const std::string &socVersion);

    void Finalize();

    const std::map<std::string, std::vector<std::string>>& GetAiCoreIntrinsicDtypeMap() const {
        return aiCoreIntrinsicDtypeMap_;
    }

    const std::map<std::string, std::vector<std::string>>& GetVectorCoreIntrinsicDtypeMap() const {
        return vectorCoreIntrinsicDtypeMap_;
    }

    bool GetAiCoreIntrinsicDtype(const std::string &intrinsic, std::vector<std::string> &dtypeVec) const;

    bool GetVectorCoreIntrinsicDtype(const std::string &intrinsic, std::vector<std::string> &dtypeVec) const;

    std::string GetFilePath() const {
        return platformFile_;
    }

    GENERATE_GET_INTPARAM_FUNC(AiCoreCnt, AICORE_CNT)
    GENERATE_GET_INTPARAM_FUNC(VecCoreCnt, VECCORE_CNT)
    GENERATE_GET_INTPARAM_FUNC(AiCpuCnt, AICPU_CNT)
    GENERATE_GET_INTPARAM_FUNC(MemorySize, MEMORY_SIZE)
    GENERATE_GET_INTPARAM_FUNC(L2Type, L2_TYPE)
    GENERATE_GET_INTPARAM_FUNC(L2Size, L2_SIZE)
    GENERATE_GET_INTPARAM_FUNC(L2PageNum, L2_PAGE_NUM)
    GENERATE_GET_INTPARAM_FUNC(AiCoreCubeFreq, AICORE_CUBE_FREQ)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL0ASize, AICORE_L0A_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL0BSize, AICORE_L0B_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL0CSize, AICORE_L0C_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL1Size, AICORE_L1_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreUbSize, AICORE_UB_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreUbBlockSize, AICORE_UB_BLOCK_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreUbBankSize, AICORE_UB_BANK_SIZE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreUbBankNum, AICORE_UB_BANK_NUM)
    GENERATE_GET_INTPARAM_FUNC(AiCoreUbBankGroupNum, AICORE_UB_BANK_GROUP_NUM)
    GENERATE_GET_INTPARAM_FUNC(AiCoreDdrRate, AICORE_DDR_RATE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreDdrReadRate, AICORE_DDR_READ_RATE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreDdrWriteRate, AICORE_DDR_WRITE_RATE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL2Rate, AICORE_L2_RATE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL2ReadRate, AICORE_L2_READ_RATE)
    GENERATE_GET_INTPARAM_FUNC(AiCoreL2WriteRate, AICORE_L2_WRITE_RATE)

    GENERATE_GET_STRPARAM_FUNC(SocVersion, SOC_VERSION)
    GENERATE_GET_STRPARAM_FUNC(ShortSocVersion, SHORT_SOC_VERSION)
    GENERATE_GET_STRPARAM_FUNC(AicVersion, AIC_VERSION)
private:
    enum class PmIntItem {
        AICORE_CNT = 0,
        VECCORE_CNT,
        AICPU_CNT,
        MEMORY_SIZE,
        L2_TYPE,
        L2_SIZE,
        L2_PAGE_NUM,
        AICORE_CUBE_FREQ,
        AICORE_L0A_SIZE,
        AICORE_L0B_SIZE,
        AICORE_L0C_SIZE,
        AICORE_L1_SIZE,
        AICORE_UB_SIZE,
        AICORE_UB_BLOCK_SIZE,
        AICORE_UB_BANK_SIZE,
        AICORE_UB_BANK_NUM,
        AICORE_UB_BANK_GROUP_NUM,
        AICORE_DDR_RATE,
        AICORE_DDR_READ_RATE,
        AICORE_DDR_WRITE_RATE,
        AICORE_L2_RATE,
        AICORE_L2_READ_RATE,
        AICORE_L2_WRITE_RATE,
        ITEM_BOTTOM
    };
    enum class PmStrItem {
        SOC_VERSION = 0,
        SHORT_SOC_VERSION,
        AIC_VERSION,
        ITEM_BOTTOM
    };
    PlatformManager();
    ~PlatformManager();
    void Reset();
    static bool ReadFileContent(const std::string &filePath,
                                std::map<std::string, std::map<std::string, std::string>> &contentMap);
    void ParseStrItem(const std::map<std::string, std::map<std::string, std::string>> &contentMap);
    void ParseIntItem(const std::map<std::string, std::map<std::string, std::string>> &contentMap);
    void ParseInstrDtypeMap(std::map<std::string, std::map<std::string, std::string>> &contentMap);
    static void MappingInstrDtypeMap(const std::map<std::string, std::string> &contentMap,
                                     std::map<std::string, std::vector<std::string>> &instrDtypeMap);
    static int64_t ParseIntValue(const std::string &value);
private:
    bool isInit_;
    std::string platformFile_;
    std::map<std::string, std::vector<std::string>> aiCoreIntrinsicDtypeMap_;
    std::map<std::string, std::vector<std::string>> vectorCoreIntrinsicDtypeMap_;
    std::array<int64_t, static_cast<size_t>(PmIntItem::ITEM_BOTTOM)> pmIntItemArray_;
    std::array<std::string, static_cast<size_t>(PmStrItem::ITEM_BOTTOM)> pmStrItemArray_;
    using PmItemParseFunc = std::function<int64_t(const std::string &)>;
    static const std::map<PmIntItem, std::tuple<std::string, std::string, PmItemParseFunc>> kPmIntItemParseFuncMap;
    static const std::map<PmStrItem, std::tuple<std::string, std::string>> kPmStrItemMap;
};
}