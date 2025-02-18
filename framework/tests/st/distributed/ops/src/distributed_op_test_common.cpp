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
 * \file distributed_op_test_common.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#ifdef BUILD_WITH_CANN_SUB
#include<hccl/hcom.h>
#else
#include "hcom.h"
#endif

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

namespace npu::tile_fwk {
namespace Distributed {
#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

int32_t MakeMc2TilingStruct(struct Mc2CommConfig &commConfig, std::string &groupName)
{
    constexpr uint32_t version = 2;
    constexpr uint32_t hcommCnt = 1;
    constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
    const char *algConfig = "AllGather=level0:ring";
    constexpr uint32_t arraySize = 128;

    commConfig.version = version;
    commConfig.hcommCnt = hcommCnt;
    commConfig.hcommCfg.skipLocalRankCopy = 0;
    commConfig.hcommCfg.skipBufferWindowCopy = 0;
    commConfig.hcommCfg.stepSize = 0;
    commConfig.hcommCfg.opType = opTypeAllToAll;
    auto ret = strcpy_s(commConfig.hcommCfg.groupName, arraySize, groupName.c_str());
    if (ret != 0) {
        return -1;
    }
    ret = strcpy_s(commConfig.hcommCfg.algConfig, arraySize, algConfig);
    if (ret != 0) {
        return -1;
    }
    return 0;
}

std::vector<uint64_t> GetHcclContext(const std::vector<std::string> &groupNames)
{
    std::vector<uint64_t> hcclContext(groupNames.size(), 0);
    for (size_t groupIndex = 0; groupIndex < groupNames.size(); ++groupIndex) {
        auto groupName = groupNames[groupIndex];
        HcclComm commHandle = nullptr;
        HcclResult ret = HcomGetCommHandleByGroup(groupName.c_str(), &commHandle);
        ASSERT(ret == 0);
        struct Mc2CommConfig commConfig;
        (void)memset_s(&commConfig, sizeof(commConfig), 0, sizeof(commConfig));
        ASSERT(MakeMc2TilingStruct(commConfig, groupName) == 0);
        ret = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(), &commConfig,
            reinterpret_cast<void **>(&hcclContext[groupIndex]));
        ASSERT((ret == 0) && (hcclContext[groupIndex] != 0UL));
        ALOG_INFO_F("groupIndex=%u, groupName=%s, hcclContext=%lu", groupIndex, groupName.c_str(),
            hcclContext[groupIndex]);
    }
    return hcclContext;
}

int64_t GetEleNumFromShape(std::vector<int64_t>& shape)
{
    int64_t eleNum = 1;
    for (int num : shape) {
        eleNum *= num;
    }
    return eleNum;
}

Tensor CreateTensorFromFile(std::vector<int64_t>& shape, DataType dtype, std::string& file, std::string tname)
{
    int eleNum = GetEleNumFromShape(shape);
    uint64_t byteSize = eleNum * BytesOf(dtype);
    uint8_t* ptr = (uint8_t*)readToDev(GetGoldenDir() + file, byteSize / sizeof(float));
    Tensor tensor(dtype, shape, ptr, tname);
    return tensor;
}

} // namespace Distributed
} // namespace npu::tile_fwk