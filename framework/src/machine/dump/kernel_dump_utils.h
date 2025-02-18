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
 * \file kernel_dump_utils.h
 * \brief dump binary and kernel.o into single kernel
 */

#pragma once

#include "machine/host/device_agent_task.h"

namespace npu::tile_fwk {
enum class KernelContextType : int64_t {
    OpBinary = 0,
    Kernel,
    Bottom
};

struct FatbinHeadInfo {
    uint64_t configKeyNum;
    std::vector<uint64_t> configKeyList;
    std::vector<size_t> binOffsets;
    FatbinHeadInfo() : configKeyNum(0) {}
    FatbinHeadInfo(uint64_t config_key_num) : configKeyNum(config_key_num) {
        configKeyList.resize(config_key_num);
        binOffsets.resize(config_key_num);
    }
};

struct JsonInfo {
    int64_t blockDim = 0;
    std::string kernelName;
    uint64_t configKey;
    int64_t workspaceSize = -1;
};

#pragma pack(8)
struct KernelHeader
{
    size_t dataOffset[static_cast<size_t>(KernelContextType::Bottom)];
    size_t dataSize[static_cast<size_t>(KernelContextType::Bottom)];
};
#pragma pack()

std::string GetDumpKernelPath();

class KernelDumpUtils {
public:
    static bool DumpKernelFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName,
                               const std::string &dumpDirPath, const std::string &dyKernelPath);
    static void WriteFatbinJson(const std::vector<JsonInfo> &allBinJsonInfo, const std::string &fatbinJsonPath,
                                const std::string &binFileName);
    static bool WriteBufferToFatbin(FatbinHeadInfo &fatbinHeadInfo, const std::string &path,
                                    const std::vector<char> &fatbinBuffer);
    static bool GetBufferFromBinFile(const std::string &binFilePath, std::vector<char> &buffer);
    static bool GetSubJsonInfo(const std::string &jsonPath, JsonInfo &kernelJsonInfo);
    static void* LoadTileFwkImplOpLib();
    static void FreeOpHandle(void *opLibHandle);
private:
    static bool DumpBinFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName,
                            const std::string &dumpDirPath, const std::string &dyKernelPath);
    static void DumpJsonFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName, const std::string &dumpDirPath);
};
}
