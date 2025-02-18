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
 * \file kernel_dump_utils.cpp
 * \brief dump binary and kernel.o into one kernel
 */

#include "machine/dump/kernel_dump_utils.h"
#include <climits>
#include <dlfcn.h>
#include "tilefwk/function.h"
#include "interface/utils/file_utils.h"
#include "interface/program/program.h"
#include "machine/platform/platform_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include <nlohmann/json.hpp>

namespace npu::tile_fwk {
namespace {
constexpr int64_t LEVEL_FOUR = 4;
constexpr int64_t MAX_BLOCK_NUM = 24;
const std::string KERNEL_FILE_PREFIX = "ast_op_";
const std::string KERNEL_BIN_FILE_SUFFIX = ".o";
const std::string KERNEL_JSON_FILE_SUFFIX = ".json";
const std::string AICORE_KERNEL_FILE_PATH = "kernel/kernel.o";
const std::string AICORE_KERNEL_FILE_NAME = "kernel.o";

inline size_t DataSizeAlign(const size_t bytes, const uint32_t aligns = 32U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}
}

std::string GetDumpKernelPath() {
  std::string currentLibPath;
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(GetDumpKernelPath), &info)) {
    currentLibPath = std::string(info.dli_fname);
    int32_t pos = currentLibPath.rfind('/');
    if (pos >= 0) {
      currentLibPath = currentLibPath.substr(0, pos);
    }
  }
  return currentLibPath;
}

bool KernelDumpUtils::DumpKernelFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName,
                                     const std::string &dumpDirPath, const std::string &dyKernelPath) {
    if (deviceAgentTask == nullptr || deviceAgentTask->GetFunction() == nullptr) {
        return false;
    }
    std::string realDumpDirPath = RealPath(dumpDirPath);
    if (realDumpDirPath.empty()) {
        if (!CreateMultiLevelDir(dumpDirPath)) {
            return false;
        }
        realDumpDirPath = RealPath(dumpDirPath);
    }
    std::string finalKernelName = kernelName.empty() ? deviceAgentTask->GetFunction()->GetMagicName() : kernelName;
    if (!DumpBinFile(deviceAgentTask, finalKernelName, realDumpDirPath, dyKernelPath)) {
        return false;
    }

    DumpJsonFile(deviceAgentTask, finalKernelName, realDumpDirPath);
    return true;
}

bool KernelDumpUtils::DumpBinFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName,
                                  const std::string &dumpDirPath, const std::string &dyKernelPath) {
    KernelHeader kernelHeader;
    size_t offset = sizeof(kernelHeader);
    // op binary
    kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)] = offset;
    Function *function = deviceAgentTask->GetFunction();
    std::vector<uint8_t> opBinData;
    bool isDynamic = function->IsFunctionType(FunctionType::DYNAMIC);
    if (isDynamic && function->GetDyndevAttribute() != nullptr) {
        opBinData = function->GetDyndevAttribute()->devProgBinary;
    }

    if (opBinData.empty()) {
        ALOG_INFO_F("OP binary data is empty, function type is [%s].", function->GetFunctionTypeStr().c_str());
        return true;
    }
    kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)] = opBinData.size();
    offset += DataSizeAlign(opBinData.size());

    // kernel.o
    std::string kernelPath = GetDumpKernelPath() + "/" + AICORE_KERNEL_FILE_NAME;
    std::vector<uint8_t> kernelBinData;
    if (isDynamic) {
      ALOG_INFO_F("Dynamic Kernel path %s.", dyKernelPath.c_str());
      kernelBinData = LoadFile(dyKernelPath);
    } else if (kernelBinData.empty()) {
        kernelPath = GetDumpKernelPath() + "/" + AICORE_KERNEL_FILE_PATH;
      ALOG_INFO_F("Kernel dump to %s.", kernelPath.c_str());
        kernelBinData = LoadFile(kernelPath);
        if (kernelBinData.empty()) {
            ALOG_ERROR_F("Kernel path[%] is not existed.", kernelPath.c_str());
            return false;
        }
    }
    kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::Kernel)] = offset;
    kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)] = kernelBinData.size();
    offset += DataSizeAlign(kernelBinData.size());

    std::vector<uint8_t> binData(offset, 0);
    memcpy_s(binData.data(), sizeof(kernelHeader), &kernelHeader, sizeof(kernelHeader));
    memcpy_s(binData.data() + kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)],
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)], opBinData.data(),
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)]);
    memcpy_s(binData.data() + kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::Kernel)],
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)], kernelBinData.data(),
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)]);
    std::string binFilePath = dumpDirPath + "/" + kernelName + KERNEL_BIN_FILE_SUFFIX;
    return DumpFile(binData, binFilePath);
}

void KernelDumpUtils::DumpJsonFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName, const std::string &dumpDirPath) {
    std::string jsonFilePath = dumpDirPath + "/" + kernelName + KERNEL_JSON_FILE_SUFFIX;
    std::ofstream file(jsonFilePath);
    Json binJson;
    binJson["binFileName"] = kernelName;
    binJson["binFileSuffix"] = KERNEL_BIN_FILE_SUFFIX;
    binJson["kernelName"] = "ast_main_0";
    binJson["coreType"] = "MIX";
    binJson["blockDim"] = PlatformManager::Instance().GetAiCoreCnt();
    binJson["magic"] = "RT_DEV_BINARY_MAGIC_ELF";
    binJson["dynamicParamMode"] = "floded_with_desc";
    uint64_t workspaceSize = deviceAgentTask->GetWorkSpaceSize() == 0 ? 1 : deviceAgentTask->GetWorkSpaceSize();
    if (deviceAgentTask->GetFunction()->IsFunctionType(FunctionType::DYNAMIC) &&
        deviceAgentTask->GetFunction()->GetDyndevAttribute() != nullptr) {
        dynamic::DevAscendProgram *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(deviceAgentTask->GetFunction()->GetDyndevAttribute()->devProgBinary.data());
        if (devProg != nullptr) {
            workspaceSize = devProg->memBudget.Total();
        }
    }
    ALOG_INFO_F("Work space size is [%lu].", workspaceSize);
    binJson["workspace"] = {
        {"num", 1},
        {"size", {workspaceSize}},
        {"type", {0}}
    };
    file << binJson.dump(LEVEL_FOUR) << std::endl;
    file.close();
}

bool KernelDumpUtils::GetBufferFromBinFile(const std::string &binFilePath, std::vector<char> &buffer) {
    char resolvedPath[PATH_MAX] = {0x00};
    if (realpath(binFilePath.c_str(), resolvedPath) == nullptr) {
        ALOG_INFO_F("Bin file path[%s] is invalid.", binFilePath.c_str());
        return false;
    }
    std::ifstream ifStream(resolvedPath, std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        ALOG_INFO_F("Fail to open bin file path[%s].", binFilePath.c_str());
        return false;
    }
    try {
        std::streamsize bufferSize = ifStream.tellg();
        if (bufferSize <= 0) {
            ifStream.close();
            ALOG_INFO_F("Get stream failed");
            return false;
        }
        if (bufferSize > INT_MAX) {
            ifStream.close();
            ALOG_INFO_F("bufferSize is invalid.");
            return false;
        }
        ifStream.seekg(0, std::ios::beg);
        size_t curSize = buffer.size();
        size_t increSize = static_cast<size_t>(bufferSize);
        buffer.resize(curSize + increSize);
        ifStream.read(&buffer[curSize], bufferSize);
        ifStream.close();
    } catch (const std::ifstream::failure &e) {
        ifStream.close();
        ALOG_ERROR_F("Failed to read bin file[%s], reason is [%s]", binFilePath.c_str(), e.what());
        return false;
    }
    return true;
}

bool KernelDumpUtils::WriteBufferToFatbin(FatbinHeadInfo &fatbinHeadInfo, const std::string &path,
        const std::vector<char> &fatbinBuffer) {
    std::ofstream fatbinFile(path, std::ios::binary);
    if (!fatbinFile.is_open()) {
        ALOG_ERROR_F("Failed to open file[%s].", path.c_str());
        return false;
    }
    size_t headSize = sizeof(fatbinHeadInfo.configKeyNum);
    size_t binOffsetListSize = fatbinHeadInfo.binOffsets.size() * sizeof(size_t);
    size_t configKeyListSize = fatbinHeadInfo.configKeyList.size() * sizeof(uint64_t);
    size_t offset = headSize + binOffsetListSize + configKeyListSize;
    for (auto &i : fatbinHeadInfo.binOffsets) {
        i += offset;
    }
    std::vector<uint8_t> fatbinData(offset, 0);
    memcpy_s(fatbinData.data(), headSize, &fatbinHeadInfo.configKeyNum, headSize);
    memcpy_s(fatbinData.data() + headSize, configKeyListSize, fatbinHeadInfo.configKeyList.data(), configKeyListSize);
    memcpy_s(fatbinData.data() + headSize + configKeyListSize, binOffsetListSize,
             fatbinHeadInfo.binOffsets.data(), binOffsetListSize);
    fatbinFile.write(reinterpret_cast<const char*>(fatbinData.data()), fatbinData.size());
    fatbinFile.write(reinterpret_cast<const char*>(fatbinBuffer.data()), fatbinBuffer.size());
    if (!fatbinFile.good()) {
        ALOG_WARN_F("Error occurred during writing");
    }
    ALOG_INFO_F("headInfoSize is: %zu, fatbin size is: %ld.", offset, fatbinFile.tellp());
    fatbinFile.close();
    return true;
}

void KernelDumpUtils::WriteFatbinJson(const std::vector<JsonInfo> &allBinJsonInfo, const std::string &fatbinJsonPath,
        const std::string &binFileName) {
    std::ofstream jsonFile(fatbinJsonPath);
    Json fatbinJson;
    fatbinJson["binFileName"] = binFileName;
    fatbinJson["binFileSuffix"] = ".o";
    fatbinJson["coreType"] = "MIX";
    fatbinJson["kernelName"] = "ast_main_0";
    fatbinJson["magic"] = "RT_DEV_BINARY_MAGIC_ELF";
    fatbinJson["dynamicParamMode"] = "folded_with_desc";
    fatbinJson["blockDim"] = -1;
    int64_t workspaceSize = -1;
    Json kernelListJson;
    for (auto &binInfo : allBinJsonInfo) {
        Json kernelInfo;
        kernelInfo["configKey"] = binInfo.configKey;
        kernelInfo["blockDim"] = binInfo.blockDim;
        kernelInfo["kernelName"] = binInfo.kernelName;
        kernelInfo["workspaceSize"] = binInfo.workspaceSize;
        workspaceSize = binInfo.workspaceSize > workspaceSize ? binInfo.workspaceSize : workspaceSize;
        kernelListJson.emplace_back(kernelInfo);
    }
    fatbinJson["workspace"] = {{"num", 1}, {"size", {workspaceSize}}, {"type", {0}}};
    fatbinJson["kernelList"] = kernelListJson;
    fatbinJson["compileInfo"] = nlohmann::json::object();
    jsonFile << fatbinJson.dump(LEVEL_FOUR) << std::endl;
    jsonFile.close();
}

bool KernelDumpUtils::GetSubJsonInfo(const std::string &jsonPath, JsonInfo &kernelJsonInfo) {
    char resolvedPath[PATH_MAX] = {0x00};
    if (realpath(jsonPath.c_str(), resolvedPath) == nullptr) {
        ALOG_INFO_F("realpath failed, val is: %s.", jsonPath.c_str());
        return false;
    }
    Json jsonValue;
    std::ifstream ifs(resolvedPath);
    try {
        if (!ifs.is_open()) {
            return false;
        }
        ifs >> jsonValue;
        ifs.close();
    } catch (const std::exception &e) {
        ifs.close();
        return false;
    }
    try {
        kernelJsonInfo.blockDim = jsonValue.at("blockDim").get<int64_t>();
        kernelJsonInfo.kernelName = jsonValue.at("kernelName").get<std::string>();
        if (jsonValue.find("workspace") != jsonValue.end()) {
            Json workspaceValue = jsonValue["workspace"];
            kernelJsonInfo.workspaceSize = workspaceValue.at("size").get<std::vector<int64_t>>().at(0);
        }
    } catch (const std::exception &e) {
        ALOG_ERROR_F("Fail to parse json, error:%s, json is %s.", e.what(), jsonValue.dump().c_str());
        return false;
    }
    return true;
}

void GetEnv(const char *envName, std::string &envValue) {
    const size_t envValueMaxLen = 1024UL * 1024UL;
    const char *envTemp = std::getenv(envName);
    if ((envTemp == nullptr) || (strnlen(envTemp, envValueMaxLen) >= envValueMaxLen)) {
        ALOG_INFO_F("Env[%s] is not found.", envName);
        return;
    }
    envValue = envTemp;
}

void* KernelDumpUtils::LoadTileFwkImplOpLib() {
    std::string tileFwkLibPath;
    GetEnv("TILE_FWK_OP_IMPL_PATH", tileFwkLibPath);
    ALOG_INFO_F("Value of env[TILE_FWK_OP_IMPL_PATH] is [%s].", tileFwkLibPath.c_str());
    if (tileFwkLibPath.empty()) {
        ALOG_WARN_F("Value of env[TILE_FWK_OP_IMPL_PATH] is empty.");
        return nullptr;
    }
    void *opLibHandle = dlopen(tileFwkLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (opLibHandle == nullptr) {
        ALOG_INFO_F("Failed to dlopen %s, reason is %s.", tileFwkLibPath.c_str(), dlerror());
    }
    return opLibHandle;
}

void KernelDumpUtils::FreeOpHandle(void *opLibHandle) {
    if (opLibHandle != nullptr) {
        (void)dlclose(opLibHandle);
    }
}
}
