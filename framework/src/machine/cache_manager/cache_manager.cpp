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
 * \file cache_manager.cpp
 * \brief
 */

#include "cache_manager.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/platform/platform_manager.h"
#include "interface/utils/op_info_manager.h"

namespace npu::tile_fwk {
namespace {
const std::string CACHE_FILE_PREFIX = "ast_op_";
const std::string CACHE_CONTROL_FILE_PREFIX = "pypto_control_";
const std::string CACHE_BIN_FILE_SUFFIX = ".o";
const std::string CACHE_CUSTOM_BIN_FILE_SUFFIX = "_control.so";
const std::string CACHE_CUSTOM_JSON_FILE_SUFFIX = "_control.json";
const std::string CACHE_KERNEL_FILE_SUFFIX = "_kernel.o";
const std::string CACHE_LOCK_FILE_SUFFIX = ".lock";
}
CacheManager& CacheManager::Instance() {
    static CacheManager cacheManager;
    return cacheManager;
}

bool CacheManager::Initialize() {
    if (isInit_) {
        return true;
    }
    if (!config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false)) {
        ALOG_INFO_F("Binary cache is not enable.");
        return true;
    }
    cacheMode_ = CacheMode::Enable;

    // create cache dir
    const char *envPath = std::getenv("HOME");
    if (envPath == nullptr) {
        ALOG_ERROR_F("Env[HOME] is not existed or empty.");
        return false;
    }
    std::string homeEnvPath(envPath);
    cacheDirPath_ = homeEnvPath + "/ast_data/" + PlatformManager::Instance().GetShortSocVersion();
    ALOG_DEBUG_F("Begin to initialize cache manager, cache dir path is [%s].", cacheDirPath_.c_str());
    if (RealPath(cacheDirPath_).empty() && !CreateMultiLevelDir(cacheDirPath_)) {
        ALOG_ERROR_F("Failed to create cache dir[%s].", cacheDirPath_.c_str());
        return false;
    }
    ALOG_INFO_F("Cache manager has been initialized at cache dir path[%s].", cacheDirPath_.c_str());
    isInit_ = true;
    return true;
}

bool CacheManager::MatchBinCache(const std::string &cacheKey) const {
    if (!IsCahceEnable()) {
        return false;
    }
    if (cacheKey.empty()) {
        return false;
    }
    std::string cacheBinFile = cacheDirPath_ + "/" + CACHE_FILE_PREFIX + cacheKey + CACHE_BIN_FILE_SUFFIX;
    ALOG_DEBUG_F("Try to check whether bin file[%s] is existed.", cacheBinFile.c_str());
    std::string customSoPath = cacheDirPath_ + "/lib" + OpInfoManager::GetInstance().GetOpFuncName() +
                                        CACHE_CUSTOM_BIN_FILE_SUFFIX;
    std::string customJsonPath = cacheDirPath_ + "/lib" + OpInfoManager::GetInstance().GetOpFuncName() +
                                        CACHE_CUSTOM_JSON_FILE_SUFFIX;                                    
    std::lock_guard<std::mutex> lock_guard(cacheMutex_);
    // check whether both json and bin file is existed
    bool ret = !RealPath(cacheBinFile).empty() && !RealPath(customSoPath).empty() &&
                !RealPath(customJsonPath).empty();
    if (ret) {
        ALOG_INFO_F("Cache matched, bin file[%s] and [%s] is existed.",
                     cacheBinFile.c_str(), customSoPath.c_str());
    } else {
        ALOG_INFO_F("Cache missed, bin file[%s] or [%s] or [%s] is not existed.",
                     cacheBinFile.c_str(), customSoPath.c_str(), customJsonPath.c_str());
    }
    return ret;
}

void CacheManager::SaveTaskFile(const DeviceAgentTask *deviceAgentTask) const {
    if (!IsCahceEnable()) {
        return;
    }
    if (deviceAgentTask == nullptr || deviceAgentTask->GetFunction() == nullptr) {
        return;
    }
    Function *function = deviceAgentTask->GetFunction();
    std::string basePath = cacheDirPath_ + "/" + CACHE_FILE_PREFIX + deviceAgentTask->compileTask->GetCacheKey();
    std::string binFilePath = basePath + CACHE_BIN_FILE_SUFFIX;
    std::string kernelFilePath = basePath + CACHE_KERNEL_FILE_SUFFIX;
    std::string customSoPath = cacheDirPath_ + "/lib" + OpInfoManager::GetInstance().GetOpFuncName() +
                                        CACHE_CUSTOM_BIN_FILE_SUFFIX;
    std::string customJsonPath = cacheDirPath_ + "/lib" + OpInfoManager::GetInstance().GetOpFuncName() +
                                        CACHE_CUSTOM_JSON_FILE_SUFFIX;
    ALOG_DEBUG_F("Try to save bin file[%s], function type is [%s], control bin file[%s].", binFilePath.c_str(),
                 function->GetFunctionTypeStr().c_str(), customSoPath.c_str());
    std::lock_guard<std::mutex> lock_guard(cacheMutex_);
    if (!RealPath(binFilePath).empty() && !RealPath(customSoPath).empty() && !RealPath(customJsonPath).empty()) {
        ALOG_INFO_F("Bin file[%s] and [%s] already exists.", binFilePath.c_str(), customSoPath.c_str());
        return;
    }
    if (function->IsFunctionType(FunctionType::DYNAMIC) && function->GetDyndevAttribute() != nullptr) {
        ALOG_INFO_F("Save devProgBinary at bin file[%s].", binFilePath.c_str());
        std::string lockFilePath =
            cacheDirPath_ + "/" + CACHE_FILE_PREFIX + deviceAgentTask->compileTask->GetCacheKey() + CACHE_LOCK_FILE_SUFFIX;
        FILE *fp = LockAndOpenFile(lockFilePath);
        if (fp == nullptr) {
            return;
        }
        if (RealPath(binFilePath).empty()) {
            SaveFile(binFilePath, function->GetDyndevAttribute()->devProgBinary);
            SaveFile(kernelFilePath, function->GetDyndevAttribute()->kernelBinary);
        }
        if (RealPath(customSoPath).empty() || RealPath(customJsonPath).empty()) {
            std::vector<uint8_t> controlBin;
            size_t binSize = OpInfoManager::GetInstance().GetControlBuffer().size();
            controlBin.resize(OpInfoManager::GetInstance().GetControlBuffer().size());
            if (memcpy_s(controlBin.data(), binSize, OpInfoManager::GetInstance().GetControlBuffer().data(),
                         binSize) != EOK) {
                ALOG_INFO_F("Control bin memCpy failed");
                return;
            }
            SaveFile(customSoPath, controlBin);
            CopyFile(OpInfoManager::GetInstance().GetCustomOpJsonPath(), customJsonPath);
        }
        UnlockAndCloseFile(fp);
    }
}

bool CacheManager::RecoverTask(const std::string &cacheKey, DeviceAgentTask *deviceAgentTask) const {
    if (!IsCahceEnable()) {
        return false;
    }
    if (deviceAgentTask == nullptr || deviceAgentTask->GetFunction() == nullptr) {
        return false;
    }
    Function *function = deviceAgentTask->GetFunction();
    std::string cacheBinFile = cacheDirPath_ + "/" + CACHE_FILE_PREFIX + cacheKey + CACHE_BIN_FILE_SUFFIX;
    std::string cacheKernelFile = cacheDirPath_ + "/" + CACHE_FILE_PREFIX + cacheKey + CACHE_KERNEL_FILE_SUFFIX;
    std::string customJsonPath = cacheDirPath_ + "/lib" + OpInfoManager::GetInstance().GetOpFuncName() +
                                 CACHE_CUSTOM_JSON_FILE_SUFFIX;
    ALOG_DEBUG_F("Try to recover device task from bin file[%s], function type is [%s].", cacheBinFile.c_str(),
                 function->GetFunctionTypeStr().c_str());
    auto attr = function->GetDyndevAttribute();
    std::lock_guard<std::mutex> lock_guard(cacheMutex_);
    if (function->IsFunctionType(FunctionType::DYNAMIC)) {
        ALOG_INFO_F("Recover binary from file[%s][%s].", cacheBinFile.c_str(), cacheKernelFile.c_str());
        attr->devProgBinary = LoadFile(cacheBinFile);
        attr->kernelBinary = LoadFile(cacheKernelFile);
        OpInfoManager::GetInstance().GetCustomOpJsonPath() = customJsonPath;
        return !attr->devProgBinary.empty() && !attr->kernelBinary.empty();
    }
    return true;
}
}