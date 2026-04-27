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
 * \file cann_host_runtime.cpp
 * \brief
 */

#include "tilefwk/cann_host_runtime.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu {
namespace tile_fwk {
const uint32_t kMaxLength = 50;
const std::string socVerFuncName = "rtGetSocVersion";
const std::string socSpecFuncName = "rtGetSocSpec";
const std::string aiCpuCntFuncName = "rtGetAiCpuCount";

void* CannHostRuntime::GetSymbol(const std::string& sym)
{
#ifdef BUILD_WITH_CANN
    if (handleDep_ != nullptr && handle_ != nullptr) {
        return dlsym(handle_, sym.c_str());
    }
#endif
    (void)sym;
    return nullptr;
}

CannHostRuntime::CannHostRuntime()
{
#ifdef BUILD_WITH_CANN
    std::string LibPathDir = std::string(ASCEND_CANN_PACKAGE_PATH) + "/lib64/";
    std::string soDepPath = RealPath(LibPathDir + "libprofapi.so");
    FE_LOGW("soDepPath = %s", soDepPath.c_str());
    handleDep_ = dlopen(soDepPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    std::string soPath = RealPath(LibPathDir + "libruntime.so");
    FE_LOGW("soPath = %s", soPath.c_str());
    handle_ = dlopen(soPath.c_str(), RTLD_LAZY);
    if (handleDep_ != nullptr && handle_ != nullptr) {
        socVerFunc_ = (GetSocVerFunc)GetSymbol(socVerFuncName);
        socSpecFunc_ = (GetSocSpecFunc)GetSymbol(socSpecFuncName);
        aiCpuCntFunc_ = (GetAiCpuCntFunc)GetSymbol(aiCpuCntFuncName);
    }
#endif
    if (handleDep_ == nullptr || handle_ == nullptr) {
        FE_LOGW("Cannot obtain so file through dlopen.");
    }
}

CannHostRuntime::~CannHostRuntime()
{
    if (handle_ != nullptr) {
        dlclose(handle_);
    }
    if (handleDep_ != nullptr) {
        dlclose(handleDep_);
    }
}

CannHostRuntime& CannHostRuntime::Instance()
{
    static CannHostRuntime instance;
    return instance;
}

bool CannHostRuntime::GetSocVersion(std::string& socVersion)
{
#ifdef BUILD_WITH_CANN
    int ret = 1;
    char socVer[kMaxLength] = {0x00};
    if (socVerFunc_ != nullptr) {
        ret = socVerFunc_(socVer, kMaxLength);
        socVer[kMaxLength - 1] = '\0';
    }
    if (ret == 0) {
        socVersion = std::string(socVer);
        return true;
    }
#endif
    socVersion.clear();
    return false;
}

bool CannHostRuntime::GetSocSpec(const std::string& column, const std::string& key, std::string& val)
{
#ifdef BUILD_WITH_CANN
    int ret = 1;
    char charVal[kMaxLength] = {0};
    if (socSpecFunc_ != nullptr) {
        ret = socSpecFunc_(column.c_str(), key.c_str(), charVal, kMaxLength);
    }
    if (ret == 0) {
        charVal[kMaxLength - 1] = '\0';
        val = std::string(charVal);
        return true;
    }
#endif
    (void)column;
    (void)key;
    (void)val;
    return false;
}

bool CannHostRuntime::GetAICPUCnt(size_t& aiCpuCnt)
{
#ifdef BUILD_WITH_CANN
    int ret = 1;
    uint32_t cpuNum = 0;
    if (aiCpuCntFunc_ != nullptr) {
        ret = aiCpuCntFunc_(&cpuNum);
    }
    if (ret == 0) {
        aiCpuCnt = static_cast<size_t>(cpuNum);
        return true;
    }
#endif
    (void)aiCpuCnt;
    return false;
}
} // namespace tile_fwk
} // namespace npu
