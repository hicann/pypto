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

void *CannHostRuntime::GetSymbol(const std::string &sym) {	 
 #ifdef BUILD_WITH_CANN	 
     if (handleDep_ != nullptr && handle_ != nullptr) {	 
         return dlsym(handle_, sym.c_str());	 
     }	 
 #endif	 
     (void)sym;	 
     return nullptr;	 
 }	 

CannHostRuntime::CannHostRuntime() {
#ifdef BUILD_WITH_CANN
    std::string LibPathDir = std::string(ASCEND_CANN_PACKAGE_PATH) + "/lib64/";
    std::string soDepPath = RealPath(LibPathDir + "libprofapi.so");
    FUNCTION_LOGW("soDepPath = %s", soDepPath.c_str());
    handleDep_ = dlopen(soDepPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    std::string soPath = RealPath(LibPathDir + "libruntime.so");
    FUNCTION_LOGW("soPath = %s", soPath.c_str());
    handle_ = dlopen(soPath.c_str(), RTLD_LAZY);
    if (handleDep_ != nullptr && handle_ != nullptr) {
        socVerFunc_ = (GetSocVerFunc)GetSymbol(socVerFuncName);
    }
#endif
    if (handleDep_ == nullptr || handle_ == nullptr) { 
        FUNCTION_LOGW("Cannot obtain so file through dlopen."); 
    }
}

CannHostRuntime::~CannHostRuntime() { 
    if (handle_ != nullptr) {
        dlclose(handle_);	 
    }
    if (handleDep_ != nullptr) {	 
        dlclose(handleDep_);	 
    }
}

CannHostRuntime &CannHostRuntime::Instance() {
    static CannHostRuntime instance;
    return instance;
}

bool CannHostRuntime::GetSocVersion(std::string& socVersion) {
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

std::string CannHostRuntime::GetPlatformFile(const std::string &socVersion) {
    std::string platformFile;
    if (socVersion.empty()) {
        return "";
    }
    // get platform file path
    const char *envPath = std::getenv("ASCEND_HOME_PATH");
    if (envPath == nullptr) {
        return "";
    }
    std::string configRelativePath = "data/platform_config/";
#ifdef PROCESSOR_SUBPATH
    const char *processorSubpath = PROCESSOR_SUBPATH;
#else
    const char *processorSubpath = "";
#endif
    std::string platformConfDir = std::string(envPath) + "/" + std::string(processorSubpath) + "/" + configRelativePath;
    if (RealPath(platformConfDir).empty()) {
        platformConfDir = std::string(envPath) + "/" + configRelativePath;
    }
    platformFile = platformConfDir + socVersion + ".ini";
    if (RealPath(platformFile).empty()) {
        return "";
    }
    return platformFile;
}
}  // namespace tile_fwk
}  // namespace npu
