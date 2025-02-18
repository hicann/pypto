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
 * \file pypto_aicpu_interface.cpp
 * \brief
 */

#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include "pypto_aicpu_interface.h"
#include "machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/machine_ws_intf.h"


namespace {
  npu::tile_fwk::BackendServerHandleManager g_handleManager;
}
using namespace npu::tile_fwk;
extern "C" {
__attribute__((visibility("default"))) uint32_t StaticPyptoKernelServer(void *args) {
    DEV_DEBUG("Start to exect static server");
    if (args == nullptr) {
        DEV_ERROR("Args is invalid");
        return 1;
    }
    auto devArgs = (DeviceArgs*)args;
    auto data = reinterpret_cast<char *>(devArgs->aicpuSoBin);
    if (!g_handleManager.SaveSoFile(data, devArgs->aicpuSoLen)) {
        DEV_ERROR("create so failed");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    DEV_DEBUG("Begin to exe static tileFwk Server");
    auto ret = g_handleManager.ExecuteFunc(args, staticFuncKey);
    DEV_DEBUG("After Get kernel func [%s], with ret[%d]",
                        staticServerKernelkFun.c_str(), static_cast<int>(ret));
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", staticServerKernelkFun.c_str());
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerNull(void *args) {
#if DEBUG_PLOG && defined(__DEVICE__)
    InitLogSwitch();
#endif
    if (args == nullptr) {
        DEV_ERROR("Server init input args is null");
        return 1;
    }
    auto kargs = (AstKernelArgs *)args;
    if (kargs == nullptr) {
        DEV_ERROR("Server init AstKernelArgs is null");
        return 1;
    }
    auto devArgs = reinterpret_cast<DeviceArgs*>(kargs->cfgdata);
    auto data = reinterpret_cast<char *>(devArgs->aicpuSoBin);
    if (!g_handleManager.SaveSoFile(data, devArgs->aicpuSoLen, devArgs->deviceId)) {
        DEV_ERROR("create so failed");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void *args) {
    auto ret = g_handleManager.ExecuteFunc(args, dyExecFuncKey);
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", dynServerKernelFun.c_str());
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void *args) {
    auto ret = g_handleManager.ExecuteFunc(args, dyInitFuncKey);
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", dynServerKernelInitFun.c_str());
        return 1;
    }
    return 0;
}
}