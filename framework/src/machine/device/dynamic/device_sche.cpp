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
 * \file device_machine.cpp
 * \brief DO NOT MODIFY THIS FILE!!!
 *      This file is an entry file which is only used to export symbol.
 *      Please modify the implementation of *DynMachineManager* when needed.
 */

#include "device_sche.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sched.h>
#include "machine/device/dynamic/device_utils.h"
#include "machine/utils/device_log.h"
#include "device_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

DynMachineManager g_machine_mgr;

void SigAct(int signum, siginfo_t* info, void* act) {
    g_machine_mgr.SigAct(signum, info, act);
}

}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerInit(void *targ);
extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServer(void *targ);

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *targ) {
    (void)targ;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *targ) {
    DeviceKernelArgs *kargs = (DeviceKernelArgs *)targ;
    DynMachineManager::KernelCtrlEntry entry = { SigAct, PyptoKernelCtrlServerInit, PyptoKernelCtrlServer };
    return g_machine_mgr.Entry(kargs, entry);
}
