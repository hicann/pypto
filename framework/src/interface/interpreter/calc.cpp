/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <dlfcn.h>
#include <iostream>

#include "tilefwk/error.h"

#include "interface/utils/file_utils.h"

namespace npu::tile_fwk::calc {

typedef struct CalcOps* (*GetCalcOpsFunc)();

struct CalcOps *GetCalcOps() {
    static std::once_flag once_;
    static struct CalcOps *calcOps = nullptr;

    std::call_once(once_, []() {
        std::string path = GetCurrentSharedLibPath() + "/libtile_fwk_calculator.so";
        if (!FileExist(path)) {
            return;
        }
        auto handle = dlopen(path.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            std::cout << "torch not found please check the library path or import torch first" << std::endl;
            return;
        }
        auto func = dlsym(handle, "GetCalcOps");
        calcOps = reinterpret_cast<GetCalcOpsFunc>(func)();
    });

    return calcOps;
}
}
