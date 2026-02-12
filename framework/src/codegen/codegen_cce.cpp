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
 * \file codegen.cpp
 * \brief
 */

#include <unistd.h>

#include "codegen_cce.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
std::string CodeGenCCE::GetEmitPath(const std::string &name) {
    std::string dirPath;
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_FIXED_OUTPUT_PATH, false)) {
        dirPath = name;
    } else {
        dirPath = config::LogTopFolder() + "/" + name;
    }
    return dirPath;
}

void CodeGenCCE::PrepareOutputPath() {
    if (ctx.IsCCEPathEmpty() || !IsPathExist(ctx.cceDir)) {
        PrepareDefaultOutputPath();
    }
}

void CodeGenCCE::PrepareDefaultOutputPath() {
    if (ctx.IsCCEPathEmpty()) {
        ctx.cceDir = GetEmitPath("kernel_aicore");
    };
    CreateMultiLevelDir(ctx.cceDir);
}

std::map<int, int> GenRealizeIdMap(const SubfuncParam &subFuncParam) {
    auto &tensorInvokeArgs = subFuncParam.tensorsArgs_;
    auto &incastInvokeArgs = subFuncParam.inCastArgs_;
    auto &outcastInvokeArgs = subFuncParam.outCastArgs_;

    std::map<int, int> idMap;
    auto f = [&idMap](size_t offset, auto &invokeArgs) {
        CODEGEN_LOGI("start offset is %d, arg size is %d", offset, invokeArgs.size());
        for (size_t i = 0; i < invokeArgs.size(); i++) {
            size_t paramOff = (offset + i);
            uint32_t paramLoc = invokeArgs[i].paramLoc;
            ALOG_DEBUG("paramLoc ", paramLoc, " --> offset ", paramOff);
            CODEGEN_LOGI(" paramLoc is %d, paramOff is %d, SymDDRId is %d, SymName is %s", paramLoc, paramOff,
                invokeArgs[i].symDDRId, invokeArgs[i].symName.c_str());
            idMap.insert({paramLoc, paramOff});
        }
    };

    CODEGEN_LOGI("---  start tensorInvokeArgs paramLoc map ---- ");
    f(0, tensorInvokeArgs);
    CODEGEN_LOGI("---  start incastInvokeArgs paramLoc map ---- ");
    f(tensorInvokeArgs.size(), incastInvokeArgs);
    CODEGEN_LOGI("---  start outcastInvokeArgs paramLoc map ---- ");
    f(tensorInvokeArgs.size() + incastInvokeArgs.size(), outcastInvokeArgs);
    return idMap;
}
} // namespace npu::tile_fwk
