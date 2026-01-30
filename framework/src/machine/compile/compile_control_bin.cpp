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
 * \file compile_control_bin.cpp
 * \brief
 */


#include "tilefwk/op_registry.h"
#include <iosfwd>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/program/program.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "machine/utils/machine_utils.h"
using Json = nlohmann::json;

namespace {
const std::string CustomOpCtrolRunFuncName = "PyptoKernelCtrlServer";
const std::string CustomOpCtrolInitFuncName = "PyptoKernelCtrlServerInit";
const std::string CustomKerneLib = "CUSTKFCKernel";

std::string GetMachineCompilerPath() {
    // ARM arch compiler
    const char *homePath = std::getenv("ASCEND_HOME_PATH");
    if (homePath == nullptr) {
        return "";
    } else {
        return std::string(homePath) + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++";
    }
}
const std::string DeviceMahineCompiler = GetMachineCompilerPath();
}
namespace npu::tile_fwk {

constexpr int DUMP_LEVEL_FOUR = 4;

void GenCustomOpInfo(const std::string &funcName, const std::string &controlAicpuPath, const std::string &constrolSoName) {
    Json customOp;
    AicpuOpConfig costomInit;
    costomInit.functionName = CustomOpCtrolInitFuncName;
    costomInit.kernelSo = constrolSoName + ".so";
    costomInit.opKernelLib = CustomKerneLib;
    costomInit.opType = funcName + "Init";

    AicpuOpConfig costomRun = costomInit;
    costomRun.opType = funcName + "Run";
    costomRun.functionName = CustomOpCtrolRunFuncName;

    GenAicpuOpInfoJson(customOp, {costomInit, costomRun});
    std::string fileName = controlAicpuPath + "/" + constrolSoName + ".json";
    if (!DumpFile(customOp.dump(DUMP_LEVEL_FOUR), fileName)) {
        ALOG_ERROR_F("Contrust custom op json failed");
        return;
    }
    OpInfoManager::GetInstance().GetCustomOpJsonPath() = fileName;
}

bool GenTilingFunc(const std::string &funcName, const std::string &controlAicpuPath) {
    std::ostringstream oss;
    oss << "#include <vector>\n";
    oss << "#include <string>\n";
    oss << "#include \"controlFlow_dev" << funcName << ".h\"\n";
    oss << "#include <map>\n";
    oss << "#include \"" << "tilefwk/aicpu_runtime.h" <<"\"\n";
    oss << "namespace npu::tile_fwk { \n";
    oss << "using controlFlowFuncPtr = uint64_t (*)(void*, int64_t*, RuntimeCallEntryType*, DevStartArgsBase*);\n";
    oss << "namespace " << funcName << "{\n";
    oss << "controlFlowFuncPtr controlFlowptr = ControlFlowEntry;\n";
    oss << "} // end namespace " << funcName << "\n";
    oss << "extern \"C\" __attribute__((visibility(\"default\"))) void* GetCtrlFlowFunc() {\n";
    oss << "return reinterpret_cast<void*>(" <<  funcName << "::controlFlowptr);\n";
    oss << "}\n";
    oss << "}\n";
    std::string fileName = controlAicpuPath + "/control_flow_kernel.cpp";
    return DumpFile(oss.str(), fileName);
}

bool TieFwkAicpuPreCompile(std::string &preCompileO, std::string &controlAicpuPath) {
    std::stringstream preCompileStream;
    std::string ext = "cpp";
    auto files = GetFiles(controlAicpuPath, ext);
    std::string includePath = GetCurrentSharedLibPath() + "/../include/tilefwk";
    for (auto file : files) {
        std::string objFile = controlAicpuPath + file.substr(0, file.find(".")) + ".o";
        std::string compileCmd = DeviceMahineCompiler + " -Wall -O2 -fPIC -c -std=gnu++17 -fno-common " +
                                 controlAicpuPath + file +
                                " -I" + includePath +
                                " -I" + includePath + "/include/" +
                                " -I" + GetCurrentSharedLibPath() + "/include/" +
                                " -o " + objFile;
        ALOG_DEBUG_F("PreCompileCmd is %s, file is %s\n", compileCmd.c_str(), file.c_str());
        int result = std::system(compileCmd.c_str());
        if (result != 0) {
            ALOG_ERROR_F("Precompile %s fail\n", file.c_str());
            return false;
        }
        preCompileStream << objFile << " ";
    }
    preCompileO = preCompileStream.str();
    return true;
}

bool SharedAicpuCompile(const std::string &funcName, const std::string &aicpuDirPath, const std::string &preCompileO) {
    std::string cmdGccCompile = "LD_PRELOAD= " + DeviceMahineCompiler +
                                " -std=gnu++17 -fno-common -shared -fPIC -O2 -Wl,--no-warn-rwx-segments -o " +
                                aicpuDirPath + "/lib" + funcName + "_control.so " + preCompileO +
                                " -Wl,--whole-archive " +
                                GetCurrentSharedLibPath() + "/libpypto_ctrl_server.a" +
                                " -Wl,--no-whole-archive";
    auto ret = std::system(cmdGccCompile.c_str());
    if (ret != 0) {
        ALOG_ERROR_F("RUNDeviceMachine compile fail\n");
        return false;
    }
    ALOG_INFO_F("CmdGcc: %s\n", cmdGccCompile.c_str());
    std::string srcSoPath = aicpuDirPath + "/lib" + funcName + "_control.so";
    std::string constrolSoName = "lib" + funcName + "_control";
    GenCustomOpInfo(funcName, aicpuDirPath, constrolSoName);
    return ReadBytesFromFile(srcSoPath, OpInfoManager::GetInstance().GetControlBuffer());
}

bool TileFwkAiCpuCompile(const std::string &funcName, const std::string &aicpuDirPath) {
    OpInfoManager::GetInstance().GetOpFuncName() = funcName;
    std::string controlAicpuPath = aicpuDirPath + "/" + funcName + "/aicpu/";
    if (!GenTilingFunc(funcName, controlAicpuPath)) {
        ALOG_ERROR_F("Gen op[%s]  not success\n", funcName.c_str());
        return false;
    }
    // preCompile
    std::string preCompileO= "";
    if (!TieFwkAicpuPreCompile(preCompileO, controlAicpuPath)) {
        ALOG_ERROR_F("Op %s preCompile fail\n", funcName);
        return false;
    }
    return SharedAicpuCompile(funcName, aicpuDirPath, preCompileO);
}
}