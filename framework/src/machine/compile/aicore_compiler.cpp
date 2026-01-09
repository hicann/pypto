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
 * \file aicore_compiler.cpp
 * \brief
 */

#include "machine/compile/aicore_compiler.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include "interface/utils/log.h"
#include "codegen/utils/parallel_execute.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/compile/gen_aicore_code.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {
namespace {
constexpr const size_t CMD_SIZE_1K = 1024;
constexpr const size_t CMD_SIZE_2K = 2048;
constexpr const char* BISHENG_PROGRAM_CMD = "bisheng";
constexpr const char* BISHENG_LD_CMD = "ld.lld";
}

static int CompileCoreMachine(const std::string &objFile, bool isCube, uint64_t tilingKey,
                              const std::string &headFile, const std::string &aicoreSrcFile) {
  ALOG_INFO_F("Compile src file is [%s], kernel type[%d].", aicoreSrcFile.c_str(), isCube);
  std::string ccecAicVersion = Platform::Instance().GetSoc().GetCCECVersion("AIC");
  std::string ccecAivVersion = Platform::Instance().GetSoc().GetCCECVersion("AIV");
  const std::string cc_opt = isCube ? ccecAicVersion : ccecAivVersion;
  const std::string coreType = isCube ? "-D__AIC__" : "-D__AIV__";
  const auto &opType = OpInfoManager::GetInstance().GetOpType();
  std::string hasSubFunc = headFile.empty() ? "" : "-D__HAS_SUB_FUNC__";
  std::string ccecCmd;
  ccecCmd.resize(CMD_SIZE_2K);
  std::string includePath = GetCurrentSharedLibPath() + "/../include/tile_fwk";
  int ret = snprintf_s(ccecCmd.data(), CMD_SIZE_2K, CMD_SIZE_2K - 1,
                   "%s -c -O3 -g -x cce -Wall -Werror -std=c++17 "
                   "--cce-aicore-only "
                   "--cce-aicore-arch=%s "
                   "-mllvm -cce-aicore-stack-size=0x8000 "
                   "-mllvm -cce-aicore-function-stack-size=0x8000 "
                   "-mllvm -cce-aicore-record-overflow=false "
                   "-mllvm -cce-aicore-addr-transform "
                   "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
                   "-D__MIX__ "
                   "-D__TILINGKEY__=%s "
                   "-D__OPTYPE__=%s "
                   "-D__HEAD_FILE__=%s "
                   "%s "
                   "%s "
                   "-I%s/tileop/arch32 "
                   "-I%s/ "
                   "-I%s/include/tileop/arch32 "
                   "-I%s/include/ "
                   "-o %s "
                   "%s",
                   BISHENG_PROGRAM_CMD, cc_opt.c_str(), std::to_string(tilingKey).c_str(), opType.c_str(),
                   headFile.c_str(), hasSubFunc.c_str(), coreType.c_str(), includePath.c_str(), includePath.c_str(),
                   GetCurrentSharedLibPath().c_str(), GetCurrentSharedLibPath().c_str(),
                   objFile.c_str(), aicoreSrcFile.c_str());
  if (ret < 0) {
    ALOG_ERROR_F("Compile aicore construct cmd failed.");
    return ret;
  }
  ALOG_DEBUG_F("Compile ccec command:[%s].", ccecCmd.c_str());
  ret = std::system(ccecCmd.c_str());
  if (ret != 0) {
    ALOG_ERROR_F("Compile ccec failed.");
  }
  return ret;
}

std::string GenSubFuncCall(std::map<uint64_t, Function *> &leafDict, CoreType coreType,
    dynamic::EncodeDevAscendFunctionParam &param, const std::string &ccePath, uint64_t tilingKey,
    std::stringstream &src_obj) {
  std::stringstream code;
  int leafIndex = 1;
  std::map<int, std::string> idxNameMap;
  // Declare extern leaf func.
  for (const auto &iter : leafDict) {
      const auto leaf = iter.second;
      leafIndex = param.calleeHashIndexDict[leaf->ComputeHash().GetHash()];
      auto leafFuncAttr = leaf->GetLeafFuncAttribute();
      ASSERT(leafFuncAttr != nullptr) << "LeafFuncAttr is null for leaf: " << leaf;
      if (coreType != leafFuncAttr->coreType) {
          continue;
      }
      src_obj << leafFuncAttr->binPath << " ";
      code << leafFuncAttr->kernelDeclare << std::endl;
      idxNameMap[leafIndex] = leafFuncAttr->kernelName;
      ALOG_DEBUG_F("Func[%d] kernel_name[%s].", leafIndex, leafFuncAttr->kernelName.c_str());
  }
  if (idxNameMap.empty()) {
    return "";
  }
  // Define call sub func inline function.
  code << "__attribute__((always_inline)) inline __aicore__ void CallSubFuncTask(uint64_t funcIdx, ";
  code << "CoreFuncParam *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext) {\n";
  code << "    switch (funcIdx) {\n";
  for (const auto &iter : idxNameMap) {
    ALOG_DEBUG_F("Call sub func id[%d] kernel_name[%s].", iter.first, iter.second.c_str());
    code << "        case " << std::to_string(iter.first) << ": {\n";
    code << "            " << iter.second <<  "(param, gmStackAddr, hcclContext, nullptr);\n";
    code << "            break;\n";
    code << "        }\n";
  }
  code << "        default:\n";
  code << "            return;\n";
  code << "    };\n    return;\n}\n";
  ALOG_DEBUG_F("Sub func call code[\n%s\n].", code.str().c_str());

  std::string head_file = (coreType == CoreType::AIC) ? "sub_func_aic_call_" : "sub_func_aiv_call_";
  head_file = ccePath + head_file + std::to_string(tilingKey) + ".h";
  FILE *fsrc = fopen(head_file.c_str(), "w");
  if (fsrc == nullptr) {
    ALOG_ERROR_F("Fail to open call.h.");
    return "";
  }
  (void)fprintf(fsrc, "%s", code.str().c_str());
  (void)fclose(fsrc);
  return head_file;
}

static int LinkObject(const std::string &src_objs, std::string &objPath, const std::string &ccePath, bool relocate,
                      const std::string &key) {
  size_t cmdSize = src_objs.size() + objPath.size() + CMD_SIZE_1K;
  std::string ccecCmd;
  ccecCmd.resize(cmdSize);
  int ret = snprintf_s(ccecCmd.data(), cmdSize, cmdSize - 1,
                       "%s -m aicorelinux -Ttext=0 -static %s -o "
                       "%s "
                       "%s", BISHENG_LD_CMD, relocate ? "-r" : "" , objPath.c_str(), src_objs.c_str());
  if (ret < 0) {
    ALOG_ERROR_F("LinkCoreMachine construct cmd failed.");
    return ret;
  }
  ALOG_DEBUG_F("Link ccec command:[%s].", ccecCmd.c_str());
  const std::string linkScript = ccePath + "link_" + key + "_" + std::to_string(getpid()) + ".sh";
  std::ofstream script(linkScript.c_str());
  script << "#!/bin/bash\n";
  script << ccecCmd.c_str();
  script.close();
  const std::string ldCmd = "bash " + linkScript;
  ret = std::system(ldCmd.c_str());
  if (ret != 0) {
    ALOG_ERROR_F("Link kernel failed.");
  }
  return ret;
}

int CompileAICoreKernel(std::map<uint64_t, Function *> &leafDict, dynamic::EncodeDevAscendFunctionParam &param,
                        const std::string &ccePath, const std::string &funcHash, std::string &kernelPath) {
  if (ccePath.empty()) {
    ALOG_ERROR_F("No cce path.");
    return -1;
  }
  uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
  std::string aic_obj = ccePath + "dy_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
  std::string aiv_obj = ccePath + "dy_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
  std::string aicoreSrcFile = ccePath + "aicore.cpp";
  if (!GenAicoreSrcFile(aicoreSrcFile, funcHash)) {
    ALOG_ERROR_F("Fail to generate aicore src file.");
    return -1;
  }
  std::deque<std::function<void(void)>> tasks;
  std::function task = [&ccePath, &funcHash, &leafDict, &param, &aic_obj, &aicoreSrcFile, &tilingKey]() {
      // gen switch case func
      std::stringstream src_aic_obj;
      auto headFile = GenSubFuncCall(leafDict, CoreType::AIC, param, ccePath, tilingKey, src_aic_obj);
      std::string mid_aic_obj = ccePath + "mid_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
      auto ret = CompileCoreMachine(mid_aic_obj, true, tilingKey, headFile, aicoreSrcFile);
      ASSERT(ret == 0) << "CompileCoreMachine failed with return code  " << ret;
      src_aic_obj << mid_aic_obj;
      ret = LinkObject(src_aic_obj.str(), aic_obj, ccePath, true, "aic");
      ASSERT(ret == 0) << "LinkObject failed with return code  " << ret;
      return;
  };
  tasks.push_back(task);

  std::function task1 = [&ccePath, &funcHash, &leafDict, &param, &aiv_obj, &aicoreSrcFile, &tilingKey]() {
      std::stringstream src_aiv_obj;
      auto headFile = GenSubFuncCall(leafDict, CoreType::AIV, param, ccePath, tilingKey, src_aiv_obj);
      std::string mid_aiv_obj = ccePath + "mid_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
      auto ret = CompileCoreMachine(mid_aiv_obj, false, tilingKey, headFile, aicoreSrcFile);
      ASSERT(ret == 0) << "CompileCoreMachine failed with return code  " << ret;
      src_aiv_obj << mid_aiv_obj;
      ret = LinkObject(src_aiv_obj.str(), aiv_obj, ccePath, true, "aiv");
      ASSERT(ret == 0) << "LinkObject failed with return code  " << ret;
      return;
  };
  tasks.push_back(task1);
  ParallelExecuteAndWait(tasks.size(), tasks);

  std::stringstream src_obj;
  src_obj << " " << aic_obj << " " << aiv_obj;
  kernelPath = ccePath + "dy_kernel_" + funcHash + "_" + std::to_string(tilingKey) + ".o";
  ALOG_DEBUG_F("Compile dynamic kernel to %s.", kernelPath.c_str());
  auto ret = LinkObject(src_obj.str(), kernelPath, ccePath, false, "mix");
  return ret;
}

}
