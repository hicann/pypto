/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "interface/function/function.h"
#ifdef BUILD_WITH_CANN
#include "profiling/aprof_pub.h"
#include "acl/acl_base_rt.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk{
struct CacheTaskInfo {
    uint32_t taskType;
    uint32_t numBlocks;
    uint64_t nodeId;
    uint64_t opType;
    uint64_t attrId{0};
    uint64_t reserve2{0};
    uint32_t opFlag;
    uint32_t tensorNum;
    MsrofTensorData tensorData[0];
};
class HostProf
{
public:
  HostProf() = default;
  ~HostProf();
  bool HostProfReportApi(const uint64_t &startTime, const uint64_t &endTime) const;
  void HostProfReportNodeInfo(const uint64_t &endTime, const uint32_t blockDim, const uint16_t taskType) const;
  void HostProfReportContextInfo(const uint64_t &endTime) const;
  void HostProfReportCacheTaskInfo(const aclrtStream stream, const uint32_t numBlocks, const uint32_t taskType) const;
  void SetProfFunction(Function *function);
  static uint64_t GetProfSwitch();
  static uint32_t GetProfType();
  static void RegHostProf();
private:
  static int32_t HostProfInit(uint32_t type, void *data, uint32_t len);
  void ReportTensoInfo(const uint32_t &groupId, const uint32_t mods, const uint64_t &endTime) const;
  void PackTensorInfo(MsprofTensorInfo *profTensorData, const uint32_t groupId, const uint32_t modId) const;
  void HostProfReportBasicInfo(const uint64_t &endTime, const uint32_t blockDim, const uint16_t taskType) const;
  void HostProfReportTensorInfo(const uint64_t &endTime) const;
  static bool IsCacheOpInfoEnable(const aclrtStream stream);
  static void BuildCacheTensorInfo(CacheTaskInfo *taskInfo);
  static void BuildTensor(const uint32_t tensorType, const RawTensorDataPtr &tensorInfo, MsrofTensorData &tensorData);
  std::string opName_;
  Function *profFunction_{nullptr};
  uint32_t inputsSize_;
  static uint64_t profSwitch_;  // prof level
  static uint32_t profType_;    // prof open/close
};
} // npu::tile_fwk
#else
namespace npu::tile_fwk{
class HostProf
{
public:
  HostProf() = default;
  ~HostProf() {};
  void SetProfFunction(Function *function) {(void)function;}
  void RegHostProf() {};
  void HostProfReportNodeInfo(uint64_t &endTime, uint32_t blockDim, uint16_t taskType) {
    (void)endTime;
    (void)blockDim;
    (void)taskType;
  }
  void HostProfReportContextInfo(uint64_t &endTime) {(void) endTime;}
  static HostProf &Get() {
    static HostProf hostProf;
    return hostProf;
  }
};
}
#endif