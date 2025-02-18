/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <sys/syscall.h>
#include "machine/runtime/host_prof.h"
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#ifdef BUILD_WITH_CANN
#include "runtime/base.h"
#include "toolchain/prof_api.h"
#include "log_types.h"
#include "prof_common.h"


namespace npu::tile_fwk {
const std::string OpType = "pyPto";
HostProf::~HostProf() {}

uint64_t HostProf::GetProfSwitch() {
  return profSwitch_;
}

uint32_t HostProf::GetProfType() {
  return profType_;
}

uint64_t HostProf::profSwitch_ = 0;
uint32_t HostProf::profType_ = 0;

int32_t HostProf::HostProfInit(uint32_t type, void *data, uint32_t len) {
  if (data == nullptr || len == 0) {
    ALOG_WARN_F("Para is invalid");
    return -1;
  }
  if (type != RT_PROF_CTRL_SWITCH) {
    ALOG_WARN_F("Prof type [%u] is invalid", type);
    return -1;
  }
  if (len < sizeof(MsprofCommandHandle)) {
    ALOG_WARN_F("Prof CommandHandle len [%u] is invalid", len);
    return -1;
  }
  MsprofCommandHandle *hostProfHandleConfig = reinterpret_cast<MsprofCommandHandle*>(data);
  profSwitch_ = hostProfHandleConfig->profSwitch;
  profType_ = hostProfHandleConfig->type;
  ALOG_DEBUG_F("Host prof profSwitch is %lu profType is %u", profSwitch_, profType_);
  return 0;
}

void HostProf::RegHostProf() {
  MsprofRegisterCallback(CCECPU, HostProfInit);
}

bool HostProf::HostProfReportApi(const uint64_t &startTime, const uint64_t &endTime) const {
  struct MsprofApi apiInfo;
  apiInfo.level = MSPROF_REPORT_NODE_LEVEL;
  apiInfo.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  apiInfo.beginTime = startTime;
  apiInfo.endTime = endTime;
  apiInfo.itemId = MsprofGetHashId(opName_.c_str(), opName_.length());
  apiInfo.threadId = syscall(SYS_gettid);
  auto ret = MsprofReportApi(true, &apiInfo);
  if (ret != 0) {
    ALOG_WARN_F("Report Api not success");
    return false;
  }
  return true;
}

void HostProf::HostProfReportNodeInfo(const uint64_t &endTime, const uint32_t blockDim, const uint16_t taskType) const {
  HostProfReportBasicInfo(endTime, blockDim, taskType);
  HostProfReportTensorInfo(endTime);
}

void HostProf::HostProfReportBasicInfo(const uint64_t &endTime, const uint32_t blockDim, const uint16_t taskType) const
{
  struct MsprofCompactInfo nodeBasicInfo;
  nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
  nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  nodeBasicInfo.timeStamp = endTime;
  nodeBasicInfo.threadId = syscall(SYS_gettid);
  nodeBasicInfo.data.nodeBasicInfo.opName = MsprofGetHashId(opName_.c_str(), opName_.length());
  nodeBasicInfo.data.nodeBasicInfo.opType = MsprofGetHashId(OpType.c_str(), OpType.length());
  nodeBasicInfo.data.nodeBasicInfo.taskType = taskType;
  nodeBasicInfo.data.nodeBasicInfo.blockDim = blockDim;
  nodeBasicInfo.data.nodeBasicInfo.opFlag = true;
  auto ret = MsprofReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
                          static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
  if (ret != 0) {
    ALOG_WARN_F("Compact node[%s] basic info failed", opName_.c_str());
  }
}

void HostProf::HostProfReportContextInfo(const uint64_t &endTime) const {
  struct MsprofAdditionalInfo contextInfo;
  contextInfo.level = MSPROF_REPORT_NODE_LEVEL;
  contextInfo.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  contextInfo.threadId = syscall(SYS_gettid);
  contextInfo.timeStamp = endTime;
  struct MsprofContextIdInfo ctxId;
  ctxId.opName = MsprofGetHashId(opName_.c_str(), opName_.length());
  ctxId.ctxIdNum = 1;
  ctxId.ctxIds[0] = 0;
  memcpy_s(contextInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &ctxId, sizeof(MsprofContextIdInfo));
  auto ret = MsprofReportAdditionalInfo(false,reinterpret_cast<void*>(&contextInfo), sizeof(MsprofAdditionalInfo));
  if (ret != 0) {
    ALOG_WARN_F("Op[%s] Msprof report context info not success", opName_.c_str());
  }
}

void HostProf::HostProfReportTensorInfo(const uint64_t &endTime) const {
  if (profFunction_ == nullptr) {
    ALOG_WARN_F("Op [%s] is null", opName_.c_str());
    return;
  }
  uint32_t iONums = profFunction_->inCasts_.size() + profFunction_->outCasts_.size();
  ALOG_DEBUG_F("Op [%s] with inputs[%zu], outputs[%zu]", opName_.c_str(),
              profFunction_->inCasts_.size(), profFunction_->outCasts_.size());
  uint32_t groupNums = iONums / MSPROF_GE_TENSOR_DATA_NUM;
  uint32_t modulus = iONums % MSPROF_GE_TENSOR_DATA_NUM;
  for (uint32_t i = 0; i < groupNums; i++) {
    ReportTensoInfo(i, MSPROF_GE_TENSOR_DATA_NUM, endTime);
  }

  if (modulus > 0) {
    ReportTensoInfo(groupNums, modulus, endTime);
  }
}

void HostProf::ReportTensoInfo(const uint32_t &groupId, const uint32_t mods, const uint64_t &endTime) const {
  struct MsprofAdditionalInfo tensorInfo;
  tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
  tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensorInfo.threadId = syscall(SYS_gettid);
  tensorInfo.timeStamp = endTime;
  auto profTensorData = reinterpret_cast<MsprofTensorInfo*>(tensorInfo.data);
  profTensorData->opName = MsprofGetHashId(opName_.c_str(), opName_.length());
  profTensorData->tensorNum = mods;
  for (uint32_t j = 0; j < mods; j++) {
    PackTensorInfo(profTensorData, groupId, j);
  }
  auto ret = MsprofReportAdditionalInfo(false, reinterpret_cast<void*>(&tensorInfo), sizeof(MsprofAdditionalInfo));
  if (ret != 0) {
    ALOG_WARN_F("Op[%s] Msprof report tensor info not success", opName_.c_str());
  }
}

void HostProf::PackTensorInfo(MsprofTensorInfo *profTensorData, const uint32_t groupId, const uint32_t modId) const {
  uint32_t iOIdx = groupId * MSPROF_GE_TENSOR_DATA_NUM + modId;
  std::shared_ptr<LogicalTensor> iOTensor;
  std::stringstream iOtensorInfo;
  if (inputsSize_ > iOIdx) {
    profTensorData->tensorData[modId].tensorType = MSPROF_GE_TENSOR_TYPE_INPUT;
    profTensorData->tensorData[modId].format = static_cast<uint32_t>(profFunction_->inCasts_[iOIdx]->Format());
    profTensorData->tensorData[modId].dataType = static_cast<uint32_t>(profFunction_->inCasts_[iOIdx]->nodetype);
    iOTensor = profFunction_->inCasts_[iOIdx];
    iOtensorInfo << "Input " << iOIdx << " shape: ";
  } else {
    auto outputIdx = iOIdx - inputsSize_;
    profTensorData->tensorData[modId].tensorType = MSPROF_GE_TENSOR_TYPE_OUTPUT;
    profTensorData->tensorData[modId].format = static_cast<uint32_t>(profFunction_->outCasts_[outputIdx]->Format());
    profTensorData->tensorData[modId].dataType = static_cast<uint32_t>(profFunction_->outCasts_[outputIdx]->nodetype);
    iOTensor = profFunction_->outCasts_[outputIdx];
    iOtensorInfo << "output " << outputIdx << " shape: ";
  }
  size_t shapeLen = iOTensor->shape.size();
  if (iOTensor->shape.size() > MSPROF_GE_TENSOR_DATA_SHAPE_LEN) {
    ALOG_WARN_F("Op [%s] tensor[%u] size[%zu] len over [%d]", opName_.c_str(), iOIdx, shapeLen,
                MSPROF_GE_TENSOR_DATA_SHAPE_LEN);
    shapeLen = MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
  }

  for (size_t j = 0; j < shapeLen; j++) {
    profTensorData->tensorData[modId].shape[j] = iOTensor->shape[j];
    iOtensorInfo << iOTensor->shape[j] << " ";
  }
  for (size_t j = shapeLen; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
    profTensorData->tensorData[modId].shape[j] = 0;
  }
  iOtensorInfo << "\n";
  ALOG_DEBUG_F("tensorInfo %s", iOtensorInfo.str().c_str());
}

void HostProf::SetProfFunction(Function *function)
{
  if (function == nullptr) {
    ALOG_WARN_F("Function is invalid, please check function");
    return;
  }
  // current using functionHashId as opName;
  opName_ = function->GetFunctionHash().Data();
  profFunction_ = function;
  inputsSize_ = profFunction_->inCasts_.size();
}
}
#endif
