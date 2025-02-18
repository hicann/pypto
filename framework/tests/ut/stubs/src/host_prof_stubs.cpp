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
 * \file host_prof_stubs.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include "toolchain/prof_api.h"
#include "log_types.h"
#include "prof_common.h"

uint64_t MsprofSysCycleTime() {
  return 1;
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) {
  (void)moduleId;
  (void)handle;
  return 0;
}

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api) {
  (void)agingFlag;
  (void)api;
  return 0;
 }

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) {
  (void)agingFlag;
  (void)data;
  (void)length;
  return 0;
}

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) {
  (void)agingFlag;
  (void)data;
  (void)length;
  return 0;
}

uint64_t MsprofGetHashId(const char *hashInfo, size_t length) {
  (void)hashInfo;
  (void)length;
  return 0;
}