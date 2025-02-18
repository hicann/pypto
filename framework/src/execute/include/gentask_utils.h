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
 * \file gentask_utils.h
 * \brief
 */

#ifndef GENTASK_UTILS_H_
#define GENTASK_UTILS_H_
#include "graph/kernel_launch_info.h"

using namespace ge;

namespace ops {
class GentaskUtils {
 public:
  static ge::graphStatus CommonOpSelectFormat(const gert::OpCheckContext *context, ge::AscendString &result);
  static ge::graphStatus CommonCalcOpParam(gert::ExeResGenerationContext *context, ge::AscendString &name,
                                           ge::AscendString &reuse_key);
  static ge::graphStatus CommonGenerateTask(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks);
 private:
  static ge::graphStatus InsertHiddenInput(const gert::ExeResGenerationContext* context,
                                           ge::KernelLaunchInfo &aicore_task);
  static ge::graphStatus GenerateAicpuTask(const gert::ExeResGenerationContext* context, const int64_t &sub_stream_id,
                                           std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
#endif

