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
 * \file gentask_utils.cpp
 * \brief
 */

#include "gentask_utils.h"
#include "graph/ge_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/arg_desc_info.h"
#include "tile_fwk_log.h"
#include "nlohmann/json.hpp"

using namespace ge;
using Json = nlohmann::json;

namespace ops {
namespace {
const std::string kTileFwkOpFlag = "tileFwkOp";
const std::string kAicpuInitTaskKernelName = "DynTileFwkKernelServerInit";
const std::string kAicpuMainTaskKernelName = "DynTileFwkKernelServer";
const std::string kAicpuSoName = "DynTileFwkKernelServer";
const uint8_t kAicpuInitTaskNum = 1;
const uint8_t kAicpuMainTaskNum = 5;
}

ge::graphStatus GentaskUtils::CommonOpSelectFormat(const gert::OpCheckContext *context, ge::AscendString &result) {
  (void)context;
  Json op_select_format;
  op_select_format[kTileFwkOpFlag] = "true";
  result = op_select_format.dump().c_str();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GentaskUtils::CommonCalcOpParam(gert::ExeResGenerationContext *context, ge::AscendString &name,
                                                ge::AscendString &reuse_key) {
  gert::StreamInfo stream_info;
  stream_info.name = name;
  stream_info.reuse_key = reuse_key;
  std::vector<int64_t> stream_depend_value_list(0);
  stream_info.depend_value_input_indices = stream_depend_value_list;
  stream_info.required = true;
  std::vector<gert::StreamInfo> stream_info_vec(0);
  stream_info_vec.push_back(stream_info);
  context->SetAttachedStreamInfos(stream_info_vec);

  gert::SyncResInfo sync_res_info;
  sync_res_info.type = gert::SyncResType::SYNC_RES_NOTIFY;
  sync_res_info.name = name;
  sync_res_info.reuse_key = reuse_key;
  sync_res_info.required = true;
  std::vector<gert::SyncResInfo> sync_info_vec(0);
  sync_info_vec.push_back(sync_res_info);
  context->SetSyncResInfos(sync_info_vec);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GentaskUtils::InsertHiddenInput(const gert::ExeResGenerationContext* context,
                                                ge::KernelLaunchInfo &aicore_task) {
  auto ori_args_format = aicore_task.GetArgsFormat();
  if (ori_args_format == nullptr) {
    TILE_FWK_LOGE("Node[%s, %s]: failed to get args format from aicore task.\n",
                  context->GetNodeName(), context->GetNodeType());
    return ge::GRAPH_FAILED;
  }
  std::vector<ge::ArgDescInfo> arg_desc_info = ge::ArgsFormatSerializer::Deserialize(ori_args_format);
  if (arg_desc_info.empty()) {
    TILE_FWK_LOGE("Node[%s, %s]: failed to parse args format.\n", context->GetNodeName(), context->GetNodeType());
    return GRAPH_FAILED;
  }
  arg_desc_info.emplace_back(ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
  auto new_args_format = ge::ArgsFormatSerializer::Serialize(arg_desc_info).GetString();
  if (aicore_task.SetArgsFormat(new_args_format) != ge::GRAPH_SUCCESS) {
    TILE_FWK_LOGE("Node[%s, %s]: failed to set args format for aicore task.\n",
                  context->GetNodeName(), context->GetNodeType());
    return GRAPH_FAILED;
  }
  TILE_FWK_LOGD("Node[%s, %s]: args format is %s.\n", context->GetNodeName(), context->GetNodeType(), new_args_format);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GentaskUtils::GenerateAicpuTask(const gert::ExeResGenerationContext* context,
                                                const int64_t &sub_stream_id,
                                                std::vector<std::vector<uint8_t>> &tasks) {
  // tasks vector only have one task, which is aicore task
  ge::KernelLaunchInfo aicore_task = ge::KernelLaunchInfo::LoadFromData(context, tasks.back());
  if (InsertHiddenInput(context, aicore_task) != ge::GRAPH_SUCCESS) {
    TILE_FWK_LOGE("Node[%s, %s]: failed to insert hidden input args format.\n",
                  context->GetNodeName(), context->GetNodeType());
    return GRAPH_FAILED;
  }
  tasks.back() = aicore_task.Serialize();
  ge::KernelLaunchInfo aicpu_init_task =
      ge::KernelLaunchInfo::CreateAicpuKfcTask(context, kAicpuSoName.c_str(), kAicpuInitTaskKernelName.c_str());
  aicpu_init_task.SetStreamId(sub_stream_id);
  aicpu_init_task.SetBlockDim(kAicpuInitTaskNum);
  TILE_FWK_LOGD("Node[%s, %s]: aicore args format is %s.\n", context->GetNodeName(), context->GetNodeType(),
                aicore_task.GetArgsFormat());
  aicpu_init_task.SetArgsFormat(aicore_task.GetArgsFormat());
  tasks.emplace_back(aicpu_init_task.Serialize());

  ge::KernelLaunchInfo aicpu_task =
      ge::KernelLaunchInfo::CreateAicpuKfcTask(context, kAicpuSoName.c_str(), kAicpuMainTaskKernelName.c_str());
  aicpu_task.SetStreamId(sub_stream_id);
  aicpu_task.SetBlockDim(kAicpuMainTaskNum);
  aicpu_task.SetArgsFormat(aicore_task.GetArgsFormat());
  tasks.emplace_back(aicpu_task.Serialize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GentaskUtils::CommonGenerateTask(const gert::ExeResGenerationContext *context,
                                                 std::vector<std::vector<uint8_t>> &tasks) {
  TILE_FWK_LOGD("Node[%s, %s]: origin tasks size %zu.\n", context->GetNodeName(), context->GetNodeType(), tasks.size());
  std::vector<gert::StreamInfo> stream_v = context->GetAttachedStreamInfos();
  if (stream_v.empty()) {
    TILE_FWK_LOGE("Node[%s, %s]: failed to get stream info.\n", context->GetNodeName(), context->GetNodeType());
    return ge::GRAPH_FAILED;
  }
  int64_t sub_stream_id = stream_v[0].stream_id;
  auto ret = GentaskUtils::GenerateAicpuTask(context, sub_stream_id, tasks);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }
  TILE_FWK_LOGD("Node[%s, %s]: current tasks size %zu.\n", context->GetNodeName(), context->GetNodeType(),
                tasks.size());
  return ge::GRAPH_SUCCESS;
}
} // namespace ops

