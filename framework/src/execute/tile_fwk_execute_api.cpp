/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/op_desc.h"
#include "graph/buffer.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "register/hidden_inputs_func_registry.h"
#include "tile_fwk_log.h"
#include "aicore_runtime_manager.h"
#include "fatbin_parser.h"

namespace npu::tile_fwk {
ge::graphStatus TileFwkHiddenInputsFunc(const ge::OpDescPtr &op_desc, std::vector<void *> &contexts) {
  if (op_desc == nullptr) {
    TILE_FWK_LOGE("Op desc is null.");
    return ge::GRAPH_FAILED;
  }
  TILE_FWK_LOGD("Begin to parse tile fwk hidden input of op[%s, %s].", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  int64_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  TILE_FWK_LOGD("Block dim of op[%s, %s] is %ld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), block_dim);
  if (block_dim < 0) {
    TILE_FWK_LOGD("Block dim[%ld] of op[%s, %s] is invalid.", block_dim, op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }

  std::vector<int64_t> workspaces = op_desc->GetWorkspaceBytes();
  if (workspaces.empty()) {
    TILE_FWK_LOGE("Workspace bytes of op[%s, %s] is empty.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  uint64_t workspace_size = static_cast<uint64_t>(workspaces[0]);
  TILE_FWK_LOGD("Workspace size of op[%s, %s] is %lu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), workspace_size);

  uint64_t config_key = 0;
  (void)ge::AttrUtils::GetInt(op_desc, "_tile_fwk_op_config_key", config_key);
  TILE_FWK_LOGD("Config key of op[%s, %s] is %lu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), config_key);

  ge::Buffer op_binary_buffer;
  ge::AttrUtils::GetBytes(op_desc, "_subkernel_op_binary", op_binary_buffer);
  if (op_binary_buffer.GetData() == nullptr || op_binary_buffer.GetSize() == 0) {
    TILE_FWK_LOGE("Subkernel binary data of op[%s, %s] is null or empty.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  std::vector<uint8_t> op_bin(op_binary_buffer.GetSize(), 0);
  memcpy_s(op_bin.data(), op_binary_buffer.GetSize(), op_binary_buffer.GetData(), op_binary_buffer.GetSize());
  TILE_FWK_LOGD("Sub op binary size of op[%s, %s] is %zu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_bin.size());

  int64_t *dev_args = AicoreRtManager::Instance().TileFwkHiddenInputWithCache(op_bin, config_key,
      static_cast<uint32_t>(block_dim), workspace_size, op_desc->GetId());
  if (dev_args == nullptr) {
    return ge::GRAPH_FAILED;
  }
  contexts.push_back(reinterpret_cast<void *>(dev_args));
  return ge::GRAPH_SUCCESS;
}
REG_HIDDEN_INPUTS_FUNC(ge::HiddenInputsType::TILEFWK, TileFwkHiddenInputsFunc);

extern "C" int64_t *ParseTileFwkHiddenInput(const std::vector<uint8_t> &op_bin, const uint64_t config_key,
    const uint32_t block_dim, const uint64_t workspace_size) {
    TILE_FWK_LOGD("Begin to parse tile fwk hidden input, bin size[%zu], config key[%lu], block dim[%u], workspace[%lu]",
                  op_bin.size(), config_key, block_dim, workspace_size);
    return AicoreRtManager::Instance().TileFwkHiddenInput(op_bin, config_key, block_dim, workspace_size);
}

extern "C" bool ParseTileFwkFatbin(const std::string &bin_file_path, const uint64_t &config_key, size_t &subkernl_index,
    std::vector<uint8_t> &op_binary_bin, std::vector<uint8_t> &kernel_bin) {
    TILE_FWK_LOGD("Begin to parse tile fwk fat bin of bin file[%s], config key[%lu], subkernl_index[%zu]",
                  bin_file_path.c_str(), config_key, subkernl_index);
    return FatbinParser::ParseFatbin(bin_file_path, config_key, subkernl_index, op_binary_bin, kernel_bin);
}
}
