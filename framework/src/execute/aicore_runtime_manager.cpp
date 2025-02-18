/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aicore_runtime_manager.h"
#include <dlfcn.h>
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#include "driver/ascend_hal_define.h"
#include "tile_fwk_log.h"

namespace npu::tile_fwk {
namespace {
const uint8_t AICORE_MAP_BUFF_LEN = 2;
const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint64_t SHARE_BUFFER_SIZE = 512;
const uint64_t AICPU_COUNT = 5;
const uint64_t SCHE_AICPU_COUNT = 3;
const uint64_t DEV_ARGS_SIZE = 4096;
const uint64_t DEVICE_TASK_CTRL_SIZE = 7168;
const uint64_t DEVICE_QUEUE_SIZE = 2048 * 3;
const uint64_t DEVICE_SHM_SIZE = DEV_ARGS_SIZE + DEVICE_TASK_CTRL_SIZE + DEVICE_QUEUE_SIZE;

bool GetPgmsk(const int32_t deviceId, uint64_t &valid) {
  uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
  int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
  auto halFuncDevInfo = (int (*)(uint32_t deviceId, int32_t moduleType, int32_t infoType,
                         void* buf, int32_t *size))dlsym(nullptr, "halGetDeviceInfoByBuff");
  if (halFuncDevInfo == nullptr) {
    TILE_FWK_LOGE("Failed to find halGetDeviceInfoByBuff function.\n");
    return false;
  }
  auto ret = halFuncDevInfo(static_cast<uint32_t>(deviceId), MODULE_TYPE_AI_CORE, INFO_TYPE_OCCUPY,
                            reinterpret_cast<void *>(&aicore_bitmap[0]), &size_n);
  if (ret != 0) {
    return false;
  }
  valid = aicore_bitmap[0];
  return true;
}
}
AicoreRtManager::AicoreRtManager() {}

AicoreRtManager::~AicoreRtManager() {
    TILE_FWK_LOGD("DeInit with mem size %zu.", allocated_addrs_.size());
    BatchFreeDevAddr(allocated_addrs_);
    cache_hidden_input_map_.clear();
}

bool AicoreRtManager::AllocDevAddr(void **dev_addr, size_t size, std::vector<void *> &allocated_addrs) {
  TILE_FWK_LOGD("Alloc size is %zu.", size);
  int res = rtMalloc(dev_addr, size, RT_MEMORY_HBM, 0);
  if (res != 0) {
    TILE_FWK_LOGE("Failed to alloc mem with size %zu.");
    return false;
  }
  allocated_addrs.emplace_back(*dev_addr);
  return true;
}

void AicoreRtManager::BatchFreeDevAddr(std::vector<void *> &allocated_addrs) {
  if (allocated_addrs.empty()) {
    return;
  }
  for (void* addr : allocated_addrs) {
    if (addr != nullptr) {
      rtFree(addr);
    }
  }
  allocated_addrs.clear();
}

void AicoreRtManager::SaveAllocatedAddrs(const std::vector<void *> &allocated_addrs) {
  allocated_addrs_.insert(allocated_addrs_.end(), allocated_addrs.begin(), allocated_addrs.end());
}

void AicoreRtManager::AddHiddenInputCache(const int64_t &cache_id, int64_t *hidden_input) {
  cache_hidden_input_map_[cache_id] = hidden_input;
}

int64_t* AicoreRtManager::GetHiddenInputCache(const int64_t &cache_id) const {
  auto iter = cache_hidden_input_map_.find(cache_id);
  if (iter == cache_hidden_input_map_.end()) {
    return nullptr;
  }
  TILE_FWK_LOGD("Cache %ld hit hidden input.", cache_id);
  return iter->second;
}

bool AicoreRtManager::GetAicoreRegInfo(const int32_t device_id, std::vector<int64_t> &aic, std::vector<int64_t> &aiv) {
  int nrCore = 25;
  int nrSubCore = 3;
  uint64_t valid = 0;
  if (!GetPgmsk(device_id, valid)) {
      TILE_FWK_LOGE("Failed to get device info or no valid core exists.");
      return false;
  }
  TILE_FWK_LOGD("The valid cores are %ld", valid);
  uint64_t coreStride = 8 * 1024 * 1024; // 8M
  uint64_t subCoreStride = 0x100000ULL;  
  auto isValid = [&valid](int id) {
      const uint64_t mask = (1ULL << 25) - 1;
      return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
  };
  auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
      size_t *outSizeRet))dlsym(nullptr, "halMemCtl");
  if (halFunc == nullptr) {
    TILE_FWK_LOGE("Failed to find halMemCtlSpeical function.");
    return false;
  }  
  struct AddrMapInPara inMapPara;
  struct AddrMapOutPara outMapPara;
  inMapPara.devid = device_id;
  inMapPara.addr_type = ADDR_MAP_TYPE_REG_AIC_CTRL;
  auto ret = halFunc(0, reinterpret_cast<void *>(&inMapPara), sizeof(struct AddrMapInPara),
      reinterpret_cast<void *>(&outMapPara), nullptr);
  if (ret != 0) {
    TILE_FWK_LOGE("CTRL_TYPE_ADDR_MAP fail. (ret=%d).", ret);
    return false;
  }
  for (int i = 0; i < nrCore; i++) {
      for (int j = 0; j < nrSubCore; j++) {
          uint64_t vaddr = 0UL;
          if (isValid(i)) {
              vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
          }
          if (j == 0) {
              aic.push_back(vaddr);
          } else {
              aiv.push_back(vaddr);
          }
      }
  }
  return true;
}

bool AicoreRtManager::InitDyBinData(const std::vector<int64_t> &aic, const std::vector<int64_t> &aiv,
                                    DevAscendProgram *host_args, std::vector<void *> &allocated_addrs) {
  host_args->devArgs.nrAic = aic.size();
  host_args->devArgs.nrAiv = aiv.size();
  std::vector<int64_t> regs;
  regs.insert(regs.end(), aic.begin(), aic.end());
  regs.insert(regs.end(), aiv.begin(), aiv.end());
  size_t shared_size = regs.size() * SHARE_BUFFER_SIZE;
  if (!AllocDevAddr((void**)&host_args->devArgs.sharedBuffer, shared_size, allocated_addrs)) {
    TILE_FWK_LOGE("Failed to alloc shared buffer.");
    return false;
  }
  if (rtMemset((void*)host_args->devArgs.sharedBuffer, shared_size, 0U, shared_size) != RT_ERROR_NONE) {
    TILE_FWK_LOGE("Failed to copy shared buffer to device.");
    return false;
  }
  uint64_t meta_addr = 0;
  if (!AllocDevAddr((void**)&meta_addr, DEVICE_SHM_SIZE, allocated_addrs)) {
    TILE_FWK_LOGE("Failed to alloc meta addr.");
    return false;
  }
  TILE_FWK_LOGD("Alloc meta size:%lu.", DEVICE_SHM_SIZE);
  host_args->devArgs.startArgsAddr = meta_addr;
  host_args->devArgs.taskCtrl = meta_addr + DEV_ARGS_SIZE;
  host_args->devArgs.taskQueue = meta_addr + DEV_ARGS_SIZE + DEVICE_TASK_CTRL_SIZE;
  host_args->devArgs.enableCtrl = 1;
  host_args->devArgs.scheCpuNum = SCHE_AICPU_COUNT;
  size_t core_reg_size = regs.size() * sizeof(uint64_t);
  if (!AllocDevAddr((void**)&host_args->devArgs.coreRegAddr, core_reg_size, allocated_addrs)) {
    TILE_FWK_LOGE("Failed to alloc core reg addr.");
    return false;
  }
  if (rtMemcpy((void*)host_args->devArgs.coreRegAddr, core_reg_size, regs.data(), core_reg_size, 
      RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
    TILE_FWK_LOGE("Failed to copy core reg addr to device.");
    return false;
  }
  TILE_FWK_LOGD("DevAscendProgram: aic %lu, aiv %lu, block dim %lu, sharedBuffer %lx, coreRegAddr %lx, workspace %lu.",
               host_args->devArgs.nrAic, host_args->devArgs.nrAiv, host_args->devArgs.nrValidAic,
               host_args->devArgs.sharedBuffer, host_args->devArgs.coreRegAddr, host_args->workspaceSize);
  return true;
}

int64_t* AicoreRtManager::TileFwkHiddenInput(const std::vector<uint8_t> &op_bin, const uint64_t config_key,
                                             const uint32_t block_dim, const uint64_t workspace_size) {
  std::vector<uint8_t> op_bin_copy = op_bin;
  DevAscendProgram *host_args = reinterpret_cast<DevAscendProgram*>(op_bin_copy.data());
  host_args->configKey = config_key;
  host_args->workspaceSize = workspace_size;
  host_args->devArgs.nrValidAic = block_dim;
  host_args->devArgs.nrAicpu = AICPU_COUNT;
  host_args->devArgs.taskType = DEVICE_TASK_TYPE_DYN;

  int32_t device_id = 0;
  int32_t user_device_id = 0;
  rtGetDevice(&user_device_id);
  if (rtGetLogicDevIdByUserDevId(user_device_id, &device_id) != RT_ERROR_NONE) {
    TILE_FWK_LOGE("Failed to trans usrDevId[%d] to logicDevId.", user_device_id);
    return nullptr;
  }
  (void)rtGetL2CacheOffset(device_id, &host_args->l2CacheOffset);
  TILE_FWK_LOGD("L2 cache offset of device id [%d] is [%lu].", device_id, host_args->l2CacheOffset);

  std::vector<int64_t> aic;
  std::vector<int64_t> aiv;
  if (!GetAicoreRegInfo(device_id, aic, aiv)) {
    TILE_FWK_LOGE("Failed to get aicore reg info.");
    return nullptr;
  }
  TILE_FWK_LOGD("After get aicore reg info, size of aic and aiv is [%zu] and [%zu].", aic.size(), aiv.size());

  std::vector<void *> allocated_addrs;
  if (!InitDyBinData(aic, aiv, host_args, allocated_addrs)) {
    TILE_FWK_LOGE("Failed to init bin data.");
    BatchFreeDevAddr(allocated_addrs);
    return nullptr;
  }

  int64_t *dev_args = nullptr;
  if (!AllocDevAddr((void**)&dev_args, op_bin_copy.size(), allocated_addrs)) {
    TILE_FWK_LOGE("Failed to alloc dev args.");
    BatchFreeDevAddr(allocated_addrs);
    return nullptr;
  }
  if (rtMemcpy(dev_args, op_bin_copy.size(), (void*)op_bin_copy.data(), op_bin_copy.size(), RT_MEMCPY_HOST_TO_DEVICE) !=
      RT_ERROR_NONE) {
    TILE_FWK_LOGE("Failed to copy bin data to device.");
    BatchFreeDevAddr(allocated_addrs);
    return nullptr;
  }
  SaveAllocatedAddrs(allocated_addrs);
  return dev_args;
}

int64_t* AicoreRtManager::TileFwkHiddenInputWithCache(const std::vector<uint8_t> &op_bin, const uint64_t config_key,
    const uint32_t block_dim, const uint64_t workspace_size, const int64_t cache_id) {
  int64_t *dev_args = GetHiddenInputCache(cache_id);
  if (dev_args != nullptr) {
    return dev_args;
  }
  dev_args = TileFwkHiddenInput(op_bin, config_key, block_dim, workspace_size);
  if (dev_args != nullptr) {
    AddHiddenInputCache(cache_id, dev_args);
  }
  return dev_args;
}
} // namespace npu::tile_fwk
