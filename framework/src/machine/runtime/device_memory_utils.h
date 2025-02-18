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
 * \file device_memory_utils.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN

#include "machine/runtime/runtime.h"
#include "machine/runtime/device_runner.h"
#include "machine/platform/platform_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk::dynamic {
struct DeviceMemoryUtils {
    static bool IsDevice() { return true; }
    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = nullptr;
        if (cachedDevAddrHolder == nullptr) {
            machine::GetRA()->AllocDevAddr(&devPtr, size);
        } else if (*cachedDevAddrHolder == nullptr) {
            machine::GetRA()->AllocDevAddr(&devPtr, size);
            *cachedDevAddrHolder = devPtr;
        } else {
            devPtr = *cachedDevAddrHolder;
        }
        return devPtr;
    }

    uint8_t *AllocZero(uint64_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = AllocDev(size, cachedDevAddrHolder);
        (void)rtMemset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = AllocDev(size, cachedDevAddrHolder);
        rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        return devPtr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedDevAddrHolder) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T), cachedDevAddrHolder);
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        rtMemcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size(), nullptr);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData &data) {
        CopyFromDev(data.data(), data.GetDevPtr(), data.size());
    }

    uint64_t GetL2Offset() {
        return machine::GetRA()->GetL2Offset();
    }
};
}
#endif
