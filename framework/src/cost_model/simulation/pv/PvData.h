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
 * \file PvData.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <cstdint>

namespace CostModel {
class PvData {
private:
    std::unordered_map<void *, std::vector<uint8_t>> data_;
    bool capture_ = false;

public:
    static PvData &Instance() {
        static PvData instance;
        return instance;
    }

    void Put(void *dev, std::vector<uint8_t> &cpu) {
        if (capture_) {
            std::vector<uint8_t> copy(cpu);
            data_[dev] = copy;  
        }
    }

    std::vector<uint8_t> Get(void *dev) {
        if (data_.find(dev) != data_.end()) {
            return data_[dev];
        } else {
            return std::vector<uint8_t>();
        }
    }

    void Enable() { capture_ = true; }

    void Disable() { capture_ = false; }
};
} // namespace CostModel
