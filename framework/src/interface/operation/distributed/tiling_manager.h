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
 * \file tiling_manager.h
 * \brief
 */

#pragma once
#include <unordered_map>
#include <optional>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cstdint>
#include <algorithm>

namespace npu::tile_fwk {
namespace Distributed {

using TilingData = std::vector<int>;

class TilingStorage {
public:
    explicit TilingStorage(int maxNum) : maxNum_(maxNum), pos_(0), storage_(maxNum, 0) {}
    ~TilingStorage() = default;

    int SaveData(const std::vector<int> &tilingData)
    {
        int offset = GetOffset(tilingData);
        if (offset < static_cast<int>(pos_)) {
            return offset;
        }
        if (pos_ + tilingData.size() > storage_.size()) {
            return -1;
        }
        std::copy(tilingData.begin(), tilingData.end(), storage_.begin() + offset);
        mappedOffset_[tilingData] = pos_;
        pos_ += tilingData.size();
        return offset;
    }

    void* GetStoragePtr()
    {
        return static_cast<void*>(storage_.data());
    }

    const void* GetConstStoragePtr() const
    {
        return static_cast<const void*>(storage_.data());
    }

    size_t GetValidsize() const
    {
        return pos_ * sizeof(int);
    }

    inline std::string PrintString() const
    {
        std::ostringstream oss;
        oss << "total size=" << maxNum_ << ", pos=" << pos_ << ", storage:{";
        for (auto &i : storage_) {
            oss << i << " ";
        }
        oss << "}";
        return oss.str();
    }
private:
    TilingStorage() = delete;

    int GetOffset(const std::vector<int> &vec)
    {
        auto it = mappedOffset_.find(vec);
        if (it != mappedOffset_.end()) {
            return it->second;
        }
        return pos_;
    }
private:
    size_t maxNum_;
    size_t pos_;
    TilingData storage_;
    std::map<std::vector<int>, size_t> mappedOffset_;
};

class TilingManager {
public:
    TilingManager() = default;
    ~TilingManager() {};

    inline int Save(const std::string &tensorSymbol, const std::vector<int> &tilingData)
    {
        auto it = tilingTensor_.find(tensorSymbol);
        if (it == tilingTensor_.end()) {
            return -1;
        }
        return it->second.SaveData(tilingData);
    }

    inline std::optional<std::pair<void*, size_t>> Get(const std::string &tensorSymbol)
    {
        auto it = tilingTensor_.find(tensorSymbol);
        if (it == tilingTensor_.end()) {
            return std::nullopt;
        }
        return std::make_pair(it->second.GetStoragePtr(), it->second.GetValidsize());
    }

    inline std::string CreateTilingStorage(const std::string &suffix, int maxNum)
    {
        static uint64_t cnt = 0;
        const std::string symbol = "DIST_TILING_INFO_" + std::to_string(cnt++) + "_" + suffix;
        if (tilingTensor_.find(symbol) == tilingTensor_.end()) {
            tilingTensor_.insert(std::make_pair(symbol, TilingStorage(maxNum)));
        }
        return symbol;
    }

    inline const std::unordered_map<std::string, TilingStorage>& GetAllTilingTensorData() const
    {
        return tilingTensor_;
    }

    inline std::string PrintString() const
    {
        std::ostringstream oss;
        oss << "distribute tiling info:";
        for (const auto& [symbol, storage] : tilingTensor_) {
            oss << "\"" << symbol << "\": " << storage.PrintString() << ", ";
        }
        return oss.str();
    }
private:
    std::unordered_map<std::string, TilingStorage> tilingTensor_;
};
} // namespace Distributed
} // namespace npu::tile_fwk
