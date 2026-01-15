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
 * \file codegen_symbol.cpp
 * \brief
 */

#include "codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
AllocKey SymbolManager::CreateAllocKey(const std::shared_ptr<LogicalTensor> &tensor) const {
    auto memType = tensor->GetMemoryTypeOriginal();
    if (OPERAND_TYPE_TO_MEMORY_TYPE.count(memType) == 0) {
        ASSERT(false) << "invalid memory type: " << static_cast<size_t>(memType) << ", tensor is " << tensor->Dump();
        return {};
    }

    const TileRange &range = tensor->memoryrange;
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);
    AllocKey key = AllocKey(bufferType, range.start, range.end);
    return key;
}

AllocKey SymbolManager::CreateAllocKey(int tensorMagicNum) const {
    std::shared_ptr<LogicalTensor> tensor = SymbolManager::GetTensorByMagic(tensorMagicNum);
    if (!tensor) {
        ALOG_ERROR_F("%s: can not query tensor object from tensor magicnum: %d", __FUNCTION__, tensorMagicNum);
        return {};
    }

    return CreateAllocKey(tensor);
}

bool SymbolManager::BindAddrWithVariableName(
    const AllocKey &key, const std::string &varName, const std::string &varNameT) {
    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return true;
    } else {
        key2VariableName_.insert(std::pair<AllocKey, std::string>(key, varName));
        key2VariableNameTileTensor_.insert(std::pair<AllocKey, std::string>(key, varNameT));
    }
    return false;
}

std::shared_ptr<LogicalTensor> SymbolManager::GetTensorByMagic(int magicNum) const {
    auto iter = tensorMap_.find(magicNum);
    if (iter != tensorMap_.end()) {
        return iter->second;
    } else {
        ASSERT(false) << "can not find tensor by magicNum:" << magicNum;
        return nullptr;
    }
}

std::string SymbolManager::FormatAllocKey(const AllocKey &key) {
    auto [bufType, start, end] = key;
    std::ostringstream os;
    os << "alloc identifier <buf_type=" << OperandTypeToStr(bufType) << ", ";
    os << "range_start=" << start << ", ";
    os << "range_end=" << end << ">";
    return os.str();
}

std::string SymbolManager::QueryVariableName(const AllocKey &key) {
    ALOG_INFO_F("%s: query varname by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());
    auto iter = key2VariableName_.find(key);
    ASSERT(iter != key2VariableName_.end())
        << "QueryVariableName Failed: UNDEFINED_VAR !!! AllocKey: " << FormatAllocKey(key);
    return iter->second;
}

std::string SymbolManager::QueryVariableNameTileTensor(const AllocKey &key) {
    ALOG_INFO_F("%s: query varname TileTensor mode by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());

    auto iter = key2VariableNameTileTensor_.find(key);
    if (iter != key2VariableNameTileTensor_.end()) {
        return iter->second;
    }

    ALOG_ERROR_F("%s: failed to query by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());
    ASSERT(false) << "QueryVariableNameTileTensor Failed: UNDEFINED_VAR !!! AllocKey: " << FormatAllocKey(key);
    return "UNDEFINED_VAR";
}

// NEXTNEXT: after TileTensor Mode is applied to all tensor, just retain TileTensor Mode
std::string SymbolManager::QueryVarNameByTensorMagic(int magic, bool isTileTensor) {
    ALOG_INFO_F("QueryVarNameByTensorMagic: magic is %d", magic);
    AllocKey key = CreateAllocKey(magic);
    std::string varName = isTileTensor ? QueryVariableNameTileTensor(key) : QueryVariableName(key);
    return varName;
}

std::string SymbolManager::FindUsingName(const TileTensorUsing &tileTensorUsing) const {
    for (const auto &usingPair : tileTensorUsing_) {
        if (usingPair.second == tileTensorUsing) {
            return usingPair.first;
        }
    }
    return "";
}

std::string SymbolManager::AddTileTensorUsing(const TileTensorUsing &tileTensorUsing) {
    std::string tensorUsingType = FindUsingName(tileTensorUsing);
    if (!tensorUsingType.empty()) {
        ALOG_INFO_F("found tensorUsingType %s", tensorUsingType.c_str());
        return tensorUsingType;
    }
    tensorUsingType = tileTensorUsing.GenName();
    tileTensorUsing_.insert({tensorUsingType, tileTensorUsing});
    ALOG_INFO_F("insert tensorUsingType %s = %s", tensorUsingType.c_str(), tileTensorUsing.ToString().c_str());
    return tensorUsingType;
}

void SymbolManager::AddTileTensor(const TileTensor &tileTensor) {
    auto result = tileTensor_.insert({tileTensor, tileTensor.tensorName});
    std::string tensorName = result.second ? tileTensor.tensorName : result.first->second;
    if (result.second) {
        tileTensorByMagic_.insert({tileTensor.magic, tileTensor});
    } else {
        tileTensorByMagic_.insert({tileTensor.magic, result.first->first});
    }

    ALOG_INFO_F("tileTensor_.insert result is %d Add TileTensor --> tensor magic: %d, tensor name: %s, tile tensor: %s",
        result.second, tileTensor.magic, tensorName.c_str(), tileTensor.ToString().c_str());
}

std::vector<TileTensor> SymbolManager::QueryTileTensorByMagic(int magic) {
    ALOG_INFO_F("QueryTileTensorByMagic magic is %d", magic);
    std::vector<TileTensor> res;
    auto [start, end] = tileTensorByMagic_.equal_range(magic);
    for (auto it = start; it != end; ++it) {
        res.emplace_back(it->second); // 或 emplace_back，性能更优
    }

    ASSERT(!res.empty()) << "tensor magic " << magic << " is not found !!! ";
    return res;
}

std::string SymbolManager::QueryTileTensorByBufVarName(const std::string &bufVarName) {
    for (const auto &tileTensorPair : tileTensor_) {
        const TileTensor &tileTensor = tileTensorPair.first;
        if (tileTensor.bufVar == bufVarName) {
            return tileTensor.tensorName;
        }
    }

    ASSERT(false) << "bufVarName " << bufVarName << " is not found !!! ";
    return "";
}

std::string SymbolManager::GenUsingList() {
    std::ostringstream oss;
    for (const auto &usingPair : tileTensorUsing_) {
        const std::string &usingName = usingPair.first;
        const TileTensorUsing &tileTensorUsing = usingPair.second;
        oss << "using " << usingName << " = " << tileTensorUsing.ToString();
    }
    return oss.str();
}

std::string SymbolManager::GenTileTensorDefList() {
    std::ostringstream oss;
    for (const auto &tensorPair : tileTensor_) {
        const TileTensor &tileTensor = tensorPair.first;
        oss << tileTensor.ToString();
    }
    return oss.str();
}

} // namespace npu::tile_fwk
