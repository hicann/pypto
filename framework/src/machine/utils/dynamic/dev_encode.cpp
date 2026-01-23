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
 * \file dev_encode.cpp
 * \brief
 */
#include "tilefwk/platform.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_workspace.h"

#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

#include "ir/function.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <queue>

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
namespace dynamic {
#define ONFILLCONTENT if (fillContent)
#ifndef PAGE_SIZE
#define PAGE_SIZE       4096
#endif

constexpr int32_t CALLOP_ARG_ATTR_BASE_INDEX = 1;
constexpr int32_t MINI_TILE_LIST_SIZE_THRESHOLD = 16;
constexpr int32_t MAX_AICORE_NUM_2210 = 75;
constexpr int32_t MAX_AICORE_NUM_3510 = 108;
constexpr int32_t SLOTS_NEED_ALLOC_SIZE = 2;
constexpr int64_t MAX_SHAPE_WARN_THRESHOLE = 512 * 512;
constexpr int32_t ALLOC_NUM_ONE_SLAB = 4;
constexpr int64_t DEFAULT_CACHE_DEVICE_TASK_NUM = 10000;
constexpr int32_t MAX_CELLMATCHSSTRIDE = 20000000;
static constexpr uint64_t GENERAL_METADATA_SIZE_MIN = 4 * MEBI;
constexpr uint32_t FRIENDLY_CACHE_ALIGN_U64_SIZE = 2; // 友好的cache对齐是2个u64

void DevAscendFunction::InitIncastOutcastAttr(
        uintdevptr_t &initOffset,
        const std::vector<std::shared_ptr<LogicalTensor>> &iList,
        const std::vector<std::shared_ptr<LogicalTensor>> &oList, bool /* fillContent */) {
    incastAddressList.HostInitDataSizeOffset(initOffset, iList.size());
    outcastAddressList.HostInitDataSizeOffset(initOffset, oList.size());
}

void DevAscendFunction::InitOperationDynamicField(
        uintdevptr_t &initOffset,
        DevAscendFunctionPredInfo predInfo,
        uint32_t outcastStitchCount,
        [[maybe_unused]]const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
        const SymbolicExpressionTable *expressionTable,
        const OrderedSet<Operation *> &callList,
        const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const std::unordered_map<Operation *, OrderedSet<Operation *>> & /* callOpSuccDict */, bool fillContent) {
    expressionList.HostInitDataSizeOffset(initOffset, expressionTable->GetPrimaryExpressionSize());

    uint64_t operationSize = callList.size();
    uint64_t incastSize = incastTensorList.size();
    uint64_t outcastSize = outcastTensorList.size();
    uint64_t expressionSize = expressionTable->GetPrimaryExpressionSize();

    uint64_t predCountListDataSize = ALIGN_UP(operationSize * sizeof(predcount_t), sizeof(uint64_t));
    uint64_t incastDataSize = ALIGN_UP(incastSize * sizeof(void *), sizeof(uint64_t));
    uint64_t outcastDataSize = ALIGN_UP(outcastSize * sizeof(void *), sizeof(uint64_t));
    uint64_t expressionDataSize = ALIGN_UP(expressionSize * sizeof(uint64_t), sizeof(uint64_t));
    uint64_t stitchDataSize = ALIGN_UP(outcastStitchCount * sizeof(DevAscendFunctionDuppedStitchList), sizeof(uint64_t));
    uint64_t totalDataSize = predCountListDataSize + incastDataSize + outcastDataSize + expressionDataSize + stitchDataSize;
    duppedDataAllocSize_ = sizeof(DevAscendFunctionDuppedData) + totalDataSize;
    duppedDataCopySize_ = sizeof(DevAscendFunctionDuppedData) + predCountListDataSize;
    duppedData_.HostInitDataSizeOffset(initOffset, duppedDataAllocSize_);
    predInfo_ = predInfo;
    ALOG_INFO("Pred: zero=", predInfo.totalZeroPred, " aiv=", predInfo.totalZeroPredAIV,
        " aic=", predInfo.totalZeroPredAIC, " hub=", predInfo.totalZeroPredHub, " aicpu=", predInfo.totalZeroPredAicpu);

    ONFILLCONTENT {
        DevAscendFunctionDuppedData *dupData = reinterpret_cast<DevAscendFunctionDuppedData *>(&At(duppedData_, 0));
        dupData->operationList_.size = operationSize;

        uint64_t offset = 0;
        dupData->operationList_.predCountBase = offset;
        offset += predCountListDataSize;

        dupData->incastList_.size = incastSize;
        dupData->incastList_.base = offset;
        offset += incastDataSize;

        dupData->outcastList_.size = outcastSize;
        dupData->outcastList_.base = offset;
        offset += outcastDataSize;

        dupData->expressionList_.size = expressionSize;
        dupData->expressionList_.base = offset;
        offset += expressionDataSize;

        dupData->operationList_.stitchBase = offset;
        dupData->operationList_.stitchCount = outcastStitchCount;
        offset += stitchDataSize;
        ASSERT(offset == totalDataSize) << "Offset mismatch:offset " << offset << " != totalDataSize " << totalDataSize;

        memset_s(dupData->data_, totalDataSize, 0, totalDataSize);

        uint8_t *dataBegin = &dupData->data_[0];
        uint8_t *incastBegin = &dupData->data_[dupData->incastList_.base];
        uint8_t *outcastBegin = &dupData->data_[dupData->outcastList_.base];
        uint8_t *expressionBegin = &dupData->data_[dupData->expressionList_.base];
        uint8_t *stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];
        uint8_t *dataEnd = &dupData->data_[totalDataSize];
        uint8_t *dataEndAlloc = &dupData->data_[duppedDataAllocSize_ - sizeof(DevAscendFunctionDuppedData)];
        ASSERT(dataEnd == dataEndAlloc) << "Pointer mismatch:dataEnd " << dataEnd << " != dataEndAlloc " << dataEndAlloc;
        for (uint64_t i = 0; i < operationSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOperationCurrPredCount(i));
            ASSERT(dataBegin <= ptr && ptr < incastBegin) << "OperationCurrPredCount out of range:  ptr "
                   << ptr << " not in [" << dataBegin << ", " << incastBegin << ")";
        }
        for (uint64_t i = 0; i < incastSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetIncastAddress(i));
            ASSERT(incastBegin <= ptr && ptr < outcastBegin) "Incast address out of range:  ptr "
                   << ptr << " not in [" << incastBegin << ", " << outcastBegin << ")";
        }
        for (uint64_t i = 0; i < outcastSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOutcastAddress(i));
            ASSERT(outcastBegin <= ptr && ptr < expressionBegin) << "Outcast address out of range:  ptr "
                   << ptr << " not in [" << outcastBegin << ", " << expressionBegin << ")";
        }
        for (uint64_t i = 0; i < expressionSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetExpression(i));
            ASSERT(expressionBegin <= ptr && ptr < stitchBegin) << "Expression address out of range:  ptr "
                   << ptr << " not in [" << expressionBegin << ", " << stitchBegin << ")";
        }
    };
}

void HandleActualRaw(const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
    const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
    const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
    const std::shared_ptr<RawTensor> &rawTensor,
    DevAscendRawTensor &encoded) {
    auto iter = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
    if (iter != rawMagicToRawTensor.end()) {
        if (iter->second->addrOffset == UINT64_MAX) {
            ALOG_ERROR_F("addrOffset is invalid actual raw magic %d, original raw magic %d",
                rawTensor->actualRawmagic, rawTensor->rawmagic);
            encoded.addrOffset = 0;
        } else {
            encoded.addrOffset = iter->second->addrOffset;
            if (outcastRawList.count(iter->second)) {
                encoded.ioIndex = outcastRawList.GetIndex(iter->second);
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
            } else if (incastRawList.count(iter->second)) {
                encoded.ioIndex = incastRawList.GetIndex(iter->second);
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
            } else {
                encoded.ioIndex = -1;
                encoded.ioProperty = DevIOProperty::NONE;
            }
            ALOG_DEBUG_F("Tensor %d use tensor %d's addr io index %d", rawTensor->rawmagic, rawTensor->actualRawmagic,
                encoded.ioIndex);
        }
    }
}

void DevAscendFunction::UpdateRawTensorDesc(const std::shared_ptr<RawTensor> &rawTensor, size_t i, size_t incastRawListSize,
    DevAscendRawTensor &encoded) {
    if (rawTensor->actualRawmagic != -1) {
        ALOG_DEBUG_F("[%3zu] raw %d, actualRaw %d, IOType <%s>, addrOffset 0x%lx, ioIndex %d.",
                i, rawTensor->rawmagic, rawTensor->actualRawmagic, DevIOProperty2String(encoded.ioProperty).c_str(),
                encoded.addrOffset, encoded.ioIndex);
    }
    uint32_t location = 0;
    uint32_t offsetOrIndex = 0;
    switch (encoded.ioProperty) {
    case DevIOProperty::ROOT_INCAST:
        location = RAW_TENSOR_LOCATION_INCAST;
        offsetOrIndex = encoded.ioIndex;
        break;
    case DevIOProperty::ROOT_OUTCAST:
        location = RAW_TENSOR_LOCATION_OUTCAST;
        offsetOrIndex = encoded.ioIndex + incastRawListSize;
        break;
    case DevIOProperty::NONE:
        location = RAW_TENSOR_LOCATION_LOCAL;
        offsetOrIndex = encoded.addrOffset;
        break;
    default:
        ASSERT(false) << "Unexpected ioProperty value: " << static_cast<int>(encoded.ioProperty);
        break;
    }
    DevRawTensorDesc *desc = GetRawTensorDesc(i);
    desc->location = location;
    desc->offsetOrIndex = offsetOrIndex;
}

static int64_t GetShapeSizeSafe(const std::vector<int64_t> &shape) {
    int64_t nelm = shape.empty() ? 0 : 1;
    for (auto x : shape) {
        if (x == -1) {
            return  0;
        }
        nelm *= x;
    }
    return nelm;
}

static void EncodeRawShape(const SymbolicExpressionTable *expressionTable, DevAscendRawTensor *encoded,
        std::shared_ptr<RawTensor> rawTensor, bool needIndependentlyAlloc, const std::string rootName = "") {
    std::vector<SymInt> shape;
    bool isDyn = false;
    for (auto x : rawTensor->GetDynRawShape()) {
        if (x.IsImmediate()) {
            shape.emplace_back(x.Concrete());
        } else {
            shape.emplace_back(true, expressionTable->LookupPrimaryExpressionIndex(x));
            isDyn = true;
        }
    }
    encoded->shape.SetShape(shape);
    encoded->dataType = rawTensor->GetDataType();
    encoded->memoryRequirement = isDyn ? 0 : AlignUp(rawTensor->GetRawDataSize(), TENSOR_ADDR_ALIGNMENT);

    if (!needIndependentlyAlloc) {
        encoded->maxStaticMemReq = 0;
        return;
    }

    int64_t nelm = std::max(GetShapeSizeSafe(rawTensor->oriRawshape), GetShapeSizeSafe(rawTensor->rawshape));
    encoded->maxStaticMemReq = AlignUp(nelm * BytesOf(rawTensor->GetDataType()), TENSOR_ADDR_ALIGNMENT);
    if (nelm > MAX_SHAPE_WARN_THRESHOLE) {
        ALOG_WARN_F("Root=[%s], symbol=[%s]: staticMemReq=[%lu] is too larger, which might indicate an error",
            rootName.c_str(), rawTensor->symbol.c_str(), encoded->maxStaticMemReq);
    }
}

static bool ShouldDropBudget(const OrderedSet<std::shared_ptr<RawTensor>>& outcastRawList,
        const IncastOutcastSlot* slotInfo,
        const std::vector<npu::tile_fwk::RuntimeSlotKindSet>& runtimeSlotKindSetList,
        const std::shared_ptr<RawTensor> rawTensor, const DevAscendRawTensor& encoded) {
    if (!outcastRawList.count(rawTensor)) {
        return false;
    }

    for (int slotIdx : slotInfo->outcastSlot[encoded.ioIndex]) {
        if (runtimeSlotKindSetList[slotIdx].Count(RuntimeSlotKind::ADDRESS_EXPRESSION)) {
            return true;
        }
    }
    return false;
}

static bool HasInputOutputOrAssembleDst(
        const std::vector<int>& outcastSlots,
        const std::vector<npu::tile_fwk::RuntimeSlotKindSet>& runtimeSlotKindSetList) {
    for (int slotIdx : outcastSlots) {
        if (runtimeSlotKindSetList[slotIdx].Count(RuntimeSlotKind::INPUT) ||
            runtimeSlotKindSetList[slotIdx].Count(RuntimeSlotKind::OUTPUT) ||
            runtimeSlotKindSetList[slotIdx].Count(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            return true;
        }
    }
    return false;
}

void DevAscendFunction::InitRawTensorAndMemoryRequirement(
        uintdevptr_t &initOffset,
        const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
        const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
        const std::vector<EncodeRawTensorAttr> &rawAttrs,
        const EncodeDevAscendFunctionParam &param,
        const SymbolicExpressionTable *expressionTable,
        bool fillContent) {
    auto inoutLink = param.inoutLink;
    auto slot = param.slot;
    rawTensorList_.HostInitDataSizeOffset(initOffset, rawList.size());
    rawTensorDescList_.HostInitDataSizeOffset(initOffset, rawList.size());

    ONFILLCONTENT {
        const std::vector<RuntimeSlotKindSet> &runtimeSlotKindSetList = inoutLink->runtimeSlotKindSetList;
        ALOG_DEBUG_F("incast raw size %zu, outcast raw size %zu, rawlist size %zu", incastRawList.size(), outcastRawList.size(),
            rawList.size());
        rootInnerTensorWsMemoryRequirement = 0;
        exclusiveOutcastWsMemoryRequirement = 0;
        for (size_t idx = 0; idx < rawList.size(); idx++) {
            const auto &rawTensor = rawList[idx];
            auto &encoded = *GetRawTensor(idx);
            // shmem need to drop budget
            bool dropBudget = ShouldDropBudget(outcastRawList, slot, runtimeSlotKindSetList, rawTensor, encoded);
            // inplace (basically reshape) need to drop budget
            bool isInplace = param.devRoot->outIncastLinkMap.count(rawTensor);
            isInplace |= rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic;
            EncodeRawShape(expressionTable, &encoded, rawTensor, !dropBudget && !isInplace,
                param.devRoot->GetRawName());
        }
        for (size_t idx = 0; idx < rawList.size(); idx++) {
            const auto &rawTensor = rawList[idx];
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                continue;
            }
            auto &encoded = *GetRawTensor(idx);
            encoded.rawMagic = rawTensor->GetRawMagic();
            if (incastRawList.count(rawTensor)) {
                // No need to allocate memory for root incasts
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
                encoded.ioIndex = incastRawList.GetIndex(rawTensor);
                rawTensor->addrOffset = 0;
            } else if (outcastRawList.count(rawTensor)) {
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
                encoded.ioIndex = outcastRawList.GetIndex(rawTensor);
                rawTensor->addrOffset = 0;
                if (!HasInputOutputOrAssembleDst(slot->outcastSlot[encoded.ioIndex], runtimeSlotKindSetList)) {
                    encoded.addrOffset = exclusiveOutcastWsMemoryRequirement;
                    rawTensor->addrOffset = exclusiveOutcastWsMemoryRequirement;
                    exclusiveOutcastWsMemoryRequirement += encoded.maxStaticMemReq;
                }
            } else {
                // For workspace tensors, the memoryRequirement property is deprecated, please don't use its value
                encoded.ioProperty = DevIOProperty::NONE;
                encoded.ioIndex = -1;
#if DEBUG_INFINITE_LIFETIME
                UNUSED(rawAttrs);
                encoded.addrOffset = rootInnerTensorWsMemoryRequirement;
                rawTensor->addrOffset = encoded.addrOffset;
                rootInnerTensorWsMemoryRequirement += encoded.maxStaticMemReq;
#else
                encoded.addrOffset = rawAttrs[idx].storage->start_ + rawAttrs[idx].storageOffset;
                rawTensor->addrOffset = encoded.addrOffset;
                rootInnerTensorWsMemoryRequirement = std::max(rootInnerTensorWsMemoryRequirement,
                    rawAttrs[idx].storage->start_ + rawAttrs[idx].storage->length_);
#endif
            }
            UpdateRawTensorDesc(rawTensor, idx, incastRawList.size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto &rawTensor = rawList[i];
            if (rawTensor->actualRawmagic == -1 || rawTensor->actualRawmagic == rawTensor->rawmagic) {
                continue;
            }
            auto &encoded = *GetRawTensor(i);
            HandleActualRaw(incastRawList, outcastRawList, rawMagicToRawTensor, rawTensor, encoded);
            UpdateRawTensorDesc(rawTensor, i, incastRawList.size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto &rawTensor = rawList[i];
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                auto it = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
                ASSERT(it != rawMagicToRawTensor.end()) << "rawMagic is not found in rawMagicToRawTensor: " <<
                       rawTensor->actualRawmagic;
                auto &actualRaw = it->second;
                const auto &rawTensorRawShape = rawTensor->GetRawShape();
                bool isDynamicShape = std::find_if(rawTensorRawShape.begin(), rawTensorRawShape.end(),
                    [](int64_t dimShape) { return dimShape < 0; }) != rawTensorRawShape.end();
                if (isDynamicShape) continue;

                auto fromType = rawTensor->datatype;
                auto toType = actualRaw->datatype;
                if (fromType != toType) {
                    int inSize = BytesOf(fromType);
                    int outSize = BytesOf(toType);
                    ASSERT(inSize != 0 && outSize != 0) << "Detected zero byte size data type, fromType: "
                           << static_cast<int>(fromType) << ", toType: " << static_cast<int>(toType);
                    if (inSize > outSize) {
                        ASSERT((rawTensor->GetRawShapeSize() * (inSize / outSize)) == actualRaw->GetRawShapeSize())
                               << "Shape size mismatch: expected " << rawTensor->GetRawShapeSize() * (inSize / outSize)
                               << ", got: " << actualRaw->GetRawShapeSize();
                    } else {
                        ASSERT(rawTensor->GetRawShapeSize() == (actualRaw->GetRawShapeSize() * (outSize / inSize)))
                               << "Shape size mismatch: expected " << actualRaw->GetRawShapeSize() * (outSize / inSize)
                               << ", got " << rawTensor->GetRawShapeSize();
                    }
                    ASSERT(rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize()) << "Data size mismatch:"
                           << rawTensor->GetRawDataSize() << "!=" << actualRaw->GetRawDataSize();
                    continue;
                }
                ASSERT(rawTensor->GetRawShapeSize() == actualRaw->GetRawShapeSize()) << "Shape size mismatch:"
                       << rawTensor->GetRawShapeSize() << "!=" << actualRaw->GetRawShapeSize();
                ASSERT(rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize()) << "Data size mismatch:"
                       << rawTensor->GetRawDataSize() << "!=" << actualRaw->GetRawDataSize();
            }
        }

        // file linkedIncastId
        auto outIncastLinkMap = param.devRoot->outIncastLinkMap;
        ALOG_DEBUG_F("devRoot is %s", param.devRoot->GetRawName().c_str());
        for (size_t i = 0; i < rawList.size(); i++) {
            auto &encoded = *GetRawTensor(i);
            if (outIncastLinkMap.find(rawList[i]) != outIncastLinkMap.end()) {
                ASSERT(outIncastLinkMap[rawList[i]]->actualRawmagic != rawList[i]->rawmagic) << "Unexpected rawmagic match: actualRawmagic "
                       << outIncastLinkMap[rawList[i]]->actualRawmagic <<
                       " == rawmagic " << rawList[i]->rawmagic;
                auto replacedIncast = outIncastLinkMap[rawList[i]];
                if (std::find(rawList.begin(), rawList.end(), replacedIncast) != rawList.end()) {
                    encoded.linkedIncastId = incastRawList.GetIndex(replacedIncast); //换成incast的下标 ioidx
                    ALOG_DEBUG_F("linkedIncastId is %d", encoded.linkedIncastId);
                } else {
                    encoded.linkedIncastId = -1;
                }
            } else {
                encoded.linkedIncastId = -1;
                ALOG_DEBUG_F("linkedIncastId is %d", encoded.linkedIncastId);
            }
        }
    }; // ONFILLCONTENT
}

void DevAscendFunction::InitTensor(
        uintdevptr_t &initOffset,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList, bool fillContent) {
    tensorList_.HostInitDataSizeOffset(initOffset, tlist.size());
    for (size_t i = 0; i < tlist.size(); i++) {
        ONFILLCONTENT {
            GetTensor(i)->rawIndex = rawList.find(tlist[i]->tensor)->second;
        };
    }
}

void DevAscendFunction::InitOperation(
        uintdevptr_t &initOffset,
        const SymbolicExpressionTable *expressionTable,
        const OrderedSet<Operation *> &callList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
        const std::unordered_map<Operation *, uint64_t> &callOpPredDict,
        const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,
        const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
        const std::vector<int32_t> &outcastStitchIndexList,
        const std::vector<int> &noPredOpList,
        const std::vector<int> &noSuccOpList,
        const std::unordered_map<Operation *, std::vector<int>> &copyOutResolveSuccIndexListDict,
        bool fillContent) {
    noPredOpList_.HostInitDataSizeOffset(initOffset, noPredOpList.size());
    noSuccOpList_.HostInitDataSizeOffset(initOffset, noSuccOpList.size());

    ONFILLCONTENT {
        memcpy_s(&At(noPredOpList_, 0), noPredOpList_.ByteSize(),
            noPredOpList.data(), noPredOpList.size() * sizeof(int));
        memcpy_s(&At(noSuccOpList_, 0), noSuccOpList_.ByteSize(),
            noSuccOpList.data(), noSuccOpList.size() * sizeof(int));

        ASSERT(noPredOpList_.size() == noPredOpList.size()) << "Size mismatch: noPredOpList size: " <<
               noPredOpList_.size() << " != noPredOpList size: " << noPredOpList.size();
        for (size_t i = 0; i < noPredOpList.size(); i++) {
            int opIdx = At(noPredOpList_, i);
            auto *op = callList[opIdx];
            ASSERT(!callOpPredDict.count(op) || callOpPredDict.at(op) == 0) << "callOpPredDict for op: " <<
                   op << " is not zero: " << callOpPredDict.at(op);
        }
        ASSERT(noSuccOpList_.size() == noSuccOpList.size()) << "Size mismatch: noSuccOpList size: " <<
               noSuccOpList_.size() << " != noSuccOpList size: " << noSuccOpList.size();
        for (size_t i = 0; i < noSuccOpList.size(); i++) {
            int opIdx = At(noSuccOpList_, i);
            auto *op = callList[opIdx];
            ASSERT(!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty()) << "callOpSuccDict for op: " <<
                   op << " is not empty";
        }
    }

    operationList_.HostInitDataSizeOffset(initOffset, callList.size());

    int operanSize = 0;
    int staticAttributeSize = 0;
    int sucSize = 0;
    int copyOutResolveSuccIdxSize = 0;
    for (size_t i = 0; i < callList.size(); i++) {
        Operation *op = callList[i];
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());

        operanSize += op->GetIOperands().size() + op->GetOOperands().size();
        staticAttributeSize += callop->GetLinearArgList().size();
        sucSize += callOpSuccDict.find(op)->second.size();
        copyOutResolveSuccIdxSize += copyOutResolveSuccIndexListDict.find(op)->second.size();
    }
    operationOperandInfoList_.HostInitDataSizeOffset(initOffset, operanSize);
    operationAttrList_.HostInitDataSizeOffset(initOffset, staticAttributeSize);
    opAttrOffsetList_.HostInitDataSizeOffset(initOffset, callList.size());
    opCalleeList_.HostInitDataSizeOffset(initOffset, callList.size());
    operationSuccList_.HostInitDataSizeOffset(initOffset, sucSize);
    operationCopyOutResolveSuccIndexList_.HostInitDataSizeOffset(initOffset, copyOutResolveSuccIdxSize);

    ONFILLCONTENT {
        DevAscendFunctionDuppedData *dupData = reinterpret_cast<DevAscendFunctionDuppedData *>(&At(duppedData_, 0));
        operanSize = 0;
        staticAttributeSize = 0;
        sucSize = 0;
        copyOutResolveSuccIdxSize = 0;
        for (size_t index = 0; index < callList.size(); index++) {
            Operation *op = callList[index];
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[index]->GetOpAttribute());
            DevAscendOperation &staticField = At(operationList_, index);

            staticField.outcastStitchIndex = outcastStitchIndexList[index];
            staticField.debugOpmagic = op->GetOpMagic();
            staticField.ioperandList.AssignRangeOffsetSize(
                operationOperandInfoList_, operanSize, op->GetIOperands().size());
            std::map<int, int> rawTensorIndex;
            for (size_t k = 0; k < op->GetIOperands().size(); k++) {
                auto coaIndex = op->GetIOpAttrOffset(k);
                const std::shared_ptr<LogicalTensor> &tensor = op->GetIOperands()[k];
                rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
                At(staticField.ioperandList, k) =
                    DevAscendOperationOperandInfo(tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
            }
            operanSize += op->GetIOperands().size();

            staticField.ooperandList.AssignRangeOffsetSize(
                operationOperandInfoList_, operanSize, op->GetOOperands().size());
            for (size_t k = 0; k < op->GetOOperands().size(); k++) {
                auto coaIndex = op->GetOOpAttrOffset(k);
                const std::shared_ptr<LogicalTensor> &tensor = op->GetOOperands()[k];
                rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
                At(staticField.ooperandList, k) =
                    DevAscendOperationOperandInfo(tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
            }
            operanSize += op->GetOOperands().size();
            ALOG_DEBUG_F("Producer %zu oOperand list size is %zu", index, op->GetOOperands().size());
            // Fill attr
            auto callArgs = callop->GetLinearArgList();
            int opStaticAttrSize = callArgs.size();
            staticField.attrList.AssignRangeOffsetSize(operationAttrList_, staticAttributeSize, opStaticAttrSize);
            At(staticField.attrList, 0) = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
            for (size_t k = CALLOP_ARG_ATTR_BASE_INDEX; k < (size_t)opStaticAttrSize; k++) {
                int fillValue = 0;
                if (callArgs[k].IsImmediate()) {
                    if (rawTensorIndex.count(k)) {
                        fillValue = rawTensorIndex[k];
                    } else {
                        fillValue = callArgs[k].Concrete();
                    }
                } else {
                    fillValue = expressionTable->LookupPrimaryExpressionIndex(callArgs[k]);
                }
                At(staticField.attrList, k) = SymInt(!callArgs[k].IsImmediate(), fillValue);
            }

            At(opAttrOffsetList_, index) = staticAttributeSize;
            At(opCalleeList_, index) = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
            staticAttributeSize += opStaticAttrSize;

            // Fill succ
            int opSuccSize = callOpSuccDict.find(op)->second.size();
            staticField.depGraphSuccList.AssignRangeOffsetSize(operationSuccList_, sucSize, opSuccSize);
            for (int k = 0; k < opSuccSize; k++) {
                int succ = callList.GetIndex(callOpSuccDict.find(op)->second[k]);
                At(staticField.depGraphSuccList, k) = succ;
                At(operationList_, succ).depGraphPredCount++;
                dupData->GetOperationCurrPredCount(succ)++;
            }
            sucSize += opSuccSize;

            // Fill copyout resolve
            const std::vector<int> &copyOutResolveSuccIndexList = copyOutResolveSuccIndexListDict.find(op)->second;
            int opCopyOutResolveSuccIndexSize = copyOutResolveSuccIndexList.size();
            staticField.depGraphCopyOutResolveSuccIndexList.AssignRangeOffsetSize(operationCopyOutResolveSuccIndexList_, copyOutResolveSuccIdxSize, opCopyOutResolveSuccIndexSize);
            for (int k = 0; k < opCopyOutResolveSuccIndexSize; k++) {
                At(staticField.depGraphCopyOutResolveSuccIndexList, k) = copyOutResolveSuccIndexList[k];
            }
            copyOutResolveSuccIdxSize += copyOutResolveSuccIndexList.size();
        }
        for (size_t idx = 0; idx < callList.size(); idx++) {
            Operation *op = callList[idx];
            ASSERT(callOpPredDict.count(op)) << "callOpPredDict does not contain op " << op;
            ASSERT(At(operationList_, idx).depGraphPredCount == callOpPredDict.find(op)->second) << "depGraphPredCount mismatch: expected " <<
                   callOpPredDict.find(op)->second << ", got " << At(operationList_, idx).depGraphPredCount;
            if(dupData->GetOperationCurrPredCount(idx) != callOpPredDict.find(op)->second) {
                ALOG_ERROR_F("OperationCurrPredCount: %d Callopsize is %u exceeds the maximum allowed value of 65535.",
                dupData->GetOperationCurrPredCount(idx), dupData ->GetOperationSize());
            }
            ASSERT(dupData->GetOperationCurrPredCount(idx) == callOpPredDict.find(op)->second) << "GetOperationCurrPredCount mismatch: expected " <<
                   dupData->GetOperationCurrPredCount(idx) << ", got " << callOpPredDict.find(op)->second <<
                   ", Callopsize is " << dupData->GetOperationSize() << " exceeds the maximum allowed value of 65535.";
        }
        dupData->GetSource() = this;
        for (size_t index = 0; index < callList.size(); index++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOperationStitch(index));
            uint8_t *stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];
            uint8_t *dataEndAlloc = &dupData->data_[duppedDataAllocSize_ - sizeof(DevAscendFunctionDuppedData)];
            ASSERT(stitchBegin <= ptr && ptr < dataEndAlloc) << "Address out of range: ptr " << ptr << " not in [" <<
                   stitchBegin << ", " << dataEndAlloc << ")";
        }
        dupData->GetSource() = nullptr;
    }
}

void DevAscendFunction::InitWrapInfo(uintdevptr_t &initOffset, const OrderedSet<Operation *> &callList, bool fillContent) {
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return;
    }
    opWrapList_.HostInitDataSizeOffset(initOffset, callList.size());
    opWrapTaskNumList_.HostInitDataSizeOffset(initOffset, callList.size());

    ONFILLCONTENT {
        std::unordered_map<int, int> wrapTaskNumMap;
        for (size_t i = 0; i < callList.size(); i++) {
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
            if (callop->wrapId != -1) {
                wrapTaskNumMap[callop->wrapId]++;
            }
        }
        wrapIdNum_ = wrapTaskNumMap.size();
        for (size_t i = 0; i < callList.size(); i++) {
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
            At(opWrapList_, i) = callop->wrapId;
            At(opWrapTaskNumList_, i) = wrapTaskNumMap[callop->wrapId];
        }
    }
}

void DevAscendFunction::InitIncastOutcast(
        uintdevptr_t &initOffset,
        const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &incastOpAttrDict,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &outcastOpAttrDict,
        const EncodeDevAscendFunctionParam &param, const std::string &initRawName, bool fillContent) {
    {
        // Fill metadata
        incastList.HostInitDataSizeOffset(initOffset, incastTensorList.size());
        outcastList.HostInitDataSizeOffset(initOffset, outcastTensorList.size());
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto &incast = At(incastList, index);
            ONFILLCONTENT {
                incast.tensorIndex = tlist.GetIndex(incastTensorList[index]);
                incast.dim = inAttr.dim;
                incast.cellMatchTableDesc = inAttr.cellMatchTableDesc;
            }
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.tensorIndex = tlist.GetIndex(outcastTensorList[index]);
                outcast.dim = outAttr.dim;
                outcast.cellMatchTableDesc = outAttr.cellMatchTableDesc;
                outcast.exprListIndex = outAttr.bindTensorExprIndex;
                outcast.desc = param.outcastDescList[index];
            }
        }
    }
    {
        const IncastOutcastSlot *slot = param.slot;
        [[maybe_unused]] const IncastOutcastLink *inoutLink = param.inoutLink;
        // Fill slot list
        slotList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t slotSize = 0;
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto &incast = At(incastList, index);
            ONFILLCONTENT {
                incast.fromSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->incastSlot[index].size());
                for (size_t j = 0; j < slot->incastSlot[index].size(); j++) {
                    At(incast.fromSlotList, j) = slot->incastSlot[index][j];
                }
            }
            slotSize += slot->incastSlot[index].size();
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.toSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->outcastSlot[index].size());
                for (size_t j = 0; j < slot->outcastSlot[index].size(); j++) {
                    At(outcast.toSlotList, j) = slot->outcastSlot[index][j];
                }
            }
            slotSize += slot->outcastSlot[index].size();
        }
        slotList.HostInitDataSizeOffset(initOffset, slotSize);

        redaccAssembleSlotList_.HostInitDataSizeOffset(initOffset, param.assembleSlotList.size());
        ONFILLCONTENT {
            for (size_t k = 0; k < param.assembleSlotList.size(); k++) {
                At(redaccAssembleSlotList_, k) = param.assembleSlotList[k];
            }
        }
    }
    {
        // Fill use list
        useList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t useSize = 0;
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto &incast = At(incastList, index);
            ONFILLCONTENT {
                incast.consumerList.AssignRangeOffsetSize(useList, useSize, inAttr.useList.size());
                for (size_t k = 0; k < inAttr.useList.size(); k++) {
                    At(incast.consumerList, k) = inAttr.useList[k];
                }
            }
            useSize += inAttr.useList.size();
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.producerList.AssignRangeOffsetSize(useList, useSize, outAttr.useList.size());
                for (size_t k = 0; k < outAttr.useList.size(); k++) {
                    At(outcast.producerList, k) = outAttr.useList[k];
                }
            }
            useSize += outAttr.useList.size();
        }
        useList.HostInitDataSizeOffset(initOffset, useSize);
    }
    {
        // Fill runtime full update table
        uint64_t cellMatchSizeOutcastTotal = 0;
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.cellMatchRuntimeFullUpdateTable.AssignRangeOffsetSize(cellMatchRuntimeFullUpdateTableList, cellMatchSizeOutcastTotal, outAttr.cellMatchSize);
                for (int k = 0; k < outAttr.cellMatchSize; k++) {
                    At(outcast.cellMatchRuntimeFullUpdateTable, k) = (uint32_t)-1;
                };
            }
            cellMatchSizeOutcastTotal += outAttr.cellMatchSize;
        }
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeOutcastTotal);
    }
    {
        // Fill stitchPolicyFullCoverProducerList
        uint64_t fullCoverTotal = 0;
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.stitchPolicyFullCoverProducerHubOpIdx = outAttr.stitchPolicyFullCoverProducerHubOpIdx;
                outcast.stitchPolicyFullCoverProducerList.AssignRangeOffsetSize(stitchPolicyFullCoverProducerList_, fullCoverTotal, outAttr.stitchPolicyFullCoverProducerList.size());
                for (size_t k = 0; k < outAttr.stitchPolicyFullCoverProducerList.size(); k++) {
                    At(outcast.stitchPolicyFullCoverProducerList, k) = outAttr.stitchPolicyFullCoverProducerList[k];
                };
            }
            fullCoverTotal += outAttr.stitchPolicyFullCoverProducerList.size();
        }
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, fullCoverTotal);
    }
    {
        // Fill stitchPolicyFullCoverOpList_
        uint32_t fullCoverOpTotal = 0;
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto &incast = At(incastList, index);
            ONFILLCONTENT {
                incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignRangeOffsetSize(stitchPolicyFullCoverOpList_, fullCoverOpTotal, inAttr.useOpList.size());
                for (size_t k = 0; k < inAttr.useOpList.size(); k++) {
                    At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, k) = inAttr.useOpList[k];
                }
            }
            fullCoverOpTotal += inAttr.useOpList.size();
        }
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.stitchPolicyFullCoverProducerAllOpIdxList.AssignRangeOffsetSize(stitchPolicyFullCoverOpList_, fullCoverOpTotal, outAttr.useOpList.size());
                for (size_t k = 0; k < outAttr.useOpList.size(); k++) {
                    At(outcast.stitchPolicyFullCoverProducerAllOpIdxList, k) = outAttr.useOpList[k];
                }
            }
            fullCoverOpTotal += outAttr.useOpList.size();
        }
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, fullCoverOpTotal);
    }
    {
        // Fill incast & outcast all full update table
        uint64_t cellMatchSizeIncastTotal = 0;
        cellMatchStaticIncastTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < incastTensorList.size(); index++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[index]);
            auto &incast = At(incastList, index);
            ONFILLCONTENT {
                incast.cellMatchStaticIncastTable.AssignRangeOffsetSize(cellMatchStaticIncastTableList, cellMatchSizeIncastTotal, inAttr.cellMatchSize);

                auto consumerList = &At(incast.consumerList, 0);
                auto tableData = &At(incast.cellMatchStaticIncastTable, 0);
                bool stitchByAllFullMatch = CellMatchFillIncastOutcast<true>(
                        this, consumerList, incast.consumerList.size(), nullptr, true, incast.cellMatchTableDesc, tableData);
                incast.stitchByAllFullMatch = stitchByAllFullMatch;
            };
            cellMatchSizeIncastTotal += inAttr.cellMatchSize;
        }
        cellMatchStaticIncastTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeIncastTotal);

        uint64_t cellMatchSizeOutcastTotal = 0;
        cellMatchStaticOutcastTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t index = 0; index < outcastTensorList.size(); index++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[index]);
            auto &outcast = At(outcastList, index);
            ONFILLCONTENT {
                outcast.cellMatchStaticOutcastTable.AssignRangeOffsetSize(cellMatchStaticOutcastTableList, cellMatchSizeOutcastTotal, outAttr.cellMatchSize);

                auto producerList = &At(outcast.producerList, 0);
                auto tableData = &At(outcast.cellMatchStaticOutcastTable, 0);
                bool stitchByAllFullMatch = CellMatchFillIncastOutcast<true>(
                        this, producerList, outcast.producerList.size(), nullptr, false, outcast.cellMatchTableDesc, tableData);
                outcast.stitchByAllFullMatch = stitchByAllFullMatch;
            }
            cellMatchSizeOutcastTotal += outAttr.cellMatchSize;
        }
        cellMatchStaticOutcastTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeOutcastTotal);
    }

    rawName_.HostInitDataSizeOffset(initOffset, (initRawName.size() / 8 + 1) * 8); // 8 byte align
    ONFILLCONTENT {
        memcpy_s(&At(rawName_, 0), rawName_.size(), initRawName.c_str(), initRawName.size());
        memset_s(&At(rawName_, initRawName.size()), rawName_.size() - initRawName.size(), 0,
            rawName_.size() - initRawName.size());
    };
}

struct EncodeDevAscendFunctionInfo {
    Function *devRoot{nullptr};
    Function *devTile{nullptr};

    const std::unordered_map<uint64_t, int> &calleeHashIndexDict;
    const std::vector<CceCodeInfo> &cceCodeInfoList;
    const SymbolicExpressionTable *expressionTable{nullptr};

    std::string rawName;

    OrderedSet<Operation *> callList;

    uint64_t totalZeroPred{0};
    uint64_t totalZeroPredAIV{0};
    uint64_t totalZeroPredAIC{0};
    uint64_t totalZeroPredHub{0};
    uint64_t totalZeroPredAicpu{0};
    uint32_t hubOpCount{0};

    std::unordered_map<Operation *, uint64_t> callOpPredDict;
    std::unordered_map<Operation *, OrderedSet<Operation *>> callOpSuccDict;
    std::unordered_map<int, std::vector<int>> colorOutGraph;
    std::vector<std::shared_ptr<Operation>> dummyOpList;

    std::vector<int> noSuccOpList;
    std::vector<int> noPredOpList;

    uint32_t outcastStitchCount{0};
    std::vector<int32_t> outcastStitchIndexList;

    OrderedSet<std::shared_ptr<RawTensor>> incastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> outcastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> rawTensorList;
    std::vector<EncodeRawTensorAttr> rawAttrs;

    std::unordered_map<int, std::shared_ptr<RawTensor>> rawMagicToRawTensor;
    OrderedSet<std::shared_ptr<LogicalTensor>> tensorList;

    std::vector<std::shared_ptr<LogicalTensor>> incastList;
    std::vector<std::shared_ptr<LogicalTensor>> outcastList;

    std::unordered_set<std::shared_ptr<LogicalTensor>> incastSet;
    std::unordered_set<std::shared_ptr<LogicalTensor>> outcastSet;

    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> incastOpAttrDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> outcastOpAttrDict;

    std::unordered_map<Operation *, std::vector<int>> copyOutResolveSuccIndexListDict;

    DyndevFunctionAttribute::ValueDependDesc valueDependDesc;

    static DevShape InitShape(const std::vector<int64_t> &shape) {
        DevShape initShape;
        initShape.dimSize = shape.size();
        for (size_t index = 0; index < DEV_SHAPE_DIM_MAX; index++) {
            if (index < shape.size()) {
                initShape.dim[index] = shape[index];
            } else {
                initShape.dim[index] = 0;
            }
        }
        return initShape;
    }
    static DevAscendStride InitStride(const std::vector<int64_t> &stride) {
        DevAscendStride initStride;
        initStride.dimSize = stride.size();
        for (size_t index = 0; index < DEV_SHAPE_DIM_MAX; index++) {
            if (index < stride.size()) {
                initStride.dimStride[index] = stride[index];
            } else {
                initStride.dimStride[index] = 0;
            }
        }
        return initStride;
    }
    static DevCellMatchTableDesc InitCellMatchTableDesc(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride) {
        DevCellMatchTableDesc desc = {
            InitShape(shape),
            InitStride(stride),
        };
        return desc;
    }

    std::vector<int> ShapeToVector(const DevShape &shape) {
        std::vector<int> data(&shape.dim[0], &shape.dim[shape.dimSize]);
        return data;
    }
    std::vector<int> StrideToVector(const DevAscendStride &stride) {
        std::vector<int> data(&stride.dimStride[0], &stride.dimStride[stride.dimSize]);
        return data;
    }

    void UpdateCellMatchShape(DevCellMatchTableDesc &cellMatchTableDesc, const std::vector<int64_t> &shape) {
        auto &cellMatchShape = cellMatchTableDesc.cellShape;
        for (size_t index = 0; index < shape.size(); ++index) {
            auto dimValue = shape[index];
            if (cellMatchShape.dim[index] > dimValue) {
                cellMatchShape.dim[index] = dimValue;
                if (cellMatchShape.dim[index] == 0) {
                    ALOG_ERROR_F("cellMatchShape.dim[%zu] is zero after assignment", index);
                }
                DEV_ASSERT(cellMatchShape.dim[index]);
            }
        }
    }

    void UpdateCellMatchStrideAndSize(int &cellMatchSize, DevCellMatchTableDesc &cellMatchTableDesc, const std::shared_ptr<LogicalTensor> &tensor, int dim) {
        auto &cellMatchShape = cellMatchTableDesc.cellShape;
        auto &cellMatchStride = cellMatchTableDesc.stride;

        cellMatchSize = 1;
        cellMatchStride.dimSize = dim;
        for (int r = (dim - 1); r >= 0; --r) {
            int tile = 0;
            if (cellMatchShape.dim[r] != 0) {
                tile = tensor->shape[r] / cellMatchShape.dim[r];
                if (tensor->shape[r] % cellMatchShape.dim[r] != 0) {
                    // should not happen
                    tile += 1;
                }
            }
            cellMatchSize *= tile;
            cellMatchStride[r] = cellMatchSize;
        }
        ALOG_DEBUG_F("Outcast is %d raw %d shape %s | cellMatchSize %d cellMatchShape %s cellMatchStride %s\n", tensor->magic, tensor->GetRawMagic(),
            IntVecToStr(tensor->shape).c_str(),
            cellMatchSize,
            IntVecToStr(ShapeToVector(cellMatchShape)).c_str(),
            IntVecToStr(StrideToVector(cellMatchStride)).c_str());
        ASSERT(cellMatchStride[0] < MAX_CELLMATCHSSTRIDE) << " Assemble outcast " << tensor->magic << "raw" << tensor->GetRawMagic()
 	         <<"stitch results in excessive memory consumption," 
 	         << "Please appropriately configure the view shape and tile shape, and ensure aligned with the input shape."; 
    }

    void RecordRawTensor(const std::shared_ptr<LogicalTensor> &tensor) {
        if (rawTensorList.Insert(tensor->GetRawTensor())) {
            EncodeRawTensorAttr &attr = rawAttrs.emplace_back();
            attr.storage = tensor->storage_;
            attr.storageOffset = tensor->storageOffset_;
            rawMagicToRawTensor[tensor->GetRawTensor()->rawmagic] = tensor->GetRawTensor();
        }
    }

    void EncodeAnalysisOpUseOutCasts(const std::shared_ptr<LogicalTensor>& o, std::set<uint32_t>& allOutcastUseOpSet, InoutOperationAttr& outcastOpAttr) {
        std::set<uint32_t> outcastUseOpSet;
        int dimSize = outcastOpAttr.dim;
        for (size_t i = 0; i < callList.size(); i++) {
            auto &op = *callList[i];
            auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
            std::vector<DevAscendFunctionCallOperandUse> useList;
            DevAscendFunctionCallOperandUse stitchPolicyFullCoverProducer;
            for (size_t j = 0; j < op.GetOOperands().size(); ++j) {
                auto &oOperand = op.GetOOperands()[j];
                if (o->tensor->rawmagic != oOperand->tensor->rawmagic) {
                    continue;
                }

                auto coaIndex = op.GetOOpAttrOffset(j) + COA_INDEX_DIM_BASE;
                std::vector<int64_t> offset = callAttr->GetLinearImmediateArgList(coaIndex, coaIndex + dimSize, true);
                std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dimSize, coaIndex + dimSize * 0x2, false);
                if (offset == std::vector<int64_t>(dimSize, 0) && shape == oOperand->GetShape()) {
                    stitchPolicyFullCoverProducer = DevAscendFunctionCallOperandUse(i, j, coaIndex, coaIndex + dimSize);
                } else {
                    useList.emplace_back(i, j, coaIndex, coaIndex + dimSize);
                }
                ALOG_DEBUG_F("Outcast oOperandIdx for outcast %d %d is %d", o->magic, o->GetRawMagic(), j);
                outcastUseOpSet.insert(i);
                auto expr = callAttr->GetOutcastSymbolicExpr(j);
                outcastOpAttr.bindTensorExprIndex = -1;
                if ((expr.has_value()) && (expressionTable != nullptr)) {
                    outcastOpAttr.bindTensorExprIndex = expressionTable->LookupPrimaryExpressionIndex(expr.value());
                }
            }

            if (stitchPolicyFullCoverProducer.operationIdx != -1) {
                outcastOpAttr.stitchPolicyFullCoverProducerList.push_back(stitchPolicyFullCoverProducer);
            } else {
                outcastOpAttr.useList.insert(outcastOpAttr.useList.end(), useList.begin(), useList.end());
                for (auto &[operationIdx, operandIdx, offsetAttrIdx, shapeAttrIdx] : useList) {
                    UNUSED(operationIdx);
                    UNUSED(operandIdx);
                    UNUSED(offsetAttrIdx);
                    auto shape = callAttr->GetLinearImmediateArgList(shapeAttrIdx, shapeAttrIdx + dimSize, false);
                    UpdateCellMatchShape(outcastOpAttr.cellMatchTableDesc, shape);
                    ALOG_DEBUG_F("minimal shape for outcast %d raw %d op %d %d is %s\n", o->magic, o->GetRawMagic(), i,
                        op.GetOpMagic(), IntVecToStr(ShapeToVector(outcastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                }
            }
        }
        UpdateCellMatchStrideAndSize(outcastOpAttr.cellMatchSize, outcastOpAttr.cellMatchTableDesc, o, dimSize);
        outcastOpAttr.useOpList.insert(outcastOpAttr.useOpList.end(), outcastUseOpSet.begin(), outcastUseOpSet.end());
        allOutcastUseOpSet.insert(outcastUseOpSet.begin(), outcastUseOpSet.end());
        outcastOpAttrDict.insert({o, outcastOpAttr});
    }

    void EncodeOutCasts() {
        std::set<uint32_t> allOutcastUseOpSet;
        for (auto &i : outcastList) {
            tensorList.Insert(i);
            outcastRawTensorList.Insert(i->GetRawTensor());
            RecordRawTensor(i);
            InoutOperationAttr outcastOpAttr;
            auto dimSize = i->shape.size();
            outcastOpAttr.dim = dimSize;
            outcastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(i->GetShape(), std::vector<int64_t>(dimSize, 1));
            EncodeAnalysisOpUseOutCasts(i, allOutcastUseOpSet, outcastOpAttr);
        }

        // Add edge from all stitchPolicyFullCoverProducerList's node to the single node
        size_t hubEntryLeast = 2;
        for (auto &i : outcastList) {
            auto &outcastOpAttr = outcastOpAttrDict[i];
            if (outcastOpAttr.stitchPolicyFullCoverProducerList.size() > hubEntryLeast) {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = callList.size();

                auto dummyOp = MakeDummyCall();
                callList.Insert(dummyOp);

                callOpPredDict[dummyOp] = outcastOpAttr.stitchPolicyFullCoverProducerList.size();
                for (auto &producer : outcastOpAttr.stitchPolicyFullCoverProducerList) {
                    auto callOp = callList[producer.operationIdx];
                    callOpSuccDict[callOp].Insert(dummyOp);
                }
                callOpSuccDict[dummyOp].Clear();
            } else {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = -1;
            }
        }

        std::set<int> noSuccOpSet;
        for (size_t index = 0; index < callList.size(); index++) {
            Operation *op = callList[index];
            if (!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty()) {
                noSuccOpList.push_back(index);
                noSuccOpSet.insert(index);
            }
        }

        /*
         * As we need a reference of null, we use the 0-th element for the reference of null for DevAscendFunctionDuppedData
         * So outcastStitchCount starts from 1, as the 0-th is for the reference of null.
         */
        outcastStitchCount = 1;
        for (size_t index = 0; index < callList.size(); index++) {
            if (allOutcastUseOpSet.count(index) || noSuccOpSet.count(index)) {
                outcastStitchIndexList.push_back(outcastStitchCount);
                outcastStitchCount++;
            } else {
                outcastStitchIndexList.push_back(0);
            }
        }
    }

    void EncodeIncasts() {
        for (auto &index : incastList) {
            tensorList.Insert(index);
            incastRawTensorList.Insert(index->GetRawTensor());
            RecordRawTensor(index);
            InoutOperationAttr incastOpAttr;
            auto dimSize = index->shape.size();
            incastOpAttr.dim = dimSize;
            incastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(index->GetShape(), std::vector<int64_t>(dimSize, 1));

            std::set<uint32_t> incastUseOpSet;
            for (size_t j = 0; j < callList.size(); j++) {
                auto &op = *callList[j];
                auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
                // add icast and oper io's relationship
                for (size_t k = 0; k < op.GetIOperands().size(); ++k) {
                    auto &iOperand = op.GetIOperands()[k];
                    auto coaIndex = op.GetIOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                    if (index->tensor->rawmagic == iOperand->tensor->rawmagic) {
                        ASSERT(iOperand->GetShape().size() == dimSize) << "Shape size mismatch: expected: " << dimSize << ", got " <<
                               iOperand->GetShape().size() << " for operand: " << k;
                        std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dimSize, coaIndex + dimSize * 0x2, false);
                        if (shape == Shape(shape.size())) { // 跳过全0
                            continue;
                        }
                        incastOpAttr.useList.emplace_back(j, k, coaIndex, coaIndex + dimSize);
                        UpdateCellMatchShape(incastOpAttr.cellMatchTableDesc, shape);
                        ALOG_DEBUG_F("minimal shape for incast %d raw %d op %d %d is %s\n", index->magic, index->GetRawMagic(), j,
                            op.GetOpMagic(), IntVecToStr(ShapeToVector(incastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                        incastUseOpSet.insert(j);
                    }
                }
            }

            UpdateCellMatchStrideAndSize(incastOpAttr.cellMatchSize, incastOpAttr.cellMatchTableDesc, index, dimSize);
            incastOpAttr.useOpList.insert(incastOpAttr.useOpList.end(), incastUseOpSet.begin(), incastUseOpSet.end());

            incastOpAttrDict.insert({index, incastOpAttr});
        }

        for (size_t index = 0; index < callList.size(); index++) {
            Operation *op = callList[index];
            if (!callOpPredDict.count(op) || callOpPredDict.at(op) == 0) {
                noPredOpList.push_back(index);
            }
        }
    }

    struct Hasher {
        template<typename T>
        std::size_t operator()(const OrderedSet<T> &operationSet) const {
            size_t res = 0;
            for (auto op : operationSet) {
                res ^= std::hash<Operation *>{}(op);
            }
            return res;
        }
    };

    Operation *MakeDummyCall() {
        LogicalTensors inputs, outputs;
        auto opAttr = std::make_shared<CallOpAttribute>();
        auto dummyOp = std::make_shared<Operation>(*devRoot, Opcode::OP_CALL);
        dummyOp->SetOpAttribute(opAttr);
        dummyOpList.push_back(dummyOp);
        ASSERT(GetCoreType(dummyOp.get()) == static_cast<int>(CoreType::HUB)) << "GetCoreType return unexpected value: " <<
               GetCoreType(dummyOp.get()) << ", expected:  " << static_cast<int>(CoreType::HUB);
        return dummyOp.get();
    }

    int GetCoreType(Operation *callop) {
        int leafIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
        return cceCodeInfoList[leafIndex].coreType;
    }

    void RemoveDeadHubCall(std::vector<Operation *> &/* callOpList */) {
        std::vector<Operation *> deadCallOps;
        for (auto &[callOp, succOps] : callOpSuccDict) {
            if (GetCoreType(callOp) == static_cast<int>(CoreType::HUB) && succOps.size() == 0) {
                /*  Find all hub callop that has no successors, mark it is no need to schedule:
                 *  1. mark pred to be zero
                 *  2. remove it from the successor of all callop
                 */
                callOpPredDict[callOp] = 0;
                deadCallOps.push_back(callOp);
            }
        }

        for (auto &[callOp, succOps] : callOpSuccDict) {
            UNUSED(callOp);
            succOps.Remove(deadCallOps);
        }
    }

    void ReplaceSuccessorWithHub(std::vector<Operation *> &callOpList, int optimizeLimit) {
        // dict from successor set to callop set that has the successor.
        OrderedMap<OrderedSet<Operation *>, OrderedSet<Operation *>, Hasher> predDict;
        for (auto &callOp : callOpList) {
            if (callOpSuccDict.count(callOp)) {
                auto succOps = callOpSuccDict[callOp];
                predDict[succOps].Insert(callOp);
            }
        }

        for (auto & [succSet, predSet] : predDict) {
            int optimizeCnt = succSet.size() * predSet.size() - succSet.size() - predSet.size();
            if (optimizeCnt > optimizeLimit) {
                auto dummyOp = MakeDummyCall();
                callOpList.push_back(dummyOp);

                callOpSuccDict[dummyOp] = succSet;
                for (auto &pred : predSet) {
                    callOpSuccDict[pred].Clear();
                    callOpSuccDict[pred].Insert(dummyOp);
                }

                callOpPredDict[dummyOp] = predSet.size();
                for (auto &succ: succSet) {
                    callOpPredDict[succ] -= predSet.size() - 1;
                }
            }
        }
    }

    void PrintColorGraph(int colorNum) {
        ALOG_INFO_F("*********** Call OP Graph ***********\n");
        for (int index = 0; index < colorNum; index++) {
            ALOG_INFO_F("%zu: %zu", index, colorOutGraph[index].size());
            ALOG_INFO_F("%s", IntVecToStr(colorOutGraph[index]).c_str());
        }
        int outCount = 0;
        for (int index = 0; index < colorNum; index++) {
            outCount += colorOutGraph[index].size();
        }
        ALOG_INFO_F("Total out: %d\n", outCount);
    }

    inline void FindAllReachableNodes(int start_node, std::unordered_map<int, std::vector<int>>& outGraph,
                                        std::vector<std::unordered_set<int>>& reachable, std::vector<int>& visited) {
        reachable[start_node].insert(start_node);
        for (int v : outGraph[start_node]) {
            if (visited[v] == 0) {
                FindAllReachableNodes(v, outGraph, reachable, visited);
            }
            reachable[start_node].insert(reachable[v].begin(), reachable[v].end());
        }
        visited[start_node] = 1;
    }

    void FindRedundantEdges(int colorNum, std::vector<std::vector<int>>& redundantColorOutGraph) {
        std::vector<std::unordered_set<int>> reachable(colorNum);
        std::vector<int> visited(colorNum, 0);
        for (int index = 0; index < colorNum; ++index) {
            if (visited[index] == 0) {
                FindAllReachableNodes(index, colorOutGraph, reachable, visited); // DFS记忆化计算
            }
        }
        for (int j = 0; j < colorNum; ++j) {
            for (int k : colorOutGraph[j]) {
                bool is_redundant = false;
                for (int w : colorOutGraph[j]) {
                    if (w == k) {
                        continue;
                    }
                    if (reachable[w].count(k)) {
                        is_redundant = true;
                        break;
                    }
                }
                if (is_redundant) {
                    redundantColorOutGraph[j].push_back(k);
                }
            }
        }
    }

    void EraseRedundantColorEdges(std::vector<Operation *> &callopList) {
        int colorNum = callopList.size();
        std::vector<std::vector<int>> redundantColorOutGraph(colorNum);
        // Find redundant edges
        FindRedundantEdges(colorNum, redundantColorOutGraph);
        // Erase redundant edges
        for (int index = 0; index < colorNum; index++) {
            // make redundantColorOutGraph[index]'s order grow
            std::sort(redundantColorOutGraph[index].begin(), redundantColorOutGraph[index].end());
            ALOG_INFO_F("Redundant outgraph of %d is %s.", index, IntVecToStr(redundantColorOutGraph[index]).c_str());
            // update color_out_graph
            std::vector<int> newGraph;
            size_t n = 0U;
            // for each index -> * -> k && index -> k
            for (int k : redundantColorOutGraph[index]) {
                // for each index -> x before x is k (so that x != k), add index -> x
                while (colorOutGraph[index][n] != k) {
                    newGraph.push_back(colorOutGraph[index][n]);
                    callOpSuccDict[callopList[index]].Insert(callopList[colorOutGraph[index][n]]);
                    callOpPredDict[callopList[colorOutGraph[index][n]]]++;
                    n++;
                }
                // until x == k, skip x
                n++;
            }
            // add index -> x, where x is the rest (larger than the largest k)
            while (n < colorOutGraph[index].size()) {
                newGraph.push_back(colorOutGraph[index][n]);
                callOpSuccDict[callopList[index]].Insert(callopList[colorOutGraph[index][n]]);
                callOpPredDict[callopList[colorOutGraph[index][n]]]++;
                n++;
            }
            colorOutGraph[index] = newGraph;
        }
    }

    void AddDummyCallsAtBeginningAndEnding(std::vector<Operation *> &callopList) {
        static constexpr size_t OPTIMIZATION_THRESHOLD = 3;

        std::vector<Operation *> zeroPreds;
        std::vector<Operation *> zeroSuccs;
        for (auto *op : callopList) {
            if (callOpPredDict[op] == 0) {
                zeroPreds.push_back(op);
            }
            if (callOpSuccDict[op].empty()) {
                zeroSuccs.push_back(op);
            }
        }

        // Zero predecessors
        if (zeroPreds.size() >= OPTIMIZATION_THRESHOLD) {
            auto *dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpPredDict[dummyOp] = 0;
            for (auto *op : zeroPreds) {
                ASSERT(callOpPredDict[op] == 0) << "callOpPredDict[op] is not zero:" << callOpPredDict[op] << ", expected 0";
                callOpSuccDict[dummyOp].Insert(op);
                callOpPredDict[op] = 1;
            }
            ASSERT(callOpSuccDict[dummyOp].size() == zeroPreds.size()) << "callOpSuccDict[dummyOp] size mismatch: expected: " <<
                   zeroPreds.size() << ", got: " << callOpSuccDict[dummyOp].size();
        }

        // Zero successors
        if (zeroSuccs.size() >= OPTIMIZATION_THRESHOLD) {
            auto *dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpSuccDict[dummyOp] = {};
            callOpPredDict[dummyOp] = zeroSuccs.size();
            for (auto *op : zeroSuccs) {
                ASSERT(callOpSuccDict[op].empty()) << "callOpSuccDict[op] is not empty, expect empty";
                callOpSuccDict[op].Insert(dummyOp);
            }
        }
    }

     void AddDependOperandsToColorGraph(std::vector<Operation *> &callopList, std::unordered_map<Operation *, int> &callopIndexDict) {
        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation *>> producerDict;
        for (auto &op : callopList) {
            for (auto &i : op->GetOOperands()) {
                producerDict[i].Insert(op);
            }
        }

        for (auto &op : callopList) {
            for (auto &o : op->GetDependOperands()) {
                for (auto &producer : producerDict[o]) {
                    if (op == producer) {
                        // Consumer and producer can not be the same.
                        continue;
                    }
                    // Index for callop from its depend operand's producer callop index list
                    colorOutGraph[callopIndexDict[producer]].push_back(callopIndexDict[op]);
                }
            }
        }
    }

    void EncodeZeroPredCount(std::vector<Operation *>& callopList) {
        std::unordered_map<Operation *, int> callopCoreTypeDict;
        for (auto &op : callopList) {
            auto callOpAttr = std::static_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
            auto calleeHash = callOpAttr->GetCalleeHash().GetHash();
            ASSERT(calleeHashIndexDict.count(calleeHash)) << "calleeHash 0x" << std::hex << calleeHash << " is not found in calleeHashIndexDict";
            int cceIndex = calleeHashIndexDict.find(calleeHash)->second;
            ASSERT(cceIndex < static_cast<int>(cceCodeInfoList.size())) << "cceIndex " << cceIndex << " exceeds cceCodeInfoList size: " << cceCodeInfoList.size();

            uint32_t coreType = cceCodeInfoList[cceIndex].coreType;
            ASSERT(coreType == static_cast<uint32_t>(CoreType::AIV) || coreType == static_cast<uint32_t>(CoreType::AIC) ||
                   coreType == static_cast<uint32_t>(CoreType::HUB) || coreType == static_cast<uint32_t>(CoreType::AICPU)) <<
                   "invalid coreType " << coreType << " for op " << op;
            callopCoreTypeDict[op] = coreType;
        }

        std::sort(callopList.begin(), callopList.end(), [&](Operation *lhs, Operation *rhs) {
            if (callOpPredDict[lhs] != callOpPredDict[rhs]) {
                return callOpPredDict[lhs] < callOpPredDict[rhs];
            }
            ASSERT(callopCoreTypeDict.count(lhs)) << "lhs operation " << lhs << " is not found in callopCoreTypeDict";
            ASSERT(callopCoreTypeDict.count(rhs)) << "rhs operation " << rhs << " is not found in callopCoreTypeDict";
            return callopCoreTypeDict[lhs] < callopCoreTypeDict[rhs];
        });

        totalZeroPred = callopList.size();
        for (size_t index = 0; index < callopList.size(); index++) {
            if (callOpPredDict[callopList[index]] != 0) {
                totalZeroPred = index;
                break;
            }
        }
        for (size_t index = totalZeroPred; index < callopList.size(); index++) {
            ASSERT(callOpPredDict[callopList[index]] != 0) << "callOpPredDict[callopList[" << index << "]] is zero, callopList[" << index <<
                   "] = " << callopList[index];
        }

        for (uint32_t index = 0; index < totalZeroPred; index++) {
            if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AIV)) {
                totalZeroPredAIV++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AIC)) {
                totalZeroPredAIC++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::HUB)) {
                totalZeroPredHub++;
            } else if (callopCoreTypeDict[callopList[index]] == static_cast<uint32_t>(CoreType::AICPU)) {
                totalZeroPredAicpu++;
            } else {
                ASSERT(false) << "Invalid coreType for callopList[" << index << "], op : " << callopList[index];
            }
        }
    }

    void EncodeCopyOutReslove(std::unordered_map<Operation *, std::unordered_map<Operation *, int>>& producerConsumerOOperandIndexDict) {
        FunctionCache &cache = Program::GetInstance().GetFunctionCache();
        for (auto &[callop, succSet] : callOpSuccDict) {
            Function *devLeafFunc = cache.GetCacheFunction(callop->GetCalleeHash());
            std::shared_ptr<LeafFuncAttribute> leafAttr = nullptr;
            if (devLeafFunc == nullptr) {
                pto::BlockFunction *devIrBlockFunc = cache.GetCacheIrBlockFunction(callop->GetCalleeHash());
                if (devIrBlockFunc == nullptr) {
                    ASSERT(GetCoreType(callop) == static_cast<int>(CoreType::HUB)) << "GetCoreType return unexpected value: " <<
                       GetCoreType(callop) << ", expectedBlockFunction: " << static_cast<int>(CoreType::HUB) << " for callop: " << callop;
                    copyOutResolveSuccIndexListDict[callop] = std::vector<int>({0});
                    continue;
                }
                leafAttr = devIrBlockFunc->GetLeafFuncAttribute();
            } else {
                leafAttr = devLeafFunc->GetLeafFuncAttribute();
            }

            if (leafAttr == nullptr) {
                ALOG_ERROR_F("Leaf Attr of leaf function %s is nullptr.", callop->GetCalleeMagicName().c_str());
                continue;
            }
            if (leafAttr->outcastCopyOutResolveCounterList.size() == 0) {
                copyOutResolveSuccIndexListDict[callop] = std::vector<int>({0});
                continue;
            }

            std::vector<OrderedSet<Operation *>> copyOutResolveSetList;
            copyOutResolveSetList.resize(leafAttr->copyOutResolveSize);

            OrderedSet<Operation *> nonCopyOutResolveSuccSet;
            for (auto &succ : succSet) {
                if (producerConsumerOOperandIndexDict.count(callop) && producerConsumerOOperandIndexDict[callop].count(succ)) {
                    auto ooperandIndex = producerConsumerOOperandIndexDict[callop][succ];
                    int copyOutResolveCounter = leafAttr->outcastCopyOutResolveCounterList[ooperandIndex];
                    copyOutResolveSetList[copyOutResolveCounter].Insert(succ);
                } else {
                    nonCopyOutResolveSuccSet.Insert(succ);
                }
            }

            std::vector<int> copyOutResolveSuccIndexList;
            std::vector<Operation *> copyOutResolveSuccList;
            for (int k = 0; k < leafAttr->copyOutResolveSize; k++) {
                OrderedSet<Operation *> &succ = copyOutResolveSetList[k];
                copyOutResolveSuccIndexList.push_back(copyOutResolveSuccList.size());
                copyOutResolveSuccList.insert(copyOutResolveSuccList.end(), succ.begin(), succ.end());
            }
            copyOutResolveSuccIndexList.push_back(copyOutResolveSuccList.size());
            ASSERT(copyOutResolveSuccIndexList[0] == 0) << "copyOutResolveSuccIndexList[0] is " << copyOutResolveSuccIndexList[0] << ", expected 0";
            copyOutResolveSuccList.insert(copyOutResolveSuccList.end(), nonCopyOutResolveSuccSet.begin(), nonCopyOutResolveSuccSet.end());

            // Assert: succ set are the same
            ASSERT(std::set<Operation *>(succSet.begin(), succSet.end()) == std::set<Operation *>(copyOutResolveSuccList.begin(), copyOutResolveSuccList.end())) <<
                   "succSet and copyOutResolveSuccList content mismatch";

            succSet.Clear();
            for (Operation *copyOutResolveSucc : copyOutResolveSuccList) {
                // Assert: no duplicated item in copyOutResolveSuccList
                ASSERT(succSet.Insert(copyOutResolveSucc)) << "Duplicate item " << copyOutResolveSucc << " found in copyOutResolveSuccList";
            }

            copyOutResolveSuccIndexListDict[callop] = copyOutResolveSuccIndexList;
        }
    }

    void InsertProducerConsmerOOperandIndexDict(std::shared_ptr<LeafFuncAttribute> leafAttr, Operation *op, Operation *consumer, std::shared_ptr<LogicalTensor> o,
        std::unordered_map<Operation *, std::unordered_map<Operation *, int>>& producerConsumerOOperandIndexDict) {
            if (producerConsumerOOperandIndexDict.count(op) && producerConsumerOOperandIndexDict[op].count(consumer)) {
                // There might be multiple ooperand of op that is consumed by the same consumer. So when
                // it happens, we need to select the ooperand with the biggest counter.
                int currIndex = producerConsumerOOperandIndexDict[op][consumer];
                int oIndex = op->GetOOperandIndex(o);
                if (leafAttr != nullptr && leafAttr->outcastCopyOutResolveCounterList.size() != 0) {
                    // When there is leaf, and the root is marked as resolve, the leafAttr records the biggest counter.
                    int currCounter = leafAttr->outcastCopyOutResolveCounterList[currIndex];
                    int oCounter = leafAttr->outcastCopyOutResolveCounterList[oIndex];
                    if (oCounter > currCounter) {
                        producerConsumerOOperandIndexDict[op][consumer] = oIndex;
                    }
                } else {
                    // Otherwise, we use any, which is the first
                }
            } else {
                producerConsumerOOperandIndexDict[op][consumer] = op->GetOOperandIndex(o);
            }
    }

    void BuildColorOutGraphAndProducerConsumerOOperandDict(std::vector<Operation *>& callopList,
        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation *>>& consumerDict,
        std::unordered_map<Operation *, int>& callopIndexDict,
        std::unordered_map<Operation *, std::unordered_map<Operation *, int>>& producerConsumerOOperandIndexDict) {
        FunctionCache &cache = Program::GetInstance().GetFunctionCache();
        for (auto &op : callopList) {
            Function *devLeafFunc = cache.GetCacheFunction(op->GetCalleeHash());
            std::shared_ptr<LeafFuncAttribute> leafAttr = devLeafFunc != nullptr ? devLeafFunc->GetLeafFuncAttribute() : nullptr;

            for (auto &o : op->GetOOperands()) {
                for (auto &consumer : consumerDict[o]) {
                    if (consumer->GetOpcode() != Opcode::OP_CALL) {
                        // This should be prevented from the above: only call op is considered as consumer
                        continue;
                    }
                    if (op == consumer) {
                        // Consumer and producer can not be the same.
                        continue;
                    }
                    // Index for callop to its ooperand's consumer callop index list
                    colorOutGraph[callopIndexDict[op]].push_back(callopIndexDict[consumer]);
                    InsertProducerConsmerOOperandIndexDict(leafAttr, op, consumer, o, producerConsumerOOperandIndexDict);
                }
            }
        }
        AddDependOperandsToColorGraph(callopList, callopIndexDict);
        for (size_t idx = 0; idx < callopList.size(); idx++) {
            std::sort(colorOutGraph[idx].begin(), colorOutGraph[idx].end());
            // remove repeated idx in ooperand's consumer callop idx list
            colorOutGraph[idx].resize(std::unique(colorOutGraph[idx].begin(), colorOutGraph[idx].end()) -
                                colorOutGraph[idx].begin());
        }
    }

    void BuildCallopList(std::vector<Operation *>& callopList, std::unordered_map<Operation *, int>& callopIndexDict,
        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation *>>& consumerDict) {
        for (auto &op : devRoot->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CALL) {
                callopIndexDict[&op] = callopList.size();
                callopList.push_back(&op);

                for (auto &i : op.GetIOperands()) {
                    tensorList.Insert(i);
                    RecordRawTensor(i);
                    consumerDict[i].Insert(&op);
                }
                for (auto &o : op.GetOOperands()) {
                    tensorList.Insert(o);
                    RecordRawTensor(o);
                }
            }
        }
        for (auto &op : callopList) {
            callOpPredDict[op] = 0;
            callOpSuccDict[op].clear();
        }
    }

    EncodeDevAscendFunctionInfo(
            Function *dyndev,
            const std::unordered_map<uint64_t, int> &tHashIndexDict,
            const std::vector<CceCodeInfo> &tCceCodeInfoList,
            const SymbolicExpressionTable *tExpressionTable,
            Function *tdevRoot)
            : devRoot(tdevRoot),
              calleeHashIndexDict(tHashIndexDict),
              cceCodeInfoList(tCceCodeInfoList),
              expressionTable(tExpressionTable) {
        (void)dyndev;
        ASSERT(dyndev->GetDyndevAttribute()->rootTileDict.count(devRoot)) << "devRoot: " << devRoot << " not found in rootTileDict of dyndev";
        devTile = dyndev->GetDyndevAttribute()->rootTileDict[devRoot];
        if (dyndev->GetDyndevAttribute()->valueDependDescDict.count(devTile)) {
            valueDependDesc = dyndev->GetDyndevAttribute()->valueDependDescDict[devTile];
        }

        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation *>> consumerDict;
        std::unordered_map<Operation *, int> callopIndexDict;
        std::vector<Operation *> callopList;
        std::unordered_map<Operation *, std::unordered_map<Operation *, int>> producerConsumerOOperandIndexDict;

        rawName = devRoot->GetRawName();
        incastList = devRoot->GetIncast();
        outcastList = devRoot->GetOutcast();
        incastSet.insert(incastList.begin(), incastList.end());
        outcastSet.insert(outcastList.begin(), outcastList.end());

        BuildCallopList(callopList, callopIndexDict, consumerDict);
        BuildColorOutGraphAndProducerConsumerOOperandDict(callopList, consumerDict, callopIndexDict, producerConsumerOOperandIndexDict);

        PrintColorGraph(callopList.size());
        EraseRedundantColorEdges(callopList);
        PrintColorGraph(callopList.size());

        RemoveDeadHubCall(callopList);
        ReplaceSuccessorWithHub(callopList, 10); // add dummp op at least 10 depends can be reduced

        AddDummyCallsAtBeginningAndEnding(callopList);

        EncodeCopyOutReslove(producerConsumerOOperandIndexDict);
        EncodeZeroPredCount(callopList);

        for (auto &op : callopList) {
            callList.Insert(op);
        }

        EncodeIncasts();
        EncodeOutCasts();

        // dummy op might be inserted in EncodeOutcast, add dummy copy out resolve.
        for (auto &op : callList) {
            if (!copyOutResolveSuccIndexListDict.count(op)) {
                copyOutResolveSuccIndexListDict[op] = std::vector<int>({0});
            }

            if (GetCoreType(op) == static_cast<int>(CoreType::HUB)) {
                hubOpCount++;
            }
        }
    }

    void Init(DevAscendFunction *devFunc, const EncodeDevAscendFunctionParam &param, bool fillContent) {
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(&devFunc->data) - reinterpret_cast<uintdevptr_t>(devFunc);
        DevAscendFunctionPredInfo predInfo = {totalZeroPred, totalZeroPredAIV, totalZeroPredAIC, totalZeroPredHub,
            totalZeroPredAicpu};
        devFunc->sourceFunc = nullptr;
        devFunc->getInputDataCount = valueDependDesc.getInputDataCount;
        devFunc->getTensorDataCount = valueDependDesc.getTensorDataCount;
        devFunc->hubOpCount_ = hubOpCount;
        devFunc->InitIncastOutcastAttr(initOffset, incastList, outcastList, fillContent);
        devFunc->InitOperationDynamicField(initOffset, predInfo, outcastStitchCount, calleeHashIndexDict,
            expressionTable, callList, incastList, outcastList, callOpSuccDict, fillContent);
        devFunc->InitRawTensorAndMemoryRequirement(initOffset, incastRawTensorList, outcastRawTensorList,
            rawTensorList, rawMagicToRawTensor, rawAttrs, param, expressionTable, fillContent);
        devFunc->InitTensor(initOffset, tensorList, rawTensorList, fillContent);
        devFunc->InitOperation(
                initOffset, expressionTable, callList, tensorList, rawTensorList,
                callOpPredDict, callOpSuccDict, calleeHashIndexDict, outcastStitchIndexList,
                noPredOpList, noSuccOpList, copyOutResolveSuccIndexListDict, fillContent);
        devFunc->InitWrapInfo(initOffset, callList, fillContent);
        devFunc->InitIncastOutcast(initOffset, incastList, outcastList, tensorList, incastOpAttrDict, outcastOpAttrDict, param, rawName, fillContent);
    }
};

void EncodeDevAscendFunction(Function *dyndev, const EncodeDevAscendFunctionParam &param, uint64_t &offset, DevAscendFunction *base) {
    EncodeDevAscendFunctionInfo encodeInfo(dyndev, param.calleeHashIndexDict, param.cceCodeInfoList, param.expressionTable, param.devRoot);

    if (base == nullptr) {
        DevAscendFunction devfunc;
        encodeInfo.Init(&devfunc, param, false);
        offset = devfunc.GetSize();
    } else {
        encodeInfo.Init(base, param, true);
        offset = base->GetSize();
    }
}

void DevAscendProgram::InitSymbolTable(
        uintdevptr_t &initOffset, SymbolicSymbolTable *symbolTableInput, bool fillContent) {
    symbolTable.HostInitDataSizeOffset(initOffset, symbolTableInput->GetSymbolTable().size());

    symbolTableNameList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t index = 0; index < symbolTableInput->GetSymbolTable().size(); index++) {
        std::string name = symbolTableInput->GetSymbolTable()[index];
        ONFILLCONTENT {
            symbolTable[index].index = index;
        };
        ONFILLCONTENT {
            symbolTable[index].name.HostAssignRangeOffsetSize(symbolTableNameList, offset, name.size());
            memcpy_s(symbolTable[index].name.Data(), symbolTable[index].name.size(), name.c_str(), name.size());
        }
        offset += ALIGN_UP(name.size(), sizeof(uint64_t));
    }
    symbolTableNameList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitExpressionTableBinary(
        uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &expressionTableBinaryListInput, bool fillContent) {
    expressionTableOffsetList.HostInitDataSizeOffset(initOffset, expressionTableBinaryListInput.size());
    preGuardPage.HostInitDataSizeOffset(initOffset, PAGE_SIZE);
    ONFILLCONTENT { memset_s(preGuardPage.Data(), PAGE_SIZE, 0, PAGE_SIZE); }

    expressionTableBinary.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < expressionTableBinaryListInput.size(); i++) {
        ONFILLCONTENT { expressionTableOffsetList[i] = offset; }
        ONFILLCONTENT { memcpy_s(expressionTableBinary.Data() + offset, expressionTableBinaryListInput[i].size(), expressionTableBinaryListInput[i].data(), expressionTableBinaryListInput[i].size()); }
        offset += expressionTableBinaryListInput[i].size();
    }
    expressionTableBinary.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitControlFlowBinary(
        uintdevptr_t &initOffset,
        const std::vector<uint8_t> &hostControlFlowBinaryInput, const std::vector<uint8_t> &devControlFlowBinaryInput,
        bool fillContent) {
    uint64_t alignedHostControlFlowBinaryInputSize = ALIGN_UP(hostControlFlowBinaryInput.size(), sizeof(uint64_t));
    hostControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedHostControlFlowBinaryInputSize);
    ONFILLCONTENT { memcpy_s(hostControlFlowBinary.Data(), hostControlFlowBinaryInput.size(), hostControlFlowBinaryInput.data(), hostControlFlowBinaryInput.size()); }

    uint64_t alignedDevControlFlowBinaryInputSize = ALIGN_UP(devControlFlowBinaryInput.size(), sizeof(uint64_t));
    devControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedDevControlFlowBinaryInputSize);
    ONFILLCONTENT { memcpy_s(devControlFlowBinary.Data(), devControlFlowBinaryInput.size(), devControlFlowBinaryInput.data(), devControlFlowBinaryInput.size()); }
}
void DevAscendProgram::InitDevEncodeList(
        uintdevptr_t &initOffset,
        const std::vector<std::vector<uint8_t>> &devEncodeListInput,
        bool fillContent) {
    devEncodeList.HostInitDataSizeOffset(initOffset, devEncodeListInput.size());
    devEncodeDataList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t index = 0; index < devEncodeListInput.size(); index++) {
        uint64_t alignedDevEncodeListInputSize = ALIGN_UP(devEncodeListInput[index].size(), sizeof(uint64_t));
        ONFILLCONTENT {
            devEncodeList[index].HostAssignRangeOffsetSize(devEncodeDataList, offset, alignedDevEncodeListInputSize);
        };
        ONFILLCONTENT {
            memcpy_s(devEncodeList[index].Data(), devEncodeList[index].size(), devEncodeListInput[index].data(),
                devEncodeListInput[index].size());
        };
        offset += alignedDevEncodeListInputSize;
    }
    devEncodeDataList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitCceCodeList(uintdevptr_t &initOffset, const std::vector<CceCodeInfo> &cceInfo,
                                       bool fillContent) {
    cceCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, 0);
    size_t offset = 0;
    for (size_t index = 0; index < cceInfo.size(); index++) {
        ONFILLCONTENT {
            cceCodeList[index].coreType = cceInfo[index].coreType;
            cceCodeList[index].psgId = cceInfo[index].psgId;
            cceCodeList[index].funcHash = cceInfo[index].funcHash;
            cceCodeList[index].wrapVecId = cceInfo[index].wrapVecId;
            cceCodeList[index].mixResourceType = cceInfo[index].mixResourceType;
            auto dataLen = cceInfo[index].aicpuLeafCode.size();
            aicpuLeafCodeList[index].aicpuLeafCode.HostAssignRangeOffsetSize(aicpuLeafCodeDataList, offset, dataLen);
            (void)memcpy_s(aicpuLeafCodeList[index].aicpuLeafCode.Data(), sizeof(int32_t) * dataLen,
                cceInfo[index].aicpuLeafCode.data(), sizeof(int32_t) * dataLen);
        };
        offset += cceInfo[index].aicpuLeafCode.size();
    }
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, offset);
}

void DevAscendProgram::InitPrefetchInfoList(uintdevptr_t &initOffset, const std::vector<L2Info> &l2InfoList,
    bool fillContent) {
    prefetchInfoList.HostInitDataSizeOffset(initOffset, l2InfoList.size());
    for (size_t index = 0; index < l2InfoList.size(); index++) {
        ONFILLCONTENT { memcpy_s(&prefetchInfoList[index], sizeof(PrefetchInfo), &l2InfoList[index], sizeof(L2Info)); };
    }
    return;
}

void DevAscendProgram::InitDisableL2List(uintdevptr_t &initOffset, const std::vector<uint8_t> &disableL2,
                                         bool fillContent) {
  disableL2List.HostInitDataSizeOffset(initOffset, disableL2.size());
  ONFILLCONTENT { (void)memcpy_s(disableL2List.Data(), disableL2.size(), disableL2.data(), disableL2.size()); };
  return;
}

void DevAscendProgram::InitStartArgsABIParamList(
        uintdevptr_t &initOffset,
        const std::vector<int> &tStartArgsInputTensorSlotIndexList,
        const std::vector<int> &tStartArgsOutputTensorSlotIndexList,
        const std::vector<int> &tStartArgsInputSymbolIndexList,
        const std::vector<SymbolHandler> &tStartArgsSymbolHandlerList,
        const std::vector<int> &tAsembleSlotIndexList,
        const std::vector<int> &tInplaceSlotIndexList,
        bool fillContent) {
    this->startArgsInputTensorSlotIndexList.HostInitDataSizeOffset(initOffset, tStartArgsInputTensorSlotIndexList.size());
    this->startArgsOutputTensorSlotIndexList.HostInitDataSizeOffset(initOffset, tStartArgsOutputTensorSlotIndexList.size());
    this->startArgsInputSymbolIndexList.HostInitDataSizeOffset(initOffset, tStartArgsInputSymbolIndexList.size());
    this->startArgsSymbolHandlerList.HostInitDataSizeOffset(initOffset, tStartArgsSymbolHandlerList.size());
    this->assembleSlotIndexList.HostInitDataSizeOffset(initOffset, tAsembleSlotIndexList.size());
    this->outputInplaceSlotList.HostInitDataSizeOffset(initOffset, tInplaceSlotIndexList.size());

    ONFILLCONTENT {
        for (size_t index = 0; index < tStartArgsInputTensorSlotIndexList.size(); index++) {
            this->startArgsInputTensorSlotIndexList[index] = tStartArgsInputTensorSlotIndexList[index];
        }
        for (size_t index = 0; index < tStartArgsOutputTensorSlotIndexList.size(); index++) {
            this->startArgsOutputTensorSlotIndexList[index] = tStartArgsOutputTensorSlotIndexList[index];
        }
        for (size_t index = 0; index < tStartArgsInputSymbolIndexList.size(); index++) {
            this->startArgsInputSymbolIndexList[index] = tStartArgsInputSymbolIndexList[index];
        }
        for (size_t index = 0; index < tStartArgsSymbolHandlerList.size(); index++) {
            this->startArgsSymbolHandlerList[index] = tStartArgsSymbolHandlerList[index];
        }
        for (size_t index = 0; index < tAsembleSlotIndexList.size(); index++) {
            this->assembleSlotIndexList[index] = tAsembleSlotIndexList[index];
        }
        for (size_t index = 0; index < tInplaceSlotIndexList.size(); index++) {
            this->outputInplaceSlotList[index] = tInplaceSlotIndexList[index];
        }
    }
}

static void InitPartialUpdateCellMatch(
        const std::vector<const DevAscendFunctionOutcast *> &outcastList,
        DevCellMatchTableDesc *partialUpdateCellMatchTableDesc) {
    std::vector<int> tensorShape;
    for (size_t index = 0; index < outcastList.size(); index++) {
        std::vector<int> outcastShape;
        for (int d = 0; d < outcastList[index]->dim; d++) {
            outcastShape.push_back(outcastList[index]->cellMatchTableDesc.GetCellShape(d) * outcastList[index]->cellMatchTableDesc.GetStrideShape(d));
            if (index > 0) {
               tensorShape[d] = std::max(tensorShape[d], outcastShape[d]);
            }
        }
        if (index == 0) {
            tensorShape = outcastShape;
        }
    }

    std::vector<int> cellShape;
    for (size_t d = 0; d < tensorShape.size(); d++) {
        int dim = 0;
        for (size_t index = 0; index < outcastList.size(); index++) {
            if (index == 0) {
                dim = outcastList[index]->cellMatchTableDesc.GetCellShape(d);
            } else if (dim != -1) {
                dim = std::gcd(dim, outcastList[index]->cellMatchTableDesc.GetCellShape(d));
            } else {
                ASSERT(outcastList[index]->cellMatchTableDesc.GetCellShape(d) == -1) << "Invalid cell shape for outcastList[" << index << "], dimension: " <<
                       d << ", expected -1, got " << outcastList[index]->cellMatchTableDesc.GetCellShape(d);
            }
        }
        cellShape.push_back(dim);
    }
    partialUpdateCellMatchTableDesc->SetCellShape(cellShape);

    std::vector<int> strideShape;
    for (size_t i = 0; i < tensorShape.size(); i++) {
        strideShape.push_back(cellShape[i] != 0 ? tensorShape[i] / cellShape[i] : 0);
    }
    partialUpdateCellMatchTableDesc->SetStrideShape(strideShape);
}

void DevAscendProgram::InitPartialUpdateSlot(
        uintdevptr_t &initOffset,
        const std::vector<std::vector<uint8_t>> &devEncodeListInput,
        const std::unordered_map<Function *, int> &rootFuncKeyDict,
        const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootIncastDict,
        const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootOutcastDict,
        const std::vector<int> &tPartialUpdateSlotIndexList,
        bool fillContent) {
    (void)slotRootIncastDict;
    (void)slotRootOutcastDict;
    this->partialUpdateList.HostInitDataSizeOffset(initOffset, slotSize);

    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
    int totalCellMatchSize = 0;
    for (size_t index = 0; index < tPartialUpdateSlotIndexList.size(); index++) {
        std::vector<const DevAscendFunctionOutcast *> outcastList;
        auto slotIndex = tPartialUpdateSlotIndexList[index];
        ASSERT(slotRootOutcastDict.count(slotIndex)) << "slotIndex: " << slotIndex << " not found in slotRootOutcastDict";
        for (auto &[root, outcastIndex] : slotRootOutcastDict.find(slotIndex)->second) {
            ASSERT(rootFuncKeyDict.count(root)) << "root: " << root << " not found in rootFuncKeyDict";
            int funcKey = rootFuncKeyDict.find(root)->second;
            DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(const_cast<uint8_t *>(devEncodeListInput[funcKey].data()));
            outcastList.push_back(&devFunc->GetOutcast(outcastIndex));
        }
        DevCellMatchTableDesc partialUpdateCellMatchTableDesc;
        InitPartialUpdateCellMatch(outcastList, &partialUpdateCellMatchTableDesc);
        size_t tableSize = partialUpdateCellMatchTableDesc.GetStride(0);

        ONFILLCONTENT {
            auto &partialUpdate = At(partialUpdateList, slotIndex);
            partialUpdate.slotIndex = slotIndex;
            partialUpdate.cellMatchTableDesc = partialUpdateCellMatchTableDesc;
            partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignRangeOffsetSize(cellMatchRuntimePartialUpdateTableList, totalCellMatchSize, tableSize);
            auto tableData = partialUpdate.cellMatchRuntimePartialUpdateTable.Data();
            for (size_t j = 0; j < tableSize; j++) {
                tableData[j] = AICORE_TASK_INIT;
            }
        }
        totalCellMatchSize += tableSize;
    }
    totalCellMatchSize = ALIGN_UP(totalCellMatchSize, sizeof(uint64_t) * FRIENDLY_CACHE_ALIGN_U64_SIZE / sizeof(uint64_t));
    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, totalCellMatchSize);
}

struct ControlFlowCacheFactor {
    const std::string name;
    uint64_t deviceElementFactor;
    uint64_t rootElementFactor;
    uint64_t leafElementFactor;
    ControlFlowCacheFactor(const std::string &name_, uint64_t device, uint64_t root, uint64_t leaf)
      : name(name_), deviceElementFactor(device), rootElementFactor(root), leafElementFactor(leaf) {}
};

static int EstimatedStitchingCount() {
    int value = config::GetRuntimeOption<int>(STITCH_FUNCTION_OUTCAST_MEMORY);
    ASSERT(value > 0) << "Invalid value for STITCH_FUNCTION_OUTCAST_MEMORY: " << value << ", must be greater than 0";
    return value;
}

static int WorkspaceRecyclePeriod() {
    int value = config::GetRuntimeOption<int>(STITCH_FUNCTION_INNER_MEMORY);
    ASSERT(value > 0) << "Invalid value for STITCH_FUNCTION_INNER_MEMORY: " << value << ", must be greater than 0";
    return value;
}

void DevAscendProgram::InitControlFlowCache(
        uintdevptr_t &initOffset,
        const std::shared_ptr<DyndevFunctionAttribute> &dyndevAttr,
        bool fillContent) {
    (void)fillContent;

    uint64_t maxDuppedDataAllocSize = 0;
    uint64_t maxIncastOutcastCount = 0;
    for (size_t index = 0; index < dyndevAttr->devEncodeList.size(); index++) {
        std::vector<uint8_t> &devEncode = dyndevAttr->devEncodeList[index];
        const DevAscendFunction *devFunc = reinterpret_cast<const DevAscendFunction *>(devEncode.data());
        uint64_t duppedDataAllocSize = devFunc->GetDuppedDataAllocSize();
        if (maxDuppedDataAllocSize < duppedDataAllocSize) {
            maxDuppedDataAllocSize = duppedDataAllocSize;
        }
        uint64_t incastOutcastCount = devFunc->GetIncastSize() + devFunc->GetOutcastSize();
        if (maxIncastOutcastCount < incastOutcastCount) {
            maxIncastOutcastCount = incastOutcastCount;
        }
    }
    uint64_t totalSize = config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE);

    initOffset = ALIGN_UP(initOffset, alignof(DevTensorData));
    controlFlowCache.inputTensorDataList.HostInitDataSizeOffset(initOffset, dyndevAttr->startArgsInputTensorList.size());
    controlFlowCache.outputTensorDataList.HostInitDataSizeOffset(initOffset, dyndevAttr->startArgsOutputTensorList.size());

    uint64_t slottedCount = slotSize * (std::min(EstimatedStitchingCount(), (int)MAX_CACHED_FUNC_NUM) + SLOTS_NEED_ALLOC_SIZE);
    controlFlowCache.runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.HostInitDataSizeOffset(initOffset, slottedCount);

    controlFlowCache.runtimeBackup.slotContext.slotList.HostInitDataSizeOffset(initOffset, slotSize);
    controlFlowCache.runtimeBackup.workspace.runtimeOutcastTensorPool.HostInitDataSizeOffset(initOffset, runtimeOutcastPoolSize);

    initOffset = ALIGN_UP(initOffset, alignof(DynFuncHeader *));
    controlFlowCache.deviceTaskCacheList.HostInitDataSizeOffset(initOffset, DEFAULT_CACHE_DEVICE_TASK_NUM);//10000
    controlFlowCache.cacheData.HostInitDataSizeOffset(initOffset, totalSize);
    controlFlowCache.isRecording = false;
    controlFlowCache.isRecordingStopped= false;
    controlFlowCache.isActivated = false;
    controlFlowCache.deviceTaskCount = 0;
    controlFlowCache.deviceTaskSkippedCount = 0;
    controlFlowCache.cacheDataOffset = 0;
    controlFlowCache.workspaceAddr = 0;
}

struct EncodeDevAscendProgramInfo {
    Function *func;
    std::shared_ptr<DyndevFunctionAttribute> dyndevAttr;
    uint64_t getTensorDataCount = 0;
    uint64_t getInputDataCount = 0;

    explicit EncodeDevAscendProgramInfo(Function *tfunc) : func(tfunc) {
        ASSERT(func->GetDyndevAttribute() != nullptr) << "DyndevAttribute is null for function: " << func;
        dyndevAttr = func->GetDyndevAttribute();
    }

    void Init(DevAscendProgram *devProg, bool fillContent) {
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(devProg->data);
        devProg->slotSize = dyndevAttr->inoutLink.totalSlot;
        devProg->runtimeOutcastPoolSize = dyndevAttr->inoutLink.totalSlot * (MAX_CACHED_FUNC_NUM + 1);
        devProg->assembleSlotSize = dyndevAttr->inoutLink.assembleSlotIndexList.size();
        devProg->InitSymbolTable(initOffset, &dyndevAttr->symbolTable, fillContent);
        devProg->InitExpressionTableBinary(initOffset, dyndevAttr->expressionTableBinaryList, fillContent);
        uint64_t expressionTableSize = 0;
        for (auto &[root, exprTable] : dyndevAttr->exprTableDictGroup.devRootCoaDict) {
            (void) root;
            expressionTableSize = std::max(expressionTableSize, (uint64_t)exprTable.GetPrimaryExpressionSize());
        }
        devProg->expressionTableSize = expressionTableSize;
        devProg->InitControlFlowBinary(initOffset, dyndevAttr->hostControlFlowBinary, dyndevAttr->devControlFlowBinary,
            fillContent);
        devProg->InitDevEncodeList(initOffset, dyndevAttr->devEncodeList, fillContent);
        devProg->InitCceCodeList(initOffset, dyndevAttr->cceCodeInfo, fillContent);
        devProg->InitStartArgsABIParamList(
            initOffset,
            dyndevAttr->inoutLink.inputSlotIndexList,
            dyndevAttr->inoutLink.outputSlotIndexList,
            dyndevAttr->startArgsInputSymbolIndexList,
            dyndevAttr->startArgsSymbolHandlerList,
            dyndevAttr->inoutLink.assembleSlotIndexList,
            dyndevAttr->inoutLink.inplaceSlotIndexList,
            fillContent);
        devProg->InitPartialUpdateSlot(
                initOffset,
                dyndevAttr->devEncodeList,
                dyndevAttr->rootFuncKeyDict,
                dyndevAttr->slotRootIncastDict,
                dyndevAttr->slotRootOutcastDict,
                dyndevAttr->inoutLink.partialUpdateSlotIdexList,
                fillContent);
        devProg->InitPrefetchInfoList(initOffset, dyndevAttr->l2InfoList, fillContent);
        devProg->commGroupNum = static_cast<uint64_t>(dyndevAttr->commGroupNames.size());
        devProg->InitDisableL2List(initOffset, dyndevAttr->disableL2List, fillContent);

        // control flow cache is always at the back of the program. So it should be the last.
        devProg->InitControlFlowCache(initOffset, dyndevAttr, fillContent);
        devProg->dataSize = initOffset - reinterpret_cast<uintdevptr_t>(devProg->data);
        ASSERT(reinterpret_cast<uint8_t *>(devProg->controlFlowCache.cacheData.end()) == reinterpret_cast<uint8_t *>(initOffset)) << "controlFlowCache.cacheData.end()"
               " does not match initOffset, expected " << reinterpret_cast<uint8_t *>(initOffset) <<
               ", got " << reinterpret_cast<uint8_t *>(devProg->controlFlowCache.cacheData.end());
        ASSERT(devProg->GetSize() == sizeof(*devProg) + devProg->dataSize) << "devProg->GetSize() does not match expected size, expected: " <<
               sizeof(*devProg) + devProg->dataSize << ", got: " << devProg->GetSize();
    }
};

struct TensorWorkspaceResult {
    uint64_t rootInnerMem{0};
    uint64_t devTaskInnerExclusiveOutcastMem{0};
    uint64_t maxStaticOutcastMem{0};
    uint64_t devTaskBoundaryOutcastNum{0};
    uint64_t perCoreSpilledMem{0};
    SymbolicScalar maxDynamicAssembleOutcastMem;
    uint64_t totalExclusiveOutcastSlot{0};
    uint64_t totalAssembleOutcastSlot{0};
};

struct SlotInfo {
    bool asWriteSlot{false};
    RuntimeSlotKindSet kindSet;
    uint64_t maxAssembleDstMemReq{0};
    SymbolicScalar dynMemReq;
};

static std::vector<SlotInfo> MarkInputOutputAssembleSlots(DevAscendProgram &devProg) {
    std::vector<SlotInfo> slotInfoList(devProg.slotSize);
    std::vector<int> inputSlotIdxList = devProg.GetInputTensorSlotIndexList();
    for (int inputSlotIdx : inputSlotIdxList) {
        slotInfoList[inputSlotIdx].kindSet.Add(RuntimeSlotKind::INPUT);
    }
    std::vector<int> outputSlotIdxList = devProg.GetOutputTensorSlotIndexList();
    for (int outputSlotIdx : outputSlotIdxList) {
        slotInfoList[outputSlotIdx].kindSet.Add(RuntimeSlotKind::OUTPUT);
    }
    for (auto slotIdx : devProg.GetAssembleTensorSlotIndexList()) {
        slotInfoList[slotIdx].kindSet.Add(RuntimeSlotKind::ASSEMBLE_OUTCAST);
    }
    for (auto &&devEncodeData : devProg.devEncodeList) {
        DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(devEncodeData.Data());
        for (size_t outcastIdx = 0; outcastIdx < devFunc->GetOutcastSize(); outcastIdx++) {
            auto &toSlotList = devFunc->GetOutcast(outcastIdx).toSlotList;
            bool isInputOutputSlot = false;
            bool isAssembleOutcastSlot = false;
            for (size_t j = 0; j < toSlotList.size(); j++) {
                SlotInfo &slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                if (slotInfo.kindSet.Count(RuntimeSlotKind::INPUT) ||
                    slotInfo.kindSet.Count(RuntimeSlotKind::OUTPUT)) {
                    isInputOutputSlot = true;
                } else if (slotInfo.kindSet.Count(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
                    isAssembleOutcastSlot = true;
                }
            }
            if (isInputOutputSlot) {

            } else if (isAssembleOutcastSlot) {
                for (size_t j = 0; j < toSlotList.size(); j++) {
                    /* If multiple outcast slot reference the same assemble outcast tensor
                     * then these slots are treated as assemble slot, even if in later
                     * function they are assigned as exclusive outcast tensor
                     */
                    SlotInfo &slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                    slotInfo.kindSet.Add(RuntimeSlotKind::ASSEMBLE_OUTCAST);
                    slotInfo.maxAssembleDstMemReq = std::max(
                        slotInfo.maxAssembleDstMemReq,
                        devFunc->GetOutcastRawTensor(outcastIdx)->maxStaticMemReq);
                }
            } else {
                for (size_t j = 0; j < toSlotList.size(); j++) {
                    SlotInfo &slotInfo = slotInfoList[devFunc->At(toSlotList, j)];
                    slotInfo.kindSet.Add(RuntimeSlotKind::EXCLUSIVE_OUTCAST);
                }
            }
        }
    }
    return slotInfoList;
}

static bool IsInputOutputSlot(const std::vector<SlotInfo> &slotInfoList, DevAscendFunction *func, size_t idx) {
    auto &toSlotList = func->GetOutcast(idx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = func->At(toSlotList, j);
        if (slotInfoList[slotIdx].kindSet.Count(RuntimeSlotKind::INPUT) ||
            slotInfoList[slotIdx].kindSet.Count(RuntimeSlotKind::OUTPUT)) {
            return true;
        }
    }
    return false;
}

static bool IsAssembleSlot(std::vector<SlotInfo> &slots, DevAscendFunction *func, size_t idx) {
    auto &toSlotList = func->GetOutcast(idx).toSlotList;
    bool isAssemble = false;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = func->At(toSlotList, j);
        if (slots[slotIdx].kindSet.Count(RuntimeSlotKind::ASSEMBLE_OUTCAST)) {
            isAssemble = true;
            break;
        }
    }
    return isAssemble;
};

static int ParseUnrollTimes(const std::string &rawName) {
    const static std::string UNROLL_MARKS[2] = {"_LoopUnroll", "_Unroll"};
    int unrollTimes = 1;
    for (auto& unrollMask : UNROLL_MARKS) {
        auto unrollPos = rawName.rfind(unrollMask);
        if (unrollPos == std::string::npos) {
            continue;
        }
        std::string suffix = rawName.substr(unrollPos + unrollMask.length());
        if (std::isdigit(suffix.front())) {
            unrollTimes *= std::stoi(suffix);
        }
    }
    return unrollTimes;
}

static uint64_t CalcUnrolledRootBudget(uint64_t budget, int unrollTimes, int configMultiplier) {
    ASSERT(unrollTimes > 0) << "Invalid unrollTimes:  " << unrollTimes << ", must be greater than 0";
    if (unrollTimes >= configMultiplier) {
        return budget;
    }
    return AlignUp((budget + unrollTimes - 1) / unrollTimes, TENSOR_ADDR_ALIGNMENT) * configMultiplier;
}

static SymbolicScalar GetDynRawTensorSize(Function *dynFunc, int funcKey, int idx) {
    auto dynAttr = dynFunc->GetDyndevAttribute();
    Function *devRoot = nullptr;

    for (auto &func: dynAttr->funcGroup.devRootList) {
        if (dynAttr->funcGroup.devRootList.GetIndex(func) == funcKey) {
            devRoot = func;
            break;
        }
    }
    ASSERT(devRoot) << "func " << funcKey << " missing";

    auto rawTensor = devRoot->GetOutcast()[idx]->GetRawTensor();
    ASSERT(!rawTensor->GetDynRawShape().empty()) << "Not dynamic shape tensor";

    SymbolicScalar size = BytesOf(rawTensor->GetDataType());
    for (auto x : rawTensor->GetDynRawShape()) {
        size = size * x;
    }
    return size;
}

// Helper: process assemble outcast branch for a single outcast
static void ProcessAssembleOutcast(Function *func, DevAscendFunction *devFunc, size_t outIdx, std::vector<SlotInfo> &slots,
                                   uint64_t staticMemReq) {
    SymbolicScalar dynMemReq;
    // memoryRequirement == 0 means dynamic memory requirement
    if (devFunc->GetOutcastRawTensor(outIdx)->memoryRequirement == 0) {
        dynMemReq = GetDynRawTensorSize(func, devFunc->funcKey, outIdx);
    }
    auto &toSlotList = devFunc->GetOutcast(outIdx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = devFunc->At(toSlotList, j);
        if (dynMemReq.IsValid() || slots[slotIdx].dynMemReq.IsValid()) {
            if (!dynMemReq.IsValid()) {
                dynMemReq = staticMemReq;
            }
            if (!slots[slotIdx].dynMemReq.IsValid()) {
                slots[slotIdx].dynMemReq = slots[slotIdx].maxAssembleDstMemReq;
                slots[slotIdx].maxAssembleDstMemReq = 0;
            }
            slots[slotIdx].dynMemReq = std::max(dynMemReq, slots[slotIdx].dynMemReq);
        } else {
            slots[slotIdx].maxAssembleDstMemReq = std::max(slots[slotIdx].maxAssembleDstMemReq, staticMemReq);
        }
    }
}

// Helper: process exclusive outcast branch for a single outcast
static void ProcessExclusiveOutcast(DevAscendFunction *devFunc, size_t outIdx, std::vector<SlotInfo> &slots) {
    auto &toSlotList = devFunc->GetOutcast(outIdx).toSlotList;
    for (size_t j = 0; j < toSlotList.size(); j++) {
        int slotIdx = devFunc->At(toSlotList, j);
        // No output slot
        slots[slotIdx].asWriteSlot = true;
    }
}

// Helper: process a single DevAscendFunction's outcasts and update slot/memory accumulators
static void ProcessDevFunctionOutcasts(Function *func, DevAscendFunction *devFunc, std::vector<SlotInfo> &slots,
                                       uint64_t &maxExclusiveOutcastMem, uint64_t &maxRootInnerMem,
                                       uint64_t &maxDevTaskInnerExclusiveOutcastMem, uint64_t &maxPerCoreSpilledMem) {
    for (size_t i = 0; i < devFunc->GetOutcastSize(); i++) {
        if (IsInputOutputSlot(slots, devFunc, i)) {
            continue;
        }

        // maxStaticMemReq could be 0 when no need of independent allocation
        uint64_t staticMemReq = devFunc->GetOutcastRawTensor(i)->maxStaticMemReq;
        if (IsAssembleSlot(slots, devFunc, i)) {
            ProcessAssembleOutcast(func, devFunc, i, slots, staticMemReq);
        } else {
            ProcessExclusiveOutcast(devFunc, i, slots);
            maxExclusiveOutcastMem = std::max(maxExclusiveOutcastMem, staticMemReq);
        }
    }

    int unroll = ParseUnrollTimes(devFunc->GetRawName());
    uint64_t funcRootInnerMem = CalcUnrolledRootBudget(
        devFunc->rootInnerTensorWsMemoryRequirement, unroll, WorkspaceRecyclePeriod());
    uint64_t funcDevTaskInnerExclusiveOutcastMem = CalcUnrolledRootBudget(
        devFunc->exclusiveOutcastWsMemoryRequirement, unroll, EstimatedStitchingCount());

    maxRootInnerMem = std::max(maxRootInnerMem, funcRootInnerMem);
    maxDevTaskInnerExclusiveOutcastMem = std::max(maxDevTaskInnerExclusiveOutcastMem, funcDevTaskInnerExclusiveOutcastMem);
    maxPerCoreSpilledMem = std::max(maxPerCoreSpilledMem, static_cast<uint64_t>(devFunc->stackWorkSpaceSize));
}

// Helper: compute assemble-outcast memory aggregates from slots
static std::pair<uint64_t, SymbolicScalar> ComputeAssembleOutcastMem(const std::vector<SlotInfo> &slots) {
    uint64_t maxStaticAssembleOutcastMem = std::accumulate(slots.begin(), slots.end(), UINT64_C(0),
        [](uint64_t acc, const SlotInfo &slot) {
            return std::max(acc, (slot.kindSet.Count(RuntimeSlotKind::ASSEMBLE_OUTCAST) ? slot.maxAssembleDstMemReq : 0));
        });

    SymbolicScalar maxDynamicAssembleOutcastMem = std::accumulate(slots.begin(), slots.end(),
        SymbolicScalar(0), [](SymbolicScalar acc, const SlotInfo &slot) {
            return std::max(acc, slot.dynMemReq.IsValid() ? slot.dynMemReq : SymbolicScalar(0));
        });

    return {maxStaticAssembleOutcastMem, maxDynamicAssembleOutcastMem};
}

static TensorWorkspaceResult CalcTensorWorkspace(Function *func, DevAscendProgram &devProg) {
    std::vector<SlotInfo> slots = MarkInputOutputAssembleSlots(devProg);

    uint64_t maxRootInnerMem = 0;
    uint64_t maxDevTaskInnerExclusiveOutcastMem = 0;
    uint64_t maxExclusiveOutcastMem = 0;
    uint64_t maxPerCoreSpilledMem = 0;

    for (auto &&devEncodeData : devProg.devEncodeList) {
        DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(devEncodeData.Data());
        ProcessDevFunctionOutcasts(func, devFunc, slots, maxExclusiveOutcastMem, maxRootInnerMem,
                                   maxDevTaskInnerExclusiveOutcastMem, maxPerCoreSpilledMem);
    }

    auto [maxStaticAssembleOutcastMem, maxDynamicAssembleOutcastMem] = ComputeAssembleOutcastMem(slots);

    TensorWorkspaceResult res;
    res.maxStaticOutcastMem = std::max(maxExclusiveOutcastMem, maxStaticAssembleOutcastMem);
    res.rootInnerMem = maxRootInnerMem;
    res.devTaskInnerExclusiveOutcastMem = maxDevTaskInnerExclusiveOutcastMem;
    res.maxDynamicAssembleOutcastMem = maxDynamicAssembleOutcastMem;

    res.totalExclusiveOutcastSlot = std::count_if(slots.begin(), slots.end(), [](const SlotInfo &slot) {
        return slot.kindSet.Count(RuntimeSlotKind::EXCLUSIVE_OUTCAST);
    });
    res.totalAssembleOutcastSlot = std::count_if(slots.begin(), slots.end(), [](const SlotInfo &slot) {
        return slot.kindSet.Count(RuntimeSlotKind::ASSEMBLE_OUTCAST);
    });
    res.devTaskBoundaryOutcastNum = res.totalExclusiveOutcastSlot * SLOTS_NEED_ALLOC_SIZE +
        res.totalAssembleOutcastSlot * std::min(EstimatedStitchingCount(), (int)MAX_CACHED_FUNC_NUM);

    res.perCoreSpilledMem = AlignUp(maxPerCoreSpilledMem, TENSOR_ADDR_ALIGNMENT);

    return res;
}

static uint64_t CalcGeneralMetadataSlotWorkspace(DevAscendProgram *devProg) {
    uint64_t generalMetadataSlotSize = 0;
    uint64_t itemPoolMemSize = DeviceWorkspaceAllocator::CalcMetadataItemPoolMemSize(devProg);
    uint64_t vectorMemSize = DeviceWorkspaceAllocator::CalcMetadataVectorMemSize(devProg);
    uint64_t slotAllocatorMemSize = DeviceWorkspaceAllocator::CalcMetadataSlotAllocatorMemSize(devProg);
    ALOG_DEBUG_F("itemPoolMemSize is: %lu, vectorMemSize is: %lu, slotAllocatorMemSize is %lu,", 
                  itemPoolMemSize, vectorMemSize, slotAllocatorMemSize);
    static constexpr uint64_t AICPU_SLOT_STATIC_MEMSIZE = 2 * MEBI;
    generalMetadataSlotSize = itemPoolMemSize + vectorMemSize + 
                              slotAllocatorMemSize + AICPU_SLOT_STATIC_MEMSIZE;
    ALOG_DEBUG_F("workspace of generalMetadataSlotSize is %lu, ", generalMetadataSlotSize);
    return generalMetadataSlotSize;
}

static uint64_t CalcGeneralMetadataSlabWorkspace(DevAscendProgram *devProg) {
    DeviceWorkspaceAllocator workspace(devProg);
    uint64_t generalMetadataSlabSize = 0;
    uint32_t slabSize = workspace.CalcSlabMemObjmaxSize() * ALLOC_NUM_ONE_SLAB;
    uint32_t slabCapacity[ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT)];
    size_t objUsedNum [ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT)] {
        MAX_CACHED_FUNC_NUM, //DevFunctionDupped
        1,// DynFuncData
        1,// VecStitchList
        1,// DynDevTask
        READY_QUEUE_SIZE, //ReadyQue
        1,
        1,
    };
    workspace.CalculateSlabCapacityPerType(slabSize, slabCapacity,
    ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT));

    for (int i=0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
        ALOG_DEBUG_F("slabCapacity[%d] is %u", i, slabCapacity[i]);
        if (slabCapacity[i] == 0) {
            continue;
        }
        uint32_t requiredSlabNum = (objUsedNum[i] + slabCapacity[i] - 1) / slabCapacity[i];
        // alloc redundant slabpage for DuppedFunction and Readyque to prevent memory border situations
        if(i == ToUnderlying(WsAicpuSlabMemType::DUPPED_FUNC_DATA) ||
         i == ToUnderlying(WsAicpuSlabMemType::READY_QUE)) requiredSlabNum++;
        ALOG_DEBUG_F("requiredSlabNum[%d] is %u", i, requiredSlabNum);
        generalMetadataSlabSize += static_cast<uint64_t>(requiredSlabNum) * slabSize;
    }
    ALOG_DEBUG_F("generalMetadataSlabSize is %u", generalMetadataSlabSize);
    generalMetadataSlabSize = (generalMetadataSlabSize < GENERAL_METADATA_SIZE_MIN) ? GENERAL_METADATA_SIZE_MIN : generalMetadataSlabSize;
    return generalMetadataSlabSize;
}

static uint64_t CalcStitchWorkspace(DevAscendProgram &devProg) {
    (void)devProg;
    static constexpr uint64_t AICPU_STITCH_SIZE = 2 * MEBI;
    return AICPU_STITCH_SIZE;
}

static uint64_t DumpTensorWorkspace() {
#if DEBUG_INFINITE_LIFETIME
    static constexpr uint64_t DUMP_TENSOR_WORKSPACE = 8 * GIBI;
    return DUMP_TENSOR_WORKSPACE;
#else
    return 0;
#endif
}

void EncodeDevAscendProgram(Function *func, uint64_t &offset, DevAscendProgram *base) {
    EncodeDevAscendProgramInfo encodeInfo(func);

    if (base == nullptr) {
        DevAscendProgram devfunc;
        encodeInfo.Init(&devfunc, false);
        offset = devfunc.GetSize();
    } else {
        encodeInfo.Init(base, true);
        offset = base->GetSize();

        // Calc workspace size
        TensorWorkspaceResult tensorWsRes = CalcTensorWorkspace(func, *base);

        base->slottableOutcastSlotSize = tensorWsRes.totalExclusiveOutcastSlot + tensorWsRes.totalAssembleOutcastSlot;

        base->memBudget.tensor.rootInner = tensorWsRes.rootInnerMem;
        base->memBudget.tensor.devTaskInnerExclusiveOutcasts = tensorWsRes.devTaskInnerExclusiveOutcastMem;
        base->memBudget.tensor.maxStaticOutcastMem = tensorWsRes.maxStaticOutcastMem;
        base->memBudget.tensor.devTaskBoundaryOutcastNum = tensorWsRes.devTaskBoundaryOutcastNum;

        int32_t maxCoreNum = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 ? MAX_AICORE_NUM_3510 : MAX_AICORE_NUM_2210;
        base->memBudget.aicoreSpilled = tensorWsRes.perCoreSpilledMem * maxCoreNum;
        base->devArgs.machineConfig = func->paramConfigs_.machineConfig_;
        base->stitchFunctionNumInitial = func->paramConfigs_.stitchFunctionNumInitial_;
        base->stitchFunctionNumStep = func->paramConfigs_.stitchFunctionNumStep_;
        base->stitchFunctionsize = config::GetRuntimeOption<uint32_t>(STITCH_FUNCTION_SIZE);
        base->memBudget.metadata.general = CalcGeneralMetadataSlotWorkspace(base);
        base->memBudget.metadata.general += CalcGeneralMetadataSlabWorkspace(base);
        base->memBudget.metadata.stitchPool = CalcStitchWorkspace(*base);
        base->memBudget.debug.dumpTensor = DumpTensorWorkspace();

        func->GetDyndevAttribute()->maxDynamicAssembleOutcastMem = tensorWsRes.maxDynamicAssembleOutcastMem;
    }
}
} // namespace dynamic
} // namespace npu::tile_fwk
