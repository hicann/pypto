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
 * \file dev_encode_program.h
 * \brief
 */

#pragma once

#include "dev_encode_program_ctrlflow_cache.h"
#include "interface/tensor/symbol_handler.h"

namespace npu::tile_fwk {
class DyndevFunctionAttribute;
}

namespace npu::tile_fwk::dynamic {
struct DevAscendProgramSymbol {
    DevRelocVector<char> name;
    uint64_t index;
};

struct DevAscendProgram {
    // shadow definition in `aicore_runtime_manager.h`, make sure the first 4 members are the same
    DeviceArgs devArgs;
    uint64_t workspaceSize;
    uint64_t l2CacheOffset;
    uint64_t configKey;
    uint64_t hashKey;
    uint32_t slotSize;
    uint32_t runtimeOutcastPoolSize;
    uint32_t assembleSlotSize;
    uint32_t slottableOutcastSlotSize;
    struct {
        struct {
            // root func inner tensors
            uint64_t rootInner;
            // root func outcasts & non-dassemble-dst & DeviceTask inner tensors
            uint64_t devTaskInnerExclusiveOutcasts;
            // root func outcasts & non-dassemble-dst & DeviceTask boundary outcasts: MaxOutcastMem() * devTaskBoundaryOutcastNum
            uint64_t maxStaticOutcastMem;
            uint64_t maxDynamicAssembleOutcastMem;
            uint64_t devTaskBoundaryOutcastNum;

            uint64_t MaxOutcastMem() const {
                return std::max(maxStaticOutcastMem, maxDynamicAssembleOutcastMem);
            }

            uint64_t Total() const {
                uint64_t total = rootInner +                     // root func inner tensors
                    devTaskInnerExclusiveOutcasts +              // root func outcasts & non-dassemble-dst & DeviceTask inner tensors
                    MaxOutcastMem() * devTaskBoundaryOutcastNum; // root func outcasts & non-dassemble-dst & DeviceTask boundary outcasts
                static constexpr uint64_t ALIGNMENT_32K = 32 * 1024;
                return AlignUp(total, ALIGNMENT_32K);
            }
        } tensor;
        uint64_t aicoreSpilled;
        struct {
            uint64_t general;
            uint64_t stitchPool;

            uint64_t Total() const {
                return general + stitchPool;
            }
        } metadata;
        struct {
            uint64_t dumpTensor;
        } debug;

        uint64_t Total() const {
            return tensor.Total() + aicoreSpilled + debug.dumpTensor;
        }
    } memBudget;
    const void *controlFlowBinaryAddr{nullptr};
    uint64_t hcclContext[HCCL_GROUP_NUM];
    uint64_t commGroupNum{0};
    uint16_t stitchFunctionNumInitial{0};
    uint16_t stitchFunctionNumStep{0};
    uint32_t stitchFunctionsize{0};
    uint32_t ctrlFlowCacheSize{0};
    DevRelocVector<DevAscendProgramSymbol> symbolTable;
    DevRelocVector<char> symbolTableNameList;
    uint64_t expressionTableSize;
    DevRelocVector<uint64_t> expressionTableOffsetList;
    DevRelocVector<uint8_t> preGuardPage;
    DevRelocVector<uint8_t> expressionTableBinary;
    DevRelocVector<uint8_t> hostControlFlowBinary;  // compiled by system gcc (host arch)
    DevRelocVector<uint8_t> devControlFlowBinary;   // compiled by CANN gcc (ARM arch)
    DevRelocVector<uint8_t> postGuardPage;
    DevRelocVector<DevRelocVector<uint8_t>> devEncodeList;
    DevRelocVector<uint8_t> devEncodeDataList;
    DevRelocVector<DevCceBinary> cceCodeList;
    DevRelocVector<DevAicpuLeafBinary> aicpuLeafCodeList;
    DevRelocVector<int32_t> aicpuLeafCodeDataList;
    DevRelocVector<uint64_t> startArgsInputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsOutputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsInputSymbolIndexList;
    DevRelocVector<SymbolHandler> startArgsSymbolHandlerList;
    DevRelocVector<uint64_t> assembleSlotIndexList;
    DevRelocVector<uint64_t> outputInplaceSlotList;
    DevRelocVector<DevAscendProgramPartialUpdate> partialUpdateList;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTableList;
    DevRelocVector<PrefetchInfo> prefetchInfoList;
    DevRelocVector<uint8_t> disableL2List;
    DevControlFlowCache *ctrlFlowCacheAnchor{nullptr};
    DevControlFlowCache controlFlowCache;
#define programLastField                              controlFlowCache.cacheData
    uint64_t dataSize;
    uint8_t data[0];

    /*
     *      DevAscendProgramSymbol symbolTableData[]
     *      char symbolTableNameListData[]
     *      uint64_t expressionTableOffsetListData[]
     *      uint8_t preGuardPageData[PAGE_SIZE]
     *      uint8_t expressionTableBinaryData[]
     *      uint8_t hostControlFlowBinaryData[]
     *      uint8_t devControlFlowBinaryData[]
     *      DevRelocVector<uint8_t> devEncodeList[]
     *      uint8_t devEncodeDataList[]
     *      DevRelocVector<uint8_t> cceCodeList[]
     *      uint64_t startArgsInputTensorSlotIndexListData[]
     *      uint64_t startArgsOutputTensorSlotIndexListData[]
     *      uint64_t startArgsInputSymbolIndexListData[]
     *      SymbolHandler startArgsSymbolHandlerListData[]
     *      uint64_t assembleSlotIndexList[]
	 *      uint64_t outputInplaceSlotList[];
     *      DevAscendProgramPartialUpdate partialUpdateList[]
     *      DevAscendProgramSlot slotList[]
     */

    template <typename T>
    const T &At(const DevRelocVector<T> &localvec, int index) const {
        return localvec[index];
    }
    template <typename T>
    T &At(DevRelocVector<T> &localvec, int index) {
        return localvec[index];
    }

    void DumpCce(std::ostringstream& oss, int indent) const;

    void DumpControlFlow(const int indent, const bool dumpAddr, std::ostringstream& oss) const;

    void DumpExpressionTable(const int indent, const bool dumpAddr, std::ostringstream& oss) const;

    std::string Dump(const int indent = 0, const bool dumpAddr = false) const;

    void DumpFile(const std::string &filePath) const;

    std::vector<int> GetInputTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsInputTensorSlotIndexList, i));
        }
        return indexList;
    }
    std::vector<int> GetOutputTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsOutputTensorSlotIndexList, i));
        }
        return indexList;
    }

    std::vector<int> GetAssembleTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
            indexList.push_back(At(assembleSlotIndexList, i));
        }
        return indexList;
    }

    std::vector<int> GetPartialUpdateTensorSlotIndexList() const {
        const int &front = At(assembleSlotIndexList, 0);
        const int &back = At(assembleSlotIndexList, assembleSlotIndexList.size() - 1);
        std::vector<int> slotIndexList(&front, &back + 1);
        return slotIndexList;
    }

    std::tuple<const void *, uint64_t> GetDevControlFlowBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(devControlFlowBinary.Data()),
            (uint64_t)devControlFlowBinary.size());
    }

    std::tuple<const void *, uint64_t> GetHostControlFlowBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(hostControlFlowBinary.Data()),
            (uint64_t)hostControlFlowBinary.size());
    }

    std::tuple<const void *, uint64_t, const uint64_t *, uint64_t> GetExpressionTableBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(expressionTableBinary.Data()),
            static_cast<uint64_t>(expressionTableBinary.size()),
            expressionTableOffsetList.Data(),
            static_cast<uint64_t>(expressionTableOffsetList.size()));
    }

    uint64_t GetSymbolTableSize() const { return symbolTable.size(); }

    uint64_t GetExpressionTableSize() const { return expressionTableSize; }

    uint64_t GetFunctionSize() const { return devEncodeList.size(); }

    DevAscendFunction *GetFunction(int index) const {
        return reinterpret_cast<DevAscendFunction *>(const_cast<uint8_t *>(devEncodeList[index].Data()));
    }

    DevAscendFunction *GetFunctionByRawName(const std::string &rawName) const {
        for (size_t i = 0; i < GetFunctionSize(); i++) {
            DevAscendFunction *func = GetFunction(static_cast<int>(i));
            if (func->GetRawName() == rawName) {
                return func;
            }
        }
        return nullptr;
    }

    const DevCceBinary *GetCceBinary(int index) const { return &cceCodeList[index]; }
    const DevAicpuLeafBinary *GetAicpuLeafBinary(int index) const { return &aicpuLeafCodeList[index]; }

    DevControlFlowCache *GetControlFlowCache() { return ctrlFlowCacheAnchor; }

    template<typename Ty>
    typename Ty::ElementType *RelocOffset(intptr_t shift, void *&offset, Ty &list) {
        typename Ty::ElementType *ptr = reinterpret_cast<typename Ty::ElementType *>(offset);
        offset = (void *)((uintptr_t)(offset) + list.ElementSize() * list.size());
        list.DeviceRelocData(shift);
        return ptr;
    }

    void RelocProgram(uint64_t srcProgram, uint64_t dstProgram, bool relocFunc = false) {
        intptr_t shift = static_cast<int64_t>(dstProgram) - static_cast<int64_t>(srcProgram);
        void *offset = data;

        auto symbolTablePtr = RelocOffset(shift, offset, symbolTable);
        for (size_t i = 0; i < symbolTable.size(); i++) {
            symbolTablePtr[i].name.DeviceRelocData(shift);
        }

        RelocOffset(shift, offset, symbolTableNameList);
        RelocOffset(shift, offset, expressionTableOffsetList);
        RelocOffset(shift, offset, preGuardPage);
        RelocOffset(shift, offset, expressionTableBinary);
        RelocOffset(shift, offset, hostControlFlowBinary);
        RelocOffset(shift, offset, devControlFlowBinary);

        auto devEncodeListPtr = RelocOffset(shift, offset, devEncodeList);
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            devEncodeListPtr[i].DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, devEncodeDataList);
        RelocOffset(shift, offset, cceCodeList);
        auto aicpuLeafCodeListPtr = RelocOffset(shift, offset, aicpuLeafCodeList);
        for (size_t i = 0; i < aicpuLeafCodeList.size(); i++) {
            aicpuLeafCodeListPtr[i].aicpuLeafCode.DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, aicpuLeafCodeDataList);

        RelocOffset(shift, offset, startArgsInputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsOutputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsSymbolHandlerList);
        RelocOffset(shift, offset, startArgsInputSymbolIndexList);
        RelocOffset(shift, offset, assembleSlotIndexList);
        RelocOffset(shift, offset, outputInplaceSlotList);
        auto partialUpdateListPtr = RelocOffset(shift, offset, partialUpdateList);
        for (size_t i = 0; i < partialUpdateList.size(); i++) {
            partialUpdateListPtr[i].cellMatchRuntimePartialUpdateTable.DeviceRelocDataMaybeNull(shift);
        }
        RelocOffset(shift, offset, cellMatchRuntimePartialUpdateTableList);

        RelocOffset(shift, offset, prefetchInfoList);
        RelocOffset(shift, offset, disableL2List);
        if (relocFunc) {
            for (int i = 0; i < static_cast<int>(GetFunctionSize()); i++) {
                DevAscendFunction *func = GetFunction(i);
                func->Reloc(reinterpret_cast<uint64_t>(func), true);
            }
        }

        RelocOffset(shift, offset, controlFlowCache.inputTensorDataList);
        RelocOffset(shift, offset, controlFlowCache.outputTensorDataList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.slotContext.slotList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.workspace.runtimeOutcastTensorPool);
        RelocOffset(shift, offset, controlFlowCache.deviceTaskCacheList);
        RelocOffset(shift, offset, controlFlowCache.cacheData);
    }

    void ResetFromLaunch() {
        memset_s(&devArgs, sizeof(devArgs), 0, sizeof(devArgs));
        controlFlowBinaryAddr = nullptr;
        workspaceSize = 0;
        ctrlFlowCacheAnchor = nullptr;
        RelocProgram(reinterpret_cast<int64_t>(this), 0);
    }

    void ResetRerun() {
        uint64_t *RuntimePartialUpdateTable = cellMatchRuntimePartialUpdateTableList.Data();
        uint64_t RuntimePartialUpdateTableSize = cellMatchRuntimePartialUpdateTableList.DataSize();
        memset_s(RuntimePartialUpdateTable, RuntimePartialUpdateTableSize, 0, RuntimePartialUpdateTableSize);
    }

    struct DevRelocRange {
        template<typename T>
        DevRelocRange(const DevRelocVector<T> &v) : begin(reinterpret_cast<uintptr_t>(v.begin())), end(reinterpret_cast<uintptr_t>(v.end())) {}

        uintptr_t begin;
        uintptr_t end;
    };

    void RuntimeVerify(uintptr_t workspaceBegin, uintptr_t workspaceEnd) const {
        (void)workspaceBegin, (void)workspaceEnd;
        DEV_IF_VERBOSE_DEBUG {
        } else {
            return;
        }
        std::vector<DevRelocRange> rangeList = {
            symbolTable, // 0
            symbolTableNameList,
            expressionTableOffsetList,
            hostControlFlowBinary,
            devControlFlowBinary,
            devEncodeList, // 5
            devEncodeDataList,
            cceCodeList,
            aicpuLeafCodeList,
            aicpuLeafCodeDataList,
            startArgsInputTensorSlotIndexList, // 10
            startArgsOutputTensorSlotIndexList,
            assembleSlotIndexList,
            outputInplaceSlotList,
            partialUpdateList,
            cellMatchRuntimePartialUpdateTableList, // 15
            prefetchInfoList,
            disableL2List,
            controlFlowCache.inputTensorDataList,
            controlFlowCache.outputTensorDataList,
            controlFlowCache.runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList, // 20
            controlFlowCache.runtimeBackup.slotContext.slotList,
            controlFlowCache.runtimeBackup.workspace.runtimeOutcastTensorPool,
            controlFlowCache.deviceTaskCacheList,
            controlFlowCache.cacheData,
        };
        if ((uintptr_t)data != rangeList[0].begin) {
            DEV_ERROR("Assertion failed: data (0x%p) != rangeList[0].begin (0x%p)", data, (void*)rangeList[0].begin);
        }
        DEV_ASSERT((uintptr_t)data == rangeList[0].begin);
        if (rangeList[0].begin > rangeList[0].end) {
            DEV_ERROR("Assertion failed: rangeList[0].begin (0x%p) > rangeList[0].end (0x%p)",
                      (void*)rangeList[0].begin, (void*)rangeList[0].end);
        }
        DEV_ASSERT(rangeList[0].begin <= rangeList[0].end);
        for (size_t k = 1; k < rangeList.size(); k++) {
            if (rangeList[k - 1].end > rangeList[k].begin) {
                DEV_ERROR("Ranges overlap: range[%d].end (0x%p) > range[%d].begin (0x%p)",
                      (int)(k - 1), (void*)rangeList[k - 1].end,
                      (int)k, (void*)rangeList[k].begin);
            }
            if (rangeList[k].begin > rangeList[k].end) {
                DEV_ERROR("Invalid range: range[%d].begin (0x%p) > range[%d].end (0x%p)",
                      (int)k, (void*)rangeList[k].begin,
                      (int)k, (void*)rangeList[k].end);
            }
            DEV_ASSERT_MSG(rangeList[k - 1].end <= rangeList[k].begin, "range:%d->%d", (int)(k - 1), (int)(k));
            DEV_ASSERT_MSG(rangeList[k].begin <= rangeList[k].end, "range:%d", (int)k);
        }
        uintptr_t lastEnd = rangeList.back().end;
        uintptr_t dataEnd = (uintptr_t)(&data[dataSize]);
        if (lastEnd != dataEnd) {
            DEV_ERROR("Last range end does not match data end: rangeList.back().end (0x%p) != dataEnd (0x%p)",
                      (void*)lastEnd, (void*)dataEnd);
        }
        DEV_ASSERT(lastEnd == dataEnd);
    }

    uint64_t GetSize() const { return reinterpret_cast<uintptr_t>(programLastField.End()) - reinterpret_cast<uintptr_t>(this); }

private:
    friend struct EncodeDevAscendProgramInfo;

    void InitSymbolTable(
            uintdevptr_t &initOffset, SymbolicSymbolTable *symbolTableInput, bool fillContent);
    void InitExpressionTableBinary(
            uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &expressionTableBinaryListInput, bool fillContent);
    void InitControlFlowBinary(
            uintdevptr_t &initOffset,
            const std::vector<uint8_t> &hostControlFlowBinaryInput,
            const std::vector<uint8_t> &devControlFlowBinaryInput,
            bool fillContent);
    void InitDevEncodeList(
            uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &devEncodeListInput, bool fillContent);
    void InitCceCodeList(uintdevptr_t &initOffset, const std::vector<CceCodeInfo> &cceInfo, bool fillContent);
    void InitPrefetchInfoList(
            uintdevptr_t &initOffset, const std::vector<L2Info> &l2InfoList, bool fillContent);
    void InitDisableL2List(uintdevptr_t &initOffset, const std::vector<uint8_t> &disableL2, bool fillContent);
    void InitStartArgsABIParamList(uintdevptr_t &initOffset, const std::vector<int> &tStartArgsInputTensorSlotIndexList,
        const std::vector<int> &tStartArgsOutputTensorSlotIndexList,
        const std::vector<int> &tStartArgsInputSymbolIndexList,
        const std::vector<SymbolHandler> &tStartArgsSymbolHandlerList,
        const std::vector<int> &tAsembleSlotIndexList,
        const std::vector<int> &tInplaceSlotIndexList, bool fillContent);
    void InitPartialUpdateSlot(
            uintdevptr_t &initOffset,
            const std::vector<std::vector<uint8_t>> &devEncodeListInput,
            const std::unordered_map<Function *, int> &rootFuncKeyDict,
            const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootIncastDict,
            const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootOutcastDict,
            const std::vector<int> &tPartialUpdateSlotIndexList,
            bool fillContent);
    void InitControlFlowCache(
            uintdevptr_t &initOffset,
            const std::shared_ptr<DyndevFunctionAttribute> &dyndevAttr,
            bool fillContent);
};
}