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
 * \file dev_workspace.h
 * \brief
 */

#ifndef DEV_WORKSPACE_H
#define DEV_WORKSPACE_H

#include "dev_start_args.h"
#include "device_task.h"
#include "item_pool.h"
#include "spsc_queue.h"
#include "../machine_ws_intf.h"
#include "allocator/allocators.h"
#include "machine/device/dynamic/device_perf.h"

namespace npu::tile_fwk::dynamic {
inline constexpr int64_t TENSOR_ADDR_ALIGNMENT = 512;
inline constexpr uint32_t SUBMMIT_TASK_QUE_SIZE = 32;
class DeviceWorkspaceAllocator {
public:
    DeviceWorkspaceAllocator() = default;
    explicit DeviceWorkspaceAllocator(DevAscendProgram *base) : devProg_(base) {}
    ~DeviceWorkspaceAllocator() = default;
    void Init(DevStartArgs *args) {
        uintdevptr_t baseAddr = args->contextWorkspaceAddr;
        DevAscendProgram *devProg = args->devProg;

        // Host coherent allocators MUST be initialized EARLIEST since some other allocators might depend on them
        InitMetadataAllocators(devProg);

        InitAICoreSpilledMemory(baseAddr, devProg);
        baseAddr += devProg->memBudget.aicoreSpilled;

        // dassembleDests contains dynamic workspace, put it to the end
        InitTensorAllocators(baseAddr, devProg->memBudget.tensor.Total(), devProg);
        baseAddr += devProg->memBudget.tensor.Total();

#if DEBUG_INFINITE_LIFETIME
        dumpTensorWsAllocator_.InitTensorAllocator(baseAddr, devProg->memBudget.debug.dumpTensor);
        DEV_DEBUG("[DumpTensor] dumpTensorWsAllocator_: ptr=0x%lx, size=%lu",
                  baseAddr, devProg->memBudget.debug.dumpTensor);
        baseAddr += devProg->memBudget.debug.dumpTensor;

        // Allocate 512 for address alignment
        dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.Malloc(TENSOR_ADDR_ALIGNMENT).As<uint64_t>();
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
#endif
        SetupVector(slotMemToBeFree_);
        slotMemToBeFree_.reserve(devProg->memBudget.tensor.devTaskBoundaryOutcastNum);

        devProg_ = devProg;

        assembleSlotBuffer_ = metadataAllocators_.general.Allocate<WsAllocation>(devProg_->assembleSlotSize).As<WsAllocation>();
    }

    uintdevptr_t StackWorkspaceAddr() const { return stackWorkspaceBase_; }
    uint64_t StandardStackWorkspacePerCore() const { return standardStackWorkspacePerCore_; }

#if DEBUG_INFINITE_LIFETIME
    uintdevptr_t DumpTensorWsBaseAddr() const { return dumpTensorWsAllocator_.MemBaseAddr(); }
    uint64_t DumpTensorWsSize() const { return dumpTensorWsAllocator_.Capacity(); }
#endif
    template <typename T, WsMemCategory category, typename WsAllocator_T>
    void SetupVector(Vector<T, category, WsAllocator_T> &vector) {
        if constexpr (std::is_same_v<WsAllocator_T, npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>) {
            vector.InitAllocator((*this));
        } else {
            vector.InitAllocator(metadataAllocators_.general);
        }
    }

    template <typename T, WsMemCategory category>
    void SetupItemPool(ItemPool<T, category> &pool, size_t count) {
        pool.Init(metadataAllocators_.general, count);
    }

    WsMemoryState VerifyTensorMemoryState(uintdevptr_t ptr, size_t size) const {
        return tensorWsVerifier_.Verify(ptr, size);
    }

    void VerifyStitchedListMemory(DevStartArgs &args, const DevAscendFunctionDupped *stitchedList, size_t size) {
        struct MemoryInfo {
            uintdevptr_t ptr;
            size_t size;
            DevAscendFunctionDupped dup;
            size_t stitchedListIndex;
            size_t rawIndex;

            void DumpError() const {
                std::string ioPropertyDump;
                switch (dup.GetSource()->GetRawTensor(rawIndex)->ioProperty) {
                    case DevIOProperty::ROOT_INCAST:
                        ioPropertyDump = " (Root Incast)";
                        break;
                    case DevIOProperty::ROOT_OUTCAST:
                        ioPropertyDump = " (Root Outcast)";
                        break;
                    default:
                        break;
                }
                DEV_ERROR("  Func (%2zu) %16s rawTensor[%2zu], @%" PRIx64 " [%zu bytes]%s.",
                    stitchedListIndex, dup.GetSource()->GetRawName(), rawIndex, ptr, size,
                    ioPropertyDump.c_str());
            }
        };

        bool verificationSuccess = true;

        std::set<uintdevptr_t> inoutAddr;
        for (int i = 0; i < args.GetInputTensorSize(); i++) {
            inoutAddr.insert(args.GetInputTensor(i).address);
        }
        for (int i = 0; i < args.GetOutputTensorSize(); i++) {
            inoutAddr.insert(args.GetOutputTensor(i).address);
        }

        for (size_t i = 0; i < size; i++) {
            const auto &dup = stitchedList[i];

            auto isValidWsTensor = [&](uintdevptr_t ptr, size_t memSize) {
                return slotVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    dassembleDestsTensorVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    rootInnerWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    devTaskInnerExclusiveOutcastsWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE;
            };

            size_t rawTensorCount = dup.GetSource()->GetRawTensorSize();
            for (size_t j = 0; j < rawTensorCount; j++) {
                auto *rawTensor = dup.GetSource()->GetRawTensor(j);
                auto memReq = rawTensor->GetMemoryRequirement(dup.GetExpressionAddr());
                MemoryInfo memInfo {
                    dup.GetRawTensorAddr(j),
                    // For workspace tensors, the memoryRequirement property is deprecated
                    rawTensor->ioProperty == DevIOProperty::NONE ? 0 : memReq,
                    dup,
                    i,
                    j,
                };
                switch (VerifyTensorMemoryState(memInfo.ptr, memInfo.size)) {
                    case WsMemoryState::INSIDE:
                        if (!isValidWsTensor(memInfo.ptr, memInfo.size)) {
                            DEV_ERROR("Invalid workspace tensor (not completely inside any workspace segment):");
                            memInfo.DumpError();
                            verificationSuccess = false;
                        }
                        break;
                    case WsMemoryState::CROSS_BOUNDARY:
                        DEV_ERROR("Memory crossing workspace boundary:");
                        memInfo.DumpError();
                        verificationSuccess = false;
                        break;
                    default:
                        if (!inoutAddr.count(memInfo.ptr)) {
                            DEV_ERROR("Non input/output tensor outside of workspace:");
                            memInfo.DumpError();
                            verificationSuccess = false;
                        }
                        break;
                }
            }
        }
        if (!verificationSuccess) {
            DEV_ERROR("verification failed");
        }
        DEV_ASSERT(verificationSuccess);
    }

private:
    bool TryAllocateFuncWs(DevAscendFunctionDupped dup, uint64_t rootInnerSize, WsAllocatorCounter *dfxCounter = nullptr) {
        if (!tensorAllocators_.rootInner.CanAllocate(rootInnerSize)) {
            tensorAllocators_.rootInner.ResetPool();
            if (!tensorAllocators_.rootInner.CanAllocate(rootInnerSize)) {
                DEV_DEBUG("Can not AllocateFuncWs, size=%lu", rootInnerSize);
            }
            DEV_DEBUG_ASSERT(tensorAllocators_.rootInner.CanAllocate(rootInnerSize));
        }
        WsAllocation allocation = tensorAllocators_.rootInner.Malloc(
            rootInnerSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        if (dfxCounter) {
            dfxCounter->LogMalloc(allocation);
        }
#else
        UNUSED(dfxCounter);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dup.RuntimeWorkspace() = allocation.ptr;
        auto &reuseInfo = dup.GetRuntimeReuseInfo();
        reuseInfo.poolResetTimes = tensorAllocators_.rootInner.ResetTimes();
        return true;
    }

public:
#if DEBUG_INFINITE_LIFETIME
    WsAllocation DebugDumpTensorAllocate(size_t memReq,
        WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        if (!dumpTensorWsAllocator_.CanAllocate(memReq)) {
            DEV_ERROR("dumpTensorWsAllocator_ CanAllocate failed, memReq=%zu", memReq);
        }
        DEV_ASSERT(dumpTensorWsAllocator_.CanAllocate(memReq));
        WsAllocation allocation = dumpTensorWsAllocator_.Malloc(memReq, category);
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
        return allocation;
    }
#endif

    bool TryAllocateFunctionMemory(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot *slotList) {
        AutoScopedPerf asp(PERF_EVT_ALLOCATE_WORKSPACE);

        WsAllocatorCounter *pDfxCounter = nullptr;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        WsAllocatorCounter funcAllocDfx;
        pDfxCounter = &funcAllocDfx;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        DevAscendFunction *devRootSrc = devRootDup.GetSource();

        // alloc outcast workspace
        size_t outcastSize = devRootSrc->exclusiveOutcastWsMemoryRequirement;
        if (outcastSize != 0) {
            if (!tensorAllocators_.devTaskInnerExclusiveOutcasts.CanAllocate(outcastSize)) {
                return false;
            }
            WsAllocation allocation = tensorAllocators_.devTaskInnerExclusiveOutcasts.Malloc(
                outcastSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
            funcAllocDfx.LogMalloc(allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
#if DEBUG_INFINITE_LIFETIME
            allocation = DebugDumpTensorAllocate(outcastSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#endif
            devRootDup.RuntimeOutcastBase() = allocation.ptr;
        } else {
            devRootDup.RuntimeOutcastBase() = 0;
        }

        // alloc inner workspace
        size_t rootInnerSize = devRootSrc->rootInnerTensorWsMemoryRequirement;
        if (rootInnerSize != 0) {
            if (!TryAllocateFuncWs(devRootDup, rootInnerSize, pDfxCounter)) {
                return false;
            }
#if DEBUG_INFINITE_LIFETIME
            WsAllocation allocation = DebugDumpTensorAllocate(rootInnerSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
            devRootDup.RuntimeWorkspace() = allocation.ptr;
#endif
        } else {
            devRootDup.RuntimeWorkspace() = 0;
        }

        // assign incast address descriptor
        for (size_t i = 0; i < devRootSrc->GetIncastSize(); ++i) {
            if (devRootSrc->GetIncast(i).fromSlotList.size() <= 0) {
                DEV_DEBUG("fromSlotList.size()=%zu <=0, i=%zu", devRootSrc->GetIncast(i).fromSlotList.size(), i);
            }
            DEV_DEBUG_ASSERT(devRootSrc->GetIncast(i).fromSlotList.size() > 0);

            int slotIndex = devRootSrc->At(devRootSrc->GetIncast(i).fromSlotList, 0);
            devRootDup.GetIncastAddress(i) = slotList[slotIndex].desc;
            DEV_VERBOSE_DEBUG("get incast %zu, from slot %d address %s.", i, slotIndex, devRootDup.GetIncastAddress(i).Dump().c_str());
        }

        /* Try assemble outcast. Should this be moved to where the need alloc mark is called? */
        size_t assembleOutcastNeedAlloc = 0;
        int *assembleSlotList = &devRootSrc->GetRedaccAssembleSlotList(0);
        for (size_t i = 0, ie = devRootSrc->GetRedaccAssembleSlotListSize(); i < ie; i++) {
            if (slotList[assembleSlotList[i]].isAssembleSlotNeedAlloc) {
                assembleOutcastNeedAlloc++;
            }
        }
        if (assembleOutcastNeedAlloc != 0) {
            if (!AllocateSlot(assembleOutcastNeedAlloc, devRootSrc->GetRawName(), assembleSlotBuffer_)) {
                return false;
            }
        }

        // assign outcast address separately first, will be reassigned when corresponding slot was replaced
        uint32_t assembleOutcastIndex = 0;
        uintdevptr_t outcastBaseAddr = devRootDup.RuntimeOutcastBase();
        for (size_t i = 0; i < devRootSrc->GetOutcastSize(); ++i) {
            int outputSlotIndex = -1;
            int assembleSlotIndex = -1;
            auto &toSlotList = devRootSrc->GetOutcast(i).toSlotList;
            for (size_t k = 0; k < toSlotList.size(); ++k) {
                auto idx = devRootSrc->At(toSlotList, k);
                if (slotList[idx].IsOutputAddress()) { // true表示固定地址，用户输出Assemble的结果
                    outputSlotIndex = idx;
                } else if (slotList[idx].IsAssembleAddress()) {
                    assembleSlotIndex = idx;
                }
            }
            AddressDescriptor desc;
            if (outputSlotIndex != -1) {
                /* Output tensor */
                desc = slotList[outputSlotIndex].desc;
                if (desc.IsNullAddress()) {
                    /* Allocate such tensor */
                    auto rawTensor = devRootSrc->GetOutcastRawTensor(i);
                    if (rawTensor->linkedIncastId == -1) {
                        auto memReq = rawTensor->GetMemoryRequirement(devRootDup.GetExpressionAddr());
                        auto allocation = tensorAllocators_.dassembleDests.Allocate<uint8_t>(memReq);
#if DEBUG_INFINITE_LIFETIME
                        allocation = DebugDumpTensorAllocate(memReq);
#endif
                        desc = AddressDescriptor(allocation.ptr);
                        slotList[outputSlotIndex].desc = desc;
                    }
                }
            } else if (assembleSlotIndex != -1) {
                /* assemble outcast tensor */
                desc = slotList[assembleSlotIndex].desc;
                if (slotList[assembleSlotIndex].isAssembleSlotNeedAlloc) {
                    if (desc.IsAddress() && desc.GetAddressValue() != 0) {
                        /* Mark recycle */
                        DelayedRecycleSlotMem(desc.GetAddressValue());
                    }
                    auto address = assembleSlotBuffer_[assembleOutcastIndex++].ptr;
                    desc = AddressDescriptor(address);
                    slotList[assembleSlotIndex].desc = desc;
                    slotList[assembleSlotIndex].isAssembleSlotNeedAlloc = false;
                }
            } else if (devRootSrc->GetOutcast(i).exprListIndex != -1) {
                uint64_t *exprTbl = devRootDup.GetExpressionAddr();
                uint64_t addr = exprTbl[devRootSrc->GetOutcast(i).exprListIndex];
                desc = AddressDescriptor(addr);
            } else {
                desc = AddressDescriptor(outcastBaseAddr + devRootSrc->GetOutcastRawTensor(i)->addrOffset);
            }

            //判断是否与incast 共地址
            auto rawTensor = devRootSrc->GetOutcastRawTensor(i);
            if (rawTensor->linkedIncastId != -1) {
                desc = devRootDup.GetIncastAddress(rawTensor->linkedIncastId);
                if (desc.IsNullAddress()) {
                    auto memReq = rawTensor->GetMemoryRequirement(devRootDup.GetExpressionAddr());
                    auto allocation = tensorAllocators_.dassembleDests.Allocate<uint8_t>(memReq);
#if DEBUG_INFINITE_LIFETIME
                    allocation = DebugDumpTensorAllocate(memReq);
#endif // DEBUG_INFINITE_LIFETIME
                    desc = AddressDescriptor(allocation.ptr);
                    if (outputSlotIndex != -1) {
                        slotList[outputSlotIndex].desc = desc;
                    }
                    devRootDup.GetIncastAddress(rawTensor->linkedIncastId) = desc;
                }
            }

            devRootDup.GetOutcastAddress(i) = desc;
            DEV_VERBOSE_DEBUG("get outcast %zu slot %d/%d address %s.", i, outputSlotIndex, assembleSlotIndex, desc.Dump().c_str());
        }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        funcAllocDfx.DelayedDumpAsRootFuncAndReset(wsMemDelayedDumper_, devRootDup.GetSource()->GetRawName());
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return true;
    }

    bool IsValidSlotMemRequirement(uint64_t memReq) const {
        return tensorAllocators_.devTaskBoundaryOutcasts.IsValidSlotMemRequirement(memReq);
    }

    uintdevptr_t AllocateSlot(const char *rootFuncName) {
        WsAllocation allocation = tensorAllocators_.devTaskBoundaryOutcasts.Allocate();
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.LogTensorMalloc(rootFuncName, allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        (void)rootFuncName;
        return allocation.ptr;
    }

    bool AllocateSlot(int count, [[maybe_unused]] const char *rootFuncName, WsAllocation *buffer) {
        bool ret = tensorAllocators_.devTaskBoundaryOutcasts.Allocate(count, buffer);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.LogTensorMalloc(rootFuncName, allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return ret;
    }

    void DelayedRecycleSlotMem(uintdevptr_t ptr) {
        slotMemToBeFree_.push_back(ptr);
    }

    void TriggerDelayedRecycle() {
        for (uintdevptr_t ptr : slotMemToBeFree_) {
            tensorAllocators_.devTaskBoundaryOutcasts.Deallocate(ptr);
        }
        slotMemToBeFree_.clear();
    }

    void RecycleDevFuncWorkspace() {
        tensorAllocators_.devTaskInnerExclusiveOutcasts.ResetPool();
        tensorAllocators_.rootInner.ResetPool();
    }

    DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func) {
        WsAllocation tinyAlloc = ControlFlowAllocateSlab(devProg_, func->GetDuppedDataAllocSize(), SlabAlloc(func->GetDuppedDataAllocSize(), WsAicpuSlabMemType::DUPPED_FUNC_DATA));
        return DevAscendFunctionDupped::DuplicateRoot(func, tinyAlloc);
    }

    void DestroyDuppedFunc(DevAscendFunctionDupped &dup) {
        dup.ReleaseDuppedMemory(metadataAllocators_.general);
    }

    DynDeviceTask *MakeDynDeviceTask() {
        WsAllocation alloc = ControlFlowAllocateSlab(devProg_, sizeof(DynDeviceTask), SlabAlloc(sizeof(DynDeviceTask), WsAicpuSlabMemType::DEV_DYN_TASK));
        DynDeviceTask *dynTask = new(reinterpret_cast<void *>(alloc.ptr)) DynDeviceTask(*this);
        dynTask->selfAlloc = alloc;
        return dynTask;
    }

    DevAscendFunctionDuppedStitch *AllocateStitch() {
        WsAllocation allocation = ControlFlowAllocateSlab(devProg_, sizeof(DevAscendFunctionDuppedStitch), SlabAlloc(sizeof(DevAscendFunctionDuppedStitch), WsAicpuSlabMemType::DUPPED_STITCH));
        DevAscendFunctionDuppedStitch *stitch = allocation.As<DevAscendFunctionDuppedStitch>();
        uint64_t *clear = PtrToPtr<DevAscendFunctionDuppedStitch, uint64_t>(stitch);
        clear[0] = 0;
        clear[1] = 0;
        return stitch;
    }

    DynFuncHeader *AllocateDynFuncData(uint64_t size) {
        WsAllocation allocation = ControlFlowAllocateSlab(devProg_, size, SlabAlloc(size, WsAicpuSlabMemType::DYN_FUNC_DATA));
        DynFuncHeader *header = allocation.As<DynFuncHeader>();
        return header;
    }

    void ResetAicpuMemCounter() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        metadataAllocators_.general.ResetCounter();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void RewindMemoryDumper() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.Rewind();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void MarkAsNewStitchWindow() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        metadataAllocators_.general.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
        aicpuStitchAllocator_.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
        wsMemDelayedDumper_.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void DumpMemoryUsage(const char *hint) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.DumpStitchWindowMemoryUsage();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        metadataAllocators_.general.DumpMemoryUsage(hint, "Metadata");
        metadataAllocators_.generalSlab.DumpMemoryUsage(hint, "Metadata slab allocator");
        metadataAllocators_.stitchSlab.DumpMemoryUsage(hint, "Metadata Stitch slab allocator");
        tensorAllocators_.rootInner.DumpMemoryUsage(hint, "Tensor (root inner) workspace");
        tensorAllocators_.devTaskInnerExclusiveOutcasts.DumpMemoryUsage(hint, "Tensor (DeviceTask inner outcasts) workspace");
        tensorAllocators_.dassembleDests.DumpMemoryUsage(hint, "Tensor (dassembleDests) workspace");
        tensorAllocators_.devTaskBoundaryOutcasts.DumpMemoryUsage(hint);

        // Dump stack memory
        DEV_MEM_DUMP("Stack workspace memory usage (%s)\n", hint);
        DEV_MEM_DUMP("            Memory pool size: %10lu bytes\n", stackWorkspaceSize_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    }

    void InitMetadataSlabAllocator() {
        if (metadataAllocators_.general.FreeMemorySize() <= 0) {
            DEV_ERROR("FreeMemorySize=%lu <= 0", metadataAllocators_.general.FreeMemorySize());
        }
        DEV_ASSERT(metadataAllocators_.general.FreeMemorySize() > 0);
        uint64_t memBase = metadataAllocators_.general.MemBaseAddr() + metadataAllocators_.general.AllocatedSize();
        uint64_t realMemBase = AlignUp(memBase, sizeof(uint64_t));
        uint32_t metaSlabMemSize = metadataAllocators_.general.FreeMemorySize() - (realMemBase - memBase);
        uint32_t slabSize = CalcAicpuMetaSlabAlloctorSlabPageSize(metaSlabMemSize);
        metadataAllocators_.generalSlab.Init(reinterpret_cast<void*>(realMemBase), metaSlabMemSize, slabSize);
        for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                if (!(metadataAllocators_.generalSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])()))) {
                    DEV_ERROR("RegisCache faild, i=%zu", i);
                }
                DEV_ASSERT(metadataAllocators_.generalSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])()));
            }
        }
    }
    uint32_t CalcSlabMemObjmaxSize () {
        uint32_t slabMemObjmaxSize = CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize();
        DEV_DEBUG ("slabMemObjmaxSize is: %u", slabMemObjmaxSize);
        return slabMemObjmaxSize;
    }
    void CalculateSlabCapacityPerType (uint32_t slabSize, uint32_t* slabCapacity, uint32_t slabTypeNum) {
        if (slabCapacity == nullptr) {
            DEV_ERROR("slabCapacity is nullptr");
            return;
        }
        constexpr uint32_t maxSlabTypes = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT);
        if (slabTypeNum > maxSlabTypes) {
            DEV_ERROR("slabTypeNum exceeds the allowed typenum %u ", maxSlabTypes);
            return;
        }
        for (size_t i = 0; i < slabTypeNum; ++i) {
          if (slabMemObjSizeFunc[i] != nullptr && (this->*slabMemObjSizeFunc[i])() !=0) {
             DEV_DEBUG("WsAicpuSlabMemType[%zu] is %u", i, (this->*slabMemObjSizeFunc[i])());
              slabCapacity[i] = slabSize / (this->*slabMemObjSizeFunc[i])();
           }
        }
    }
    WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type) {
        void* ptr = nullptr;
        DEV_VERBOSE_DEBUG("SlabAlloc type = %u, size = %u.", ToUnderlying(type), objSize);
        SlabTryDynAddCache(type, objSize); // ready que need dyn add cache
        do {
            if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
                ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
            } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
                ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
            }
            if (ptr != nullptr) {
                break;
            }

            if (submmitTaskSlabMemQueue_.IsEmpty()) {
                // should not happen, first task alloc failed
                metadataAllocators_.generalSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
                metadataAllocators_.stitchSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
                DEV_ERROR("Slab alloc null,type=%u,objsize=%u.", ToUnderlying(type), objSize);
                DEV_ASSERT_MSG(false, "Slab alloc null,type=%u,objsize=%u.", ToUnderlying(type), objSize);
            }
            uint64_t ttlstart = GetCycles();
            while (!SlabStageAllocMemTryRecycle()) {  // wait sch aicpu finish task
                if (GetCycles() - ttlstart > TIMEOUT_CYCLES) {
                    ttlstart = GetCycles();
                    DEV_WARN("Waiting for device task finished for too long.");
                }
            };
        } while (true);

        WsAllocation allocation;
        allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
        return allocation;
    }

    WsSlabStageAllocMem SlabGetStageAllocMem(bool keepTail, WsAicpuSlabMemType keepType) {
        WsSlabStageAllocMem stageMem;
        stageMem.generalMetadataStageMem = metadataAllocators_.generalSlab.PopStageAllocMem(keepTail, ToUnderlying(keepType));
        stageMem.stitchStageMem = metadataAllocators_.stitchSlab.PopStageAllocMem(false, 0); // not support keep alloc memory
        return stageMem;
    }

    void SlabStageAllocMemSubmmit(WsSlabStageAllocMem* submmitSlabMem) {
        while (!submmitTaskSlabMemQueue_.TryEnqueue(submmitSlabMem)) {
            // maybe que is full, need wait task finish and recycle aicpu meta memory
            SlabStageAllocMemTryRecycle();
        }
        return;
    }

    /* support vector allocator,so need have this fucntion member */
    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category) {
        if (category != WsMemCategory::VECTOR_STITCHED_LIST) {
            DEV_ERROR("category=%d != WsMemCategory::VECTOR_STITCHED_LIST=%d", (int)category, (int)WsMemCategory::VECTOR_STITCHED_LIST);
        }
        DEV_ASSERT(category == WsMemCategory::VECTOR_STITCHED_LIST);
        return SlabAlloc(count * sizeof(T), WsAicpuSlabMemType::VEC_STITCHED_LIST);
    }

    void Deallocate(WsAllocation) {} // just for support vector allocator,so need have this fucntion member

private:
    void InitMetadataAllocators(DevAscendProgram *devProg) {
        // Initialize aicpu memory
        metadataAllocators_.general.InitMetadataAllocator(devProg->devArgs.generalAddr, devProg->memBudget.metadata.general);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceMetadataGeneral(Range(devProg->devArgs.generalAddr, devProg->devArgs.generalAddr + devProg->memBudget.metadata.general))));

        InitAicpuStitchSlabAllocator(reinterpret_cast<void*>(devProg->devArgs.stitchPoolAddr), devProg->memBudget.metadata.stitchPool);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceMetadataStitch(Range(devProg->devArgs.stitchPoolAddr, devProg->devArgs.stitchPoolAddr + devProg->memBudget.metadata.stitchPool))));
    }

    void InitTensorAllocators(uintdevptr_t workspaceAddr,
                              uint64_t tensorWorkspaceSize,
                              DevAscendProgram *devProg) {
        uint64_t baseAddr = workspaceAddr;

        // Initialize tensor workspace memory verifier
        tensorWsVerifier_.Init(
            baseAddr,
            tensorWorkspaceSize);

        // Initialize root function slotted outcast tensor memory
        auto devTaskBoundaryOutcastsBudget = devProg->memBudget.tensor.devTaskBoundaryOutcastNum * devProg->memBudget.tensor.MaxOutcastMem();
        slotVerifier_.Init(baseAddr, devTaskBoundaryOutcastsBudget);
        tensorAllocators_.devTaskBoundaryOutcasts.InitTensorAllocator(
            baseAddr,
            devProg->memBudget.tensor.devTaskBoundaryOutcastNum,
            devProg->memBudget.tensor.MaxOutcastMem(),
            metadataAllocators_.general);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceCrossDeviceTaskOutcast(Range(baseAddr, baseAddr + devTaskBoundaryOutcastsBudget))));
        baseAddr += devTaskBoundaryOutcastsBudget;

        // Initialize root function non-outcast tensor memory
        auto rootInnerBudget = devProg->memBudget.tensor.rootInner;
        rootInnerWsVerifier_.Init(baseAddr, rootInnerBudget);
        tensorAllocators_.rootInner.InitTensorAllocator(baseAddr, rootInnerBudget);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceInnerTensor(Range(baseAddr, baseAddr + rootInnerBudget))));
        baseAddr += rootInnerBudget;

        // Initialize root function sequential outcast tensor memory
        auto devTaskInnerOutcastBudget = devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts;
        devTaskInnerExclusiveOutcastsWsVerifier_.Init(baseAddr, devTaskInnerOutcastBudget);
        tensorAllocators_.devTaskInnerExclusiveOutcasts.InitTensorAllocator(baseAddr, devTaskInnerOutcastBudget);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceInDeviceTaskOutcast(Range(baseAddr, baseAddr + devTaskInnerOutcastBudget))));
        baseAddr += devTaskInnerOutcastBudget;

        // Initialize dassembleDests tensor memory
        // dassembleDests contains dynamic workspace, put it to the end
        auto dassembleDestsTensorBudget = workspaceAddr + tensorWorkspaceSize - baseAddr;
        DEV_ASSERT(devProg->memBudget.tensor.DAssembleDests() <= dassembleDestsTensorBudget);
        dassembleDestsTensorVerifier_.Init(baseAddr, dassembleDestsTensorBudget);
        tensorAllocators_.dassembleDests.InitTensorAllocator(baseAddr, dassembleDestsTensorBudget);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspacePartialOutcast(Range(baseAddr, baseAddr + dassembleDestsTensorBudget))));
        baseAddr += dassembleDestsTensorBudget;
        if (!(workspaceAddr <= baseAddr && baseAddr <= workspaceAddr + tensorWorkspaceSize)) {
            DEV_ERROR("Address range check failed: workspaceAddr=%lu, baseAddr=%lu, tensorWorkspaceSize=%lu",
            workspaceAddr, baseAddr, tensorWorkspaceSize);
        }
        DEV_ASSERT(workspaceAddr <= baseAddr && baseAddr <= workspaceAddr + tensorWorkspaceSize);
    }

    void InitAICoreSpilledMemory(uintdevptr_t workspaceAddr,
                                 DevAscendProgram *devProg) {
        uint64_t coreNum = devProg->devArgs.GetBlockNum();
        if (coreNum == 0) {
            return;
        }
        // Compile time `aicoreSpilled` per single core is required to be aligned by 512.
        // This formula will never result into a value smaller than compile time one.
        uint64_t perCoreMem = devProg->memBudget.aicoreSpilled / TENSOR_ADDR_ALIGNMENT / coreNum * TENSOR_ADDR_ALIGNMENT;

        // Initialize in-core stack memory
        stackWorkspaceBase_ = workspaceAddr;
        standardStackWorkspacePerCore_ = perCoreMem;
        stackWorkspaceSize_ = devProg->memBudget.aicoreSpilled;
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceSpill(
            mem(perCoreMem), coreNum,
            Range(stackWorkspaceBase_, stackWorkspaceBase_ + stackWorkspaceSize_))));
    }

    uint32_t DevFunctionDuppedSlabMemObjSize() {
        if (maxDevFuncDuppedSize_ == 0) {
            for (uint32_t i = 0; i < devProg_->GetFunctionSize(); i++) {
                uint64_t curSize = devProg_->GetFunction(i)->GetDuppedDataAllocSize();
                if (curSize > maxDevFuncDuppedSize_) {
                    maxDevFuncDuppedSize_ = curSize;
                }
            }
        }

        return maxDevFuncDuppedSize_;
    }

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t DynFuncDataSlabMemObjSize() {
        return (sizeof(DynFuncHeader) + MAX_CACHED_FUNC_NUM * sizeof(DynFuncData));
    }

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t VecStitchListSLabMemObjSize() {
        return MAX_CACHED_FUNC_NUM * sizeof(DevAscendFunctionDupped);
    }

    uint32_t DynDevTaskSlabMemObjSize() {
        return sizeof(struct DynDeviceTask);
    }

    uint32_t DuppedStitchSlabMemObjSize() {
        return sizeof(struct DevAscendFunctionDuppedStitch);
    }

    uint32_t ReadyQueSlabMemObjSize() {
        return sizeof(ReadyCoreFunctionQueue) + devProg_-> stitchFunctionsize * sizeof(uint32_t);
    }
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
    uint32_t WrapQueSlabMemObjSize() {
        return sizeof(ReadyCoreFunctionQueue) + devProg_-> stitchFunctionsize * sizeof(uint32_t);
    }

    uint32_t WrapTasklistSlabMemObjSize() {
        return devProg_-> stitchFunctionsize * sizeof(uint32_t);
    }
#endif

    uint32_t (DeviceWorkspaceAllocator::*slabMemObjSizeFunc[ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)])() = {
        &DeviceWorkspaceAllocator::DevFunctionDuppedSlabMemObjSize,
        &DeviceWorkspaceAllocator::DynFuncDataSlabMemObjSize,
        &DeviceWorkspaceAllocator::VecStitchListSLabMemObjSize,
        &DeviceWorkspaceAllocator::DynDevTaskSlabMemObjSize,
        &DeviceWorkspaceAllocator::ReadyQueSlabMemObjSize,
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
        &DeviceWorkspaceAllocator::WrapQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::WrapTasklistSlabMemObjSize,
#endif
        nullptr, // invalid type
        &DeviceWorkspaceAllocator::DuppedStitchSlabMemObjSize,
    };

    /* 根据当前算子的业务模型分析计算出slab 管理内存页大小, 基于当前可评估的所有内存类型的最大值评估 */
    uint32_t CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize() {
        uint32_t slabMemObjmaxSize = 0;
        constexpr uint32_t extendBuf = 1024;
        for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); ++i) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                uint32_t currentSize = (this->*slabMemObjSizeFunc[i])();
                if (currentSize > slabMemObjmaxSize) {
                    slabMemObjmaxSize = currentSize;
                }
            }
        }
        slabMemObjmaxSize += extendBuf;
        return slabMemObjmaxSize;
    }
    uint32_t CalcAicpuMetaSlabAlloctorSlabPageSize(uint32_t totalMemSize) {
        uint32_t allocNumOneSlab = 4; // default
        uint32_t slabSize = CalcAicpuMetaSlabAlloctorSlabMemObjmaxSize();
        uint32_t leastSlabReqMem = (ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)) * slabSize;
        if (leastSlabReqMem >= totalMemSize) {
            DEV_ERROR("leastSlabReqMem=%u >= totalMemSize=%u", leastSlabReqMem, totalMemSize);
        }
        DEV_ASSERT(leastSlabReqMem < totalMemSize);
        uint32_t realMaxAllocNum = totalMemSize / leastSlabReqMem;
        if (realMaxAllocNum < allocNumOneSlab) {
            allocNumOneSlab = realMaxAllocNum;
        }
        slabSize *= allocNumOneSlab;
        return ALIGN_UP(slabSize, sizeof(uint64_t));
    }

    void InitAicpuStitchSlabAllocator(void* memBase, uint32_t totalSize) {
        if (!(memBase != nullptr && totalSize > 0)) {
            DEV_ERROR("memBase=%p, totalSize=%u", memBase, totalSize);
        }
        DEV_ASSERT(memBase != nullptr && totalSize > 0);
        constexpr uint32_t slabSize = 4 * 1024; // fix size
        metadataAllocators_.stitchSlab.Init(memBase, totalSize, slabSize);
        for (size_t i = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) + 1;
            i < ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT); ++i) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                uint32_t objSize = (this->*slabMemObjSizeFunc[i])();
                if (slabSize <= objSize) {
                    DEV_ERROR("slabSize=%u <= objSize=%u", slabSize, objSize);
                }
                DEV_ASSERT(slabSize > objSize);
                if (!(metadataAllocators_.stitchSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])()))) {
                    DEV_ERROR("stitchSlab RegistCache failed, i=%zu", i);
                }
                DEV_ASSERT(metadataAllocators_.stitchSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])()));
            }
        }
    }

    void SlabTryDynAddCache(WsAicpuSlabMemType type, uint32_t objSize) {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            if (!metadataAllocators_.generalSlab.ExistCache(ToUnderlying(type), objSize)) {
                if (!(metadataAllocators_.generalSlab.RegistCache(ToUnderlying(type), objSize))) {
                    DEV_ERROR("generalSlab RegistCache failed, type=%u, objSize=%u", ToUnderlying(type), objSize);
                }
                DEV_ASSERT(metadataAllocators_.generalSlab.RegistCache(ToUnderlying(type), objSize));
            }
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            if (!metadataAllocators_.stitchSlab.ExistCache(ToUnderlying(type), objSize)) {
                if (!(metadataAllocators_.generalSlab.RegistCache(ToUnderlying(type), objSize))) {
                    DEV_ERROR("stitchSlab RegistCache failed, type=%u, objSize=%u", ToUnderlying(type), objSize);
                }
                DEV_ASSERT(metadataAllocators_.stitchSlab.RegistCache(ToUnderlying(type), objSize));
            }
        } else {
            DEV_ERROR("Invalid slab memory type: %u", (unsigned int)type);
            DEV_ASSERT(false);
        }
    }
public:
    TensorAllocator &GetTensorAllocator() { return tensorAllocators_; }
private:
    bool SlabStageAllocMemTryRecycle() {
        auto FreeTaskSlabMemfunc = [this] (WsSlabStageAllocMem* slabStageMem) -> bool {
            if (slabStageMem->canFree.load(std::memory_order_relaxed)) {
                // recycle slab alloc memory
                metadataAllocators_.generalSlab.FreeStageAllocMem(slabStageMem->generalMetadataStageMem);
                metadataAllocators_.stitchSlab.FreeStageAllocMem(slabStageMem->stitchStageMem);
                return true;
            }
            return false;
        };

        // try free finished task and recycle aicpu meta memory
        return submmitTaskSlabMemQueue_.FreeUntil(FreeTaskSlabMemfunc);
    }

private:
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    DelayedDumper wsMemDelayedDumper_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

    MetadataAllocator metadataAllocators_;
    TensorAllocator tensorAllocators_;

#if DEBUG_INFINITE_LIFETIME
    SeqWsAllocator dumpTensorWsAllocator_;
    uint64_t *dumpTensorWsAllocatorCounter_; // used in host-side when reading back npu memory
#endif

    uintdevptr_t stackWorkspaceBase_{0};
    uint64_t standardStackWorkspacePerCore_{0};
    uint64_t stackWorkspaceSize_{0};

    uint32_t maxDevFuncDuppedSize_{0};
    DevAscendProgram *devProg_{nullptr};
    WsAllocation *assembleSlotBuffer_{nullptr};

    WsMemoryVerifier tensorWsVerifier_;
    WsMemoryVerifier slotVerifier_;
    WsMemoryVerifier dassembleDestsTensorVerifier_;
    WsMemoryVerifier rootInnerWsVerifier_;
    WsMemoryVerifier devTaskInnerExclusiveOutcastsWsVerifier_;

    Vector<uintdevptr_t, WsMemCategory::VECTOR_SLOT_RECYCLE_LIST> slotMemToBeFree_;
    SPSCQueue<WsSlabStageAllocMem *, SUBMMIT_TASK_QUE_SIZE> submmitTaskSlabMemQueue_;
};
} // namespace npu::tile_fwk::dynamic
#endif
