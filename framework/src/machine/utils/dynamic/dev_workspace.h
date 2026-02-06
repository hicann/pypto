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
#include "machine/utils/dynamic/runtime_outcast_tensor.h"

namespace npu::tile_fwk::dynamic {
inline constexpr int64_t TENSOR_ADDR_ALIGNMENT = 512;
inline constexpr uint32_t SUBMMIT_TASK_QUE_SIZE = 32;
class DeviceWorkspaceAllocator {
public:
    DeviceWorkspaceAllocator() = default;
    explicit DeviceWorkspaceAllocator(DevAscendProgram *base) : devProg_(base) {}
    ~DeviceWorkspaceAllocator() = default;
    void Init(DevStartArgs *devStartArgs) {
        uintdevptr_t baseAddr = devStartArgs->contextWorkspaceAddr;
        DevAscendProgram *devProg = devStartArgs->devProg;

        // Host coherent allocators MUST be initialized EARLIEST since some other allocators might depend on them
        InitMetadataAllocators(devProg, devStartArgs);

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
        SetupVector(rtBoundaryOutcastToBeFree_);
        rtBoundaryOutcastToBeFree_.reserve(devProg->memBudget.tensor.devTaskBoundaryOutcastNum);

        SetupItemPool(runtimeOutcastTensorPool_, devProg->runtimeOutcastPoolSize, WsMemCategory::ITEMPOOL_RUNTIME_OUTCAST);

        devProg_ = devProg;
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

    template <typename T>
    void SetupItemPool(ItemPool<T> &pool, size_t count, WsMemCategory category) {
        pool.Init(metadataAllocators_.general, count, category);
    }

private:
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

    WsMemoryState VerifyTensorMemoryState(uintdevptr_t ptr, size_t size) const {
        return tensorWsVerifier_.Verify(ptr, size);
    }

    bool IsValidWsTensor(uintdevptr_t ptr, size_t memSize) const {
        return slotVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
            dassembleDestsTensorVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
            rootInnerWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
            devTaskInnerExclusiveOutcastsWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE;
    }

public:
    void VerifyStitchedListMemory(DevStartArgs &args, const DevAscendFunctionDupped *stitchedList, size_t size) {
        std::set<uintdevptr_t> inoutAddr;
        for (int i = 0; i < args.GetInputTensorSize(); i++) {
            inoutAddr.insert(args.GetInputTensor(i).address);
        }
        for (int i = 0; i < args.GetOutputTensorSize(); i++) {
            inoutAddr.insert(args.GetOutputTensor(i).address);
        }

        bool verificationSuccess = true;
        for (size_t i = 0; i < size; i++) {
            const auto &dup = stitchedList[i];

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
                        if (!IsValidWsTensor(memInfo.ptr, memInfo.size)) {
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
        DEV_ASSERT(verificationSuccess);
    }

private:
    void AllocateFunctionInnerWorkspace(DevAscendFunctionDupped dup, uint64_t rootInnerMemReq,
                                        [[maybe_unused]] WsAllocatorCounter *dfxCounter) {
        if (!tensorAllocators_.rootInner.CanAllocate(rootInnerMemReq)) {
            tensorAllocators_.rootInner.ResetPool();
            DEV_ASSERT_MSG(tensorAllocators_.rootInner.CanAllocate(rootInnerMemReq),
                "After reset, still cannot allocate root inner workspace unexpectedly, memReq=%" PRIu64,
                rootInnerMemReq);
        }
        WsAllocation allocation = tensorAllocators_.rootInner.Malloc(
            rootInnerMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        if (dfxCounter) {
            dfxCounter->LogMalloc(allocation);
        }
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dup.RuntimeWorkspace() = allocation.ptr;
        auto &reuseInfo = dup.GetRuntimeReuseInfo();
        reuseInfo.poolResetTimes = tensorAllocators_.rootInner.ResetTimes();
    }

    // Helper: allocate outcast workspace for a duplicated root function
    void AllocateOutcastWorkspaceForDup(DevAscendFunctionDupped devRootDup,
                                        [[maybe_unused]] WsAllocatorCounter *pDfxCounter) {
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        size_t outcastMemReq = devRootSrc->exclusiveOutcastWsMemoryRequirement;
        if (outcastMemReq != 0) {
            DEV_ASSERT(tensorAllocators_.devTaskInnerExclusiveOutcasts.CanAllocate(outcastMemReq));
            WsAllocation allocation = tensorAllocators_.devTaskInnerExclusiveOutcasts.Malloc(
                outcastMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
            if (pDfxCounter) {
                pDfxCounter->LogMalloc(allocation);
            }
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
#if DEBUG_INFINITE_LIFETIME
            allocation = DebugDumpTensorAllocate(outcastMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#endif
            devRootDup.RuntimeOutcastBase() = allocation.ptr;
        } else {
            devRootDup.RuntimeOutcastBase() = 0;
        }
    }

    // Helper: allocate inner workspace for a duplicated root function
    void AllocateInnerWorkspaceForDup(DevAscendFunctionDupped devRootDup, WsAllocatorCounter *pDfxCounter) {
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        size_t rootInnerMemReq = devRootSrc->rootInnerTensorWsMemoryRequirement;
        if (rootInnerMemReq != 0) {
            AllocateFunctionInnerWorkspace(devRootDup, rootInnerMemReq, pDfxCounter);
#if DEBUG_INFINITE_LIFETIME
            WsAllocation allocation = DebugDumpTensorAllocate(rootInnerMemReq, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
            devRootDup.RuntimeWorkspace() = allocation.ptr;
#endif
        } else {
            devRootDup.RuntimeWorkspace() = 0;
        }
    }

    // Helper: assign incast address descriptors for a duplicated root function
    void AssignIncastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot *slotList) {
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        for (size_t i = 0; i < devRootSrc->GetIncastSize(); ++i) {
            DEV_ASSERT_MSG(devRootSrc->GetIncast(i).fromSlotList.size() > 0,
                "Root [%s] Incast %zu has no fromSlotList.", devRootSrc->GetRawName(), i);

            int slotIndex = devRootSrc->At(devRootSrc->GetIncast(i).fromSlotList, 0);
            DEV_ASSERT_MSG(slotList[slotIndex].rtOutcastIter != ITEM_POOL_INVALID_INDEX,
                "Root incast read from empty address.");
            auto &incastDesc = devRootDup.GetIncastAddress(i);
            incastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[slotIndex].rtOutcastIter);
            RuntimeOutcastTensorRef(incastDesc.GetRtOutcastIter());
            DEV_VERBOSE_DEBUG("get incast %zu, from slot %d address %s.", i, slotIndex, incastDesc.Dump().c_str());
        }
    }

    // Helper: assign outcast address descriptors for a duplicated root function
    void AssignOutcastAddresses(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot *slotList) {
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        uintdevptr_t outcastBaseAddr = devRootDup.RuntimeOutcastBase();
        for (size_t i = 0; i < devRootSrc->GetOutcastSize(); ++i) {
            int outputSlotIndex = -1;
            int assembleSlotIndex = -1;
            auto &toSlotList = devRootSrc->GetOutcast(i).toSlotList;
            for (size_t k = 0; k < toSlotList.size(); ++k) {
                auto idx = devRootSrc->At(toSlotList, k);
                if (slotList[idx].IsOutputAddress()) {
                    outputSlotIndex = idx;
                } else if (slotList[idx].IsAssembleAddress()) {
                    assembleSlotIndex = idx;
                }
            }

            AddressDescriptor &outcastDesc = devRootDup.GetOutcastAddress(i);
            auto rawTensor = devRootSrc->GetOutcastRawTensor(i);

            if (outputSlotIndex != -1) {
                /* Output tensor */
                outcastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[outputSlotIndex].rtOutcastIter);
                RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
            } else if (assembleSlotIndex != -1) {
                /* assemble outcast tensor */
                if (slotList[assembleSlotIndex].isAssembleSlotNeedAlloc) {
                    RuntimeOutcastTensorDerefSafe(slotList[assembleSlotIndex].rtOutcastIter);
                    slotList[assembleSlotIndex].rtOutcastIter = MakeRuntimeOutcastTensor(
                        AllocateSlot(devRootSrc->GetRawName()), RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
                    slotList[assembleSlotIndex].isAssembleSlotNeedAlloc = false;
                } else {
                    DEV_ASSERT_MSG(slotList[assembleSlotIndex].rtOutcastIter != ITEM_POOL_INVALID_INDEX,
                        "Missing RUNTIME_SlotMarkNeedAlloc for assemble slot %d.", assembleSlotIndex);
                }
                outcastDesc = AddressDescriptor::MakeFromRtOutcast(slotList[assembleSlotIndex].rtOutcastIter);
                RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
            } else if (devRootSrc->GetOutcast(i).exprListIndex != -1) {
                /* something like an expression address, probably shmem */
                uint64_t *exprTbl = devRootDup.GetExpressionAddr();
                uint64_t addr = exprTbl[devRootSrc->GetOutcast(i).exprListIndex];
                outcastDesc = AddressDescriptor::MakeFromRtOutcast(
                    MakeRuntimeOutcastTensor(addr, RuntimeTensorMemProperty::EXTERNAL));
            } else if (rawTensor->linkedIncastId != -1) {
                /* reshape inplace or something */
                auto &incastDesc = devRootDup.GetIncastAddress(rawTensor->linkedIncastId);
                DEV_ASSERT(incastDesc.IsRtOutcast());
                DEV_ASSERT(incastDesc.GetRtOutcastIter() != ITEM_POOL_INVALID_INDEX);
                outcastDesc = incastDesc;
                RuntimeOutcastTensorRef(outcastDesc.GetRtOutcastIter());
            } else {
                outcastDesc = AddressDescriptor::MakeFromRtOutcast(
                    MakeRuntimeOutcastTensor(outcastBaseAddr + devRootSrc->GetOutcastRawTensor(i)->addrOffset,
                                       RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST));
            }

            DEV_VERBOSE_DEBUG("get outcast %zu slot %d/%d address %s.", i, outputSlotIndex, assembleSlotIndex, outcastDesc.Dump().c_str());
        }
    }

    bool CanAllocateFunctionMemory(DevAscendFunctionDupped devRootDup) {
        DevAscendFunction *devRootSrc = devRootDup.GetSource();

        // check allocation of outcast workspace
        size_t outcastMemReq = devRootSrc->exclusiveOutcastWsMemoryRequirement;
        if (!tensorAllocators_.devTaskInnerExclusiveOutcasts.CanAllocate(outcastMemReq)) {
            return false;
        }

        // allocation of inner workspace will never fail

        // check if reallocated-assemble-slots and the stitch-ending slotMem (secondary allocation) can be allocated
        if (devProg_->slottableOutcastSlotSize > tensorAllocators_.devTaskBoundaryOutcasts.AvailableSlots()) {
            return false;
        }

        // check if runtimeOutcastTensorPool_ has enough items left, estimatedly
        if (devRootSrc->GetOutcastSize() > runtimeOutcastTensorPool_.FreeItemNum()) {
            return false;
        }

        return true;
    }

public:
#if DEBUG_INFINITE_LIFETIME
    WsAllocation DebugDumpTensorAllocate(size_t memReq,
        WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        DEV_ASSERT_MSG(dumpTensorWsAllocator_.CanAllocate(memReq),
            "dumpTensorWsAllocator_ cannot allocate requested memory unexpectedly, memReq=%zu", memReq);
        WsAllocation allocation = dumpTensorWsAllocator_.Malloc(memReq, category);
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
        return allocation;
    }
#endif

    bool TryAllocateFunctionMemory(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot *slotList) {
        AutoScopedPerf asp(PERF_EVT_ALLOCATE_WORKSPACE);

        if (!CanAllocateFunctionMemory(devRootDup)) {
            return false;
        }

        WsAllocatorCounter *pDfxCounter = nullptr;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        WsAllocatorCounter funcAllocDfx;
        pDfxCounter = &funcAllocDfx;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        // Allocate required workspaces and assign descriptors using helpers
        AllocateOutcastWorkspaceForDup(devRootDup, pDfxCounter);
        AllocateInnerWorkspaceForDup(devRootDup, pDfxCounter);

        AssignIncastAddresses(devRootDup, slotList);
        AssignOutcastAddresses(devRootDup, slotList);

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        funcAllocDfx.DelayedDumpAsRootFuncAndReset(wsMemDelayedDumper_, devRootDup.GetSource()->GetRawName());
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return true;
    }

    bool IsValidSlotMemRequirement(uint64_t memReq) const {
        return tensorAllocators_.devTaskBoundaryOutcasts.IsValidSlotMemRequirement(memReq);
    }

    uintdevptr_t AllocateSlot([[maybe_unused]] const char *rootFuncName = nullptr) {
        WsAllocation allocation;
#if !DEBUG_INFINITE_LIFETIME
        allocation = tensorAllocators_.devTaskBoundaryOutcasts.Allocate();
#else
        allocation = DebugDumpTensorAllocate(tensorAllocators_.devTaskBoundaryOutcasts.SlotByteSize(),
            WsMemCategory::TENSOR_ROOTFUNC_OUTCAST_SLOT);
#endif
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.LogTensorMalloc(rootFuncName == nullptr ? "unspecified_root" : rootFuncName, allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return allocation.ptr;
    }

    ItemPoolIter MakeRuntimeOutcastTensor(uintdevptr_t addr, RuntimeTensorMemProperty property) {
        return runtimeOutcastTensorPool_.Allocate(addr, property, 1);
    }

    ItemPool<RuntimeOutcastTensor>::ItemBlock *GetRuntimeOutcastTensorPoolBase() {
        return reinterpret_cast<ItemPool<RuntimeOutcastTensor>::ItemBlock *>(&runtimeOutcastTensorPool_.At(0));
    }

    RuntimeOutcastTensor &GetRuntimeOutcastTensor(ItemPoolIter iter) {
        DEV_ASSERT(iter != ITEM_POOL_INVALID_INDEX);
        return runtimeOutcastTensorPool_.At(iter);
    }

    void RuntimeOutcastTensorDeref(ItemPoolIter iter) {
        DEV_ASSERT(iter != ITEM_POOL_INVALID_INDEX);
        auto &outcast = runtimeOutcastTensorPool_.At(iter);
        DEV_ASSERT(outcast.refCnt > 0);
        outcast.refCnt--;
        if (outcast.refCnt == 0) {
            RuntimeOutcastTensorDestruct(outcast);
        }
    }

    void RuntimeOutcastTensorRef(ItemPoolIter iter) {
        DEV_ASSERT(iter != ITEM_POOL_INVALID_INDEX);
        auto &outcast = runtimeOutcastTensorPool_.At(iter);
        DEV_ASSERT_MSG(outcast.refCnt > 0, "Shouldn't ref a possibly destroyed tensor, iter=%" PRId64, iter);
        outcast.refCnt++;
    }

    void RuntimeOutcastTensorDerefSafe(ItemPoolIter iter) {
        if (iter != ITEM_POOL_INVALID_INDEX) {
            RuntimeOutcastTensorDeref(iter);
        }
    }

    void RuntimeOutcastTensorRefSafe(ItemPoolIter iter) {
        if (iter != ITEM_POOL_INVALID_INDEX) {
            RuntimeOutcastTensorRef(iter);
        }
    }

    void RuntimeOutcastTensorAssign(ItemPoolIter &dst, ItemPoolIter src) {
        if (dst == src) {
            return;
        }
        RuntimeOutcastTensorDerefSafe(dst);
        dst = src;
        RuntimeOutcastTensorRefSafe(src);
    }

    void RuntimeOutcastTensorReplaceAddrWithoutRecycle(ItemPoolIter iter, uintdevptr_t addr, RuntimeTensorMemProperty property) {
        DEV_ASSERT(iter != ITEM_POOL_INVALID_INDEX);
        auto &outcast = runtimeOutcastTensorPool_.At(iter);
        outcast.addr = addr;
        outcast.property = property;
    }

private:
    void RuntimeOutcastTensorDestruct(RuntimeOutcastTensor &outcast) {
#if !DEBUG_INFINITE_LIFETIME
        if (outcast.property == RuntimeTensorMemProperty::BOUNDARY_OUTCAST) {
            rtBoundaryOutcastToBeFree_.push_back(outcast);
        }
#endif // !DEBUG_INFINITE_LIFETIME
        runtimeOutcastTensorPool_.Destroy(&outcast);
    }

public:
    void TriggerDelayedRecycle() {
        for (auto &&outcast : rtBoundaryOutcastToBeFree_) {
            tensorAllocators_.devTaskBoundaryOutcasts.Deallocate(outcast.addr);
        }
        rtBoundaryOutcastToBeFree_.clear();
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
        tensorAllocators_.devTaskBoundaryOutcasts.DumpMemoryUsage(hint);

        // Dump stack memory
        DEV_MEM_DUMP("Stack workspace memory usage (%s)\n", hint);
        DEV_MEM_DUMP("            Memory pool size: %10lu bytes\n", stackWorkspaceSize_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    }

    void InitMetadataSlabAllocator() {
        DEV_ASSERT(metadataAllocators_.general.FreeMemorySize() > 0);
        uint64_t memBase = metadataAllocators_.general.MemBaseAddr() + metadataAllocators_.general.AllocatedSize();
        uint64_t realMemBase = AlignUp(memBase, sizeof(uint64_t));
        uint32_t metaSlabMemSize = metadataAllocators_.general.FreeMemorySize() - (realMemBase - memBase);
        uint32_t slabSize = CalcAicpuMetaSlabAlloctorSlabPageSize(metaSlabMemSize);
        metadataAllocators_.generalSlab.Init(reinterpret_cast<void*>(realMemBase), metaSlabMemSize, slabSize);
        for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                [[maybe_unused]] bool registCacheRes =
                    metadataAllocators_.generalSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])());
                DEV_ASSERT(registCacheRes);
            }
        }
    }

    static uint64_t CalcMetadataItemPoolMemSize(const DevAscendProgram* devProg) {
        size_t itemBlockSize = sizeof(ItemPool<RuntimeOutcastTensor>::ItemBlock);
        DEV_DEBUG("itemBlockSize is: %zu, OutcastPoolSize is %u",
                   itemBlockSize, devProg->runtimeOutcastPoolSize);
        uint64_t itemPoolMemSize = itemBlockSize * devProg->runtimeOutcastPoolSize;
        return itemPoolMemSize;
}

    static uint64_t CalcMetadataVectorMemSize(const DevAscendProgram* devProg) {
        // 1. symbolTable
        uint64_t symbolTableCapacity = CalculateVectorCapacity(devProg->symbolTable.size());
        uint64_t symbolTableMemory = symbolTableCapacity * sizeof(int64_t);
        DEV_DEBUG("symbolTableMemory is %lu,", symbolTableMemory);
        // 2. slotList_
        uint64_t slotListCapacity = CalculateVectorCapacity(devProg->slotSize);
        uint64_t slotListMemory = slotListCapacity * sizeof(DeviceExecuteSlot);
        DEV_DEBUG("slotListMemory is %lu,", slotListMemory);
        // 3. slotInfosInDecidingSlotMem_
        uint64_t slotInfosCapacity = CalculateVectorCapacity(devProg->slotSize);
        uint64_t slotInfosMemory = slotInfosCapacity * sizeof(ItemPoolIter);
        DEV_DEBUG("slotInfosMemory is %lu,", slotInfosMemory);
        // 4. rtBoundaryOutcastToBeFree_
        uint64_t boundaryOutcastToFreeListSize = CalculateVectorCapacity(devProg->memBudget.tensor.devTaskBoundaryOutcastNum);
        uint64_t boundaryOutcastToFreeMemory = boundaryOutcastToFreeListSize * sizeof(RuntimeOutcastTensor);
        DEV_DEBUG("Memory of list for Boundary outcast to free is %lu,", boundaryOutcastToFreeMemory);
        // total
        uint64_t totalSetupVectorMemory = symbolTableMemory + slotListMemory +
                                          slotInfosMemory + boundaryOutcastToFreeMemory;
        return totalSetupVectorMemory;
    }

    static uint64_t CalcMetadataSlotAllocatorMemSize(const DevAscendProgram* devProg) {
        size_t blockHeaderSize = sizeof(WsSlotAllocator::BlockHeader);
        uint64_t slotNum = devProg->memBudget.tensor.devTaskBoundaryOutcastNum;
        DEV_DEBUG("slotNum of boundary outcast is %lu", slotNum);
        return slotNum * blockHeaderSize;
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
        DEV_ASSERT_MSG(category == WsMemCategory::VECTOR_STITCHED_LIST,
            "Unexpected category=%s", GetCategoryName(category));
        return SlabAlloc(count * sizeof(T), WsAicpuSlabMemType::VEC_STITCHED_LIST);
    }

    void Deallocate(WsAllocation) {} // just for support vector allocator,so need have this fucntion member

private:
    void InitMetadataAllocators(DevAscendProgram *devProg, DevStartArgs *devStartArgs) {
        // Initialize aicpu memory
        uint64_t generalAddr = devStartArgs->deviceRuntimeDataDesc.generalAddr;
        metadataAllocators_.general.InitMetadataAllocator(generalAddr, devProg->memBudget.metadata.general);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceMetadataGeneral(Range(generalAddr, generalAddr + devProg->memBudget.metadata.general))));

        uint64_t stitchPoolAddr = devStartArgs->deviceRuntimeDataDesc.stitchPoolAddr;
        InitAicpuStitchSlabAllocator(reinterpret_cast<void*>(stitchPoolAddr), devProg->memBudget.metadata.stitchPool);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceMetadataStitch(Range(stitchPoolAddr, stitchPoolAddr + devProg->memBudget.metadata.stitchPool))));
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

    /*计算使用vector的元数据的数据结构大小*/
    static uint64_t CalculateVectorCapacity(uint64_t size) {
        if (size == 0) {
            return 0;
        }
        constexpr uint64_t MIN_CAPACITY = 8;
        uint64_t capacity = std::max(MIN_CAPACITY, size);
        // 向上取整到 2 的幂次
        capacity = (capacity == 0) ? 0 : (1ULL << (64 - __builtin_clzll(capacity - 1)));
        return capacity;
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

    uint32_t WrapQueSlabMemObjSize() {
        if (devProg_->devArgs.archInfo == ArchInfo::DAV_3510) {
            return sizeof(ReadyCoreFunctionQueue) + devProg_-> stitchFunctionsize * sizeof(uint32_t);
        } else {
            return 1;
        }
    }

    uint32_t WrapTasklistSlabMemObjSize() {
        if (devProg_->devArgs.archInfo == ArchInfo::DAV_3510) {
            return devProg_-> stitchFunctionsize * sizeof(uint32_t);
        } else {
            return 1;
        }
    }

    uint32_t (DeviceWorkspaceAllocator::*slabMemObjSizeFunc[ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)])() = {
        &DeviceWorkspaceAllocator::DevFunctionDuppedSlabMemObjSize,
        &DeviceWorkspaceAllocator::DynFuncDataSlabMemObjSize,
        &DeviceWorkspaceAllocator::VecStitchListSLabMemObjSize,
        &DeviceWorkspaceAllocator::DynDevTaskSlabMemObjSize,
        &DeviceWorkspaceAllocator::ReadyQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::WrapQueSlabMemObjSize,
        &DeviceWorkspaceAllocator::WrapTasklistSlabMemObjSize,
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
        DEV_ASSERT_MSG(leastSlabReqMem < totalMemSize,
            "leastSlabReqMem=%u >= totalMemSize=%u", leastSlabReqMem, totalMemSize);
        uint32_t realMaxAllocNum = totalMemSize / leastSlabReqMem;
        if (realMaxAllocNum < allocNumOneSlab) {
            allocNumOneSlab = realMaxAllocNum;
        }
        slabSize *= allocNumOneSlab;
        return AlignUp(slabSize, sizeof(uint64_t));
    }

    void InitAicpuStitchSlabAllocator(void* memBase, uint32_t totalSize) {
        DEV_ASSERT_MSG(memBase != nullptr && totalSize > 0,
            "memBase %s null, totalSize=%u", memBase == nullptr ? "is" : "is not", totalSize);
        constexpr uint32_t slabSize = 4 * 1024; // fix size
        metadataAllocators_.stitchSlab.Init(memBase, totalSize, slabSize);
        for (size_t i = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) + 1;
            i < ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT); ++i) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                uint32_t objSize = (this->*slabMemObjSizeFunc[i])();
                DEV_ASSERT_MSG(slabSize > objSize, "slabSize=%u <= objSize=%u", slabSize, objSize);
                [[maybe_unused]] bool registCacheRes =
                    metadataAllocators_.stitchSlab.RegistCache(i, (this->*slabMemObjSizeFunc[i])());
                DEV_ASSERT(registCacheRes);
            }
        }
    }

    void SlabTryDynAddCache(WsAicpuSlabMemType type, uint32_t objSize) {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            if (!metadataAllocators_.generalSlab.ExistCache(ToUnderlying(type), objSize)) {
                [[maybe_unused]] bool registCacheRes =
                    metadataAllocators_.generalSlab.RegistCache(ToUnderlying(type), objSize);
                DEV_ASSERT(registCacheRes);
            }
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            if (!metadataAllocators_.stitchSlab.ExistCache(ToUnderlying(type), objSize)) {
                [[maybe_unused]] bool registCacheRes =
                    metadataAllocators_.stitchSlab.RegistCache(ToUnderlying(type), objSize);
                DEV_ASSERT(registCacheRes);
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

    WsMemoryVerifier tensorWsVerifier_;
    WsMemoryVerifier slotVerifier_;
    WsMemoryVerifier dassembleDestsTensorVerifier_;
    WsMemoryVerifier rootInnerWsVerifier_;
    WsMemoryVerifier devTaskInnerExclusiveOutcastsWsVerifier_;

    Vector<RuntimeOutcastTensor, WsMemCategory::VECTOR_RUNTIME_OUTCAST_RECYCLE_LIST> rtBoundaryOutcastToBeFree_;
    SPSCQueue<WsSlabStageAllocMem *, SUBMMIT_TASK_QUE_SIZE> submmitTaskSlabMemQueue_;

    ItemPool<RuntimeOutcastTensor> runtimeOutcastTensorPool_;
};
} // namespace npu::tile_fwk::dynamic
#endif
