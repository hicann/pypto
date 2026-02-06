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
 * \file dev_start_args.h
 * \brief
 */

#pragma once

#include <thread>

#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/device_task.h"

namespace npu::tile_fwk::dynamic {
const uint32_t DUMP_INDEX_SIZE_2 = 2;
const uint32_t DUMP_INDEX_SIZE_4 = 4;

struct DevInputSymbol {
    int64_t value;
};

struct DeviceRuntimeDataDesc {
    DeviceTaskCtrl *taskCtrlPool{nullptr};
    DeviceTaskCtrlQueue *taskQueueList{nullptr};
    uint64_t generalAddr;
    uint64_t stitchPoolAddr;
};

struct DevCtrlState {
    /* state used by control */
    uint32_t schAicpuNum{MAX_SCHEDULE_AICPU_NUM};
    uint32_t taskCtrlIndex{0};
};

#define CTRL_THREAD_INDEX 0

struct DevScheState {
    /* state used by schedule */
    std::atomic<int> threadIdx{0};
    std::atomic<int> finished{0};
};

struct DevStartArgs : DevStartArgsBase {
    uint64_t contextWorkspaceAddr;
    uint64_t contextWorkspaceSize;
    DevAscendProgram *devProg;

    DevInputSymbol *inputSymbolList;
    uint64_t inputSymbolSize;
    const void *controlFlowEntry;
    std::atomic<uint64_t> syncFlag{0}; // sche and ctrl soft sync flag

    DeviceRuntimeDataDesc deviceRuntimeDataDesc;
    DevCtrlState devCtrlState;
    DevScheState devScheState;

    void InitProgram(DevAscendProgram *prog, uint64_t base) {
        devProg = prog;
        deviceRuntimeDataDesc.taskCtrlPool = reinterpret_cast<DeviceTaskCtrl *>(base + devProg->GetDeviceRuntimeOffset().taskCtrlPoolOffset);
        deviceRuntimeDataDesc.taskQueueList = reinterpret_cast<DeviceTaskCtrlQueue *>(base + devProg->GetDeviceRuntimeOffset().taskQueueOffset);
        deviceRuntimeDataDesc.generalAddr = base + devProg->GetDeviceRuntimeOffset().generalOffset;
        deviceRuntimeDataDesc.stitchPoolAddr = base + devProg->GetDeviceRuntimeOffset().stitchPoolOffset;
    }

public:
    void InitWorkspace(DevAscendProgram *tDevProg, void *workspace) {
        contextWorkspaceAddr = reinterpret_cast<uint64_t>(workspace);
        devProg = tDevProg;
        inputSymbolList = nullptr;
        inputSymbolSize = 0;
    }

public:
    template<typename T>
    const T &At(const DevLocalVector<T> &localvec, int index) const {
        return *reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(this) + localvec.Offset(index));
    }
    template<typename T>
    T &At(const DevLocalVector<T> &localvec, int index) {
        return *reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(this) + localvec.Offset(index));
    }

    int GetInputTensorSize() const { return inputTensorSize; }
    const DevTensorData &GetInputTensor(int index) const { return devTensorList[index]; }
    DevTensorData &GetInputTensor(int index) { return devTensorList[index]; }

    int GetOutputTensorSize() const { return outputTensorSize; }
    const DevTensorData &GetOutputTensor(int index) const { return devTensorList[index + inputTensorSize]; }
    DevTensorData &GetOutputTensor(int index) { return devTensorList[index + inputTensorSize]; }

    int GetInputSymbolSize() const { return inputSymbolSize; }
    const DevInputSymbol &GetInputSymbol(int index) const { return inputSymbolList[index]; }
    DevInputSymbol &GetInputSymbol(int index) { return inputSymbolList[index]; }

    std::string Dump(int indent = 0) const {
        std::string INDENTINNER(indent + DUMP_INDEX_SIZE_2, ' ');
        std::string INDENTINNERINNER(indent + DUMP_INDEX_SIZE_4, ' ');
        std::ostringstream oss;
        oss << "DevStartArgs {" << "\n";
        for (int i = 0; i < GetInputTensorSize(); i++) {
            const DevTensorData &input = GetInputTensor(i);
            oss << INDENTINNER << "#input-" << i << ": #address:" << AddressDescriptor::DumpAddress(input.address);
            oss << " #shape:[";
            for (int j = 0; j < input.shape.dimSize; j++) {
                oss << Delim(j != 0, ",");
                oss << input.shape.dim[j];
            }
            oss << "]\n";
        }
        for (int i = 0; i < GetOutputTensorSize(); i++) {
            const DevTensorData &output = GetOutputTensor(i);
            oss << INDENTINNER << "#output-" << i << ": #address:" << AddressDescriptor::DumpAddress(output.address);
            oss << " #shape:[";
            for (int j = 0; j < output.shape.dimSize; j++) {
                oss << Delim(j != 0, ",");
                oss << output.shape.dim[j];
            }
            oss << "]\n";
        }
        oss << INDENTINNER << "#workspaceAddr:" << AddressDescriptor::DumpAddress(contextWorkspaceAddr) << "\n";
        oss << INDENTINNER << "#tensorMemBudget:" << devProg->memBudget.tensor.Total() << "\n";
        oss << INDENTINNER << "#metadataMemBudget:" << devProg->memBudget.metadata.Total() << "\n";
        oss << INDENTINNER << "#devProg:" << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(devProg)) << "\n";
        oss << "}";
        return oss.str();
    }
    static std::unordered_map<std::string, SymbolHandlerId> symbolIndexDict;
};

static_assert(sizeof(DevStartArgs) < DEV_ARGS_SIZE, "dev start args is too large");

static inline void RuntimeYield(uint64_t microseconds = 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

#define DEFAULT_RUNTIME_DATA_RING_BUFFER_COUNT 4
struct RuntimeDataRingBufferHead {
public:
    void Initialize(uint64_t runtimeDataSize, uint64_t runtimeDataCount) {
        runtimeDataSize_ = GetAlignedSize(runtimeDataSize);
        runtimeDataCount_ = runtimeDataCount;

        /* finish starts from round 0 */
        indexFinished_ = 0;
        indexPending_ = 0;
    }

    bool Full() const {
        return indexFinished_ + runtimeDataCount_ <= indexPending_;
    }

    bool Empty() const {
        return indexFinished_ == indexPending_;
    }

    void AllocateWait() {
        while (Full()) {
            RuntimeYield();
        }
    }

    uint8_t *Allocate() {
        AllocateWait();

        /* allocate next element from the ring buffer */
        uint64_t index = ++indexPending_;
        return GetRuntimeData(index);
    }

    uint8_t *AllocatePrepare() {
        AllocateWait();
        return GetRuntimeData(indexPending_ + 1);
    }

    void AllocateSubmit() {
        ++indexPending_;
    }

    void Deallocate(uint8_t *ptr) {
        uint8_t *nextFree = GetRuntimeData(indexFinished_ + 1);
        ASSERT(nextFree == ptr);
        /* deallocate from the ring buffer */
        indexFinished_ += 1;
    }

    uint64_t GetRuntimeDataSize() { return runtimeDataSize_; }
    uint64_t GetRuntimeDataCount() { return runtimeDataCount_; }
    uint64_t GetIndexFinished() { return indexFinished_; }
    uint64_t GetIndexPending()  { return indexPending_; }
    uint64_t GetIndexCurrent() { return indexFinished_ + 1; }

    uint64_t GetIndexPendingIndex() { return GetIndexPending() % GetRuntimeDataCount(); }

    uint8_t *GetRuntimeData(uint64_t index) {
        return &data_[runtimeDataSize_ * (index % runtimeDataCount_)];
    }
    uint8_t *GetRuntimeData() {
        return &data_[0];
    }
    uint8_t *GetRuntimeDataCurrent() { return GetRuntimeData(GetIndexCurrent()); }
    uint8_t *GetRuntimeDataPending() { return GetRuntimeData(GetIndexPending()); }

    static constexpr int AlignSize = 0x10;

    static constexpr uint64_t GetAlignedSize(uint64_t size) {
        return (size + AlignSize - 1) & ~(AlignSize - 1);
    }

    static constexpr uint64_t GetRingBufferSize(uint64_t runtimeDataSize, uint64_t runtimeDataCount) {
        return sizeof(RuntimeDataRingBufferHead) + GetAlignedSize(runtimeDataSize) * runtimeDataCount;
    }

private:
    uint64_t runtimeDataSize_;
    uint64_t runtimeDataCount_;

    /* ringbuffer's end and begin */
    std::atomic<uint64_t> indexFinished_;
    std::atomic<uint64_t> indexPending_;
    unsigned char data_[0];
};

}
