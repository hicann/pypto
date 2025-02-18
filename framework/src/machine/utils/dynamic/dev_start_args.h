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

#include "machine/utils/dynamic/dev_encode.h"

namespace npu::tile_fwk::dynamic {
const uint32_t DUMP_INDEX_SIZE_2 = 2;
const uint32_t DUMP_INDEX_SIZE_4 = 4;

struct DevInputSymbol {
    int64_t value;
};

struct DevStartArgs : DevStartArgsBase {
    uint64_t contextWorkspaceAddr;
    uint64_t contextWorkspaceSize;
    DevAscendProgram *devProg;

    DevInputSymbol *inputSymbolList;
    uint64_t inputSymbolSize;
    const void *controlFlowEntry;
    std::atomic<uint64_t> syncFlag{0}; // sche and ctrl soft sync flag

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
}