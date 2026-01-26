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
 * \file aot_binary.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "securec.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/device_perf.h"
#include "tilefwk/aicpu_runtime.h"
#include "tilefwk/aikernel_data.h"

#ifndef STR
#define STR_(n)         #n
#define STR(n)          STR_(n)
#endif

#define AOT_CODE_POOL_CODE_SIZE     (4096 * 0x200)
extern uint8_t aotCodePoolCode[];

namespace npu::tile_fwk::dynamic {
asm(
    "\n\t.pushsection .bss." STR(aotCodePoolCode) ",\"axwG\",@nobits," STR(aotCodePoolCode) ",comdat"
    "\n\t.p2align 12"
    "\n\t.weak " STR(aotCodePoolCode)
    "\n\t.type " STR(aotCodePoolCode) ", @gnu_unique_object"
    "\n\t.size " STR(aotCodePoolCode) ", " STR(AOT_CODE_POOL_CODE_SIZE)
    "\n" STR(aotCodePoolCode) ":"
    "\n\t.zero " STR(AOT_CODE_POOL_CODE_SIZE)
    "\n\t.popsection"
);

const size_t TUBLE_INDEX_2 = 2;
const size_t TUBLE_INDEX_3 = 3;

struct AOTCodePool {
    uintptr_t base{0};
    uintptr_t offset{0};

    void MapExec() {}

    static AOTCodePool &GetCodePool() {
        static AOTCodePool pool = {0};
        if (pool.base == 0) {
            pool.base = (uintptr_t)aotCodePoolCode;
        }
        return pool;
    };
};

struct AOTBinary {
    AOTBinary() {}

    void InitCodeSize(const void *data, uint64_t size) {
        auto &pool = AOTCodePool::GetCodePool();
        PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        memcpy_s(reinterpret_cast<void *>(pool.base), size, data, size);
        __builtin___clear_cache(reinterpret_cast<char *>(pool.base), reinterpret_cast<char *>(pool.base) + size);
        PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        code_ = reinterpret_cast<unsigned char *>(pool.base);
        size_ = size;
    }
    void InitCode(const void *data) {
        code_ = reinterpret_cast<const unsigned char *>(data);
    }

    const unsigned char *code_{nullptr};
    size_t size_{0};
};

struct DeviceExecuteContext;

struct AOTBinaryControlFlow : AOTBinary {
    typedef void (*controlFlowEntry)(
            struct DeviceExecuteContext *ctx, int64_t *symbolTable,
            RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX], DevStartArgsBase *startArgsBase);

    AOTBinaryControlFlow() = default;

    AOTBinaryControlFlow(const std::tuple<const void *, uint64_t> &code, controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(std::get<0>(code), std::get<1>(code), entry) {}

    AOTBinaryControlFlow(const std::vector<uint8_t> &code, controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(code.data(), code.size(), entry) {}

    AOTBinaryControlFlow(const void *code, uint64_t codeSize, controlFlowEntry entry = nullptr) {
        if (entry != nullptr) {
            InitCode(reinterpret_cast<void *>(entry));
        } else {
            InitCodeSize(code, codeSize);
        }
    }

    void CallControlFlow(
            struct DeviceExecuteContext *ctx, int64_t *symbolTable,
            RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX], DevStartArgsBase *startArgsBase) {
        (reinterpret_cast<controlFlowEntry>(const_cast<unsigned char *>(code_)))(ctx, symbolTable, runtimeCallList, startArgsBase);
    }
};

struct AOTBinaryExpressionTable : AOTBinary {
    using exprEntry = uint64_t (*)(struct DeviceExecuteContext *ctx, int64_t *symbolTable);
    AOTBinaryExpressionTable() {}
    AOTBinaryExpressionTable(const std::tuple<const void *, uint64_t, const uint64_t *, uint64_t> &table)
        : offsetList(std::get<TUBLE_INDEX_2>(table)), offsetSize(std::get<TUBLE_INDEX_3>(table)) {
        InitCodeSize(std::get<0>(table), std::get<1>(table));
    }

    uint64_t CallExpr(struct DeviceExecuteContext *ctx, int64_t *symbolTable, uint64_t index) {
        return (reinterpret_cast<exprEntry>(const_cast<unsigned char *>(code_ + offsetList[index])))(ctx, symbolTable);
    }

    const uint64_t *offsetList{nullptr};
    uint64_t offsetSize{0};
};

struct DeviceExecuteProgram {
    DevAscendProgram *prog{nullptr};

    AOTBinaryControlFlow controlFlowBinary;
    AOTBinaryExpressionTable exprBinary;

    DeviceExecuteProgram() {}
    DeviceExecuteProgram(DevAscendProgram *prog_, AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
        : prog(prog_),
          controlFlowBinary(IsDeviceMode() ? prog_->GetDevControlFlowBinary() : prog_->GetHostControlFlowBinary(), entry),
          exprBinary(prog_->GetExpressionTableBinary()) {}

    const void *GetControlFlowEntry() {
        return controlFlowBinary.code_;
    }
};
}
