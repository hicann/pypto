/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_asc_dump.h
 * \brief AICore thin wrappers around CANN asc_dump for GM / UB / L1 / L0C pointers.
 */

#pragma once

#if IS_AICORE && defined(__ENABLE_ASC_PRINTF__)
#include "utils/debug/asc_dump.h"

// Pointer-based dump (PyPTO machine / codegen has no AscendC LocalTensor / GlobalTensor).
// desc: user tag (e.g. line id); dump_size: element count. Disabled when ASCENDC_DUMP=0.

template <typename T>
INLINE void AicoreAscDumpGm(__gm__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __asc_aicore::asc_dump_gm(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDumpUbuf(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __asc_aicore::asc_dump_ubuf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDumpL1(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __asc_aicore::asc_dump_l1buf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDumpL0c(__cc__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __asc_aicore::asc_dump_cbuf(input, desc, dumpSize);
}

// Route generic asc_dump overloads through explicit asc_dump_* APIs so older CANN
// packages that only declare asc_dump_gm/ubuf/l1buf/cbuf still compile.
template <typename T>
INLINE void AicoreAscDump(__gm__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicoreAscDumpGm(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDump(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicoreAscDumpUbuf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDump(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicoreAscDumpL1(input, desc, dumpSize);
}

template <typename T>
INLINE void AicoreAscDump(__cc__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicoreAscDumpL0c(input, desc, dumpSize);
}

#define PYPTO_AICORE_DUMP_GM(ptr, desc, n) AicoreAscDumpGm((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_UBUF(ptr, desc, n) AicoreAscDumpUbuf((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_L1(ptr, desc, n) AicoreAscDumpL1((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_L0C(ptr, desc, n) AicoreAscDumpL0c((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP(ptr, desc, n) AicoreAscDump((ptr), (desc), (n))
#else
#define PYPTO_AICORE_DUMP_GM(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_UBUF(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_L1(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_L0C(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP(ptr, desc, n) ((void)0)
#endif
