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
 * \file aicore_asc_printf.h
 * \brief AICore wrapper around CANN asc_printf with core id and sys_cnt prefix.
 */

#pragma once

#if IS_AICORE && defined(__ENABLE_ASC_PRINTF__)
// Do NOT include CANN's umbrella utils/debug/asc_printf.h: for arch 3510 it also drags in
// asc_printf_simt_impl.h -> __clang_cce_simt_atomic.h, which defines atomicExch function
// templates. When the same translation unit also compiles aicore_entry_drco.h (DRCO uses the
// native atomicExch intrinsic), that produces an undefined atomicExch symbol at link time.
// Instead include only the AICore printf implementation (__asc_aicore::printf) directly.
#ifndef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __AICORE_ASC_PRINTF_DEFINED_INTERNAL__
#endif
// Skip CANN print_common_head(): ASC_DEVKIT_TIMESTAMP is a devkit build id, not runtime time.
#ifndef __NPU_DEVICE__
#define __NPU_DEVICE__
#define __AICORE_ASC_PRINTF_DEFINED_NPU_DEVICE__
#endif
#include "impl/utils/debug/asc_aicore_printf_impl.h"
#ifdef __AICORE_ASC_PRINTF_DEFINED_NPU_DEVICE__
#undef __NPU_DEVICE__
#undef __AICORE_ASC_PRINTF_DEFINED_NPU_DEVICE__
#endif
#ifdef __AICORE_ASC_PRINTF_DEFINED_INTERNAL__
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __AICORE_ASC_PRINTF_DEFINED_INTERNAL__
#endif

// Prefer PYPTO_AICORE_PRINTF_WITH_CORE(ctx->blockIdx, ...): get_subblockdim() may become 0 after
// KernelEntry, so recomputing the mix formula collapses to get_block_num().
#if defined(__AIV__) && defined(__MIX__)
#define PYPTO_AICORE_PRINTF_CORE_ID() (get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num())
#else
#define PYPTO_AICORE_PRINTF_CORE_ID() (get_block_idx())
#endif

#define PYPTO_AICORE_PRINTF_WITH_CORE(core_id, fmt, ...) \
    __asc_aicore::printf("[core=%u][t=%lu] " fmt "\n", static_cast<uint32_t>(core_id), get_sys_cnt(), ##__VA_ARGS__)
#define PYPTO_AICORE_PRINTF(fmt, ...) PYPTO_AICORE_PRINTF_WITH_CORE(PYPTO_AICORE_PRINTF_CORE_ID(), fmt, ##__VA_ARGS__)
#else
#define PYPTO_AICORE_PRINTF_WITH_CORE(core_id, fmt, ...) ((void)0)
#define PYPTO_AICORE_PRINTF(fmt, ...) ((void)0)
#endif
