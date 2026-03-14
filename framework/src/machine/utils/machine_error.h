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
 * \file machine_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
enum class MachineError : uint32_t {
    SCHEDULE       = 70000U, // 调度链路
    CONTROL_FLOW   = 71000U, // 控制流执行
    WORKSPACE      = 72000U, // Workspace / Slab
    DUMP_DFX       = 73000U, // Dump / DFX / Profiling
    PROGRAM_ENCODE = 74000U, // 编解码与一致性
    TENSOR_META    = 75000U, // 张量元信息
    SERVER_KERNEL  = 76000U, // AICPU server / kernel
    THREAD_MACHINE = 77000U, // 线程/机器级
    DATA_STRUCTURE = 78000U, // 内部数据结构
    UNKNOWN        = 79000U, // 未知/预留
};

enum class SchedErr : uint32_t {
    PREFETCH_CHECK_FAILED     = 70001U,
    AIC_TASK_WAIT_TIMEOUT     = 70002U,
    AIV_TASK_WAIT_TIMEOUT     = 70003U,
    TAIL_TASK_WAIT_TIMEOUT    = 70004U,
    ALL_AICORE_SYNC_TIMEOUT   = 70005U,
    HANDSHAKE_TIMEOUT         = 70006U,
    READY_QUEUE_OVERFLOW      = 70007U,
    SIGNAL_QUEUE_OVERFLOW     = 70008U,
    QUEUE_DEQUEUE_WHEN_EMPTY  = 70009U,
    CORE_TASK_PROCESS_FAILED  = 70010U,
    AICPU_TASK_SYNC_TIMEOUT   = 70011U,
    EXCEPTION_RESET_TRIGGERED = 70012U,
    EXCEPTION_SIGNAL_RECEIVED = 70013U,
    THREAD_INIT_ARGS_INVALID  = 70014U,
    UNKNOWN                   = 70099U
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED     = 71001U,
    ROOT_ALLOC_CTX_NULL       = 71002U,
    ROOT_STITCH_CTX_NULL      = 71003U,
    SYNC_FLAG_WAIT_TIMEOUT    = 71004U,
    DEVICE_TASK_BUILD_FAILED  = 71005U,
    READY_QUEUE_INIT_FAILED   = 71006U,
    DEP_DUMP_FAILED           = 71007U,
    READY_QUEUE_DUMP_FAILED   = 71008U,
    TASK_STATS_ABNORMAL       = 71009U,
    UNKNOWN                   = 71099U
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED       = 72001U,
    SLAB_STAGE_LIST_INCONSISTENT = 72002U,
    SLAB_TYPE_INVALID           = 72003U,
    WORKSPACE_INIT_RESOURCE_ERROR = 72004U,
    WORKSPACE_INIT_PARAM_INVALID  = 72005U,
    WS_TENSOR_ADDRESS_OUT_OF_RANGE = 72006U,
    SLAB_CAPACITY_CALC_INVALID  = 72007U,
    UNKNOWN                     = 72099U
};

enum class DumpDfxErr : uint32_t {
    DUMP_MEMCPY_FAILED            = 73001U,
    DUMP_TENSOR_INFO_FAILED       = 73002U,
    DUMP_TENSOR_DATA_FAILED       = 73003U,
    METRIC_ALLOC_OR_WAIT_TIMEOUT  = 73004U,
    PERF_TRACE_FORMAT_ERROR       = 73005U,
    PERF_TRACE_DUMP_ERROR         = 73006U,
    DFX_AICPU_TIMEOUT            = 73007U,
    UNKNOWN                       = 73099U
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = 74001U,
    FUNC_OP_SIZE_MISMATCH        = 74002U,
    STITCH_PRED_SUCC_MISMATCH     = 74003U,
    STITCH_LIST_TOO_LARGE         = 74004U,
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = 74005U,
    CELL_MATCH_PARAM_INVALID     = 74006U,
    PROGRAM_RANGE_VERIFY_FAILED   = 74007U,
    CACHE_RELOC_KIND_INVALID      = 74008U,
    UNKNOWN                       = 74099U
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED   = 75001U,
    TENSOR_ENCODE_PTR_MISMATCH  = 75002U,
    RAW_TENSOR_INDEX_OUT_OF_RANGE = 75003U,
    SHAPE_VALUE_MISMATCH        = 75004U,
    TENSOR_DUMP_INFO_INCONSISTENT = 75005U,
    UNKNOWN                      = 75099U
};

enum class ServerKernelErr : uint32_t {
    DYN_SERVER_ARGS_NULL      = 76001U,
    DYN_SERVER_SAVE_SO_FAILED  = 76002U,
    KERNEL_EXEC_FUNC_FAILED    = 76003U,
    KERNEL_SO_OR_FUNC_LOAD_FAILED = 76004U,
    DYN_SERVER_RUN_FAILED      = 76005U,
    DYN_SERVER_INIT_FAILED     = 76006U,
    UNKNOWN                    = 76099U
};

enum class ThreadErr : uint32_t {
    DEVICE_ARGS_INVALID     = 77001U,
    SIGNAL_HANDLER_ABNORMAL = 77002U,
    RESET_REG_ALL_TRIGGERED = 77003U,
    UNKNOWN                 = 77099U
};

enum class DataStructErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = 78001U,
    SMALL_ARRAY_RESIZE_OOB     = 78002U,
    UNKNOWN                    = 78099U
};

enum class MachineFunctionErr : uint32_t {
    RESERVED = 87000U
};
enum class MachinePassErr : uint32_t {
    RESERVED = 87500U
};

enum class MachineCodegenErr : uint32_t {
    RESERVED = 88000U
};

enum class MachineSimulationErr : uint32_t {
    RESERVED = 88500U
};

enum class MachineDistributedErr : uint32_t {
    RESERVED = 89000U
};

enum class MachineOperationErr : uint32_t {
    RESERVED = 89500U 
};

}  // namespace npu::tile_fwk

