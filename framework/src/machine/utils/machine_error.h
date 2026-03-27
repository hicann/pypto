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
#include <type_traits>
#include "interface/utils/common.h"

namespace npu::tile_fwk {
enum class MachineError : uint32_t {
    HOST_BACKEND   = 0x70000U, // Host 后端（总段标识）
    HOST_LAUNCHER  = 0x71000U, // Host 启动器（总段标识）
    SCHEDULE       = 0x72000U, // 调度链路
    CONTROL_FLOW   = 0x73000U, // 控制流执行
    WORKSPACE      = 0x74000U, // Workspace / Slab
    DUMP_DFX       = 0x75000U, // Dump / DFX / Profiling
    PROGRAM_ENCODE = 0x76000U, // 编解码与一致性
    TENSOR_META    = 0x77000U, // 张量元信息
    SERVER_KERNEL  = 0x78000U, // AICPU server / kernel
    THREAD_MACHINE = 0x79000U, // 线程/机器级
    DEV_DATA       = 0x7A000U, // 设备侧数据结构子码段（DevDataErr：0x01–0x0A）
    DEV_COMMON     = 0x7A000U, // 设备侧通用子码段（DevCommonErr：0x0B–0x14，与 DEV_DATA 同段基址）
    RUNTIME_ERROR  = 0x7B000U, // aclRt 调用错误
    UNKNOWN        = 0x7F000U, // 未知/预留
};

enum class DevDataErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB      = ToUnderlying(MachineError::DEV_DATA) + 0x01U, // 设备 reloc 向量索引越界
    SMALL_ARRAY_RESIZE_OOB          = ToUnderlying(MachineError::DEV_DATA) + 0x02U, // 小数组 resize 越界
    VECTOR_UNINITIALIZED            = ToUnderlying(MachineError::DEV_DATA) + 0x03U, // vector 未初始化
    VECTOR_INDEX_OUT_OF_RANGE       = ToUnderlying(MachineError::DEV_DATA) + 0x04U, // vector 索引越界
    VECTOR_EMPTY_ACCESS             = ToUnderlying(MachineError::DEV_DATA) + 0x05U, // 空 vector 访问 front/back
    ITEM_POOL_UNINITIALIZED         = ToUnderlying(MachineError::DEV_DATA) + 0x06U, // item pool 未初始化
    ITEM_POOL_FREE_LIST_INVALID     = ToUnderlying(MachineError::DEV_DATA) + 0x07U, // freelist 非法
    ITEM_POOL_INDEX_OUT_OF_RANGE    = ToUnderlying(MachineError::DEV_DATA) + 0x08U, // item pool 索引越界
    SHEET_COLUMN_MISMATCH           = ToUnderlying(MachineError::DEV_DATA) + 0x09U, // 列数量不匹配
    SHEET_COLUMN_INDEX_OUT_OF_RANGE = ToUnderlying(MachineError::DEV_DATA) + 0x0AU, // 列索引越界
};

enum class DevCommonErr : uint32_t {
    MEMCPY_FAILED      = ToUnderlying(MachineError::DEV_COMMON) + 0x01U, // memcpy_s 失败
    ALLOC_FAILED       = ToUnderlying(MachineError::DEV_COMMON) + 0x02U, // 分配失败
    MALLOC_FAILED      = ToUnderlying(MachineError::DEV_COMMON) + 0x03U,
    NULLPTR            = ToUnderlying(MachineError::DEV_COMMON) + 0x04U, // 空指针访问
    PARAM_INVALID      = ToUnderlying(MachineError::DEV_COMMON) + 0x05U, // 参数非法
    PARAM_CHECK_FAILED = ToUnderlying(MachineError::DEV_COMMON) + 0x06U, // 参数校验失败
    FILE_ERROR         = ToUnderlying(MachineError::DEV_COMMON) + 0x07U, // 文件错误
    CMD_ERROR          = ToUnderlying(MachineError::DEV_COMMON) + 0x08U, // 命令错误
    GET_ENV_FAILED     = ToUnderlying(MachineError::DEV_COMMON) + 0x09U, // 获取环境变量路径失败
    GET_HANDLE_FAILED  = ToUnderlying(MachineError::DEV_COMMON) + 0x0AU, // 获取句柄失败
    FREE_FAILED        = ToUnderlying(MachineError::DEV_COMMON) + 0x0BU, // 释放失败
};

enum class HostBackEndErr : uint32_t {
    COMPILE_AICORE_FAILED    = ToUnderlying(MachineError::HOST_BACKEND) + 0x01U, // 生成/执行 AICore 编译命令失败
    COMPILE_CCEC_FAILED      = ToUnderlying(MachineError::HOST_BACKEND) + 0x02U, // ccec 编译失败
    LINK_FAILED              = ToUnderlying(MachineError::HOST_BACKEND) + 0x03U, // 链接 core machine 命令失败
    GEN_AICORE_FILE_FAILED   = ToUnderlying(MachineError::HOST_BACKEND) + 0x04U, // 生成 AICore 源码失败
    GEN_DYNAMIC_OP_FAILED    = ToUnderlying(MachineError::HOST_BACKEND) + 0x05U, // 生成动态算子失败
    PRECOMPILE_FAILED        = ToUnderlying(MachineError::HOST_BACKEND) + 0x06U, // 算子预编译失败
    FUNCTION_CACHE_HASH_MISS = ToUnderlying(MachineError::HOST_BACKEND) + 0x07U, // 函数缓存 hash 未命中/不一致
    DUPLICATE_LEAF_FUNC_HASH = ToUnderlying(MachineError::HOST_BACKEND) + 0x08U, // leaf 函数 hash 重复
};

enum class HostLauncherErr : uint32_t {
    LAUNCH_AICPU_FAILED                 = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x01U, // 启动 AICPU 任务失败
    LAUNCH_PREPARE_FAILED               = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x02U, // launch 准备阶段失败
    LAUNCH_CUSTOM_AICPU_FAILED          = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x03U, // 启动 custom AICPU 失败
    LAUNCH_AICORE_FAILED                = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x04U, // 启动 AICore 失败
    LAUNCH_BUILTIN_OP_NULL_FAILED       = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x05U, // 内置算子句柄/函数为空
    REGISTER_KERNEL_FAILED              = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x06U, // 注册 kernel bin 失败
    PREPARE_ARGS_FAILED                 = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x07U, // kernel 参数准备失败
    MAP_REG_ADDR_FAILED                 = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x08U, // 寄存器地址映射失败
    MEM_POOL_CHECK_ALL_SENTINELS_FAILED = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x09U, // 哨兵校验失败
    TRIPLE_STREAM_ERROR                 = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x0AU, // 三级流水报错
    SYNC_FAILED                         = ToUnderlying(MachineError::HOST_LAUNCHER) + 0x0BU, // 同步失败
};

enum class SchedErr : uint32_t {
    TASK_WAIT_TIMEOUT        = ToUnderlying(MachineError::SCHEDULE) + 0x01U, // AIC 任务等待超时
    HANDSHAKE_TIMEOUT        = ToUnderlying(MachineError::SCHEDULE) + 0x02U, // 线程握手超时
    READY_QUEUE_OVERFLOW     = ToUnderlying(MachineError::SCHEDULE) + 0x03U, // 就绪队列溢出
    CORE_TASK_EXEC_FAILED    = ToUnderlying(MachineError::SCHEDULE) + 0x04U, // core 任务执行返回错误
    CORE_TASK_PROCESS_FAILED = ToUnderlying(MachineError::SCHEDULE) + 0x05U, // core 任务处理失败
    RINGBUFFER_WAIT_TIMEOUT    = ToUnderlying(MachineError::SCHEDULE) + 0x06U, // ring buf 等待超时
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED    = ToUnderlying(MachineError::CONTROL_FLOW) + 0x01U, // 控制流执行失败
    ROOT_ALLOC_CTX_NULL      = ToUnderlying(MachineError::CONTROL_FLOW) + 0x02U, // root alloc 上下文为空
    ROOT_STITCH_CTX_NULL     = ToUnderlying(MachineError::CONTROL_FLOW) + 0x03U, // root stitch 上下文为空
    DEVICE_TASK_BUILD_FAILED = ToUnderlying(MachineError::CONTROL_FLOW) + 0x04U, // device task 构建失败
    TASK_STATS_ABNORMAL      = ToUnderlying(MachineError::CONTROL_FLOW) + 0x05U, // 任务统计异常
    CTRL_INIT_FAILED         = ToUnderlying(MachineError::CONTROL_FLOW) + 0x06U, // 控制流初始化失败
    CTRL_SIM_FAILED          = ToUnderlying(MachineError::CONTROL_FLOW) + 0x07U, // 模拟控制流失败
    CTRL_ALLOC_TIMEOUT       = ToUnderlying(MachineError::CONTROL_FLOW) + 0x08U, // 控制流分配资源超时
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED             = ToUnderlying(MachineError::WORKSPACE) + 0x01U, // slab 添加 cache 失败
    SLAB_STAGE_LIST_INCONSISTENT      = ToUnderlying(MachineError::WORKSPACE) + 0x02U, // slab stage 列表不一致
    SLAB_TYPE_INVALID                 = ToUnderlying(MachineError::WORKSPACE) + 0x03U, // slab 类型非法
    WORKSPACE_INIT_RESOURCE_ERROR     = ToUnderlying(MachineError::WORKSPACE) + 0x04U, // workspace 初始化资源错误
    WORKSPACE_INIT_PARAM_INVALID      = ToUnderlying(MachineError::WORKSPACE) + 0x05U, // workspace 初始化参数非法
    WS_TENSOR_ADDRESS_OUT_OF_RANGE    = ToUnderlying(MachineError::WORKSPACE) + 0x06U, // tensor 地址越界
    WORKSPACE_ITER_INVALID            = ToUnderlying(MachineError::WORKSPACE) + 0x07U, // workspace 迭代器非法
    WORKSPACE_REFCOUNT_INVALID        = ToUnderlying(MachineError::WORKSPACE) + 0x08U, // 引用计数非法
    WORKSPACE_ALLOCATOR_REGIST_FAILED = ToUnderlying(MachineError::WORKSPACE) + 0x09U, // allocator 注册失败
    WORKSPACE_CATEGORY_INVALID        = ToUnderlying(MachineError::WORKSPACE) + 0x0AU, // category 非法
    WORKSPACE_CAPACITY_INSUFFICIENT   = ToUnderlying(MachineError::WORKSPACE) + 0x0BU, // 容量不足
    WORKSPACE_BASE_ADDR_OUT_OF_RANGE  = ToUnderlying(MachineError::WORKSPACE) + 0x0CU, // 基址越界
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR     = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x01U, // 动态函数数据对齐错误
    FUNC_OP_SIZE_MISMATCH            = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x02U, // func op 大小不一致
    STITCH_PRED_SUCC_MISMATCH        = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x03U, // stitch 前驱后继不一致
    STITCH_LIST_TOO_LARGE            = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x04U, // stitch 列表过大
    STITCH_HANDLE_INDEX_OUT_OF_RANGE = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x05U, // stitch handle 索引越界
    CELL_MATCH_PARAM_INVALID         = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x06U, // cell match 参数非法
    PROGRAM_RANGE_VERIFY_FAILED      = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x07U, // program 范围校验失败
    CACHE_RELOC_KIND_INVALID         = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x08U, // cache reloc 类型非法
    ADDR_OFFSET_RAW_MAGIC_MISMATCH   = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x09U, // 地址偏移与 raw magic 不匹配
    CALL_OP_COUNT_EXCEEDS_UINT16_MAX = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x0AU, // call op 数量超过 uint16 上限
    CELL_MATCH_DIM_ZERO              = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x0BU, // cell match 维度为 0
    ASSEMBLE_STITCH_MEMORY_EXCESS    = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x0CU, // assemble stitch 内存超限
    LEAF_CALLEE_ATTR_NULL            = ToUnderlying(MachineError::PROGRAM_ENCODE) + 0x0DU, // leaf 被调属性为空
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED     = ToUnderlying(MachineError::TENSOR_META) + 0x01U, // 维度个数超限
    TENSOR_ENCODE_PTR_MISMATCH    = ToUnderlying(MachineError::TENSOR_META) + 0x02U, // 编码指针不一致
    RAW_TENSOR_INDEX_OUT_OF_RANGE = ToUnderlying(MachineError::TENSOR_META) + 0x03U, // raw tensor 索引越界
    SHAPE_VALUE_MISMATCH          = ToUnderlying(MachineError::TENSOR_META) + 0x04U, // shape 数值不一致
    INCAST_ADDRESS_NULL           = ToUnderlying(MachineError::TENSOR_META) + 0x05U, // incast 地址为空
    OUTCAST_ADDRESS_NULL          = ToUnderlying(MachineError::TENSOR_META) + 0x06U, // outcast 地址为空
    RUNTIME_WORKSPACE_NULL        = ToUnderlying(MachineError::TENSOR_META) + 0x07U, // runtime workspace 为空
};

enum class ServerKernelErr : uint32_t {
    KERNEL_EXEC_FAILED = ToUnderlying(MachineError::SERVER_KERNEL) + 0x01U, // kernel 执行函数失败
};

enum class ThreadErr : uint32_t {
    SIGNAL_HANDLER_ABNORMAL = ToUnderlying(MachineError::THREAD_MACHINE) + 0x01U, // 信号处理异常
    RESET_REG_ALL_TRIGGERED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x02U, // 触发全量寄存器复位
    THREAD_CPU_ALLOC_FAILED = ToUnderlying(MachineError::THREAD_MACHINE) + 0x03U, // CPU 分配线程失败
};

enum class RtErr : uint32_t {
    RT_INIT_FAILED     = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x01U, // ACL/RT 初始化失败
    RT_MEMCPY_FAILED   = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x02U, // rtMemcpy 失败
    RT_MEMSET_FAILED   = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x03U, // rtMemset 失败
    RT_MALLOC_FAILED   = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x04U, // rtMalloc 失败
    RT_LAUNCH_FAILED   = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x05U, // rt 启动失败
    RT_EVENT_FAILED    = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x06U, // rt event/stream 相关失败
    RT_CAPTURE_FAILED  = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x07U, // capture 信息/状态失败
    RT_REGISTER_FAILED = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x08U, // kernel 注册/注销失败
    RT_LOAD_FAILED     = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x09U, // 加载失败
    RT_GET_FUNC_FAILED = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x0AU, // 获取函数句柄失败
    RT_DEVICE_FAILED   = ToUnderlying(MachineError::RUNTIME_ERROR) + 0x0BU, // 设备 id/设备相关失败
};
}  // namespace npu::tile_fwk
