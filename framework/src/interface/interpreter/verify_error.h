/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file verify_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// 一级错误大类，用于和 ASSERT(err_code, cond) 搭配使用
// 具体场景（Scene）可以在后续根据需要扩展
enum class VerifyErrorCategory : uint32_t {
    VERIFY_ENABLE     = 0xB0000U, // 0: 校验环境
    CONTROL_FLOW      = 0xB1000U, // 1: 执行控制流
    EXECUTE_OPERATION = 0xB2000U, // 2：op执行
    OP_DUMP           = 0xB3000U, // 3：op Dump
    VERIFY_RESULT     = 0xB4000U, // 4: 精度比对
};

// 预留二级场景枚举：
//  - VerifyErrorCategory 仅表示大类范围，不直接作为具体错误码使用
//  - 具体错误码请使用对应 Scene 枚举值
enum class VerifyEnableScene : uint32_t {
    VERIFY_NOT_ENABLE        = 0xB0001U, // 校验功能未开启
    VERIFY_LOAD_CALC_OPS_FAILED = 0xB0002U, // 校验依赖的 CalcOps / 校验模块加载失败
};

// 控制流相关场景：函数图/控制流结构不合法
enum class ControlFlowScene : uint32_t {
    INVALID_FUNC_IO_SPEC              = 0xB1001U, // 函数 incast/outcast 规格不一致（通用兜底）
    INVALID_INPLACE_CHAIN             = 0xB1002U, // inplace 链路不满足预期约束（通用兜底）
    INVALID_CALLEE_MAPPING           = 0xB1003U, // callee hash 映射不一致（通用兜底）

    // Function IO / DataView 细分错误
    FUNC_IO_DATAVIEW_NULL            = 0xB1004U, // FunctionIODataPair：incast/outcast DataView 为空
    FUNC_INCAST_COUNT_MISMATCH       = 0xB1005U, // func.GetIncast().size 与 incastDataViewList.size 不一致
    FUNC_OUTCAST_COUNT_MISMATCH      = 0xB1006U, // func.GetOutcast().size 与 outcastDataViewList.size 不一致
    FUNC_TENSOR_DATAVIEW_MISMATCH    = 0xB1007U, // 同一 LogicalTensor 映射到不同 LogicalTensorData
    FUNC_TENSOR_DATAVIEW_LIST_SIZE_MISMATCH = 0xB1008U, // tensorList.size 与 dataViewList.size 不一致
    FUNC_INPLACE_ALLOC_CONFLICT      = 0xB1009U, // AllocateDataView 中 inplaceTensor 非空但期望新分配 RawTensor
    FUNC_TENSOR_DATAVIEW_DUP         = 0xB100AU, // tensorDataViewDict 已包含该 LogicalTensor
    FUNC_SPILL_RAW_TENSOR_DUP        = 0xB100BU, // spillRawTensorDict 已包含该 LogicalTensor
    FUNC_INPLACE_GROUP_NO_FUNC_IO    = 0xB100CU, // inplaceTensorSet 不包含当前函数的 incast/outcast
    FUNC_SLOT_IO_COUNT_MISMATCH      = 0xB100DU, // 函数 incast/outcast 数量与槽位映射个数不一致
    FUNC_SLOT_MISSING                = 0xB100EU, // 指定 slot 在 slotDataViewDict_ 中不存在
    FUNC_UNKNOWN_IO_TYPE             = 0xB100FU, // 未知或不支持的 iotype 分支
};

// 执行算子相关场景：shape/dtype/参数非法等
enum class ExecuteOperationScene : uint32_t {
    INVALID_TENSOR_SHAPE      = 0xB2001U, // 张量 shape / validShape 不匹配
    INVALID_TENSOR_DTYPE      = 0xB2002U, // 张量 dtype 不符合预期
    INVALID_TENSOR_SIZE       = 0xB2003U, // 数据长度/字节数不匹配
    CTX_NULL                  = 0xB2004U, // ExecuteOperationContext 指针为空
    CTX_OP_NULL               = 0xB2005U, // ctx 存在，但 ctx->op 为空
    CTX_INPUT_COUNT_MISMATCH  = 0xB2006U, // 输入 DataView 个数与预期不一致
    CTX_OUTPUT_COUNT_MISMATCH = 0xB2007U, // 输出 DataView 个数与预期不一致
    CTX_INPUT_VIEW_NULL       = 0xB2008U, // 某个输入 DataView 为空
    CTX_OUTPUT_VIEW_NULL      = 0xB2009U, // 某个输出 DataView 为空
    UNSUPPORTED_OPCODE        = 0xB200AU, // 不支持的 Opcode
    EMPTY_VALIDSHAPE          = 0xB200BU, // logiclTensor的validShape为空

    // OP_VIEW_TYPE 专属错误
    VIEWTYPE_BYTES_MISMATCH        = 0xB200CU, // OP_VIEW_TYPE：输入/输出底层字节数不一致
    // CUBE MatMul / Copy 专属错误
    AMULACC_ACC_DTYPE_UNSUPPORTED  = 0xB200DU, // OP_A_MULACC_B：lhs=int8 & acc=fp32 的组合不受支持
    L0C_TO_L1_SHAPE_NOT_2D         = 0xB200EU, // OP_L0C_TO_L1：输入/输出 shape 不是 2D

    // ExecuteOperation 通用兜底错误：运行期抛出的 std::exception 统一归此
    RUNTIME_EXCEPTION              = 0xB200FU,
};


// Dump/IO 相关场景：文件读写错误等
enum class OpDumpScene : uint32_t {
    DUMP_OPEN_FILE_FAILED     = 0xB3001U, // 打开 dump 文件失败
    DUMP_WRITE_FILE_FAILED    = 0xB3002U, // 写入 dump 文件失败
};

// 精度/结果验证相关场景
enum class VerifyResultScene : uint32_t {
    VERIFY_RESULT_MISMATCH    = 0xB4001U, // 精度比对失败
    VERIFY_RESULT_SHAPE_DIFF  = 0xB4002U, // 比对双方 shape 不一致
    VERIFY_RESULT_DTYPE_DIFF  = 0xB4003U, // 比对双方 dtype 不一致
};

// Element 封装/标量计算相关场景：Element 内部 dtype/运算非法
enum class ElementScene : uint32_t {
    INVALID_ELEMENT_DTYPE = 0xB5001U, // Element：dtype 不符合预期或不支持的 Element 运算
};
} // namespace npu::tile_fwk