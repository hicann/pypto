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
 * \file aikernel_root_function.h
 * \brief
 */

#ifndef AIKERNEL_ROOT_FUNCTION_H
#define AIKERNEL_ROOT_FUNCTION_H

#include "tilefwk/aikernel_tensor.h"

namespace npu::tile_fwk {

constexpr uint32_t DUPPED_STITCH_NODE_U32_SIZE = 0x10;
constexpr uint32_t DUPPED_STITCH_SIZE = DUPPED_STITCH_NODE_U32_SIZE - (sizeof(void*) / sizeof(uint32_t)) - 0x1;
// nodeNext 编码（全部落在低 6 位，节点地址 64 字节对齐 → 低 6 位恒为 0）：
// nodeNext == 0 表示无后继节点；非 0 时：
//   bit0-5   = 后续节点数 R（以本节点为头的子链内、除自己外的节点数，≤63）
constexpr uint64_t DUPPED_STITCH_NODE_ADDR_MASK = 0xFFFFFFFFFFFFFFC0ULL;
constexpr uint32_t DUPPED_STITCH_NODE_REMAIN_COUNT_MASK = 0x3F;
constexpr uint64_t DUPPED_STITCH_NODE_ALIGN = DUPPED_STITCH_NODE_REMAIN_COUNT_MASK + 1;

struct DevAscendFunctionOperationSuccInfo {
    // 单个 op 的后继描述信息，打包为一个 u64（静态编码，只读）。所有 DRCO 静态后继统一
    // 以 succNode 节点存储（大扇出分块与小平铺同构），低 32bit 恒为节点区语义：
    /*
        就地模式（bit63 = 0，DRCO_SUCC_PARALLEL_BIT 未置位，生产核串行解依赖）：
        |------31bit------|-----16bit------|-----16bit------|
        |---stitchIndex---|--succNodeSize--|--succNodeIdx---|
        散射模式（bit63 = 1，大扇出分块 op，节点散射由消费核并行解）：
        |------31bit------|-----16bit------|-----16bit------|
        |---stitchIndex---|--succNodeSize--|--succNodeIdx---|
    */
    uint64_t staticIndexSizeAndStitchIndex;
};

struct DevAscendFunctionDuppedStitchNode {
#ifdef __TILE_FWK_HOST__
    void InitWithNext(DevAscendFunctionDuppedStitchNode* next)
    {
        nodeNext = next;
        nodeSize = 0;
    }

    void SafePushBack(uint32_t taskId) { nodeTaskList[nodeSize++] = taskId; }

    uint32_t Size() const { return nodeSize; }

    // 遍历接口：内部解码 nodeNext 低 6 位编码后返回真实后继地址
    DevAscendFunctionDuppedStitchNode* Next() const
    {
        return (DevAscendFunctionDuppedStitchNode*)((uint64_t)nodeNext & DUPPED_STITCH_NODE_ADDR_MASK);
    }

    // 原始字段引用：仅内部编码写入 / 控制流缓存重定位使用，外部禁止
    DevAscendFunctionDuppedStitchNode* const& NextRaw() const { return nodeNext; }
    DevAscendFunctionDuppedStitchNode*& NextRaw() { return nodeNext; }

    // 函数在核心流程，已在Size()内循环，校验会影响性能
    uint32_t At(uint32_t idx) const { return nodeTaskList[idx]; }

    template <typename Callback>
    void ForEach(Callback&& callback) const
    {
        for (uint32_t i = 0; i < nodeSize; i++) {
            callback(nodeTaskList[i]);
        }
    }
#endif
    __gm__ DevAscendFunctionDuppedStitchNode* nodeNext;
    uint32_t nodeSize;
    uint32_t nodeTaskList[DUPPED_STITCH_SIZE];
};

// 静态后继节点（全量 DRCO 视图后继，大扇出分块与小扇出就地同构）：devprog 常驻扁平数组
// （无链指针），仅复用 stitch matrix 的 slot 通路；散射节点内类型同质（coreType 记于
// 头，就地节点混合类型、头存首条目 ct 仅 debug）；64B，自然对齐（与原平铺表同口径）
struct DevAscendFunctionStaticSuccNode {
    static constexpr uint32_t MAX_SUCC_LIST_SIZE = 15;
#ifdef __TILE_FWK_HOST__
    void Init(uint16_t nodeCoreType)
    {
        coreType = nodeCoreType;
        succSize = 0;
    }

    void SafePushBack(uint32_t taskId) { succList[succSize++] = taskId; }
#endif
    uint16_t coreType;
    uint16_t succSize;
    uint32_t succList[MAX_SUCC_LIST_SIZE];
};
static_assert(sizeof(DevAscendFunctionStaticSuccNode) == 64, "static succ node must span one cacheline");
} // namespace npu::tile_fwk

#endif
