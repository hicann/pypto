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
 * \file opcode.h
 * \brief
 */

#pragma once

#include <string>
#include <map>
#include <array>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "verifier.h"
namespace npu::tile_fwk {
enum class Opcode {
    // Unary Vector
    OP_EXP,
    OP_NEG,
    OP_RSQRT,
    OP_SQRT,
    OP_RECIPROCAL,
    OP_CAST,
    OP_EXPAND,
    OP_ONEHOT,
    OP_COMPACT,
    OP_LOGICALNOT,
    OP_ROWMAX,
    OP_ROWSUM,
    OP_ROWEXPMAX,
    OP_ROWEXPSUM,
    OP_ROWSUMLINE,
    OP_ROWMAXLINE,
    OP_ROWMINLINE,
    OP_ADDS,
    OP_SUBS,
    OP_MULS,
    OP_DIVS,
    OP_MAXS,
    OP_MINS,
    OP_POW,
    OP_S_ADDS,
    OP_S_SUBS,
    OP_S_MULS,
    OP_S_DIVS,
    OP_S_MAXS,
    OP_S_MINS,
    OP_TRANSPOSE_MOVEIN,
    OP_TRANSPOSE_MOVEOUT,
    OP_TRANSPOSE_VNCHWCONV,
    OP_ABS,
    OP_LN,
    OP_HUB,
    OP_BRCB,
    // Binary Vector
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_ADD_BRC,
    OP_SUB_BRC,
    OP_MUL_BRC,
    OP_DIV_BRC,
    OP_MAX_BRC,
    OP_MIN_BRC,
    OP_S_ADD,
    OP_S_SUB,
    OP_S_MUL,
    OP_S_DIV,
    OP_S_MAX,
    OP_S_MIN,
    OP_MAXIMUM,
    OP_MINIMUM,
    OP_GATHER_FROM_UB,
    OP_GATHER,
    OP_GATHER_ELEMENT,
    OP_SCATTER_ELEMENT,
    OP_SCATTER,
    OP_INDEX_PUT,
    OP_INDEX_ADD,
    OP_CONCAT,
    OP_CUM_SUM,
    OP_SCATTER_UPDATE,
    OP_SCATTER_SCALAR,
    OP_PAIRMAX,
    OP_PAIRMIN,
    OP_PAIRSUM,
    OP_ROWMAX_SINGLE,
    OP_ROWMIN_SINGLE,
    OP_ROWSUM_SINGLE,
    OP_ROWMAX_COMBINE_AXIS_SINGLE,
    OP_ROWSUM_COMBINE_AXIS_SINGLE,
    OP_CMP,
    OP_CMPS,
    OP_WHERE_TT,
    OP_WHERE_TS,
    OP_WHERE_ST,
    OP_WHERE_SS,
    OP_LOGICALAND,
    // Cube
    OP_A_MUL_B,
    OP_A_MULACC_B,
    OP_A_MUL_BT,
    OP_A_MULACC_BT,
    OP_AT_MUL_B,
    OP_AT_MUL_BT,

    OP_CONV,
    OP_CONV_ADD,
    OP_CUBE_CONV_D2S,
    OP_CUBE_CONCAT_C,
    OP_L1_TO_L0A,
    OP_L1_TO_L0B,
    OP_L1_TO_BT,
    // ANY
    OP_DUPLICATE,
    // View
    OP_RESHAPE,
    OP_RESHAPE_COPY_IN,
    OP_RESHAPE_COPY_OUT,
    OP_ASSEMBLE,
    OP_ASSEMBLE_SSA,
    OP_VIEW,
    OP_VIEW_TYPE,
    // Move
    OP_INDEX_OUTCAST,
    OP_REGISTER_COPY,
    OP_CONVERT,
    OP_COPY_IN,
    OP_COPY_OUT,
    OP_GATHER_IN_L1,
    OP_GATHER_IN_UB,
    OP_LOAD,
    // Special
    OP_CALL,
    OP_CALL_NOT_EXPAND,
    OP_PRINT,
    OP_NOP,

    OP_UB_ALLOC,
    OP_UB_COPY_IN,
    OP_UB_COPY_OUT,
    OP_VEC_DUP,
    OP_REG_ALLOC,
    OP_VLD,
    OP_VST,

    // Cube
    OP_L1_ALLOC,
    OP_L0A_ALLOC,
    OP_L0B_ALLOC,
    OP_L0C_ALLOC,
    OP_FIX_ALLOC,
    OP_BT_ALLOC,

    // MTE
    OP_L1_COPY_IN,
    OP_L1_COPY_IN_FRACTAL_Z,
    OP_L1_COPY_OUT,
    OP_L1_LOOP_ENHANCE,

    OP_L1_COPY_IN_DMA,
    OP_L1_COPY_OUT_DMA,

    OP_L0C_COPY_OUT,
    OP_L1_TO_L0_AT,
    OP_L1_TO_L0_BT,
    OP_L1_TO_FIX,
    OP_L1_TO_FIX_QUANT_PRE,
    OP_L1_TO_FIX_RELU_PRE,
    OP_L1_TO_FIX_RELU_POST,
    OP_L1_TO_FIX_QUANT_POST,
    OP_L1_TO_FIX_ELT_ANTIQ,
    OP_L1_TO_FIX_MTE2_ANTIQ,
    OP_L1_COPY_UB,
    OP_L0C_COPY_UB,
    OP_UB_COPY_L1,
    OP_UB_COPY_L1_ND,
    OP_L1_TO_L1,
    OP_COPY_UB_TO_UB,
    OP_L0C_TO_L1,
    OP_UB_COPY_ND2NZ,

    // Scala
    OP_SYNC_SRC,
    OP_SYNC_DST,
    OP_CV_SYNC_SRC,
    OP_CV_SYNC_DST,
    OP_PHASE1,
    OP_PHASE2,
    OP_BAR_V,
    OP_BAR_M,
    OP_BAR_ALL,
    OP_PAD,

    // Distributed
    OP_SEND_TO_ROUTING_EXPERT,
    OP_SEND_TO_SHARED_EXPERT,
    OP_COPY_TO_LOCAL_EXPERT,
    OP_DISPATCH_SET_FLAG,
    OP_FFN_SCHED,
    OP_FFN_BATCHING,
    OP_FFN_COMBINEINFO,
    OP_FFN_VALIDCNT,
    OP_SHMEM_SET,
    OP_SHMEM_PUT,
    OP_SHMEM_PUT_UB2GM,
    OP_SHMEM_SIGNAL,
    OP_SHMEM_WAIT_UNTIL,
    OP_SHMEM_GET,
    OP_SHMEM_GET_GM2UB,
    OP_SHMEM_REDUCE,
    OP_BIND_TENSOR,
    OP_SHMEM_MOE_COMBINE_SEND,
    OP_SHMEM_MOE_COMBINE_RECEIVE,
    // Begin: add for TOPK and ArgSort
    OP_TOPK,
    OP_TILEDMRGSORT,
    OP_BITSORT,
    OP_MRGSORT,
    OP_ARGSORT,
    OP_EXTRACT,
    OP_FUSED_OP,
    // End: add for TOPK and ArgSort
    // Begin: topk for DS3.2-Day0
    OP_TOPK_SORT,
    OP_TOPK_MERGE,
    OP_TOPK_EXTRACT,
    // End: topk for DS3.2-Day0
    // Begin: add for Reduce Atomic
    OP_REDUCE_ACC,
    // End: add for Reduce Atomic
    // Begin: aicpu-aicore communication
    OP_AICPU_CALL_AIC,
    OP_AICPU_CALL_AIV,
    // End: aicpu-aicore communication
    OP_MAX_POOL,
    OP_RANGE,
    // Begin: parallel sort
    OP_SORT,
    OP_COMPARE_SWAP,
    OP_MERGE,
    // End: parallel sort
    OP_UNKNOWN
};

enum class OpCoreType { AIC, AIV, ANY, AICPU, HUB, GMATOMIC };

enum class OpCalcType {
    ELMWISE,
    CAST,
    BROADCAST,
    OTHER,
    REDUCE,
    MATMUL,
    CONV,
    MOVE_IN,
    MOVE_OUT,
    MOVE_LOCAL,
    SYNC,        // 同步
    DISTRIBUTED, // 通信
    SYS,         // 框架
    CALC_TYPE_BOTTOM
};

class TileOpCfg {
public:
    TileOpCfg() {};
    TileOpCfg(std ::string code, PipeType pipeIdStart, PipeType pipeIdEnd, CoreType coreType)
        : tileOpCode_(code), pipeIdStart_(pipeIdStart), pipeIdEnd_(pipeIdEnd), coreType_(coreType) {}
    std::string tileOpCode_;
    PipeType pipeIdStart_{PipeType::PIPE_S};
    PipeType pipeIdEnd_{PipeType::PIPE_S};
    CoreType coreType_{CoreType::AIV};
};

class OpcodeManager {
public:
    static OpcodeManager &Inst() {
        static OpcodeManager inst;
        return inst;
    }

    bool HasOpcode(Opcode opcode) const {
        return static_cast<int>(opcode) >= 0 && static_cast<size_t>(opcode) < opcodeInfos_.size();
    }
    bool HasOpcode(const std::string &str) const { return strToEnum_.count(str) > 0; }

    Opcode GetOpcode(const std::string &str) const {
        auto it = strToEnum_.find(str);
        ASSERT(it != strToEnum_.end());
        return it->second;
    }
    const std::string &GetOpcodeStr(Opcode opcode) const {
        ASSERT(HasOpcode(opcode));
        return opcodeInfos_[static_cast<int>(opcode)].str;
    }

    OpCoreType GetCoreType(Opcode opcode) const {
        ASSERT(HasOpcode(opcode)) << "Can't find op " << static_cast<int>(opcode) << std::endl;
        return opcodeInfos_[static_cast<int>(opcode)].coreType;
    }

    const TileOpCfg &GetTileOpCfg(Opcode opcode) const {
        ASSERT(HasOpcode(opcode)) << "Can't find op " << static_cast<int>(opcode) << std::endl;
        return opcodeInfos_[static_cast<int>(opcode)].tileOpCfg;
    }

    bool MemTypeSensitive(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return !info.inputsMemType.empty() || !info.outputsMemType.empty();
    }

    const std::vector<MemoryType> &GetInputsMemType(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.inputsMemType;
    }

    const std::vector<MemoryType> &GetOutputsMemType(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.outputsMemType;
    }

    OpCalcType GetOpCalcType(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.calcType;
    }

    const std::vector<std::string> &GetAttrs(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.attrs;
    }

    VerifyOperationEntry GetVerifyOperationEntry(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.verifyOperationEntry;
    }

    bool HasStaticAttribute(Opcode opcode, const std::string &attribute) const {
        if (attribute.empty()) {
            return false;
        }
        std::vector<std::string> attrs = GetAttrs(opcode);
        return std::find(attrs.begin(), attrs.end(), attribute) != attrs.end();
    }

    std::string PrintSupportOpcodes() const {
        std::stringstream ss;
        ss << "[";
        bool isFirst = true;
        for (const auto &info : opcodeInfos_) {
            if (!isFirst) {
                ss << ", ";
            }
            isFirst = false;
            ss << info.str;
        }
        ss << "]";
        return ss.str();
    }

    bool IsBoundaryIn(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.calcType == OpCalcType::MOVE_IN;
    }

    bool IsBoundaryOut(Opcode opcode) const {
        auto &info = opcodeInfos_[static_cast<int>(opcode)];
        return info.calcType == OpCalcType::MOVE_OUT;
    }

    inline bool IsCopyIn(Opcode opCode) const {
        return opCode == Opcode::OP_COPY_IN || opCode == Opcode::OP_UB_COPY_IN || opCode == Opcode::OP_L1_COPY_IN ||
               opCode == Opcode::OP_TRANSPOSE_MOVEIN || opCode == Opcode::OP_RESHAPE_COPY_IN ||
               opCode == Opcode::OP_L1_TO_FIX_QUANT_PRE || opCode == Opcode::OP_L1_TO_BT;
    }

    inline bool IsCopyOut(Opcode opCode) const {
        return opCode == Opcode::OP_COPY_OUT || opCode == Opcode::OP_UB_COPY_OUT || opCode == Opcode::OP_L0C_COPY_OUT ||
               opCode == Opcode::OP_L1_COPY_OUT || opCode == Opcode::OP_TRANSPOSE_MOVEOUT ||
               opCode == Opcode::OP_INDEX_OUTCAST ||
               opCode == Opcode::OP_FFN_SCHED || opCode == Opcode::OP_FFN_BATCHING ||
               opCode == Opcode::OP_FFN_COMBINEINFO || opCode == Opcode::OP_FFN_VALIDCNT ||
               opCode == Opcode::OP_COPY_TO_LOCAL_EXPERT || opCode == Opcode::OP_SHMEM_PUT ||
               opCode == Opcode::OP_SHMEM_SIGNAL || opCode == Opcode::OP_SHMEM_GET ||
               opCode == Opcode::OP_SHMEM_REDUCE || opCode == Opcode::OP_RESHAPE_COPY_OUT ||
               opCode == Opcode::OP_SHMEM_PUT_UB2GM || opCode == Opcode::OP_SHMEM_GET_GM2UB ||
               opCode == Opcode::OP_SHMEM_MOE_COMBINE_SEND || opCode == Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE;
    }

    inline bool IsCopyInOrOut(Opcode opCode) const { return IsCopyIn(opCode) || IsCopyOut(opCode); }

    inline bool IsSync(Opcode opcode) const { return opcode == Opcode::OP_SYNC_SRC || opcode == Opcode::OP_SYNC_DST; }

    inline bool IsSharedMemory(Opcode opCode) const {
        return opCode == Opcode::OP_SHMEM_WAIT_UNTIL || opCode == Opcode::OP_SHMEM_PUT ||
            opCode == Opcode::OP_SHMEM_SIGNAL || opCode == Opcode::OP_SHMEM_GET ||
            opCode == Opcode::OP_SHMEM_REDUCE || opCode == Opcode::OP_SHMEM_SET ||
            opCode == Opcode::OP_SHMEM_PUT_UB2GM || opCode == Opcode::OP_SHMEM_GET_GM2UB ||
            opCode == Opcode::OP_SHMEM_MOE_COMBINE_SEND || opCode == Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE ||
            opCode == Opcode::OP_FFN_BATCHING || opCode == Opcode::OP_SEND_TO_ROUTING_EXPERT ||
            opCode == Opcode::OP_COPY_TO_LOCAL_EXPERT || opCode == Opcode::OP_DISPATCH_SET_FLAG ||
            opCode == Opcode::OP_FFN_SCHED || opCode == Opcode::OP_FFN_COMBINEINFO ||
            opCode == Opcode::OP_FFN_VALIDCNT;
    }

private:
    struct OpcodeInfo {
        Opcode opcode;
        OpCoreType coreType;
        std::string str;
        std::vector<MemoryType> inputsMemType;
        std::vector<MemoryType> outputsMemType;
        TileOpCfg tileOpCfg;
        OpCalcType calcType;
        std::vector<std::string> attrs;
        VerifyOperationEntry verifyOperationEntry;
    };

private:
    OpcodeManager();

private:
    std::array<OpcodeInfo, static_cast<int>(Opcode::OP_UNKNOWN)> opcodeInfos_{};
    std::unordered_map<std::string, Opcode> strToEnum_;
};

inline Opcode FindOpcode(const std::string &op) {
    std::string originOp = op;
    constexpr int32_t END_VALUE_4 = 4;
    constexpr int32_t END_VALUE_5 = 5;
    if (op.substr(0, END_VALUE_5) == "TILE_") {
        originOp = originOp.substr(END_VALUE_5);
    }
    if (op.substr(0, END_VALUE_4) == "CALL") {
        originOp = "CALL";
    }

    if (!OpcodeManager::Inst().HasOpcode(originOp)) {
        ASSERT(0) << "Can't find op " << originOp << "\n" << OpcodeManager::Inst().PrintSupportOpcodes();
    }

    return OpcodeManager::Inst().GetOpcode(originOp);
}

const std::unordered_set<Opcode> ALLOC_OPCODE = {Opcode::OP_UB_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC,
    Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_FIX_ALLOC, Opcode::OP_BT_ALLOC};
const std::unordered_set<Opcode> BINARY_OPS{
    Opcode::OP_ADD,
    Opcode::OP_SUB,
    Opcode::OP_MUL,
    Opcode::OP_DIV,
    Opcode::OP_S_ADD,
    Opcode::OP_S_SUB,
    Opcode::OP_S_MUL,
    Opcode::OP_S_DIV,
    Opcode::OP_S_MAX,
    Opcode::OP_S_MIN,
    Opcode::OP_MAXIMUM,
    Opcode::OP_MINIMUM,
    Opcode::OP_PAIRSUM,
    Opcode::OP_PAIRMAX,
    Opcode::OP_PAIRMIN,
};

const std::unordered_set<Opcode> BINARY_WITH_BRC_OPS{
    Opcode::OP_ADD_BRC,
    Opcode::OP_SUB_BRC,
    Opcode::OP_MUL_BRC,
    Opcode::OP_DIV_BRC,
    Opcode::OP_MAX_BRC,
    Opcode::OP_MIN_BRC,
};

const std::unordered_set<Opcode> UNARY_OPS{Opcode::OP_EXP, Opcode::OP_NEG, Opcode::OP_RSQRT, Opcode::OP_SQRT,
    Opcode::OP_EXPAND, Opcode::OP_RECIPROCAL, Opcode::OP_ROWSUM, Opcode::OP_ROWMAX, Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWEXPMAX, Opcode::OP_L1_TO_L1, Opcode::OP_COPY_UB_TO_UB, Opcode::OP_ROWSUMLINE, Opcode::OP_ABS,
    Opcode::OP_LN, Opcode::OP_HUB};

const std::unordered_set<Opcode> UNARY_OPS_WITH_TMP{Opcode::OP_COMPACT, Opcode::OP_ROWSUM_SINGLE,
    Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE, Opcode::OP_TRANSPOSE_VNCHWCONV,
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE};

const std::unordered_set<Opcode> VECTOR_SCALAR_OPS{
    Opcode::OP_ADDS, Opcode::OP_SUBS, Opcode::OP_MULS, Opcode::OP_DIVS, Opcode::OP_MAXS, Opcode::OP_MINS};

const std::unordered_set<Opcode> SCLAR_VECTOR_SCALAR_OPS{
    Opcode::OP_S_ADDS, Opcode::OP_S_SUBS, Opcode::OP_S_MULS, Opcode::OP_S_DIVS, Opcode::OP_S_MAXS, Opcode::OP_S_MINS};

const std::unordered_set<Opcode> CAST_OPS{Opcode::OP_CAST};

const std::unordered_set<Opcode> GATHER_OPS{Opcode::OP_GATHER};

const std::unordered_set<Opcode> WHERE_OPS{
    Opcode::OP_WHERE_TT, Opcode::OP_WHERE_TS, Opcode::OP_WHERE_ST, Opcode::OP_WHERE_SS};

const std::unordered_set<Opcode> GATHER_ELEMENT_OPS{Opcode::OP_GATHER_ELEMENT};
const std::unordered_set<Opcode> SCATTER_ELEMENT_OPS{Opcode::OP_SCATTER_ELEMENT};
const std::unordered_set<Opcode> SCATTER_OPS{Opcode::OP_SCATTER};
const std::unordered_set<Opcode> INDEX_ADD_OPS{Opcode::OP_INDEX_ADD};
const std::unordered_set<Opcode> CUM_SUM_OPS{Opcode::OP_CUM_SUM};

const std::unordered_set<Opcode> SUPPORT_DYNAMIC_UNALIGNED_OPS{Opcode::OP_RANGE, Opcode::OP_TRANSPOSE_VNCHWCONV,
    Opcode::OP_GATHER_ELEMENT, Opcode::OP_INDEX_ADD, Opcode::OP_CUM_SUM, Opcode::OP_COPY_IN, Opcode::OP_UB_COPY_IN, Opcode::OP_L1_COPY_IN,
    Opcode::OP_COPY_OUT, Opcode::OP_UB_COPY_OUT, Opcode::OP_L1_COPY_OUT, Opcode::OP_L0C_COPY_OUT,
    Opcode::OP_TRANSPOSE_MOVEOUT, Opcode::OP_INDEX_OUTCAST, Opcode::OP_ADD, Opcode::OP_SUB, Opcode::OP_MUL,
    Opcode::OP_DIV, Opcode::OP_EXP, Opcode::OP_NEG, Opcode::OP_LN, Opcode::OP_HUB, Opcode::OP_ABS, Opcode::OP_RSQRT,
    Opcode::OP_SQRT, Opcode::OP_RECIPROCAL, Opcode::OP_CAST, Opcode::OP_ADDS, Opcode::OP_SUBS, Opcode::OP_MULS,
    Opcode::OP_DIVS, Opcode::OP_MAXS, Opcode::OP_MINS, Opcode::OP_PAIRMAX, Opcode::OP_PAIRSUM, Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWSUM_SINGLE, Opcode::OP_EXPAND, Opcode::OP_VEC_DUP, Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM,
    Opcode::OP_L1_TO_L0A, Opcode::OP_LOGICALNOT, Opcode::OP_LOGICALAND, Opcode::OP_ONEHOT, Opcode::OP_POW,
    Opcode::OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0_AT, Opcode::OP_A_MUL_B, Opcode::OP_A_MULACC_B,
    Opcode::OP_A_MUL_BT, Opcode::OP_AT_MUL_B, Opcode::OP_AT_MUL_BT, Opcode::OP_WHERE_TT, Opcode::OP_WHERE_TS,
    Opcode::OP_WHERE_ST, Opcode::OP_WHERE_SS, Opcode::OP_ROWSUMLINE, Opcode::OP_ADD_BRC, Opcode::OP_ADD_BRC,
    Opcode::OP_SUB_BRC, Opcode::OP_MUL_BRC, Opcode::OP_DIV_BRC, Opcode::OP_MAX_BRC, Opcode::OP_MIN_BRC,
    Opcode::OP_GATHER, Opcode::OP_S_ADDS, Opcode::OP_S_SUBS, Opcode::OP_S_DIVS, Opcode::OP_S_MULS, Opcode::OP_S_MAXS,
    Opcode::OP_S_MINS, Opcode::OP_BITSORT, Opcode::OP_MRGSORT, Opcode::OP_CMP, Opcode::OP_CMPS, Opcode::OP_EXTRACT,
    Opcode::OP_TILEDMRGSORT, Opcode::OP_ROWMAXLINE, Opcode::OP_PAIRMIN, Opcode::OP_ROWMIN_SINGLE, Opcode::OP_ROWMINLINE,
    Opcode::OP_TOPK_SORT, Opcode::OP_TOPK_MERGE, Opcode::OP_TOPK_EXTRACT, Opcode::OP_SCATTER_ELEMENT,
    Opcode::OP_TRANSPOSE_MOVEIN, Opcode::OP_SORT, Opcode::OP_COMPARE_SWAP, Opcode::OP_MERGE, Opcode::OP_L0C_TO_L1,
    Opcode::OP_SCATTER, Opcode::OP_GATHER_FROM_UB, Opcode::OP_RESHAPE_COPY_IN, Opcode::OP_RESHAPE_COPY_OUT, Opcode::OP_L1_TO_FIX_QUANT_PRE,
    Opcode::OP_L1_TO_BT, Opcode::OP_BRCB};

const std::unordered_set<Opcode> UNSUPPORT_BF16_OPS{Opcode::OP_EXP, Opcode::OP_RSQRT, Opcode::OP_SQRT,
    Opcode::OP_RECIPROCAL, Opcode::OP_ABS, Opcode::OP_LN, Opcode::OP_LOGICALNOT,
    Opcode::OP_LOGICALAND, Opcode::OP_ADDS, Opcode::OP_SUBS, Opcode::OP_MULS, Opcode::OP_DIVS,
    Opcode::OP_MAXS, Opcode::OP_MINS, Opcode::OP_S_ADDS, Opcode::OP_S_SUBS, Opcode::OP_S_MULS, Opcode::OP_S_DIVS,
    Opcode::OP_S_MAXS, Opcode::OP_S_MINS, Opcode::OP_NEG, Opcode::OP_ADD, Opcode::OP_SUB, Opcode::OP_MUL, Opcode::OP_DIV,
    Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_ADD_BRC, Opcode::OP_SUB_BRC, Opcode::OP_MUL_BRC,
    Opcode::OP_DIV_BRC, Opcode::OP_MAX_BRC, Opcode::OP_MIN_BRC, Opcode::OP_S_ADD, Opcode::OP_S_SUB, Opcode::OP_S_MUL,
    Opcode::OP_S_DIV, Opcode::OP_S_MAX, Opcode::OP_S_MIN,Opcode::OP_WHERE_TT, Opcode::OP_WHERE_TS,
    Opcode::OP_WHERE_ST, Opcode::OP_WHERE_SS, Opcode::OP_ROWMAX, Opcode::OP_ROWSUM, Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM, Opcode::OP_ROWSUMLINE, Opcode::OP_ROWMAXLINE, Opcode::OP_ROWMINLINE, Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE, Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, Opcode::OP_SCATTER};

const std::unordered_set<Opcode> FIX_COPY_IN_OPS{Opcode::OP_L1_TO_FIX, Opcode::OP_L1_TO_FIX_QUANT_PRE,
    Opcode::OP_L1_TO_FIX_RELU_PRE, Opcode::OP_L1_TO_FIX_RELU_POST, Opcode::OP_L1_TO_FIX_QUANT_POST,
    Opcode::OP_L1_TO_FIX_ELT_ANTIQ, Opcode::OP_L1_TO_FIX_MTE2_ANTIQ};

const std::unordered_set<Opcode> CROSS_L1_UB_OPS{Opcode::OP_L1_COPY_UB, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND, Opcode::OP_L0C_TO_L1};

const std::unordered_set<Opcode> LOGICALNOT_OPS{Opcode::OP_LOGICALNOT};

const std::unordered_set<Opcode> LOGICALAND_OPS{Opcode::OP_LOGICALAND};

const std::unordered_set<Opcode> DISTRIBUTED_OPS{Opcode::OP_SEND_TO_ROUTING_EXPERT,
    Opcode::OP_SEND_TO_SHARED_EXPERT, Opcode::OP_COPY_TO_LOCAL_EXPERT, Opcode::OP_DISPATCH_SET_FLAG,
    Opcode::OP_FFN_SCHED, Opcode::OP_FFN_BATCHING, Opcode::OP_FFN_COMBINEINFO, Opcode::OP_FFN_VALIDCNT,
    Opcode::OP_SHMEM_PUT, Opcode::OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_GET, Opcode::OP_SHMEM_REDUCE,
    Opcode::OP_BIND_TENSOR, Opcode::OP_SHMEM_PUT_UB2GM, Opcode::OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_SET,
    Opcode::OP_SHMEM_MOE_COMBINE_SEND, Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE};

inline bool IsAllocOpCode(Opcode opCode) {
    return (ALLOC_OPCODE.count(opCode) != 0);
}

inline bool IsCopyIn(const Opcode opCode) {
    return opCode == Opcode::OP_COPY_IN || opCode == Opcode::OP_UB_COPY_IN || opCode == Opcode::OP_L1_COPY_IN ||
           opCode == Opcode::OP_TRANSPOSE_MOVEIN || opCode == Opcode::OP_RESHAPE_COPY_IN;
}

inline bool IsCopyOut(const Opcode &op) {
    return (op == Opcode::OP_COPY_OUT || op == Opcode::OP_L0C_COPY_OUT || op == Opcode::OP_TRANSPOSE_MOVEOUT ||
            op == Opcode::OP_INDEX_OUTCAST || op == Opcode::OP_FFN_SCHED || op == Opcode::OP_FFN_BATCHING ||
            op == Opcode::OP_FFN_COMBINEINFO || op == Opcode::OP_FFN_VALIDCNT || op == Opcode::OP_COPY_TO_LOCAL_EXPERT ||
            op == Opcode::OP_SHMEM_PUT || op == Opcode::OP_SHMEM_SIGNAL || op == Opcode::OP_SHMEM_GET ||
            op == Opcode::OP_SHMEM_REDUCE || op == Opcode::OP_RESHAPE_COPY_OUT || op == Opcode::OP_SHMEM_PUT_UB2GM ||
            op == Opcode::OP_SHMEM_GET_GM2UB || op == Opcode::OP_SHMEM_SET || op == Opcode::OP_SHMEM_MOE_COMBINE_SEND ||
            op == Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE);
}

inline bool IsOpCodeSupportMultiProducers(Opcode opCode) {
    return opCode == Opcode::OP_ASSEMBLE || IsCopyOut(opCode) || opCode == Opcode::OP_CALL ||
           opCode == Opcode::OP_INDEX_OUTCAST;
}

extern std::unordered_map<Opcode, std::string> SUPPORT_TILETENSOR_OPS;
// NEXTNEXT: for test case use only
inline void InsertTileTensorOp(Opcode opCode, const std::string &tileOpName) {
    SUPPORT_TILETENSOR_OPS.emplace(opCode, tileOpName);
}
} // namespace npu::tile_fwk
