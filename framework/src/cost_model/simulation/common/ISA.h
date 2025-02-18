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
 * \file ISA.h
 * \brief
 */

#pragma once

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <set>
#include <map>
#include "cost_model/simulation/common/CommonType.h"
#include "cost_model/simulation/common/CommonTools.h"
#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "interface/operation/operation.h"

namespace CostModel {

constexpr uint64_t RAW_MAGIC_MAX_SIZE = 1024 * 1024;

const std::map<std::string, CorePipeType> SCHED_CORE_PIPE_TYPE { // Unary Vector
    {"EXP", CorePipeType::PIPE_VECTOR_ALU},
    {"NEG", CorePipeType::PIPE_VECTOR_ALU},
    {"SQRT", CorePipeType::PIPE_VECTOR_ALU},
    {"RECIPROCAL", CorePipeType::PIPE_VECTOR_ALU},
    {"ABS", CorePipeType::PIPE_VECTOR_ALU},
    {"LN", CorePipeType::PIPE_VECTOR_ALU},
    {"CAST", CorePipeType::PIPE_VECTOR_ALU},
    {"EXPAND", CorePipeType::PIPE_VECTOR_ALU},
    {"COMPACT", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMAX", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWSUM", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWEXPMAX", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWEXPSUM", CorePipeType::PIPE_VECTOR_ALU},
    {"ADDS", CorePipeType::PIPE_VECTOR_ALU},
    {"SUBS", CorePipeType::PIPE_VECTOR_ALU},
    {"MULS", CorePipeType::PIPE_VECTOR_ALU},
    {"DIVS", CorePipeType::PIPE_VECTOR_ALU},
    {"MAXS", CorePipeType::PIPE_VECTOR_ALU},
    {"S_ADDS", CorePipeType::PIPE_VECTOR_ALU},
    {"S_SUBS", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MULS", CorePipeType::PIPE_VECTOR_ALU},
    {"S_DIVS", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MAXS", CorePipeType::PIPE_VECTOR_ALU},
    {"ADD_BRC", CorePipeType::PIPE_VECTOR_ALU},
    {"SUB_BRC", CorePipeType::PIPE_VECTOR_ALU},
    {"MUL_BRC", CorePipeType::PIPE_VECTOR_ALU},
    {"DIV_BRC", CorePipeType::PIPE_VECTOR_ALU},
    {"MAX_BRC", CorePipeType::PIPE_VECTOR_ALU},
    {"LOGICALNOT", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MINS", CorePipeType::PIPE_VECTOR_ALU},
    {"TRANSPOSE", CorePipeType::PIPE_VECTOR_ALU},
    {"TRANSPOSE_MOVEOUT", CorePipeType::PIPE_MTE_OUT},
    {"TRANSPOSE_VNCHWCONV", CorePipeType::PIPE_VECTOR_ALU},
    {"BITSORT", CorePipeType::PIPE_VECTOR_ALU},
    {"MRGSORT", CorePipeType::PIPE_VECTOR_ALU},
    {"EXTRACT", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWSUMLINE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMAXLINE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMINLINE", CorePipeType::PIPE_VECTOR_ALU},
    {"MINIMUM", CorePipeType::PIPE_VECTOR_ALU},
    {"TILEDMRGSORT", CorePipeType::PIPE_VECTOR_ALU},
    // Binary Vector
    {"ADD", CorePipeType::PIPE_VECTOR_ALU},
    {"SUB", CorePipeType::PIPE_VECTOR_ALU},
    {"MUL", CorePipeType::PIPE_VECTOR_ALU},
    {"DIV", CorePipeType::PIPE_VECTOR_ALU},
    {"MAXIMUM", CorePipeType::PIPE_VECTOR_ALU},
    {"S_ADD", CorePipeType::PIPE_VECTOR_ALU},
    {"S_SUB", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MUL", CorePipeType::PIPE_VECTOR_ALU},
    {"S_DIV", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MAX", CorePipeType::PIPE_VECTOR_ALU},
    {"S_MIN", CorePipeType::PIPE_VECTOR_ALU},
    {"GATHER", CorePipeType::PIPE_VECTOR_ALU},
    {"GATHER_ELEMENT", CorePipeType::PIPE_VECTOR_ALU},
    {"SCATTER_ELEMENT", CorePipeType::PIPE_VECTOR_ALU},
    {"INDEX_PUT", CorePipeType::PIPE_VECTOR_ALU},
    {"CONCAT", CorePipeType::PIPE_VECTOR_ALU},
    {"SCATTER_UPDATE", CorePipeType::PIPE_VECTOR_ALU},
    {"SCATTER_SCALAR", CorePipeType::PIPE_VECTOR_ALU},
    {"PAIRMAX", CorePipeType::PIPE_VECTOR_ALU},
    {"PAIRSUM", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMAX_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMIN_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWSUM_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWMAX_COMBINE_AXIS_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROWSUM_COMBINE_AXIS_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"REDUCE_SUM_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"REDUCE_MAX_SINGLE", CorePipeType::PIPE_VECTOR_ALU},
    {"ROW_SUM_EXPAND", CorePipeType::PIPE_VECTOR_ALU},
    {"ROW_MAX_EXPAND", CorePipeType::PIPE_VECTOR_ALU},
    {"SCATTERUPDATE", CorePipeType::PIPE_VECTOR_ALU},
    {"BRCB", CorePipeType::PIPE_VECTOR_ALU},
    // Range
    {"RANGE", CorePipeType::PIPE_VECTOR_ALU},
    // Cube
    {"A_MUL_B", CorePipeType::PIPE_CUBE},
    {"A_MULACC_B", CorePipeType::PIPE_CUBE},
    {"A_MUL_Bt", CorePipeType::PIPE_CUBE},
    {"A_MULACC_Bt", CorePipeType::PIPE_CUBE},
    // Ternary Vector
    {"WHERE_TT", CorePipeType::PIPE_VECTOR_ALU},
    {"WHERE_TS", CorePipeType::PIPE_VECTOR_ALU},
    {"WHERE_ST", CorePipeType::PIPE_VECTOR_ALU},
    {"WHERE_SS", CorePipeType::PIPE_VECTOR_ALU},
    // ANY
    {"DUPLICATE", CorePipeType::PIPE_VECTOR_ALU},
    // Move
    {"RESHAPE", CorePipeType::PIPE_S},
    {"ASSEMBLE", CorePipeType::PIPE_S},
    {"VIEW", CorePipeType::PIPE_S},
    {"INDEX_OUTCAST", CorePipeType::PIPE_VECTOR_ALU},
    {"REGISTER_COPY", CorePipeType::PIPE_VECTOR_ALU},
    {"FROM_INCAST", CorePipeType::PIPE_MTE_IN},
    {"TO_OUTCAST", CorePipeType::PIPE_MTE_OUT},
    {"TO_INCAST", CorePipeType::PIPE_MTE_OUT},
    {"FROM_OUTCAST", CorePipeType::PIPE_MTE_IN},
    {"CONVERT", CorePipeType::PIPE_VECTOR_ALU},
    {"COPY_IN", CorePipeType::PIPE_MTE_IN},
    {"COPY_OUT", CorePipeType::PIPE_MTE_OUT},
    {"GATHER_IN_UB", CorePipeType::PIPE_MTE_IN},
    {"GATHER_IN_L1", CorePipeType::PIPE_MTE_IN},
    {"LOAD", CorePipeType::PIPE_S},
    // Specia
    {"CALL", CorePipeType::PIPE_CALL},
    {"CALL_NOT_EXPAND", CorePipeType::PIPE_VECTOR_ALU},
    {"VEC_DUP", CorePipeType::PIPE_VECTOR_ALU},
    {"COPY_UB_TO_UB", CorePipeType::PIPE_VECTOR_ALU},
    {"UB_TO_UB", CorePipeType::PIPE_VECTOR_ALU},
    {"VECTORMAX", CorePipeType::PIPE_VECTOR_ALU},
    {"UB_ALLOC", CorePipeType::PIPE_VECTOR_BMU},
    {"UB_COPY_IN", CorePipeType::PIPE_MTE_IN},
    {"UB_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
    {"L1_COPY_IN", CorePipeType::PIPE_MTE_IN},
    {"L0C_COPY_OUT", CorePipeType::PIPE_MTE_OUT},

    // Cube
    {"L1_ALLOC", CorePipeType::PIPE_CUBE_BMU_L1},
    {"L0A_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0A},
    {"L0B_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0B},
    {"L0C_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0C},
    {"FIX_ALLOC", CorePipeType::PIPE_CUBE_BMU_L0C},
    // MTE
    {"L1_COPY_IN", CorePipeType::PIPE_MTE_IN},
    {"L1_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
    {"L0C_COPY_OUT", CorePipeType::PIPE_MTE_OUT},
    {"L1_TO_L0A", CorePipeType::PIPE_MTE1},
    {"L1_TO_L0At", CorePipeType::PIPE_MTE1},
    {"L1_TO_L0B", CorePipeType::PIPE_MTE1},
    {"L1_TO_L0Bt", CorePipeType::PIPE_MTE1},
    {"L1_TO_BT", CorePipeType::PIPE_MTE1},
    {"FIX_COPY_IN", CorePipeType::PIPE_MTE_IN},
    {"L1_COPY_UB", CorePipeType::PIPE_MTE_IN},
    {"L0C_COPY_UB", CorePipeType::PIPE_MTE_IN},
    {"UB_COPY_L1", CorePipeType::PIPE_MTE_IN},
    // Scala
    {"NOP", CorePipeType::TOTAL_CORE_PIPE_TYPE},
    {"SYNC_DST", CorePipeType::PIPE_S},
    {"SYNC_SRC", CorePipeType::PIPE_S},
    {"PHASE1", CorePipeType::PIPE_S},
    {"PHASE2", CorePipeType::PIPE_S}
};

struct ExecuteInfo {
    int exePipeId = -1;
    uint64_t latency = 1;
    bool isIncast = false;
    bool isOutcast = false;
    // For Scheduler Sort TileOp and Tile.
    int copyOutIdx = -1;
    int domCount = 1;
    int sequenceToIssue = 0;

    // For get sequence for schedule_ooo pass
    bool visited = false;

    bool issued = false;
    bool retired = false;

    // For Example Function: tile[BUF_UNKNOWN] -> copy_out -> tile[BUF_UNKNOWN]
    bool noSrcWakeup = false;
    bool noDstWakeup = false;

    // Tile Execute Info
    bool isAllocated = false;
    bool isWritten = false;
    uint64_t writeReference = 0;
    uint64_t readReference = 0;

    CycleInfo cycleInfo;
    void Reset();
};
class TileOp;
using TileOpPtr = std::shared_ptr<TileOp>;
class Function;
using FunctionPtr = std::shared_ptr<CostModel::Function>;

class Tile {
public:
    int magic = -1;
    int subgraphId = -1;

    std::string symbol = "";
    std::string dataTypeStr = "";
    npu::tile_fwk::DataType dataType = npu::tile_fwk::DataType::DT_FP16;
    NodeType nodeType = NodeType::LOCAL;
    std::string bufferType = "";
    OperandType bufType = BUF_UNKNOWN;
    CorePipeType pipeType = CorePipeType::PIPE_UNKNOW;

    std::vector<int> offset;
    std::vector<int> shape;

    int rawMagic = -1;
    std::vector<int> rawShape;

    TileOpPtr producer = nullptr;
    std::vector<TileOpPtr> producers;
    std::vector<TileOpPtr> consumers;

    bool isSubGraphBoundary = false;

    FunctionPtr funcPtr = nullptr;

    ExecuteInfo exeInfo;
    std::string semanticLabels;

    // For Calculate
    bool isScale = false;
    bool isAddr = false;
    int scaleData = 0;
    uint64_t addrReg = 0;
    std::vector<std::vector<uint64_t>> tileData;

    explicit Tile() {}
    explicit Tile(const std::string &str);
    void GetPipeType();
    void Print();
    std::string Dump();
    int SizeinBytes();
};

using TilePtr = std::shared_ptr<Tile>;

class TileOp {
public:
    int magic = -1;
    int subgraphId = -1;
    uint64_t taskId = 0;
    bool specialOp = false; // For empty tileOp, (reshape)

    std::string opcode = "";
    OperandType bufType = BUF_UNKNOWN;
    CorePipeType pipeType = CorePipeType::PIPE_UNKNOW;
    ExecuteInfo exeInfo;
    uint64_t calleeHash{};

    std::string semanticLabel;

    bool scalarVld = false;
    Element scalarVal;
    std::vector<TilePtr> iOperand;
    std::vector<TilePtr> oOperand;

    npu::tile_fwk::Operation *operation{};
    FunctionPtr funcPtr = nullptr;

    explicit TileOp() = default;

    void Print();
    void GetPipeType();
    // For MTE_IN/MTE_OUT access cache.
    uint64_t GetAddress();
    uint64_t GetSize();
    bool IsCall();
    bool IsNOP();
    bool IsSpecial();
    std::string Dump(bool outDetail = false);
};

class Subgraph {};


class FunctionInvokeInfo
{
public:
    std::unordered_set<int> args;
    std::unordered_map<int, TilePtr> binds;

    TilePtr Bind(int magic) {
        if (binds.count(magic)) {
            return binds[magic];
        }
        return nullptr;
    }
};

class Function {
public:
    int magic = -1;
    int pSgId = -1; // SubgraphId
    CostModel::MachineType machineType;
    npu::tile_fwk::Function *parentFunction;
    uint64_t functionHash = -1;
    std::string funcName = "";
    std::vector<TileOpPtr> tileOps;
    std::vector<TilePtr> tiles;
    std::unordered_map<int, TilePtr> tileMap;
    std::unordered_map<int, TileOpPtr> tileOpMap;

    std::vector<int> incastMagic;
    std::vector<int> outcastMagic;
    std::vector<int> tileMagic;
    std::vector<int> opMagic;
    int esgId;
    std::unordered_map<int, FunctionInvokeInfo> invoke;

    // For root function
    bool topoFromRootFunc = false;
    std::vector<TopoInfoEntry> inputTopo;

    // Schedule info.
    bool hasSchedule = false;
    std::vector<std::vector<int>> tileAllocSequence;

    // TILEOP sequence from ooo pass. <magic, seq>
    std::unordered_map<int, uint64_t> opSequenceAfterOOO_;
    std::vector<int> opMagicSequence;
    void GetOpSequeceAfterOOO(int opmagic, uint64_t &index);

    // semanticLabels
    std::string semanticLabels;

    bool hasRecordInfo = false;
    uint64_t startCycles = 0;
    uint64_t totalCycles = 0;
    std::unordered_map<CostModel::CorePipeType, uint64_t> pipeLastEndCycle;
    std::unordered_map<CostModel::CorePipeType, uint64_t> pipeExecuteTime;
    void InitPipeExecTime();
    Json DumpExecuteInfo();
    uint64_t GetOpRelativeReadyCycle(TileOpPtr tileOp, uint64_t newBaseCycle);
    void CalculateRelativeCycle(uint64_t newBaseCycle, double proportion);
};
}  // namespace CostModel