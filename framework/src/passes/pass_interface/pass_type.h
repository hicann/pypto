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
 * \file pass_type.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "tilefwk/error.h"
namespace npu::tile_fwk {
enum class PassType : int32_t {
    TYPE_INVALID = -1,
    TYPE_TENSOR_GRAPH = 0,
    TYPE_TILE_GRAPH = 1,
    TYPE_BLOCK_GRAPH = 2,
    TYPE_BOTTOM
};

enum class PassName {
    LOOP_UNROLL,
    AUTO_CAST,
    REMOVE_REDUNDANT_RESHAPE,
    INFER_MEMORY_CONFLICT,
    REMOVE_UNDRIVEN_VIEW,
    EXPAND_FUNCTION,
    DUPLICATE_OP,
    MERGE_VIEW_ASSEMBLE,
    SPLIT_RESHAPE,
    SPLIT_RAW_TENSOR,
    SPLIT_LARGE_FANOUT_TENSOR,
    ASSIGN_MEMORY_TYPE,
    INFER_DISCONTINUOUS_INPUT,
    REMOVE_REDUNDANT_OP,
    INSERT_OP_FOR_VIEWASSEMBLE,
    SPLIT_K,
    GRAPH_PARTITION,
    REDUCE_COPY_MERGE,
    N_BUFFER_MERGE,
    INTRA_SUBGRAPH_ADAPTER,
    GENERATE_MOVE_OP,
    COMMON_OPERATION_ELIMINATE,
    L1_COPY_IN_REUSE_MERGE,
    AXIS_COMBINE,
    PAD_LOCAL_BUFFER,
    REMOVE_UNALIGNED_RESHAPE,
    REPLACE_TENSOR,
    PRE_GRAPH_PROCESS,
    INFER_DYN_SHAPE,
    SUBGRAPH_TO_FUNCTION,
    INFER_PARAM_INDEX,
    SRC_DST_BUFFER_MERGE,
    ADD_ALLOC,
    OOO_SCHEDULE,
    GLOBAL_MEMORY_REUSE,
    REMOVE_ALLOC,
    COPY_OUT_RESOLVE,
    INSERT_SYNC,
    MIX_SUBGRAPH_SPLIT,
    CODEGEN_PREPROC,
    DYN_ATTR_TO_STATIC,
    LOOPAXES_PROC,
    NOT_DEFINED
};

inline constexpr const char *PassNameStr(PassName name){
    switch (name) {
        case PassName::LOOP_UNROLL: return "LoopUnroll";
        case PassName::REMOVE_REDUNDANT_RESHAPE: return "RemoveRedundantReshape";
        case PassName::AUTO_CAST: return "AutoCast";
        case PassName::INFER_MEMORY_CONFLICT: return "InferMemoryConflict";
        case PassName::REMOVE_UNDRIVEN_VIEW: return "RemoveUndrivenView";
        case PassName::EXPAND_FUNCTION: return "ExpandFunction";
        case PassName::DUPLICATE_OP: return "DuplicateOp";
        case PassName::MERGE_VIEW_ASSEMBLE: return "MergeViewAssemble";
        case PassName::SPLIT_RESHAPE: return "SplitReshape";
        case PassName::SPLIT_RAW_TENSOR: return "SplitRawTensor";
        case PassName::SPLIT_LARGE_FANOUT_TENSOR: return "SplitLargeFanoutTensor";
        case PassName::ASSIGN_MEMORY_TYPE: return "AssignMemoryType";
        case PassName::INFER_DISCONTINUOUS_INPUT: return "InferDiscontinuousInput";
        case PassName::REMOVE_REDUNDANT_OP: return "RemoveRedundantOp";
        case PassName::INSERT_OP_FOR_VIEWASSEMBLE: return "InsertOpForViewAssemble";
        case PassName::SPLIT_K: return "SplitK";
        case PassName::GRAPH_PARTITION: return "GraphPartition";
        case PassName::REDUCE_COPY_MERGE: return "ReduceCopyMerge";
        case PassName::N_BUFFER_MERGE: return "NBufferMerge";
        case PassName::INTRA_SUBGRAPH_ADAPTER: return "IntraSubgraphAdapter";
        case PassName::GENERATE_MOVE_OP: return "GenerateMoveOp";
        case PassName::COMMON_OPERATION_ELIMINATE: return "CommonOperationEliminate";
        case PassName::AXIS_COMBINE: return "AxisCombine";
        case PassName::L1_COPY_IN_REUSE_MERGE: return "L1CopyInReuseMerge";
        case PassName::PAD_LOCAL_BUFFER: return "PadLocalBuffer";
        case PassName::REMOVE_UNALIGNED_RESHAPE: return "RemoveUnalignedReshape";
        case PassName::REPLACE_TENSOR: return "ReplaceTensor";
        case PassName::PRE_GRAPH_PROCESS: return "PreGraphProcess";
        case PassName::INFER_DYN_SHAPE: return "InferDynShape";
        case PassName::SUBGRAPH_TO_FUNCTION: return "SubgraphToFunction";
        case PassName::INFER_PARAM_INDEX: return "InferParamIndex";
        case PassName::SRC_DST_BUFFER_MERGE: return "SrcDstBufferMerge";
        case PassName::ADD_ALLOC: return "AddAlloc";
        case PassName::OOO_SCHEDULE: return "OoOSchedule";
        case PassName::GLOBAL_MEMORY_REUSE: return "GlobalMemoryReuse";
        case PassName::REMOVE_ALLOC: return "RemoveAlloc";
        case PassName::COPY_OUT_RESOLVE: return "CopyOutResolve";
        case PassName::INSERT_SYNC: return "InsertSync";
        case PassName::MIX_SUBGRAPH_SPLIT: return "MixSubgraphSplit";
        case PassName::CODEGEN_PREPROC: return "CodegenPreproc";
        case PassName::DYN_ATTR_TO_STATIC: return "DynAttrToStatic";
        case PassName::LOOPAXES_PROC: return "LoopaxesProc";
        case PassName::NOT_DEFINED: return "NotDefined";
        default: 
            ASSERT(false) << "[PassDependency][Manager][ERROR]: PassName not defined.";
            return "Invalid";
    }
}

inline std::ostream &operator<<(std::ostream &os, PassName name) {
    return os << PassNameStr(name);
}
} // PASSES_PASS_TYPE_H_