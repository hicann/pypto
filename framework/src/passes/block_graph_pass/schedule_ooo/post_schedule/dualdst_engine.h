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
 * \file dualdst_engine.h
 * \brief DualDst execution engine — DualDst fuse identification, graph modification,
 *        and dual-buffer allocation extracted from OoOScheduler.
 *        Public interface = RunDualDstFuse + IsDualDstAlloc + AllocateDualDstAtCurrent + ResolveCoreForFree.
 */

#ifndef PASS_DUALDST_ENGINE_H
#define PASS_DUALDST_ENGINE_H

#include <optional>
#include <memory>
#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "interface/operation/attribute.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "DualDstEngine"

namespace npu::tile_fwk {

class DualDstEngine {
public:
    DualDstEngine(ScheduleState& state, Function& function)
        : state_(state), function_(function), enableDualDst_(false) {}
    ~DualDstEngine() {}

    Status RunDualDstFuse();
    bool IsDualDstAlloc(Operation* allocOp);
    Status AllocateDualDstAtCurrent(Operation* allocA, bool& allocated);
    CoreLocationType ResolveCoreForFree(int memId);
    Status ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx);

    bool IsDualDstEnabled() const { return enableDualDst_; }
    void SetEnableDualDst(bool enable) { enableDualDst_ = enable; }
    void SetDualDstL0CDirection(const std::unordered_map<LogicalTensorPtr, int64_t>& dir) { dualDstL0CDirection_ = dir; }
    const auto& GetL02L0MXMap() const { return l02L0MXMap_; }
    auto& GetL02L0MXMap() { return l02L0MXMap_; }

private:
    static constexpr int64_t kInvalidCoord = INT64_MIN;
    static constexpr int kCopyUbGeometryDimCount = 2;
    static constexpr int kMinDualDstPairCount = 2;
    static constexpr int kMaxConsumerSearchDepth = 16;

    struct CopyUbGeometry {
        int64_t fromM{kInvalidCoord};
        int64_t fromN{kInvalidCoord};
        int64_t tileM{kInvalidCoord};
        int64_t tileN{kInvalidCoord};
        std::vector<int64_t> ubShape;
        std::vector<SymbolicScalar> ubValidShape;
        LogicalTensorPtr ubOut;
    };

    struct CandidatePair {
        Operation* opEarly;
        Operation* opLate;
        int64_t earlyOffsetOnAxis;
    };

    static int64_t SpecifiedInt(const OpImmediate& imm);
    static bool ReadGeometry(Operation* op, CopyUbGeometry& g);
    static bool LoadGeometries(const std::vector<Operation*>& copyUbs, std::vector<CopyUbGeometry>& geos);
    static void GreedyNonOverlapPick(std::vector<CandidatePair>& cands, std::vector<CandidatePair>& picked);

    CoreLocationType ConsumerCore(Operation* copyUbOp);
    Operation* FindAllocPred(Operation* op);
    void BuildAdjacencyCandidates(const std::vector<Operation*>& copyUbs,
                                  const std::vector<CopyUbGeometry>& geos,
                                  std::vector<CandidatePair>& candM,
                                  std::vector<CandidatePair>& candN);
    void PickAllocOrder(Operation* a1, Operation* a2, Operation*& early, Operation*& late);

    Operation* GetDualDstCopyOpFor(Operation* allocOp);
    int GetDualDstPairedMemId(Operation* allocOp);
    void EraseFromOrderedOps(Operation* op);
    void IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor,
                                const std::vector<Operation*>& copyUbs,
                                std::vector<DualDstPair>& pairs);
    Status IdentifyDualDstPairs(std::vector<DualDstPair>& pairs);
    Status FuseDualDstPairs(const std::vector<DualDstPair>& pairs);
    Operation* CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn);
    void SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn, const DualDstPair& p,
                            std::shared_ptr<CopyOpAttribute> attrE,
                            std::shared_ptr<CopyOpAttribute> attrL);
    void RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate,
                               Operation* A, Operation* B, Operation* C);
    void DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn, Operation* B);
    void RegisterFusedOpInMaps(Operation* C, int execOrder);
    void SyncBufRefCountForFuse(const DualDstPair& p, Operation* B, Operation* C);
    Status FuseOnePair(const DualDstPair& p);
    Status ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx);
    Status ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx);
    void CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off);
    std::optional<uint64_t> FindCommonFreeOffset(BufferPool& poolA, BufferPool& poolB, uint64_t size);

    ScheduleState& state_;
    Function& function_;
    bool enableDualDst_;
    std::unordered_map<LogicalTensorPtr, int64_t> dualDstL0CDirection_;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;
};

} // namespace npu::tile_fwk
#endif // PASS_DUALDST_ENGINE_H
