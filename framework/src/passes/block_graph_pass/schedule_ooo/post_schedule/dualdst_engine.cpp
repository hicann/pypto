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
 * \file dualdst_engine.cpp
 * \brief DualDstEngine 实现 — OP_L0C_COPY_UB → OP_L0C_COPY_UB_DUAL_DST 融合的识别与改图,
 *        以及 UB 联合分配。从 OoOScheduler 提取为独立引擎。
 */

#include "dualdst_engine.h"
#include "interface/operation/attribute.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "DualDstEngine"

namespace npu::tile_fwk {

namespace {

bool ShapeEq(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    return a == b;
}

bool DynShapeEq(const std::vector<SymbolicScalar>& a, const std::vector<SymbolicScalar>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].Dump() == b[i].Dump()) continue;
        if (a[i].ConcreteValid() && b[i].ConcreteValid() &&
            a[i].Concrete() == b[i].Concrete()) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

int64_t DualDstEngine::SpecifiedInt(const OpImmediate& imm)
{
    if (!imm.IsSpecified()) {
        return kInvalidCoord;
    }
    const auto& s = imm.GetSpecifiedValue();
    if (!s.ConcreteValid()) {
        return kInvalidCoord;
    }
    return s.Concrete();
}

bool DualDstEngine::ReadGeometry(Operation* op, CopyUbGeometry& g)
{
    if (op == nullptr) return false;
    if (op->GetIOperands().size() != 1 || op->GetOOperands().size() != 1) return false;
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (attr == nullptr) return false;

    const auto& fromOff = attr->GetFromOffset();
    if (fromOff.size() != kCopyUbGeometryDimCount) return false;
    g.fromM = SpecifiedInt(fromOff[0]);
    g.fromN = SpecifiedInt(fromOff[1]);
    if (g.fromM == kInvalidCoord || g.fromN == kInvalidCoord) return false;

    const auto& shape = attr->GetShape();
    if (shape.size() != kCopyUbGeometryDimCount) return false;
    g.tileM = SpecifiedInt(shape[0]);
    g.tileN = SpecifiedInt(shape[1]);
    if (g.tileM <= 0 || g.tileN <= 0) return false;

    g.ubOut = op->GetOutputOperand(0);
    if (g.ubOut == nullptr) return false;
    g.ubShape = g.ubOut->GetShape();
    if (op->HasAttribute(OpAttributeKey::staticValidShape)) {
        auto staticVals = op->GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
        g.ubValidShape.clear();
        g.ubValidShape.reserve(staticVals.size());
        for (auto v : staticVals) g.ubValidShape.emplace_back(v);
    } else {
        APASS_LOG_INFO_F(Elements::Operation, "DualDst op[%d] is dynValidshape", op->GetOpMagic());
        g.ubValidShape = g.ubOut->GetDynValidShape();
    }
    return true;
}

bool DualDstEngine::LoadGeometries(const std::vector<Operation*>& copyUbs, std::vector<CopyUbGeometry>& geos)
{
    geos.assign(copyUbs.size(), CopyUbGeometry{});
    int okCnt = 0;
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (ReadGeometry(copyUbs[i], geos[i])) okCnt++;
    }
    return okCnt >= kMinDualDstPairCount;
}

void DualDstEngine::GreedyNonOverlapPick(std::vector<CandidatePair>& cands, std::vector<CandidatePair>& picked)
{
    std::sort(cands.begin(), cands.end(),
        [](const CandidatePair& a, const CandidatePair& b) {
            return a.earlyOffsetOnAxis < b.earlyOffsetOnAxis;
        });
    std::unordered_set<Operation*> used;
    for (auto& c : cands) {
        if (used.count(c.opEarly) || used.count(c.opLate)) continue;
        picked.push_back(c);
        used.insert(c.opEarly);
        used.insert(c.opLate);
    }
}

CoreLocationType DualDstEngine::ConsumerCore(Operation* copyUbOp)
{
    auto out = copyUbOp->GetOutputOperand(0);
    if (out == nullptr) return CoreLocationType::UNKNOWN;
    const auto& cons = out->GetConsumers();
    if (cons.empty()) return CoreLocationType::UNKNOWN;
    Operation* cur = *cons.begin();
    for (int hop = 0; hop < kMaxConsumerSearchDepth && cur != nullptr; ++hop) {
        auto it = state_.schedInfoMap.find(cur);
        if (it != state_.schedInfoMap.end() &&
            (it->second.coreLocation == CoreLocationType::AIV0 || it->second.coreLocation == CoreLocationType::AIV1)) {
            return it->second.coreLocation;
        }
        if (cur->GetOOperands().empty()) break;
        auto outT = cur->GetOutputOperand(0);
        if (outT == nullptr) break;
        const auto& nextCons = outT->GetConsumers();
        if (nextCons.size() != 1) break;
        cur = *nextCons.begin();
    }
    auto it = state_.schedInfoMap.find(*cons.begin());
    return (it == state_.schedInfoMap.end()) ? CoreLocationType::UNKNOWN : it->second.coreLocation;
}

Operation* DualDstEngine::FindAllocPred(Operation* op)
{
    for (auto* pre : state_.depManager.GetPredecessors(op)) {
        if (pre == nullptr) continue;
        auto it = state_.schedInfoMap.find(pre);
        if (it != state_.schedInfoMap.end() && it->second.isAlloc) {
            return pre;
        }
    }
    return nullptr;
}

void DualDstEngine::BuildAdjacencyCandidates(const std::vector<Operation*>& copyUbs,
                                              const std::vector<CopyUbGeometry>& geos,
                                              std::vector<CandidatePair>& candM,
                                              std::vector<CandidatePair>& candN)
{
    auto consumerSplit = [this](Operation* early, Operation* late) {
        return ConsumerCore(early) == CoreLocationType::AIV0 &&
               ConsumerCore(late) == CoreLocationType::AIV1;
    };
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (geos[i].tileM <= 0) continue;
        for (size_t j = i + 1; j < copyUbs.size(); j++) {
            if (geos[j].tileM <= 0) continue;
            const auto& a = geos[i];
            const auto& b = geos[j];
            if (!ShapeEq(a.ubShape, b.ubShape)) continue;
            if (!DynShapeEq(a.ubValidShape, b.ubValidShape)) continue;
            if (a.tileM != b.tileM || a.tileN != b.tileN) continue;
            const int64_t tileM = a.tileM;
            const int64_t tileN = a.tileN;
            Operation* opA = copyUbs[i];
            Operation* opB = copyUbs[j];
            if (a.fromN == b.fromN && std::abs(a.fromM - b.fromM) == tileM) {
                Operation* early = (a.fromM < b.fromM) ? opA : opB;
                Operation* late  = (a.fromM < b.fromM) ? opB : opA;
                if (consumerSplit(early, late)) {
                    candM.push_back({early, late, std::min(a.fromM, b.fromM)});
                }
            }
            if (a.fromM == b.fromM && std::abs(a.fromN - b.fromN) == tileN) {
                Operation* early = (a.fromN < b.fromN) ? opA : opB;
                Operation* late  = (a.fromN < b.fromN) ? opB : opA;
                if (consumerSplit(early, late)) {
                    candN.push_back({early, late, std::min(a.fromN, b.fromN)});
                }
            }
        }
    }
}

void DualDstEngine::PickAllocOrder(Operation* a1, Operation* a2, Operation*& early, Operation*& late)
{
    const bool has1 = state_.schedInfoMap.count(a1) > 0;
    const bool has2 = state_.schedInfoMap.count(a2) > 0;
    if (!has1 || !has2) {
        APASS_LOG_WARN_F(Elements::Operation,
            "PickAllocOrder: alloc op missing in schedInfoMap (a1 has=%d magic=%d; a2 has=%d magic=%d). "
            "Falling back to INT_MAX; order may be non-deterministic when both missing.",
            static_cast<int>(has1), a1 != nullptr ? a1->GetOpMagic() : -1,
            static_cast<int>(has2), a2 != nullptr ? a2->GetOpMagic() : -1);
    }
    const int o1 = has1 ? state_.schedInfoMap.at(a1).execOrder : INT_MAX;
    const int o2 = has2 ? state_.schedInfoMap.at(a2).execOrder : INT_MAX;
    if (o1 <= o2) {
        early = a1;
        late = a2;
    } else {
        early = a2;
        late = a1;
    }
}

bool DualDstEngine::IsDualDstAlloc(Operation* allocOp)
{
    if (!enableDualDst_) return false;
    if (allocOp == nullptr) return false;
    auto it = state_.schedInfoMap.find(allocOp);
    if (it == state_.schedInfoMap.end() || !it->second.isAlloc) return false;
    for (auto* succ : state_.depManager.GetSuccessors(allocOp)) {
        if (succ != nullptr && succ->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            return true;
        }
    }
    return false;
}

Operation* DualDstEngine::GetDualDstCopyOpFor(Operation* allocOp)
{
    if (allocOp == nullptr) return nullptr;
    for (auto* succ : state_.depManager.GetSuccessors(allocOp)) {
        if (succ != nullptr && succ->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            return succ;
        }
    }
    return nullptr;
}

int DualDstEngine::GetDualDstPairedMemId(Operation* allocOp)
{
    if (allocOp == nullptr || allocOp->GetOOperands().empty()) return -1;
    int selfMemId = allocOp->GetOutputOperand(0)->memoryrange.memId;
    Operation* dual = GetDualDstCopyOpFor(allocOp);
    if (dual == nullptr) return -1;
    for (auto& out : dual->GetOOperands()) {
        if (out == nullptr) continue;
        int mid = out->memoryrange.memId;
        if (mid != selfMemId) return mid;
    }
    return -1;
}

void DualDstEngine::EraseFromOrderedOps(Operation* op)
{
    if (op == nullptr) return;
    auto it = std::find(state_.orderedOps.begin(), state_.orderedOps.end(), op);
    if (it != state_.orderedOps.end()) {
        state_.orderedOps.erase(it);
    }
    state_.schedInfoMap.erase(op);
    state_.opReqMemIdsMap.erase(op);
    state_.inOutOperandsCache.erase(op);
}

void DualDstEngine::IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor,
                                           const std::vector<Operation*>& copyUbs,
                                           std::vector<DualDstPair>& pairs)
{
    APASS_LOG_INFO_F(Elements::Operation,
        "DualDst l0cTensor->GetShape().size: %zu, copyUbs.size: %zu) for L0C tensor[%d]",
        l0cTensor->GetShape().size(), copyUbs.size(), l0cTensor->GetMagic());
    if (l0cTensor->GetShape().size() != kCopyUbGeometryDimCount) return;
    if (copyUbs.size() < kMinDualDstPairCount) return;

    std::vector<CopyUbGeometry> geos;
    if (!LoadGeometries(copyUbs, geos)) return;

    std::vector<CandidatePair> candM;
    std::vector<CandidatePair> candN;
    BuildAdjacencyCandidates(copyUbs, geos, candM, candN);

    std::vector<CandidatePair> pickedM;
    std::vector<CandidatePair> pickedN;
    GreedyNonOverlapPick(candM, pickedM);
    GreedyNonOverlapPick(candN, pickedN);

    bool chooseM = (pickedM.size() >= pickedN.size());
    std::vector<CandidatePair>& chosen = chooseM ? pickedM : pickedN;
    if (!chosen.empty()) {
        dualDstL0CDirection_[l0cTensor] = chooseM ? 0 : 1;
    }
    APASS_LOG_INFO_F(Elements::Operation,
        "DualDst pick direction: %s (M=%zu, N=%zu) for L0C tensor[%d]",
        chooseM ? "SplitM" : "SplitN", pickedM.size(), pickedN.size(), l0cTensor->GetMagic());

    for (auto& cp : chosen) {
        DualDstPair pair;
        pair.opEarly = cp.opEarly;
        pair.opLate  = cp.opLate;
        pair.tensorEarly = cp.opEarly->GetOutputOperand(0);
        pair.tensorLate  = cp.opLate->GetOutputOperand(0);
        pair.allocEarly = FindAllocPred(cp.opEarly);
        pair.allocLate  = FindAllocPred(cp.opLate);
        if (pair.allocEarly == nullptr || pair.allocLate == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation,
                "DualDst skip pair: cannot find alloc preds for op[%d]/op[%d]",
                cp.opEarly->GetOpMagic(), cp.opLate->GetOpMagic());
            continue;
        }
        pairs.push_back(pair);
    }
}

Status DualDstEngine::IdentifyDualDstPairs(std::vector<DualDstPair>& pairs)
{
    pairs.clear();
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> l0cToCopyUb;
    for (auto* op : state_.orderedOps) {
        if (op == nullptr) continue;
        if (op->GetOpcode() != Opcode::OP_L0C_COPY_UB) continue;
        if (op->GetIOperands().empty()) continue;
        auto l0cIn = op->GetInputOperand(0);
        if (l0cIn == nullptr) continue;
        l0cToCopyUb[l0cIn].push_back(op);
    }
    for (auto& kv : l0cToCopyUb) {
        IdentifyPairsForOneL0C(kv.first, kv.second, pairs);
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst identify done: %zu pairs.", pairs.size());
    return SUCCESS;
}

Status DualDstEngine::FuseDualDstPairs(const std::vector<DualDstPair>& pairs)
{
    if (pairs.empty()) return SUCCESS;
    size_t fusedCnt = 0;
    for (const auto& p : pairs) {
        if (FuseOnePair(p) != SUCCESS) {
            continue;
        }
        fusedCnt++;
    }
    if (fusedCnt > 0) {
        function_.EraseOperations(false, true);
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst fuse done: %zu / %zu pairs fused.",
        fusedCnt, pairs.size());
    return SUCCESS;
}

Operation* DualDstEngine::CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn)
{
    Operation& cRef = function_.AddRawOperation(
        Opcode::OP_L0C_COPY_UB_DUAL_DST,
        {l0cIn},
        {p.tensorEarly, p.tensorLate});
    Operation* C = &cRef;
    C->UpdateInternalSubgraphID(p.opEarly->GetInternalSubgraphID());
    C->SetAttribute(OpAttributeKey::isCube, true);
    return C;
}

void DualDstEngine::SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn,
                                       const DualDstPair& p,
                                       std::shared_ptr<CopyOpAttribute> attrE,
                                       std::shared_ptr<CopyOpAttribute> attrL)
{
    auto eShapeImms = attrE->GetShape();
    auto lShapeImms = attrL->GetShape();
    if (eShapeImms.size() != kCopyUbGeometryDimCount || lShapeImms.size() != kCopyUbGeometryDimCount) {
        APASS_LOG_WARN_F(Elements::Operation,
            "DualDst SetCopyAttr: expect 2D shape, got E=%zu L=%zu",
            eShapeImms.size(), lShapeImms.size());
        return;
    }
    int64_t eM = SpecifiedInt(eShapeImms[0]);
    int64_t eN = SpecifiedInt(eShapeImms[1]);
    int64_t lM = SpecifiedInt(lShapeImms[0]);
    int64_t lN = SpecifiedInt(lShapeImms[1]);
    if (eM <= 0 || eN <= 0 || lM <= 0 || lN <= 0) {
        APASS_LOG_WARN_F(Elements::Operation,
            "DualDst SetCopyAttr: shape not specified for op[%d]", C->GetOpMagic());
        return;
    }
    int64_t direction = dualDstL0CDirection_.count(l0cIn) ? dualDstL0CDirection_[l0cIn] : 0;
    std::vector<int64_t> realShape = (direction == 0) ? std::vector<int64_t>{eM + lM, eN}
                                                       : std::vector<int64_t>{eM, eN + lN};

    std::vector<SymbolicScalar> validShape;
    validShape.reserve(realShape.size());
    for (auto dim : realShape) validShape.push_back(SymbolicScalar(dim));

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        attrE->GetFromOffset(),
        p.tensorEarly->GetMemoryTypeOriginal(),
        OpImmediate::Specified(realShape),
        OpImmediate::Specified(l0cIn->tensor->GetDynRawShape()),
        OpImmediate::Specified(validShape));
    copyAttr->SetToOffset(attrE->GetToOffset());
    C->SetOpAttribute(copyAttr);
    C->SetAttribute(OpAttributeKey::splitMN, direction);
}

void DualDstEngine::RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate,
                                          Operation* A, Operation* B, Operation* C)
{
    auto rewireInOut = [this, A, B, C](Operation* op) {
        auto preds = state_.depManager.GetPredecessors(op);
        auto succs = state_.depManager.GetSuccessors(op);
        for (auto* pre : preds) {
            if (pre != A && pre != B) {
                if (pre->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                    state_.depManager.AddAllocDependency(pre, C);
                } else {
                    state_.depManager.AddDependency(pre, C);
                }
            }
            state_.depManager.RemoveDependency(pre, op);
        }
        for (auto* suc : succs) {
            if (suc->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                state_.depManager.AddAllocDependency(C, suc);
            } else {
                state_.depManager.AddDependency(C, suc);
            }
            state_.depManager.RemoveDependency(op, suc);
        }
    };
    rewireInOut(opEarly);
    rewireInOut(opLate);

    auto bPreds = state_.depManager.GetPredecessors(B);
    auto bSuccs = state_.depManager.GetSuccessors(B);
    for (auto* pre : bPreds) state_.depManager.RemoveDependency(pre, B);
    for (auto* suc : bSuccs) state_.depManager.RemoveDependency(B, suc);
}

void DualDstEngine::DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn, Operation* B)
{
    if (!B->GetOOperands().empty()) {
        B->GetOutputOperand(0)->RemoveProducer(B);
    }
    p.tensorEarly->RemoveProducer(p.opEarly);
    p.tensorLate->RemoveProducer(p.opLate);
    l0cIn->RemoveConsumer(p.opEarly);
    l0cIn->RemoveConsumer(p.opLate);
}

void DualDstEngine::RegisterFusedOpInMaps(Operation* C, int execOrder)
{
    state_.schedInfoMap[C].execOrder = execOrder;
    state_.schedInfoMap[C].pipeType = RescheduleUtils::GetOpPipeType(C);
    state_.schedInfoMap[C].isAlloc = false;
    state_.schedInfoMap[C].isRetired = false;
    state_.schedInfoMap[C].coreLocation = CoreLocationType::AIC;
    state_.schedInfoMap[C].viewOps = {};
    state_.InsertOrdered(C);
}

void DualDstEngine::SyncBufRefCountForFuse(const DualDstPair& p, Operation* B, Operation* C)
{
    auto sub = [this](Operation* op) {
        auto it = state_.opReqMemIdsMap.find(op);
        if (it == state_.opReqMemIdsMap.end()) return;
        for (int mid : it->second) {
            auto rit = state_.bufRefCount.find(mid);
            if (rit != state_.bufRefCount.end()) rit->second--;
        }
    };
    sub(p.opEarly);
    sub(p.opLate);
    sub(B);

    std::vector<int> cMemIds;
    auto add = [this, &cMemIds](LogicalTensorPtr t) {
        if (t == nullptr) return;
        if (t->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) return;
        int mid = t->memoryrange.memId;
        cMemIds.push_back(mid);
        state_.bufRefCount[mid]++;
    };
    for (auto& t : C->GetOOperands()) add(t);
    for (auto& t : C->GetIOperands()) add(t);
    state_.SetOpMemIds(C, cMemIds);
}

Status DualDstEngine::FuseOnePair(const DualDstPair& p)
{
    if (p.opEarly == nullptr || p.opLate == nullptr ||
        p.allocEarly == nullptr || p.allocLate == nullptr) {
        return FAILED;
    }
    auto l0cIn = p.opEarly->GetInputOperand(0);
    if (l0cIn == nullptr || l0cIn != p.opLate->GetInputOperand(0)) {
        APASS_LOG_WARN_F(Elements::Operation,
            "DualDst skip pair: l0c input mismatch op[%d] vs op[%d]",
            p.opEarly->GetOpMagic(), p.opLate->GetOpMagic());
        return FAILED;
    }
    auto attrE = std::dynamic_pointer_cast<CopyOpAttribute>(p.opEarly->GetOpAttribute());
    auto attrL = std::dynamic_pointer_cast<CopyOpAttribute>(p.opLate->GetOpAttribute());
    if (attrE == nullptr || attrL == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation,
            "DualDst skip pair: missing CopyOpAttribute op[%d]/op[%d]",
            p.opEarly->GetOpMagic(), p.opLate->GetOpMagic());
        return FAILED;
    }

    Operation* C = CreateDualDstFusedOp(p, l0cIn);
    SetDualDstCopyAttr(C, l0cIn, p, attrE, attrL);

    Operation* A = nullptr;
    Operation* B = nullptr;
    PickAllocOrder(p.allocEarly, p.allocLate, A, B);
    state_.depManager.AddAllocDependency(A, C);

    RewireEdgesForFusedOp(p.opEarly, p.opLate, A, B, C);
    DetachOldOpsFromTensors(p, l0cIn, B);

    int earlyOrder = state_.schedInfoMap.count(p.opEarly) ? state_.schedInfoMap[p.opEarly].execOrder : 0;
    SyncBufRefCountForFuse(p, B, C);
    p.opEarly->SetAsDeleted();
    p.opLate->SetAsDeleted();
    B->SetAsDeleted();
    EraseFromOrderedOps(p.opEarly);
    EraseFromOrderedOps(p.opLate);
    EraseFromOrderedOps(B);

    RegisterFusedOpInMaps(C, earlyOrder);

    APASS_LOG_INFO_F(Elements::Operation,
        "DualDst fused: opEarly[%d] + opLate[%d] -> dualOp[%d]; alloc keep[%d] drop[%d]",
        p.opEarly->GetOpMagic(), p.opLate->GetOpMagic(), C->GetOpMagic(),
        A->GetOpMagic(), B->GetOpMagic());
    return SUCCESS;
}

Status DualDstEngine::ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (allocOp == nullptr || allocOp->GetOOperands().empty()) return FAILED;
    ctx.memIdA = allocOp->GetOutputOperand(0)->memoryrange.memId;
    ctx.memIdB = GetDualDstPairedMemId(allocOp);
    if (ctx.memIdB < 0) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "DualDst[%d]: cannot resolve paired memId.", allocOp->GetOpMagic());
        return FAILED;
    }
    ctx.bufA = state_.localBufferMap[ctx.memIdA];
    ctx.bufB = state_.localBufferMap[ctx.memIdB];
    if (ctx.bufA == nullptr || ctx.bufB == nullptr || ctx.bufA->size != ctx.bufB->size) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "DualDst[%d]: missing localBuffer or size mismatch (A=%lu B=%lu).",
            allocOp->GetOpMagic(),
            ctx.bufA ? ctx.bufA->size : 0, ctx.bufB ? ctx.bufB->size : 0);
        return FAILED;
    }
    return SUCCESS;
}

Status DualDstEngine::ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx)
{
    Operation* dualOp = GetDualDstCopyOpFor(allocOp);
    if (dualOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "DualDst[%d]: cannot resolve fused dual_dst op for alloc.", allocOp->GetOpMagic());
        return FAILED;
    }
    LogicalTensorPtr ubA, ubB;
    for (auto& t : dualOp->GetOOperands()) {
        if (t == nullptr) continue;
        if (t->memoryrange.memId == ctx.memIdA) ubA = t;
        if (t->memoryrange.memId == ctx.memIdB) ubB = t;
    }
    auto coreOf = [this](LogicalTensorPtr ub) -> CoreLocationType {
        if (ub == nullptr) return CoreLocationType::UNKNOWN;
        const auto& cons = ub->GetConsumers();
        if (cons.empty()) return CoreLocationType::UNKNOWN;
        auto it = state_.schedInfoMap.find(*cons.begin());
        return (it == state_.schedInfoMap.end()) ? CoreLocationType::UNKNOWN : it->second.coreLocation;
    };
    ctx.coreA = coreOf(ubA);
    ctx.coreB = coreOf(ubB);
    if (ctx.coreA == CoreLocationType::UNKNOWN || ctx.coreB == CoreLocationType::UNKNOWN ||
        ctx.coreA == ctx.coreB) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "DualDst[%d]: paired memIds[%d/%d] not split across AIV0/AIV1 pools "
            "(consumer core: %d / %d).",
            allocOp->GetOpMagic(), ctx.memIdA, ctx.memIdB,
            static_cast<int>(ctx.coreA), static_cast<int>(ctx.coreB));
        return FAILED;
    }
    return SUCCESS;
}

Status DualDstEngine::ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (ResolveDualDstMemAndBuf(allocOp, ctx) != SUCCESS) return FAILED;
    if (ResolveDualDstCores(allocOp, ctx) != SUCCESS) return FAILED;
    return SUCCESS;
}

void DualDstEngine::CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off)
{
    state_.tensorOccupyMap[ctx.memIdA] = allocA;
    state_.tensorOccupyMap[ctx.memIdB] = allocA;
    state_.dualDstMemIdCoreOverride[ctx.memIdA] = ctx.coreA;
    state_.dualDstMemIdCoreOverride[ctx.memIdB] = ctx.coreB;
    ctx.bufA->startCycle = state_.clock;
    ctx.bufB->startCycle = state_.clock;
    APASS_LOG_DEBUG_F(Elements::Operation,
        "DualDst alloc[%d]: placed memId[%d]/[%d] at offset %lu (size %lu).",
        allocA->GetOpMagic(), ctx.memIdA, ctx.memIdB, off, ctx.bufA->size);
}

std::optional<uint64_t> DualDstEngine::FindCommonFreeOffset(
    BufferPool& poolA, BufferPool& poolB, uint64_t size)
{
    if (size == 0) {
        return std::optional<uint64_t>{0};
    }
    auto listA = poolA.GetSortedFreeIntervals();
    auto listB = poolB.GetSortedFreeIntervals();
    size_t i = 0;
    size_t j = 0;
    while (i < listA.size() && j < listB.size()) {
        uint64_t s = std::max(listA[i].first, listB[j].first);
        uint64_t e = std::min(listA[i].second, listB[j].second);
        if (e >= s && (e - s) >= size) {
            return std::optional<uint64_t>{s};
        }
        if (listA[i].second <= listB[j].second) {
            i++;
        } else {
            j++;
        }
    }
    return std::nullopt;
}

Status DualDstEngine::AllocateDualDstAtCurrent(Operation* allocA, bool& allocated)
{
    allocated = false;
    DualDstAllocCtx ctx;
    if (ResolveDualDstAllocCtx(allocA, ctx) != SUCCESS) return FAILED;

    auto& poolForA = state_.bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
    auto& poolForB = state_.bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];

    auto off = FindCommonFreeOffset(poolForA, poolForB, ctx.bufA->size);
    if (!off.has_value()) {
        return SUCCESS;
    }
    if (poolForA.AllocateAtOffset(ctx.bufA, *off) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "DualDst alloc[%d]: AllocateAtOffset poolForA failed at offset %lu.",
            allocA->GetOpMagic(), *off);
        return FAILED;
    }
    if (poolForB.AllocateAtOffset(ctx.bufB, *off) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "DualDst alloc[%d]: AllocateAtOffset poolForB failed at offset %lu, rollback.",
            allocA->GetOpMagic(), *off);
        (void)poolForA.Free(ctx.memIdA);
        return FAILED;
    }
    CommitDualDstAlloc(allocA, ctx, *off);
    allocated = true;
    return SUCCESS;
}

CoreLocationType DualDstEngine::ResolveCoreForFree(int memId)
{
    auto overrideIt = state_.dualDstMemIdCoreOverride.find(memId);
    if (overrideIt != state_.dualDstMemIdCoreOverride.end()) {
        return overrideIt->second;
    }
    return state_.schedInfoMap[state_.tensorAllocMap[memId]].coreLocation;
}

Status DualDstEngine::RunDualDstFuse()
{
    if (!enableDualDst_) return SUCCESS;
    if (state_.coreInitConfigs.find(CoreLocationType::AIV1) == state_.coreInitConfigs.end()) {
        return SUCCESS;
    }
    dualDstL0CDirection_.clear();
    std::vector<DualDstPair> pairs;
    if (IdentifyDualDstPairs(pairs) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IdentifyDualDstPairs failed.");
        return FAILED;
    }
    if (pairs.empty()) return SUCCESS;
    if (FuseDualDstPairs(pairs) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FuseDualDstPairs failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk