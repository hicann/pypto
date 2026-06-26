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
 * \file dualdst_fuse.cpp
 * \brief OoOScheduler 中 OP_L0C_COPY_UB → OP_L0C_COPY_UB_DUAL_DST 融合的识别与改图实现。
 *        阶段 1 (Identify) + 阶段 2 (Fuse) 都在这里;
 *        分配/spill 侧的 dualdst 分支仍在 ooo_scheduler.cpp / spill_buffer.cpp 内。
 */

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"
#include "interface/operation/attribute.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "DualDstFuse"

namespace npu::tile_fwk {

namespace {
// ===== 工具: OpImmediate / 形状抽取 =====

constexpr int64_t kInvalidCoord = INT64_MIN;
constexpr int kCopyUbGeometryDimCount = 2;   // CopyUb 几何始终为 2D (M, N)
constexpr int kMinDualDstPairCount = 2;      // DualDst 融合至少需要 2 个可配对的 copyUb

int64_t SpecifiedInt(const OpImmediate& imm)
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
        // 1) 字符串严格相等(快速路径,适用于 immediate / 同符号)
        if (a[i].Dump() == b[i].Dump()) continue;
        // 2) 两侧都已 lower 到 concrete 数值且相等
        //    (覆盖动态 shape lower 后两 view 引用不同 RUNTIME_COA base 但数值一致的情况)
        if (a[i].ConcreteValid() && b[i].ConcreteValid() &&
            a[i].Concrete() == b[i].Concrete()) {
            continue;
        }
        // 3) 仍然不同的符号引用,保守判 false。
        //    后续可接入 SymbolicScalar evaluator 做语义等价判定。
        return false;
    }
    return true;
}

// 一次性读出识别所需的几何信息。任意字段缺失返回 false。
struct CopyUbGeometry {
    int64_t fromM{kInvalidCoord};
    int64_t fromN{kInvalidCoord};
    int64_t tileM{kInvalidCoord};
    int64_t tileN{kInvalidCoord};
    std::vector<int64_t> ubShape;
    std::vector<SymbolicScalar> ubValidShape;
    LogicalTensorPtr ubOut;
};

bool ReadGeometry(Operation* op, CopyUbGeometry& g)
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
    // 优先读取 InferDynShape 阶段在转换前快照的 staticValidShape (op 属性):
    // 转换后 validShape 都被 normalize 成动态表达式 (即使原本是静态),直接比较会
    // 因表达式串差异误判 false。staticValidShape 缺失时回退到 GetDynValidShape。
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
} // namespace

// ===== 派生判定: 不依赖 OpAttributeKey =====

bool OoOScheduler::IsDualDstAlloc(Operation* allocOp)
{
    // switch off 时 RunDualDstFuse 直接 return, 不会生成任何 OP_L0C_COPY_UB_DUAL_DST,
    // 此函数恒返回 false。提前 short-circuit, 避免走 GetSuccessors -> succ->GetOpcode()
    // 这条路径 —— 主线 spill 调用 Function::EraseOperations(false, false) 之后,
    // depManager_ 内部 outGraph_ 仍保留指向已释放 Operation 的悬挂指针
    // 解引用就是 heap-use-after-free。
    if (!enableDualDst_) return false;
    if (allocOp == nullptr) return false;
    auto it = opIsAllocMap.find(allocOp);
    if (it == opIsAllocMap.end() || !it->second) return false;
    for (auto* succ : depManager_.GetSuccessors(allocOp)) {
        if (succ != nullptr && succ->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            return true;
        }
    }
    return false;
}

Operation* OoOScheduler::GetDualDstCopyOpFor(Operation* allocOp)
{
    if (allocOp == nullptr) return nullptr;
    for (auto* succ : depManager_.GetSuccessors(allocOp)) {
        if (succ != nullptr && succ->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            return succ;
        }
    }
    return nullptr;
}

int OoOScheduler::GetDualDstPairedMemId(Operation* allocOp)
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

// ===== 工具: 从 orderedOps + opExecOrderMap 中安全删除一个 op =====

void OoOScheduler::EraseFromOrderedOps(Operation* op)
{
    if (op == nullptr) return;
    auto it = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (it != orderedOps.end()) {
        orderedOps.erase(it);
    }
    opExecOrderMap.erase(op);
    opPipeTypeMap.erase(op);
    opIsAllocMap.erase(op);
    opIsRetiredMap.erase(op);
    opViewOpsMap.erase(op);
    opCoreLocationMap.erase(op);
    opReqMemIdsMap.erase(op);
    inOutOperandsCache_.erase(op);
    // 还有两处 DepManager 内部 map 的残留无法在此清理:
    //   1) inGraph_ / outGraph_ 的 key 条目: RemoveDependency 只 erase set 内元素,
    //      不删 outer map 的 KEY。fuse 后死 op 的 key 留空 set,等 EraseOperations
    //      物理析构 Operation 后,key 变悬挂指针 (不被解引用所以暂安全)。
    //   2) opConsumers / opProducers: 由 InitOpConsumerAndProducer 写入,只在
    //      InitDependencies(needView=true) 路径被读;当前 pass 走 needView=false,无影响。
    // 后续可在 DependencyManager 加 public EraseOp(Operation*) 接口,在这里调用清理。
}

// ===== 阶段 1: 识别 =====

namespace {

struct CandidatePair {
    Operation* opEarly;
    Operation* opLate;
    int64_t earlyOffsetOnAxis;   // 用于排序的较小 offset
};

// ConsumerCore 沿 UB tile 下游 consumer 单链遍历的最大深度上限。
// 用途: 在 chained_ops 类长 vec chain (例如 add → mul → copy_out) 中跳过中间未被
// DualDstProcess 显式归核的 op, 一直找到 chain 末端被标到 AIV0/AIV1 的 op。
// 16 为经验上限: 当前真实 kernel 中 vec chain 长度均 <= 12 (test_dual_dst_link_chain
// N_LINK=12 为已知最长场景), 留 4 步余量同时防止万一拓扑出现意外环路时陷入死循环。
// 超过 16 步的 chain 会被静默跳过、漏配 pair; 调长该值需要同步评估
// IdentifyDualDstPairs 的最坏复杂度。
constexpr int kMaxConsumerSearchDepth = 16;

// 返回 op 输出 UB tile 的下游 consumer 链上第一个 AIV0/AIV1 的核归属。
// 不直接看 first consumer 的核, 因为 chain 中间 vec op (如 chained_ops 的 add→mul)
// 在 core_assign 阶段可能被划到默认 AIV 或与同形态对侧 op 同一个 task,
// 导致 DualDstProcess 不能把"第一个" consumer 标到 AIV0/AIV1。沿 consumer 链下行
// 寻找第一个被 DualDstProcess 标到 AIV0/AIV1 的下游 op (通常是 chain 末端 COPY_OUT
// 或被 core_assign 拆分到的末端 task), 让长 vec chain 也能被 dualdst 识别。

// 限制: 单链 (每个中间 op 只 1 个 consumer) + 最多向下走 kMaxConsumerSearchDepth 步,
// 避免环 / 分叉。
CoreLocationType ConsumerCore(Operation* copyUbOp,
                              const std::unordered_map<Operation*, CoreLocationType>& coreMap)
{
    auto out = copyUbOp->GetOutputOperand(0);
    if (out == nullptr) return CoreLocationType::UNKNOWN;
    const auto& cons = out->GetConsumers();
    if (cons.empty()) return CoreLocationType::UNKNOWN;
    Operation* cur = *cons.begin();
    for (int hop = 0; hop < kMaxConsumerSearchDepth && cur != nullptr; ++hop) {
        auto it = coreMap.find(cur);
        if (it != coreMap.end() &&
            (it->second == CoreLocationType::AIV0 || it->second == CoreLocationType::AIV1)) {
            return it->second;
        }
        // 沿 cur 的唯一输出 -> 唯一下游 consumer 继续找
        if (cur->GetOOperands().empty()) break;
        auto outT = cur->GetOutputOperand(0);
        if (outT == nullptr) break;
        const auto& nextCons = outT->GetConsumers();
        if (nextCons.size() != 1) break;   // 分叉/末端 -> 停
        cur = *nextCons.begin();
    }
    // 全链都没找到 AIV0/AIV1 -> 退回原行为: 取 first consumer 的核 (可能是 UNKNOWN)
    auto it = coreMap.find(*cons.begin());
    return (it == coreMap.end()) ? CoreLocationType::UNKNOWN : it->second;
}

void GreedyNonOverlapPick(std::vector<CandidatePair>& cands, std::vector<CandidatePair>& picked)
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

// 在 op 的依赖前驱中找其输出 tensor 的 ALLOC op
Operation* FindAllocPred(Operation* op,
                         DependencyManager& depManager,
                         const std::unordered_map<Operation*, bool>& isAllocMap)
{
    for (auto* pre : depManager.GetPredecessors(op)) {
        if (pre == nullptr) continue;
        auto it = isAllocMap.find(pre);
        if (it != isAllocMap.end() && it->second) {
            return pre;
        }
    }
    return nullptr;
}

// 读组内每个 OP_L0C_COPY_UB 的几何信息;不做组级校验。
// ReadGeometry 失败的 op 在 geos 中保留默认值 (tileM=kInvalidCoord),
// 由 BuildAdjacencyCandidates 的 per-pair 校验跳过。
// 至少有 2 个 op 加载成功才返回 true(否则不可能配对)。
bool LoadGeometries(const std::vector<Operation*>& copyUbs, std::vector<CopyUbGeometry>& geos)
{
    geos.assign(copyUbs.size(), CopyUbGeometry{});
    int okCnt = 0;
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (ReadGeometry(copyUbs[i], geos[i])) okCnt++;
    }
    return okCnt >= kMinDualDstPairCount;
}

// O(n²) 两两判定:per-pair 校验 (ubShape / ubValidShape / tile 尺寸一致) + M/N 相邻 + consumer 核校验。
void BuildAdjacencyCandidates(const std::vector<Operation*>& copyUbs,
                              const std::vector<CopyUbGeometry>& geos,
                              const std::unordered_map<Operation*, CoreLocationType>& coreMap,
                              std::vector<CandidatePair>& candM,
                              std::vector<CandidatePair>& candN)
{
    auto consumerSplit = [&coreMap](Operation* early, Operation* late) {
        return ConsumerCore(early, coreMap) == CoreLocationType::AIV0 &&
               ConsumerCore(late,  coreMap) == CoreLocationType::AIV1;
    };
    for (size_t i = 0; i < copyUbs.size(); i++) {
        if (geos[i].tileM <= 0) continue;   // ReadGeometry 失败,跳过
        for (size_t j = i + 1; j < copyUbs.size(); j++) {
            if (geos[j].tileM <= 0) continue;
            const auto& a = geos[i];
            const auto& b = geos[j];
            // per-pair 形状校验:UB 输出 shape / dynValidShape / tile 尺寸都要一致
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
} // namespace

void OoOScheduler::IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor,
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
    BuildAdjacencyCandidates(copyUbs, geos, opCoreLocationMap, candM, candN);

    std::vector<CandidatePair> pickedM;
    std::vector<CandidatePair> pickedN;
    GreedyNonOverlapPick(candM, pickedM);
    GreedyNonOverlapPick(candN, pickedN);

    // 方向选择: 等量优先 M;记入 L0C -> direction 映射,fuse 阶段读取
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
        pair.allocEarly = FindAllocPred(cp.opEarly, depManager_, opIsAllocMap);
        pair.allocLate  = FindAllocPred(cp.opLate,  depManager_, opIsAllocMap);
        if (pair.allocEarly == nullptr || pair.allocLate == nullptr) {
            APASS_LOG_WARN_F(Elements::Operation,
                "DualDst skip pair: cannot find alloc preds for op[%d]/op[%d]",
                cp.opEarly->GetOpMagic(), cp.opLate->GetOpMagic());
            continue;
        }
        pairs.push_back(pair);
    }
}

Status OoOScheduler::IdentifyDualDstPairs(std::vector<DualDstPair>& pairs)
{
    pairs.clear();
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> l0cToCopyUb;
    for (auto* op : orderedOps) {
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

// ===== 阶段 2: 改图 =====

namespace {
// 在两个 ALLOC 中按 execOrder 选出"早"和"晚"。
// 防御性: 当前 fuse 调用路径上两个 alloc 都已被写进 opExecOrderMap, 但若未来调用方
// 直接传入未注册到 opExecOrderMap 的 op, 我们用 INT_MAX 回退避免崩溃; 同时打 WARN
// 提醒, 避免两者都缺失时排序退化为依赖传入参数顺序的静默非确定性行为。
void PickAllocOrder(Operation* a1, Operation* a2,
                    const std::unordered_map<Operation*, int>& orderMap,
                    Operation*& early, Operation*& late)
{
    const bool has1 = orderMap.count(a1) > 0;
    const bool has2 = orderMap.count(a2) > 0;
    if (!has1 || !has2) {
        APASS_LOG_WARN_F(Elements::Operation,
            "PickAllocOrder: alloc op missing in opExecOrderMap (a1 has=%d magic=%d; a2 has=%d magic=%d). "
            "Falling back to INT_MAX; order may be non-deterministic when both missing.",
            static_cast<int>(has1), a1 != nullptr ? a1->GetOpMagic() : -1,
            static_cast<int>(has2), a2 != nullptr ? a2->GetOpMagic() : -1);
    }
    const int o1 = has1 ? orderMap.at(a1) : INT_MAX;
    const int o2 = has2 ? orderMap.at(a2) : INT_MAX;
    if (o1 <= o2) {
        early = a1;
        late = a2;
    } else {
        early = a2;
        late = a1;
    }
}
} // namespace

Status OoOScheduler::FuseDualDstPairs(const std::vector<DualDstPair>& pairs)
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
        // 物理移除被 SetAsDeleted 的 op (含 opEarly / opLate / B)。
        // 第二个参数 true: EraseOperations 内部同步刷新 opPosition_,
        // 因此调用方 (schedule_ooo.cpp 后续的 Function::ScheduleBy) 无需再带
        // needRefresh=true 提示, 也不需要外部 functionOpsMutated_ 标志位。
        function_.EraseOperations(false, true);
    }
    APASS_LOG_INFO_F(Elements::Operation, "DualDst fuse done: %zu / %zu pairs fused.",
        fusedCnt, pairs.size());
    return SUCCESS;
}

Operation* OoOScheduler::CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn)
{
    Operation& cRef = function_.AddRawOperation(
        Opcode::OP_L0C_COPY_UB_DUAL_DST,
        {l0cIn},
        {p.tensorEarly, p.tensorLate});
    Operation* C = &cRef;
    C->UpdateInternalSubgraphID(p.opEarly->GetInternalSubgraphID());
    // OP_L0C_COPY_UB_DUAL_DST 在 cube 侧执行,不涉及 AIV 选择;isCube 恒为 true。
    C->SetAttribute(OpAttributeKey::isCube, true);
    return C;
}

void OoOScheduler::SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn,
                                      const DualDstPair& p,
                                      std::shared_ptr<CopyOpAttribute> attrE,
                                      std::shared_ptr<CopyOpAttribute> attrL)
{
    // 对齐 GenerateMoveOp::SetL0C2UBCopyAttr 模板;新规格各 attr 单值。
    // realShape (实际搬运) = 沿 SplitMN 轴对 opEarly/opLate 各自的 shape 求和;
    // 其余字段沿用 attrE,dstValidShape 由 realShape 派生。
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
    // 0=SplitM, 1=SplitN;identify 阶段缺失则默认 SplitM
    int64_t direction = dualDstL0CDirection_.count(l0cIn) ? dualDstL0CDirection_[l0cIn] : 0;
    std::vector<int64_t> realShape = (direction == 0) ? std::vector<int64_t>{eM + lM, eN}
                                                       : std::vector<int64_t>{eM, eN + lN};

    std::vector<SymbolicScalar> validShape;
    validShape.reserve(realShape.size());
    for (auto dim : realShape) validShape.push_back(SymbolicScalar(dim));

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        attrE->GetFromOffset(),
        p.tensorEarly->GetMemoryTypeOriginal(),
        OpImmediate::Specified(realShape),                              // shape (实际搬运)
        OpImmediate::Specified(l0cIn->tensor->GetDynRawShape()),         // rawshape (srcValidShape)
        OpImmediate::Specified(validShape));                              // dstValidShape
    copyAttr->SetToOffset(attrE->GetToOffset());
    C->SetOpAttribute(copyAttr);
    // SplitMN: 0=SplitM, 1=SplitN;DualDst flag 由 codegen 通过 opcode 隐式判定,无需显式设置。
    C->SetAttribute(OpAttributeKey::splitMN, direction);
}

void OoOScheduler::RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate,
                                         Operation* A, Operation* B, Operation* C)
{
    auto rewireInOut = [this, A, B, C](Operation* op) {
        auto preds = depManager_.GetPredecessors(op);
        auto succs = depManager_.GetSuccessors(op);
        for (auto* pre : preds) {
            if (pre != A && pre != B) {
                if (pre->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                    depManager_.AddAllocDependency(pre, C);
                } else {
                    depManager_.AddDependency(pre, C);
                }
            }
            depManager_.RemoveDependency(pre, op);
        }
        for (auto* suc : succs) {
            if (suc->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                depManager_.AddAllocDependency(C, suc);
            } else {
                depManager_.AddDependency(C, suc);
            }
            depManager_.RemoveDependency(op, suc);
        }
    };
    rewireInOut(opEarly);
    rewireInOut(opLate);

    // 当前 B 是 ALLOC,pred 恒空,succ 仅含 opLate
    // 已在 rewireInOut(opLate) 的 pred 循环里被摘;下面两个循环实际不动任何边。
    // 保留为防御性代码,避免未来 dep 拓扑变化导致 B 有别的边时漏清理。
    // 评估稳定后可直接删除这 4 行。
    auto bPreds = depManager_.GetPredecessors(B);
    auto bSuccs = depManager_.GetSuccessors(B);
    for (auto* pre : bPreds) depManager_.RemoveDependency(pre, B);
    for (auto* suc : bSuccs) depManager_.RemoveDependency(B, suc);
}

void OoOScheduler::DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn, Operation* B)
{
    if (!B->GetOOperands().empty()) {
        B->GetOutputOperand(0)->RemoveProducer(B);
    }
    p.tensorEarly->RemoveProducer(p.opEarly);
    p.tensorLate->RemoveProducer(p.opLate);
    l0cIn->RemoveConsumer(p.opEarly);
    l0cIn->RemoveConsumer(p.opLate);
}

void OoOScheduler::RegisterFusedOpInMaps(Operation* C, int execOrder)
{
    opExecOrderMap[C] = execOrder;
    opPipeTypeMap[C] = RescheduleUtils::GetOpPipeType(C);
    opIsAllocMap[C] = false;
    opIsRetiredMap[C] = false;
    opCoreLocationMap[C] = CoreLocationType::AIC;   // OP_L0C_COPY_UB_DUAL_DST 是 cube 侧
    opViewOpsMap[C] = {};
    // opReqMemIdsMap[C] 由 SyncBufRefCountForFuse 设置;此处不再清空,以免覆盖 sync 写入。
    InsertOrdered(C);
}

void OoOScheduler::SyncBufRefCountForFuse(const DualDstPair& p, Operation* B, Operation* C)
{
    auto sub = [this](Operation* op) {
        auto it = opReqMemIdsMap.find(op);
        if (it == opReqMemIdsMap.end()) return;
        for (int mid : it->second) {
            auto rit = bufRefCount_.find(mid);
            if (rit != bufRefCount_.end()) rit->second--;
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
        bufRefCount_[mid]++;
    };
    for (auto& t : C->GetOOperands()) add(t);
    for (auto& t : C->GetIOperands()) add(t);
    SetOpMemIds(C, cMemIds);
}

Status OoOScheduler::FuseOnePair(const DualDstPair& p)
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
    PickAllocOrder(p.allocEarly, p.allocLate, opExecOrderMap, A, B);
    depManager_.AddAllocDependency(A, C);

    RewireEdgesForFusedOp(p.opEarly, p.opLate, A, B, C);
    DetachOldOpsFromTensors(p, l0cIn, B);

    int earlyOrder = opExecOrderMap.count(p.opEarly) ? opExecOrderMap[p.opEarly] : 0;
    SyncBufRefCountForFuse(p, B, C);    // 必须在 EraseFromOrderedOps 之前
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

// ===== 阶段 3: UB 联合分配 =====

// 拆出: 解析 ctx 的 memId / buf。失败已打 log。
Status OoOScheduler::ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (allocOp == nullptr || allocOp->GetOOperands().empty()) return FAILED;
    ctx.memIdA = allocOp->GetOutputOperand(0)->memoryrange.memId;
    ctx.memIdB = GetDualDstPairedMemId(allocOp);
    if (ctx.memIdB < 0) {
        APASS_LOG_ERROR_F(Elements::Operation,
            "DualDst[%d]: cannot resolve paired memId.", allocOp->GetOpMagic());
        return FAILED;
    }
    ctx.bufA = localBufferMap_[ctx.memIdA];
    ctx.bufB = localBufferMap_[ctx.memIdB];
    if (ctx.bufA == nullptr || ctx.bufB == nullptr || ctx.bufA->size != ctx.bufB->size) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "DualDst[%d]: missing localBuffer or size mismatch (A=%lu B=%lu).",
            allocOp->GetOpMagic(),
            ctx.bufA ? ctx.bufA->size : 0, ctx.bufB ? ctx.bufB->size : 0);
        return FAILED;
    }
    return SUCCESS;
}

// 拆出: 从 dual_dst op 的 ub output tensor 的 consumer (vec op) 反推 core。
// 不依赖 alloc op 自身的归核 (后者通常是 AIC 与 ub 实际服务核脱钩)。
Status OoOScheduler::ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx)
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
        auto it = opCoreLocationMap.find(*cons.begin());
        return (it == opCoreLocationMap.end()) ? CoreLocationType::UNKNOWN : it->second;
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

Status OoOScheduler::ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx)
{
    if (ResolveDualDstMemAndBuf(allocOp, ctx) != SUCCESS) return FAILED;
    if (ResolveDualDstCores(allocOp, ctx) != SUCCESS) return FAILED;
    return SUCCESS;
}

void OoOScheduler::CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off)
{
    // 主线把 tensorOccupyMap 从 map<MemType, map<int, Op*>> 拍平成 map<int, Op*>。
    tensorOccupyMap[ctx.memIdA] = allocA;
    tensorOccupyMap[ctx.memIdB] = allocA;
    // 记录每个 memId 实际所在 UB pool 的 core; FreeBuffer 会优先查这张表, 避免
    // 用 alloc op 的归核误推到错的 pool 而 free 失败。
    dualDstMemIdCoreOverride_[ctx.memIdA] = ctx.coreA;
    dualDstMemIdCoreOverride_[ctx.memIdB] = ctx.coreB;
    ctx.bufA->startCycle = clock;
    ctx.bufB->startCycle = clock;
    APASS_LOG_DEBUG_F(Elements::Operation,
        "DualDst alloc[%d]: placed memId[%d]/[%d] at offset %lu (size %lu).",
        allocA->GetOpMagic(), ctx.memIdA, ctx.memIdB, off, ctx.bufA->size);
}

std::optional<uint64_t> OoOScheduler::FindCommonFreeOffset(
    BufferPool& poolA, BufferPool& poolB, uint64_t size)
{
    // size==0 时任意 offset 都满足 → 直接返回 0 (与原 BufferPool 版同义)。
    if (size == 0) {
        return std::optional<uint64_t>{0};
    }
    auto listA = poolA.GetSortedFreeIntervals();
    auto listB = poolB.GetSortedFreeIntervals();
    // 两侧按起点排序的 [start, end) 列表归并求交集: 取重叠段 [max(starts), min(ends)),
    // 长度 ≥ size 即命中, 返回 max(starts);否则推进结束点较小的指针继续。
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

Status OoOScheduler::AllocateDualDstAtCurrent(Operation* allocA, bool& allocated)
{
    allocated = false;
    DualDstAllocCtx ctx;
    if (ResolveDualDstAllocCtx(allocA, ctx) != SUCCESS) return FAILED;

    auto& poolForA = bufferManagerMap[ctx.coreA][MemoryType::MEM_UB];
    auto& poolForB = bufferManagerMap[ctx.coreB][MemoryType::MEM_UB];

    auto off = FindCommonFreeOffset(poolForA, poolForB, ctx.bufA->size);
    if (!off.has_value()) {
        // 两池不存在共同的连续空闲段,等价于 Full,由调用方触发 spill。
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

// ===== 阶段 4: Spill (严格同区间) =====
// 历史上这里有 SpillDualDstAllocBuffer + SpillOneSideWithCoreSwap + PickMatchingSpillGroups
// 三个函数, 自成一套 dualdst spill 路径。重构后:
//   - 选组逻辑改由 OoOScheduler::GetDualSpillGroup 实现 (在 spill_buffer.cpp 内),
//     SelectSpillBuffers 内按 IsDualDstAlloc 分叉调用
//   - spill 执行 (逐 memId SpillBuffer + RearrangeBuffer + HasEnoughBuffer 兜底)
//     与单池路径共用 GenBufferSpill
//   - SpillBuffer 内部通过 ResolveCoreForFree(memId) 反查正确的池, 不再需要 core-swap
// 净效果: 删 ~90 行 dualdst 专用 spill 代码, 同时让 dualdst 路径享受到原本漏掉的
//        RearrangeBuffer 与 HasEnoughBuffer 兜底诊断。

// ===== 入口 =====

Status OoOScheduler::RunDualDstFuse()
{
    if (!enableDualDst_) return SUCCESS;
    if (CORE_INIT_CONFIGS.find(CoreLocationType::AIV1) == CORE_INIT_CONFIGS.end()) {
        // 非 Mix / 非 3510,只有 AIV0,不存在 dualdst
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
