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
 * \file ooo_scheduler.h
 * \brief
 */

#ifndef PASS_SCHEDULER_H
#define PASS_SCHEDULER_H

#include <climits>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/dep_manager.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_base.h"
#include "passes/statistics/schedule_observer.h"
#include "schedule_main_loop_base.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) { return BytesOf(dataType); }

// AIV0/AIV1 split: each has its own BufferPool. Observer collapses via ToCoreLocation.
enum class CoreLocationType { AIC = 0, AIV0 = 1, AIV1 = 2, UNKNOWN = 3 };

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_NOP,         Opcode::OP_RESHAPE,   Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_BIND_TENSOR, Opcode::OP_VIEW_TYPE, Opcode::OP_HUB};

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,        Opcode::OP_UB_COPY_IN, Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA, Opcode::OP_L1_COPY_UB, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_ONE = {
    CoreLocationType::AIC, CoreLocationType::AIV0};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

struct IssueQueue {
    bool busy{false};
    Operation* curIssue = nullptr;
    int curOpRetireCycle{-1};
    std::vector<Operation*> queue;
    // 比较函数指针，由OoOScheduler设置
    std::function<bool(Operation*, Operation*)> compareFunc;

    IssueQueue() {}
    ~IssueQueue() {}

    void SetCompareFunc(std::function<bool(Operation*, Operation*)> func)
    {
        compareFunc = func;
    }

    void Insert(Operation* op)
    {
        queue.push_back(op);
        if (compareFunc) {
            std::push_heap(queue.begin(), queue.end(), compareFunc);
        }
    }

    void ForceInsertAfterFirst(Operation* op) {
        // 插入到第一个元素之后
        auto insertPos = queue.begin() + 1;
        queue.insert(insertPos, op);
    }

    bool Empty() { return queue.size() == 0; }

    Operation* Front()
    {
        return queue[0];
    }

    Operation* PopFront()
    {
        if (compareFunc) {
            std::pop_heap(queue.begin(), queue.end(), compareFunc);
        }
        Operation* op = queue.back();
        queue.pop_back();
        return op;
    }

    void DeleteOp(Operation* op) {
        auto it = std::find(queue.begin(), queue.end(), op);
        if (it != queue.end()) {
            queue.erase(it);
            if (compareFunc) {
                std::make_heap(queue.begin(), queue.end(), compareFunc);
            }
        }
    }
};

struct SpillContext {
    std::vector<Operation*> newCopyoutOps;    // spill产生的copyout ops
    std::vector<Operation*> newAllocOps; // 需要加入allocIssueQueue的alloc ops
    std::vector<int> spillMemIds;             // 需要从tensorOccupyMap清理的memIds
    int newNotRetiredCopyOutSize{0};      // spill 新插的未执行的 copy_out 数量
    int deleteRetiredOpSize{0};       // 删除 op 中 已执行的 op 数量
    int deleteNotRetiredOpSize{0};      // 删除 op 中 未被执行的 op 数量
    std::vector<std::tuple<Operation*, MemoryType, CoreLocationType>> deleteAllocOps;
};

// === DualDst 融合候选对 ===
// opEarly: fromOffset 较小,其 UB 输出 consumer 在 AIV0
// opLate : fromOffset 较大,其 UB 输出 consumer 在 AIV1
// allocEarly / allocLate: 两侧输出 tensor 的 ALLOC op (按 opExecOrderMap 决定保留哪条)
// SplitM / SplitN 的方向不在此结构持久化,由 opEarly / opLate 的 fromOffset 直接推导。
struct DualDstPair {
    Operation* opEarly{nullptr};
    Operation* opLate{nullptr};
    Operation* allocEarly{nullptr};
    Operation* allocLate{nullptr};
    LogicalTensorPtr tensorEarly;
    LogicalTensorPtr tensorLate;
};

// === DualDst 分配 / spill 共享上下文 ===
// 由 IsDualDstAlloc 之后的 ResolveDualDstAllocCtx 填充,供 alloc / spill 两路使用。
struct DualDstAllocCtx {
    int memIdA{-1};
    int memIdB{-1};
    LocalBufferPtr bufA;
    LocalBufferPtr bufB;
    CoreLocationType coreA{CoreLocationType::UNKNOWN};
    CoreLocationType coreB{CoreLocationType::UNKNOWN};
};

struct SingleSpillCreatedOps {
    Operation* copyoutOp{nullptr};
    Operation* allocOp{nullptr};
    Operation* copyinOp{nullptr};
    LogicalTensorPtr gmTensor{nullptr};

    void Record(Operation* copyout = nullptr,
                Operation* alloc = nullptr,
                Operation* copyin = nullptr,
                LogicalTensorPtr gm = nullptr) {
        if (copyout) copyoutOp = copyout;
        if (alloc)   allocOp   = alloc;
        if (copyin)  copyinOp  = copyin;
        if (gm)      gmTensor  = gm;
    }
};

class OoOScheduler : public ScheduleBase, public ScheduleMainLoopBase {
public:
    Status Schedule(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function) : function_(function) {}

    // Non-owning observer. Caller must ensure the observer outlives the whole Schedule() call.
    void AddObserver(ScheduleObserver* observer) { observers_.push_back(observer); }
    // Defensive dedupe: dualdst 跨核 wake / spill 重入路径下,
    // 上层 Function::ScheduleBy -> RefreshOpPosition 不允许重复 op 出现。
    // 返回保留首次出现顺序的去重副本(内部 newOperations_ 不变,便于诊断)。
    std::vector<Operation*> GetNewOperations();
    int64_t workspaceOffset{0};
    std::unordered_map<PipeType, int> pipeEndTime;

    // === DualDst 开关 ===
    // OoOSchedule 在调用 Schedule() 之前设置(默认 false 保持原行为)。
    void SetEnableDualDst(bool v) { enableDualDst_ = v; }

private:
    // 存储排好顺序的Operation指针
    std::vector<Operation*> orderedOps;
    // Operation属性管理
    std::unordered_map<Operation*, int> opExecOrderMap;
    std::unordered_map<Operation*, PipeType> opPipeTypeMap;
    std::unordered_map<Operation*, bool> opIsAllocMap;
    std::unordered_map<Operation*, bool> opIsRetiredMap;
    std::unordered_map<Operation*, std::vector<Operation*>> opViewOpsMap;
    std::unordered_map<Operation*, CoreLocationType> opCoreLocationMap;

    std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS;

    // 分核数据结构
    std::unordered_map<CoreLocationType, std::map<npu::tile_fwk::MemoryType, BufferPool>> bufferManagerMap;

    std::unordered_map<int, Operation*> tensorOccupyMap;
    // tensor和其初始化时对应的alloc的core类型 memId-core类型
    std::unordered_map<int, Operation*> tensorAllocMap;

    std::unordered_map<CoreLocationType, std::map<MemoryType, IssueQueue>> allocIssueQueue;

    std::unordered_map<CoreLocationType, std::map<PipeType, IssueQueue>> issueQueues;

    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;

    // === DualDst: L0C tensor -> 该 L0C 整组选定的方向 (0=SplitM, 1=SplitN) ===
    // identify 阶段为每个被融合的 L0C 写入一次,fuse 阶段读出供 CopyOpAttribute 设置 SplitMN。
    // 一次 Schedule() 内复用;每次 RunDualDstFuse 入口处清空。
    std::unordered_map<LogicalTensorPtr, int64_t> dualDstL0CDirection_;

    // === DualDst memId -> 实际所在 UB pool 的 core (覆盖表) ===
    // 主线 FreeBuffer 用 opCoreLocationMap[tensorAllocMap[memId]] 推 core, 但 dualdst
    // 把一对 ub memId 各自放进 AIV0 / AIV1 两池, 而它们的 alloc op (tensorAllocMap
    // 指向的) 在 opCoreLocationMap 里通常是同一核 (alloc op 自己的归核 != UB 服务的
    // AIV 核)。该覆盖表在 CommitDualDstAlloc 阶段写入, FreeBuffer 优先查询此表以拿到
    // 正确的 pool core, 避免 "Tensor[X] not in bufferSlices" 错配。
    std::unordered_map<int, CoreLocationType> dualDstMemIdCoreOverride_;

    Function& function_;
    int issueId{0};
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    std::vector<Operation*> newOperations_;
    std::vector<Operation*> operations_;
    std::vector<ScheduleObserver*> observers_;

    bool enableDualDst_{false};

    // dualdst free 路径推核: 优先用 dualDstMemIdCoreOverride_ (CommitDualDstAlloc 写入
    // 的 memId -> 实际 UB pool core 映射), 否则回退到 alloc op 自身的 opCoreLocationMap。
    // 同时被 FreeBuffer / RetireIssue 两处调用以避免重复 code block 命中 codecheck dup-lines。
    CoreLocationType ResolveCoreForFree(int memId);

    IRBuilder irBuilder_;
    // memId → DDRBufferKind. Doubles as the seen-set for EmitInitDDRBuffer.
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;

    // Notification helpers — bodies live in ooo_scheduler_notify.cpp.
    void NotifyOpLaunch(Operation* op, int cycleEnd);
    void NotifyOpRetire(Operation* op, const std::vector<int>& freedMemIds);
    void NotifyAllocExec(Operation* op, int memId);
    void NotifySpill(LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp,
        const SingleSpillCreatedOps& created);
    void NotifyBufferRearrange(Operation* triggerOp, MemoryType memType,
        std::vector<BufferRearrangeEvent::Change> changes);
    void NotifyAllocFail(Operation* triggerOp, MemoryType memType, uint64_t requestSize);
    void NotifyScheduleEnd(bool success);
    void NotifyInitDDRBuffers();
    void NotifyMainLoopBegin();
    void NotifyMainLoopEnd();
    // Emit a single INIT_DDR_BUFFER event for `t` if its memId hasn't been registered yet.
    void EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind);

    static CoreLocation ToCoreLocation(CoreLocationType c);

    std::vector<DDRRef> BuildDDRRefs(Operation* op) const;

    // scheduler
    Status Init(
        const std::vector<Operation*>& opList,
        const std::unordered_map<Operation*, CoreLocationType>& opCoreMap =
            std::unordered_map<Operation*, CoreLocationType>(),
        const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);

    Status InitOpEntry(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    Status InitOpCoreType(Operation* op, const std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    void InitOpViewOps(Operation* op);
    std::string GetOpInfo(Operation* op) const;
    int GetOOperandIdx(Operation* op, int curMemId);

    void InitCoreConfig(const std::vector<Operation *> &opList);
    void InitTensorCoreMap();
    void InitIssueQueuesAndBufferManager();

    void AllocWorkspaceGM(const std::vector<Operation *> &opList);
    Status SeqSchedule();
    Status ExecuteAllocIssue(Operation* op, size_t &pcIdx);
    Status RetireIssue(Operation* op);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireCoreIssue(CoreLocationType targetCore, uint64_t& commitCnt, int& nextCycle);
     // ScheduleMainLoopBase 钩子实现
    Status PreMainLoop() override;
    Status PostMainLoop() override;
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle) override;
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op, std::vector<int>& freedMemIds);
    Status BufferAllocStage(uint64_t& commitCnt) override;
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType,
        IssueQueue &pipe);
    // ExecuteAllocIssue 的两个分支被拆出,统一以 (allocated 是否成功) 与调用方约定:
    //   返回 SUCCESS && allocated=true  : 已分配,调用方应 PopFront 并继续;
    //   返回 SUCCESS && allocated=false : 暂时无法分配 (Full),调用方应 break;
    //   返回 FAILED                       : 真错误。
    Status TryDualDstAllocOnce(Operation* op, uint64_t& commitCnt, bool& allocated);
    Status TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                               const std::vector<int>& reqMemIds,
                               uint64_t& commitCnt, bool& allocated);
    // 新增：基于Operation*的版本
    void HandleViewOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle) override;
    Status AllocTensorMemRange(Operation* op);
    Status AllocViewTensorMemRange(Operation &operation);
    Status CheckAndUpdateLifecycle();
    Status ApplySpillContext(SpillContext& ctx, Operation* allocOp);

    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation *> opList);
    Status PrintSpillFailedInfo(Operation* allocOp);

    // gen spill
    Status GenBufferSpill(Operation* allocOp, SpillContext &ctx);
    std::vector<int> SelectSpillBuffers(Operation* allocOp);
    // 单池选组: 薄壳, 委托 pool.GetSpillGroup, 与 dualdst 路径形成对称接口。
    std::vector<std::vector<int>> GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill);
    // 双池选组: 嵌套两次单池滑窗, 匹配条件是两侧 startAddr 一致 (即两池同地址段都能腾出
    // ≥ sizeNeedSpill 的空闲)。返回 vector<combined memId list>, 每个组里前半来自 poolA、
    // 后半来自 poolB。memId 跨 core 全局唯一, SpillBuffer 用 ResolveCoreForFree 反查正确池。
    std::vector<std::vector<int>> GetDualSpillGroup(
        BufferPool& poolA, BufferPool& poolB, size_t sizeNeedSpill);
    Status GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int> &groupNextUseTime,
        std::unordered_map<int, size_t> &nextUseTimeCache);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* op);
    void FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags);
    bool CheckMachineAndL1(Operation* spillOp, Operation* allocOp);

    Status SpillBuffer(int memId, Operation* spillAllocOp, SpillContext &ctx);
    Status HandleSpillMode(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillBufferFromDDR(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillL1BufferFor3510(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillReshapeFromDDRFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillReshapeL1BufferFor3510(int memId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillL0CBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status SpillMultiProducerBufferFor3510(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx, SingleSpillCreatedOps& created);
    Status CopyoutParticalBuffer(LogicalTensorPtr spillTensor, LogicalTensorPtr gmTensor, SpillContext &ctx);
    Status CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleOOperand, Operation* copyoutOp, Operation* spillAllocOp);
    Status FillSpillAssembleBuffer(int spillMemid, LogicalTensorPtr spillTensor, LogicalTensorPtr assembleTensor,
        Operation* copyoutOp, LogicalTensorPtr gmTensor, Operation* spillAllocOp, Operation*& wholeCopyinOut);
    LogicalTensorPtr CreateLocalTensor(LogicalTensorPtr spillTensor);
    LogicalTensorPtr CreateGMTensor(LogicalTensorPtr spillTensor, LogicalTensorPtr actualSpillTensor, int spillMemId,
        DataType gmDtype = DT_BOTTOM);
    LogicalTensorPtr CreateParticalTensor(LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand, LogicalTensorPtr spillTensor, std::vector<int64_t> toOffset);
    Operation* CreateAllocOp(LogicalTensorPtr oOperand);
    Operation* CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset, bool isND2NZ = false);
    Operation* CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<OpImmediate> offset);
    Operation* CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand);
    Operation* CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<int64_t> toOffset, std::vector<SymbolicScalar> toDynOffset, std::vector<SymbolicScalar> fromDynValidShape);

    const std::vector<int64_t>& GetLargerShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);

    bool IsSmallShapeSpill(Operation* op);
    bool IsUnusedTensor(Operation* spillOp);
    void UpdateSuccessorDependencies(Operation* succOp, Operation* spillOp,
        Operation* reloadCopyin, int spillMemId, int reloadMemId);
    void UpdatePredecessorAllocDependencies(Operation* succOp, Operation* reloadAlloc, int spillMemId);
    Status UpdateSmallShapeDependAndBuf(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap,
        int spillMemId, Operation* spillOp);

    // RemoveSmallShapeSpillResources 辅助函数
    void CollectUBSceneOpsAndTensors(Operation* producerOp, std::vector<Operation*>& opsToDelete,
        std::vector<LogicalTensorPtr>& tensorsToDelete);
    void CollectProducerChainForDeletion(
        LogicalTensorPtr spillTensor, std::vector<Operation*>& opsToDelete,
        std::vector<LogicalTensorPtr>& tensorsToDelete);
    void ReleaseDeletedOpBufRefs(Operation* op, const std::vector<LogicalTensorPtr>& tensorsToDelete);
    size_t CleanupCollectedOperations(
        const std::vector<Operation*>& opsToDelete, const std::vector<LogicalTensorPtr>& tensorsToDelete);
    void CleanupCollectedTensors(
        const std::vector<LogicalTensorPtr>& tensorsToDelete);
    void EraseOrphanedTensors(
        const std::vector<LogicalTensorPtr>& tensorsToDelete, const std::vector<Operation*>& opsToDelete);
    Status RemoveSmallShapeSpillResources(int spillMemId, LogicalTensorPtr spillTensor, SpillContext &ctx);

    Operation* GetSpillOp(int memId);
    LogicalTensorPtr GetSpillTensor(Operation* spillOp, int spillMemId);
    void CollectL0CConsumers(LogicalTensorPtr spillTensor, std::vector<Operation*> &consumers);
    Status GetActualSpillForNd2nz(Operation* &spillOp, LogicalTensorPtr &spillTensor);
    Status GetActualSpill(Operation* op, Operation* &actualOp, LogicalTensorPtr &actualTensor);
    void EraseSchedulerSideMaps(Operation* op);
    Status UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status SpillOnBlock() override;
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> coreLocation);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType> &orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType &spillMemType);
    Operation* SkipViewChain(Operation* start, bool followProducers);

    // 新增：插入Operation到orderedOps
    void InsertOrdered(Operation* insertOp);

    // === DualDst 融合 (实现位于 dualdst_fuse.cpp) ===
    // 入口: 在 Init 之后、GenSpillSchedule 之前调用
    Status RunDualDstFuse();
    // 阶段 1: 识别可融合 pair
    Status IdentifyDualDstPairs(std::vector<DualDstPair>& pairs);
    // 阶段 1 子步骤: 单个 L0C tensor 的处理
    void IdentifyPairsForOneL0C(LogicalTensorPtr l0cTensor,
                                const std::vector<Operation*>& copyUbs,
                                std::vector<DualDstPair>& pairs);
    // 阶段 2: 改图,把 pair 替换成单个 OP_L0C_COPY_UB_DUAL_DST
    Status FuseDualDstPairs(const std::vector<DualDstPair>& pairs);
    // 工具: allocOp 是否是 dualdst 保留下来的那条 ALLOC
    bool IsDualDstAlloc(Operation* allocOp);
    // 工具: 给定 dualdst alloc,返回其配对的另一个 memId; 失败返回 -1
    int GetDualDstPairedMemId(Operation* allocOp);
    // 工具: 给定 dualdst alloc,返回它服务的那个 OP_L0C_COPY_UB_DUAL_DST
    Operation* GetDualDstCopyOpFor(Operation* allocOp);
    // 工具: 把一个 op 从 orderedOps 与各 map 中安全移除
    void EraseFromOrderedOps(Operation* op);
    // 阶段 2 内部: 单对融合
    Status FuseOnePair(const DualDstPair& p);
    // 阶段 2 子步骤: 仅构造 OP_L0C_COPY_UB_DUAL_DST 节点 + 继承结构性属性
    // (InternalSubgraphID / AIVCore / isCube)。返回新建的 op 指针。
    Operation* CreateDualDstFusedOp(const DualDstPair& p, LogicalTensorPtr l0cIn);
    // 阶段 2 子步骤: 给已建的 dualdst op 写 CopyOpAttribute (对齐 GenerateMoveOp::SetL0C2UBCopyAttr 模板)。
    // 新规格各 attr 单值: srcOffset / shape / srcValidShape / dstValidShape。
    // realShape = 沿 SplitMN 轴对 attrE/attrL shape 相加;SplitMN 通过 OpAttributeKey::splitMN 落到 C。
    void SetDualDstCopyAttr(Operation* C, LogicalTensorPtr l0cIn,
                            const DualDstPair& p,
                            std::shared_ptr<CopyOpAttribute> attrE,
                            std::shared_ptr<CopyOpAttribute> attrL);
    // 阶段 2 子步骤: 把 opEarly / opLate / B 的所有依赖边迁到 C / 摘除
    void RewireEdgesForFusedOp(Operation* opEarly, Operation* opLate,
                               Operation* A, Operation* B, Operation* C);
    // 阶段 2 子步骤: 摘除被替代 op 在 tensor producer/consumer 链上的引用
    void DetachOldOpsFromTensors(const DualDstPair& p, LogicalTensorPtr l0cIn, Operation* B);
    // 阶段 2 子步骤: 把 C 写入 orderedOps 与各属性 map
    void RegisterFusedOpInMaps(Operation* C, int execOrder);
    // 阶段 2 子步骤: 同步 bufRefCount_ 与 opReqMemIdsMap[C]。
    // InitBufRefCount 已在 fuse 前跑过,反映的是 pre-fuse 图的引用计数;
    // 这里减掉 opEarly/opLate/B 的贡献,加上 C 的贡献,保证 FreeBuffer 能正确归零。
    // 必须在 EraseFromOrderedOps(opEarly/opLate/B) 之前调用 (依赖它们的 opReqMemIdsMap)。
    void SyncBufRefCountForFuse(const DualDstPair& p, Operation* B, Operation* C);

    // === DualDst 分配 ===
    // 调用方先用 IsDualDstAlloc 判定;此函数负责在 AIV0/AIV1 两个 UB 池上同地址放置
    // dualdst 的两个输出 tensor。
    //   返回 SUCCESS && allocated=true   : 两侧都已 AllocateAtOffset 成功
    //   返回 SUCCESS && allocated=false  : 当前两池找不到共同空闲段(语义=Full),
    //                                       由调用方触发 spill / break canAlloc
    //   返回 FAILED                       : 真错误(回滚已尝试)
    Status AllocateDualDstAtCurrent(Operation* allocA, bool& allocated);
    // 双池公共空闲偏移查找: 分别取两池 GetSortedFreeIntervals(), 归并求首个 ≥ size 的交集,
    // 命中返回起点 offset, 否则 nullopt。原 BufferPool::FindCommonFreeOffset 已上提至此,
    // 与 GetDualSpillGroup 同源 — 单池查询留 BufferPool, 跨池决策归 OoOScheduler。
    std::optional<uint64_t> FindCommonFreeOffset(BufferPool& poolA, BufferPool& poolB, uint64_t size);
    // 共享: 解析 dualdst alloc 的两 memId / 两 localBuffer / 两核归属,失败时已打日志。
    // 内部拆为 ResolveDualDstMemAndBuf + ResolveDualDstCores 两步, 控制单函数行数与圈复杂度。
    Status ResolveDualDstAllocCtx(Operation* allocOp, DualDstAllocCtx& ctx);
    Status ResolveDualDstMemAndBuf(Operation* allocOp, DualDstAllocCtx& ctx);
    Status ResolveDualDstCores(Operation* allocOp, DualDstAllocCtx& ctx);
    // 分配成功后的 bookkeeping (tensorOccupyMap / startCycle / 通知)
    void CommitDualDstAlloc(Operation* allocA, const DualDstAllocCtx& ctx, uint64_t off);
    // === DualDst spill ===
    // SelectSpillBuffers 内部按 IsDualDstAlloc 分叉, dualdst 双池选组 + 单池执行流程
    // (GenBufferSpill 内的 SpillBuffer 循环) 共用同一执行段。详见 spill_buffer.cpp。
    // 不再有 SpillDualDstAllocBuffer / SpillOneSideWithCoreSwap 两个独立函数。
    // 新增：Tensor输入更新辅助函数
    void UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr tensor, int spillMemId);
    void UpdateTensorInputForView(Operation& op, Operation* spillSrcOp, LogicalTensorPtr tensor);

    void ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId);
    void ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId);
    Status UpdateRemainMemid(int oldMemId, int newMemId);
    void UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp);
    int GetBufNextUseTime(int curMemId);
    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType);
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType);
    Status UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId, 
        Operation* spillAllocOp, bool isRetired = true);
    void UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp);
    Status InsertOps(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, Operation* spillAllocOp, int memId);
    Status UpdateScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
        Operation* spillAllocOp, LogicalTensorPtr localTensor, Operation* spillOp);
    Status UpdateNeedDeleteScheduleStatus(std::vector<std::pair<Operation*, std::vector<int>>> opMemidMap, int memId,
        Operation* spillAllocOp, LogicalTensorPtr spillTensor, Operation* spillOp, SpillContext &ctx);
    bool IsMultiProducerTensor(LogicalTensorPtr tensor);
    Status GetPartialWriteReplayAttr(Operation* producerOp, std::vector<int64_t> &toOffset,
 	    std::vector<SymbolicScalar> &toDynOffset, std::vector<SymbolicScalar> &fromDynValidShape) const;

    // ============ 辅助函数：获取Operation属性 ============
    int GetExecOrder(Operation* op) const { return opExecOrderMap.at(op); }
    PipeType GetPipeType(Operation* op) const { return opPipeTypeMap.at(op); }
    bool IsAlloc(Operation* op) const { return opIsAllocMap.at(op); }
    bool IsRetired(Operation* op) const { return opIsRetiredMap.at(op); }
    CoreLocationType& GetCoreLocation(Operation* op) { return opCoreLocationMap[op]; }
    const CoreLocationType& GetCoreLocation(Operation* op) const { return opCoreLocationMap.at(op); }
    std::vector<Operation*>& GetViewOps(Operation* op) { return opViewOpsMap[op]; }

    // 辅助函数：设置Operation属性
    void SetExecOrder(Operation* op, int order) { opExecOrderMap[op] = order; }
    void SetPipeType(Operation* op, PipeType type) { opPipeTypeMap[op] = type; }
    void SetIsAlloc(Operation* op, bool isAlloc) { opIsAllocMap[op] = isAlloc; }
    void SetIsRetired(Operation* op, bool isRetired) { opIsRetiredMap[op] = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { opCoreLocationMap[op] = loc; }

    // 辅助函数：检查Operation是否在调度中
    bool IsOpInSchedule(Operation* op) const { return opExecOrderMap.find(op) != opExecOrderMap.end(); }

    void UpdateL0MXMap(const std::vector<Operation*> &opList);
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
