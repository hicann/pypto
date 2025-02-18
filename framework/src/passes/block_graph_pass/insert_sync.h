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
 * \file insert_sync.h
 * \brief
 */

#ifndef PASS_INSERT_SYNC_H
#define PASS_INSERT_SYNC_H

#include <queue>
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t EVENTID_DEADLOCK_ENTER_TIME = 5;
constexpr uint64_t DEADLOCK_TIME_THRESHOLD = 2;
constexpr uint64_t LEFT_OFFSET1 = 32;
constexpr uint64_t LEFT_OFFSET2 = 16;
constexpr uint64_t LEFT_OFFSET3 = 8;
constexpr uint64_t LEFT_OFFSET4 = 24;
constexpr uint64_t MAX_POP = 8;
// 两个op之间插入的set_flag/wait_flag数量的最大值为192
constexpr uint64_t SEQUENCE_IDX = 200;
constexpr uint64_t HALF_SEQUENCE_IDX = 100;
struct Interval {
    int start;
    int end;
    int idx;
    Interval(int s, int e, int id) : start(s), end(e), idx(id) {}
};

struct IntervalTreeNode {
    Interval interval;
    int max;
    IntervalTreeNode *left;
    IntervalTreeNode *right;
    explicit IntervalTreeNode(Interval i) : interval(i), max(i.end), left(nullptr), right(nullptr) {}
};

class RangeSearchTree {
public:
    ~RangeSearchTree() {
        FreeTree();
    }
    void Insert(int left, int right, int idx);
    std::set<int> GetCovered(int left, int right);

private:
    void FreeTree();
    Status ProcessTreeNode(const Interval &interval, IntervalTreeNode *currPtr, std::vector<IntervalTreeNode*> &intervalStack);
    Status InsertInterval(const Interval &interval);
    void OverlapSearch(const Interval &interval, std::set<int> &result);
    IntervalTreeNode *treeRoot = nullptr;
};

class DataDependencySearcher {
public:
    std::set<int> Find(Operation *opWait);
    void Insert(const Operation *opSet, int idx);

private:
    void CheckRAWSearchTree(Operation *opWait, std::set<int> &res);
    void CheckWAWSearchTree(Operation *opWait, std::set<int> &res);
    void CheckWARSearchTree(Operation *opWait, std::set<int> &res);
    void InsertWAWSearchTree(const Operation *opSet, int idx);
    void InsertRAWSearchTree(const Operation *opSet, int idx);
    void InsertWARSearchTree(const Operation *opSet, int idx);

    std::unordered_map<MemoryType, RangeSearchTree> wawSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> warSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> rawSearchTree_;
    std::unordered_map<int, std::set<int>> readDdrMemMap;
    std::unordered_map<int, std::set<int>> writeDdrMemMap;
};

using IndexOp = std::pair<uint64_t, std::reference_wrapper<Operation>>;
enum class PipeSeq { AIC_MTE2 = 0, AIC_MTE1, AIC_M, AIC_FIX, AIV_MTE2, AIV_V, AIV_MTE3, AIC_MTE3, AIV_S, AIC_S, PIPE_END };

class PipeSync {
public:
    PipeSync() { InitIssueQueue(); }
    Status InsertSync(Function &function, std::vector<Operation *> &syncedOpLog);
    void PhaseKernelProcess(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog);
    Status ProcessViewOrder(Operation &op, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap);
    Status ProcessAssembleOrder(Operation &op, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap);
    Status ProcessViewAssembleOrder(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew);

private:
    struct PipeCoreReal {
        PipeCoreReal(PipeType p, CoreType c) :pipe(p), core(c) {}
        PipeType pipe;
        CoreType core;

        bool operator==(const PipeCoreReal &t) const { return (this->pipe == t.pipe && this->core == t.core); }

        bool operator!=(const PipeCoreReal &t) const { return !(*this == t); }
    };

    struct PipeCoreRealCompare {
        bool operator()(const PipeCoreReal &lhs, const PipeCoreReal &rhs) const {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET2) | (static_cast<uint64_t>(lhs.pipe) << LEFT_OFFSET3))
                 < ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET2) | (static_cast<uint64_t>(rhs.pipe) << LEFT_OFFSET3));
        }
    };

    struct PipeCore {
        PipeCore(PipeType ps, PipeType pe, CoreType c) : pipeStart(ps), pipeEnd(pe), core(c) {}
        PipeType pipeStart;
        PipeType pipeEnd;
        CoreType core;

        bool operator==(const PipeCore &t) const { return (this->pipeStart == t.pipeStart && this->pipeEnd == t.pipeEnd && this->core == t.core); }

        bool operator!=(const PipeCore &t) const { return !(*this == t); }
    };

    struct PipeCoreCompare {
        bool operator()(const PipeCore &lhs, const PipeCore &rhs) const {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET4) | (static_cast<uint64_t>(lhs.pipeStart) << LEFT_OFFSET2) | (static_cast<uint64_t>(lhs.pipeEnd) << LEFT_OFFSET3))
                 < ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET4) | (static_cast<uint64_t>(rhs.pipeStart) << LEFT_OFFSET2) | (static_cast<uint64_t>(rhs.pipeEnd) << LEFT_OFFSET3));
        }
    };

    using PipePair = std::pair<PipeCoreReal, PipeCoreReal>; // setPipe, waitPipe

    struct PipePairHash {
        std::size_t operator()(const PipePair &pp) const noexcept {
            std::size_t res = 0;
            HashCombine(res, pp.first.pipe);
            HashCombine(res, pp.first.core);
            HashCombine(res, pp.second.pipe);
            HashCombine(res, pp.second.core);
            return res;
        };
    };

    struct IndexVecHash {
        std::size_t operator()(const std::pair<size_t, size_t> &pp) const noexcept {
            std::size_t res = 0;
            HashCombine(res, pp.first);
            HashCombine(res, pp.second);
            return res;
        };
    };

    struct DepOp {
        DepOp(size_t i, PipeCore pipeCore) : idx(i), selfPipeCore(pipeCore) {}
        size_t idx = SIZE_MAX;       // idx in oplog
        size_t idxInPipe = SIZE_MAX; // idx in the pipe belonging to
        PipeCore selfPipeCore;
        bool issued{false};

        std::vector<size_t> setPipe;  // this op will set_flag for op in setPipe; 后
        std::vector<size_t> waitPipe; // this op will wait_flag for op in waitPipe; 前
        std::string DumpDepOp(std::vector<Operation *> opLog = {});
    };

    struct IssueQueue {
        explicit IssueQueue(PipeCoreReal pipe) : selfPipeCore(pipe) {}
        PipeCoreReal selfPipeCore;
        size_t currOp{0};
        std::vector<size_t> ops;
        std::string DumpIssueQueue(std::vector<Operation *> opLogPtr = {});
    };

    struct PipeDepInfo {
        size_t waitIdx;
        std::map<PipeCoreReal, size_t, PipeCoreRealCompare> setPipes;
        std::string DumpPipeDepInfo();
    };

    struct DataDepInfo {
        PipeType setp;
        CoreType setc;
        PipeType waitp;
        CoreType waitc;
        std::vector<int> setOpIdList{};
        std::vector<int> setOpEventIdList{};
        std::vector<std::pair<int, int>> opDepList{};
    };

    struct IssueNum {
        // max op can be issued this round
        std::unordered_map<PipePair, size_t, PipePairHash> maxIssueNum;
        // already issued op this round
        std::unordered_map<PipePair, size_t, PipePairHash> currIssueNum;
    };

    std::string PipeSeqName(PipeSeq seq) const;
    PipeSeq GetPipeSeq(PipeCoreReal pipe);
    PipeCoreReal GetPipeFromSeq(PipeSeq seq);
    Status PipeDispatch(const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog);
    Status AdjustReshapeCfg(TileOpCfg &opcfg, Operation &op);
    Status AdjustCopyInCfg(TileOpCfg &opcfg, Operation &op);
    Status AdjustCopyOutCfg(TileOpCfg &opcfg, Operation &op);
    Status AdjustOpCfg(TileOpCfg &opcfg, Operation &op);
    void InitIssueQueue();
    void EnqueueOp(DepOp &op, const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog);
    void RemoveOpDep(DepOp &setOp, DepOp &waitOp) const;
    void AddPhaseOp1(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog, size_t &i, size_t &prerun);
    void AddPhaseOp2(Function &function, std::vector<Operation *> &dstLog, size_t &prerun);
    Status AddOpDep(DepOp &setOp, DepOp &waitOp);
    Status AdjustOpDep(DepOp &op, size_t waitOpIdx, IssueQueue &issueQ, bool &failedFlag);
    Status HandleEventID(DepOp &op, IssueQueue &issueQ, IssueNum &issuenum, bool &deadlock, bool &res);
    Status PopFromQueue(IssueQueue &issueQ, std::vector<size_t> &poped, bool &deadlock);
    Status InjectWaitFlag(Function &function, size_t idx, std::vector<IndexOp> &syncedOpLog);
    Status InjectSetFlag(Function &function, size_t idx, std::vector<IndexOp> &syncedOpLog);
    Status InjectSync(Function &function, std::vector<Operation *> opLogPtr, size_t idx, std::vector<IndexOp> &syncedOpLog);
    Status IssueOpPipeSeq(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, bool &eventIdDeadlock, size_t &issued);
    Status IssueSyncOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, size_t &totalIssued, size_t &allIssued);
    Status IssueOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog);
    Status ProcessDeadLock(uint64_t &eventIdDeadlockEnterTimes, bool &eventIdDeadlock, std::vector<IndexOp> &syncedOpLog);
    Status SynDependency(int maxOverlapDepIdx, const DataDepInfo &depInfo, const PipePair &pipePair, std::vector<IndexOp> &syncedOpLog);
    Status GetDepInfo(std::vector<IndexOp> &syncedOpLog, const PipePair &pipePair, DataDepInfo &depInfo);
    Status RelaxFakeDataDep(std::vector<IndexOp> &syncedOpLog);
    bool CheckIssuedOp(const DepOp &op);
    bool ConstructDepInfo(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i);
    bool FindDataDep(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i);
    bool FindMaxOverlap(DataDepInfo &depInfo, int &maxOverlapDepIdx);
    bool GenSyncOp(PipeCoreReal set, PipeCoreReal wait, int eventId, bool isSet, Operation &op);
    Status GetEventId(const PipePair &pp, int &eventId);
    bool HasFreeEventId(const PipePair &pp);
    bool BufOverlap(const TileRange &range1, int magic1, const TileRange &range2, int magic2) const;
    bool CheckWawDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) const;
    bool CheckRawDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) const;
    bool CheckWarDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) const;
    bool HasDataDependency(const Operation &opSet, const Operation &opWait, size_t k, size_t idx) const;
    void UpdateDep(DepOp &currOp, DepOp &prevOp);
    bool IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation *> opLogPtr);
    void FindDep(DepOp &op, const std::vector<Operation *> opLogPtr, size_t idx, DataDependencySearcher& dataDependencySearcher);
    std::deque<int> &GetFreeEventIdQueue(const PipePair &pp);
    int GetSyncSrcLogIdx(std::vector<IndexOp> &syncedOpLog, int i);
    int GetMaxEventId(const PipePair &pp);
    Status ProcessView(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair);
    Status ProcessAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair);
    Status ProcessViewAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair);
    Status ReorderViewAssemble(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew, const std::unordered_map<Operation *, Operation *> &changeMap);
    std::string DumpLatestPipeDepMap();

    std::vector<DepOp> depOps_;
    // Cube: MTE2, MTE1, M, FIX, Vector: MTE2, V, MTE3
    std::vector<IssueQueue> issueState_;
    std::unordered_map<PipePair, std::deque<int>, PipePairHash> freeEventId_;
    std::unordered_map<std::pair<size_t, size_t>, int, IndexVecHash> setWaitPairMap_;
    std::map<PipeCoreReal, PipeDepInfo, PipeCoreRealCompare> latestPipeDep_;
    static std::map<PipeCoreReal, PipeSeq, PipeCoreRealCompare> pipe2Seq;
    static std::map<PipeSeq, PipeCoreReal> seq2pipe;
    static std::vector<PipePair> dataDepPair;

    static constexpr int EVENT_NUM = 8;
    static constexpr int CV_EVENT_NUM = 16;
    static constexpr int EVENT_ID7 = 7;
    int minimalMergeOverlap{25};
    std::unordered_map<PipePair, std::vector<int>, PipePairHash> doublePipeOp; // pipepair, opmagic
    std::queue<size_t> orderedOplist_;
};

class InsertSync : public Pass {
public:
    InsertSync() : Pass("InsertSync") {}
    ~InsertSync() override {}
    void SetEnableDebug(bool enableDebug) { enableDebug_ = enableDebug; }

private:
    Status RunOnFunction(Function &function) override;
    void InsertPipeAll(Function *subGraphFunc);
    Status GenNewOpList(Function *subGraphFunc, std::vector<Operation *> &opListNew);
    Status CheckNewOpListSeq(const std::vector<Operation *> &oriOpList, const std::vector<Operation *> &opListNew);
    Status InsertSyncMainLoop(Function *subGraphFunc);
    bool enableDebug_{false};
};
} // namespace tile_fwk
} // namespace npu

#endif // PASS_INSERT_SYNC_H