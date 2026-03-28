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
 * \file function.h
 * \brief
 */
/* for flow verify tool */

#pragma once

#include "tilefwk/pypto_fwk_log.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/interpreter/operation.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "calc.h"
#include "interface/interpreter/verify_error.h"
#include <algorithm>

namespace npu::tile_fwk {

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1);
    }
};

struct FunctionIODataPair {
    std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList;

    FunctionIODataPair() {}
    FunctionIODataPair(std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList_,
        std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList_)
        : incastDataViewList(incastDataViewList_), outcastDataViewList(outcastDataViewList_) {}

    std::vector<std::shared_ptr<LogicalTensorData>> &GetIncastDataViewList() { return incastDataViewList; }
    std::vector<std::shared_ptr<LogicalTensorData>> &GetOutcastDataViewList() { return outcastDataViewList; }

    // One tensor might occur in both incast and outcast simultaneously for multiple times, so we must copy
    // simultaneously
    static void CopyWithLinkRelationship(FunctionIODataPair &dst, const FunctionIODataPair &src) {
        struct CopyInfo {
            bool isIncast;
            int index;
            CopyInfo(bool isIncast_, int index_) : isIncast(isIncast_), index(index_) {}
        };

        std::unordered_map<std::shared_ptr<LogicalTensorData>, std::vector<CopyInfo>> copyInfoDict;
        for (size_t k = 0; k < src.incastDataViewList.size(); k++) {
            copyInfoDict[src.incastDataViewList[k]].emplace_back(true, k);
        }
        for (size_t k = 0; k < src.outcastDataViewList.size(); k++) {
            copyInfoDict[src.outcastDataViewList[k]].emplace_back(false, k);
        }

        dst.incastDataViewList.resize(src.incastDataViewList.size());
        dst.outcastDataViewList.resize(src.outcastDataViewList.size());
        for (auto &[srcDataView, copyInfoList] : copyInfoDict) {
            auto dstData = srcDataView->DeepCopy();
            for (auto &[isIncast, index] : copyInfoList) {
                if (isIncast) {
                    dst.incastDataViewList[index] = dstData;
                } else {
                    dst.outcastDataViewList[index] = dstData;
                }
            }
        }
        for (size_t k = 0; k < dst.incastDataViewList.size(); k++) {
            ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, dst.incastDataViewList[k] != nullptr);
        }
        for (size_t k = 0; k < dst.outcastDataViewList.size(); k++) {
            ASSERT(ControlFlowScene::FUNC_IO_DATAVIEW_NULL, dst.outcastDataViewList[k] != nullptr);
        }
    }
};

struct FunctionFrame {
    const Function *func;
    const Operation *callop;
    const std::shared_ptr<CallOpAttribute> callopAttr;
    std::shared_ptr<FunctionIODataPair> inoutDataPair;
    std::unordered_map<std::shared_ptr<RawTensor>, std::shared_ptr<RawTensorData>> rawTensorDataDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<RawTensor>> spillRawTensorDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>> tensorDataViewDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::string> tensorDataBinDict;
    std::unordered_map<std::shared_ptr<LogicalTensorData>, std::shared_ptr<LogicalTensor>> callopDataViewTensorDict;  // Record the relationship between the callop data view and the tensor
    int frameIndex;
    std::set<int> indexOutcastOpIndices;
    std::vector<std::set<LogicalTensorPtr>> inplaceTensorSetList;
    int funcIndex;
    size_t funcHash;
    std::string funcType;
    std::string funcGraphType;
    int rootFuncIndex{-1};
    size_t rootFuncHash;
    std::string rootFuncType;
    std::string rootFuncGraphType;
    int passIndex{-1};

    Operation *currentOperation;

    const std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>> &GetTensorDataViewDict() const { return tensorDataViewDict; }

    int GetFrameIndex() const { return frameIndex; }

    void InitInplaceDataViewList() {
        inplaceTensorSetList.clear();
        if (func == nullptr || indexOutcastOpIndices.empty()) {
            return;
        }

        auto ops = const_cast<Function *>(func)->Operations(false);
        // 建立 Operation 指针到其在 Operations(false) 中下标的映射
        std::unordered_map<Operation *, int> opIndexMap;
        opIndexMap.reserve(ops.size());
        for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
            opIndexMap[&ops[static_cast<size_t>(i)]] = i;
        }

        // 使用拷贝，避免在遍历时直接修改 indexOutcastOpIndices
        std::set<int> indices = indexOutcastOpIndices;

        for (auto index : indices) {
            // 该 index 可能在前一次遍历链路时被删除，这里跳过已删除的 INDEX_OUTCAST
            if (indexOutcastOpIndices.count(index) == 0) {
                continue;
            }
            if (index < 0 || static_cast<size_t>(index) >= ops.size()) {
                continue;
            }

            Operation &idxOutcastOp = ops[static_cast<size_t>(index)];
            auto iOps = idxOutcastOp.GetIOperands();
            auto oOps = idxOutcastOp.GetOOperands();
            // 需要至少 3 个输入，且至少 1 个输出
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, iOps.size() > 2);
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, !oOps.empty());

            LogicalTensorPtr startTensor = iOps[2];
            LogicalTensorPtr endTensor = oOps[0];
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, startTensor != nullptr);
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, endTensor != nullptr);

            std::set<LogicalTensorPtr> tensorGroup;
            std::unordered_set<LogicalTensorPtr> visitedTensor;
            std::unordered_set<Operation *> visitedOp;
            bool chainValid = true;
            Operation *rootIndexOutcastOp = &idxOutcastOp;

            TraverseBackward(startTensor, rootIndexOutcastOp, tensorGroup, visitedTensor, visitedOp, chainValid, opIndexMap);
            if (!chainValid) {
                continue;
            }
            TraverseForward(endTensor, rootIndexOutcastOp, tensorGroup, visitedTensor, visitedOp, chainValid, opIndexMap);
            if (!chainValid) {
                continue;
            }

            inplaceTensorSetList.emplace_back(std::move(tensorGroup));
        }

    }
    FunctionFrame(const Function *func_, const Operation *callop_,
        const std::shared_ptr<CallOpAttribute> &callopAttr_, std::shared_ptr<FunctionIODataPair> inoutDataPair_,
        int frameIndex_)
        : func(func_),
          callop(callop_),
          callopAttr(callopAttr_),
          inoutDataPair(inoutDataPair_),
          frameIndex(frameIndex_) {
        if (func != nullptr) {
            int idx = 0;
            auto ops = const_cast<Function *>(func)->Operations(false);
            for (auto &op : ops) {
                if (op.GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                    indexOutcastOpIndices.insert(idx);
                }
                ++idx;
            }
        }
        InitInplaceDataViewList();
        if (inoutDataPair != nullptr) {
            ASSERT(ControlFlowScene::FUNC_INCAST_COUNT_MISMATCH,
                   func->GetIncast().size() == inoutDataPair->incastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->incastDataViewList.size(); i++) {
                AddDataView(func->GetIncast()[i], inoutDataPair->incastDataViewList[i]);
            }
            ASSERT(ControlFlowScene::FUNC_OUTCAST_COUNT_MISMATCH,
                   func->GetOutcast().size() == inoutDataPair->outcastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->outcastDataViewList.size(); i++) {
                AddDataView(func->GetOutcast()[i], inoutDataPair->outcastDataViewList[i]);
            }
            DoAddCallopInOutDataView();
        }
    }

    void UpdateCurrentOperation(Operation *op) { currentOperation = op; }

    std::shared_ptr<LogicalTensorData> GetDataView(const std::shared_ptr<LogicalTensor> &tensor) {
        if (!tensorDataViewDict.count(tensor)) {
            return nullptr;
        }
        auto view = tensorDataViewDict[tensor];
        return view;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> GetDataViewList(
        const std::vector<std::shared_ptr<LogicalTensor>> &tensorList) {
        std::vector<std::shared_ptr<LogicalTensorData>> viewList(tensorList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            viewList[i] = GetDataView(tensorList[i]);
        }
        return viewList;
    }

    void AddDataView(
        const std::shared_ptr<LogicalTensor> &tensor, const std::shared_ptr<LogicalTensorData> &dataView) {
        if (tensorDataViewDict.count(tensor)) {
            ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_MISMATCH,
                   tensorDataViewDict[tensor] == dataView);
        } else {
            DoAddTensorDataView(tensor, dataView);
            DoAddRawTensorDataView(tensor->GetRawTensor(), dataView->GetData());
        }
    }
    void AddDataViewList(const std::vector<std::shared_ptr<LogicalTensor>> &tensorList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &dataViewList) {
        ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_LIST_SIZE_MISMATCH,
               tensorList.size() == dataViewList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            AddDataView(tensorList[i], dataViewList[i]);
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(const std::shared_ptr<LogicalTensor> &tensor,
        const std::vector<int64_t> &offset, const std::vector<int64_t> &validShape,
        const std::vector<int64_t> &rawShape, DataType dtype,
        const std::shared_ptr<LogicalTensor> &inplaceTensor = nullptr) {
        if (tensorDataViewDict.count(tensor)) {
            tensorDataViewDict[tensor]->UpdateValidShape(validShape);
            return tensorDataViewDict[tensor];
        }

        auto raw = inplaceTensor ? inplaceTensor->GetRawTensor() : tensor->GetRawTensor();
        bool isSpilled = false;

        std::string spillRawMaigc = "1056964608";
        std::string rawMagic = std::to_string(raw->GetRawMagic());
        if (rawMagic.find(spillRawMaigc) != std::string::npos) {
            if (spillRawTensorDict.count(tensor)) {
                raw = spillRawTensorDict[tensor];
            } else {
                raw = std::make_shared<RawTensor>(dtype, rawShape);
                DoAddSpillRawTensor(tensor, raw);
            }
            isSpilled = true;
        }
        std::shared_ptr<RawTensorData> rawData;
        if (rawTensorDataDict.count(raw)) {
            rawData = rawTensorDataDict[raw];
        } else {
            ASSERT(ControlFlowScene::FUNC_INPLACE_ALLOC_CONFLICT, inplaceTensor == nullptr);
            rawData = std::make_shared<RawTensorData>(dtype, rawShape);
            rawData->resize(rawData->GetElementSize() * rawData->GetSize());
        }
        DoAddRawTensorDataView(tensor->GetRawTensor(), rawData);
        std::shared_ptr<LogicalTensorData> view =
            std::make_shared<LogicalTensorData>(rawData, tensor->GetShape(), validShape, offset);
        view->SetIsSpilled(isSpilled);
        DoAddTensorDataView(tensor, view);
        return view;
    }

private:
    bool IsAllowedInplaceChainOpcode(Opcode opcode) const {
        return opcode == Opcode::OP_INDEX_OUTCAST ||
               opcode == Opcode::OP_VIEW ||
               opcode == Opcode::OP_RESHAPE ||
               opcode == Opcode::OP_ASSEMBLE ||
               opcode == Opcode::OP_PRINT;
    }

    void TraverseBackward(LogicalTensorPtr t,
                          Operation *rootIndexOutcastOp,
                          std::set<LogicalTensorPtr> &tensorGroup,
                          std::unordered_set<LogicalTensorPtr> &visitedTensor,
                          std::unordered_set<Operation *> &visitedOp,
                          bool &chainValid,
                          const std::unordered_map<Operation *, int> &opIndexMap) {
        if (!chainValid || t == nullptr) {
            return;
        }
        if (visitedTensor.insert(t).second) {
            tensorGroup.insert(t);
        }
        for (auto producer : t->GetProducers()) {
            if (producer == nullptr) {
                continue;
            }
            if (!IsAllowedInplaceChainOpcode(producer->GetOpcode())) {
                chainValid = false;
                return;
            }
            if (visitedOp.insert(producer).second) {
                // 如果向前遍历到其他 INDEX_OUTCAST，将其从 indexOutcastOpIndices 中移除，避免之后重复遍历
                if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST &&
                    producer != rootIndexOutcastOp) {
                    auto it = opIndexMap.find(producer);
                    if (it != opIndexMap.end()) {
                        indexOutcastOpIndices.erase(it->second);
                    }
                }
                // IndexOutcast 只从第三个输入继续向前，其余（view/reshape/assemble）只有一个输入
                auto &producerInputs = producer->GetIOperands();
                if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                    if (producerInputs.size() > 2 && producerInputs[2] != nullptr) {
                        TraverseBackward(producerInputs[2], rootIndexOutcastOp, tensorGroup, visitedTensor,
                                         visitedOp, chainValid, opIndexMap);
                    }
                } else {
                    if (!producerInputs.empty() && producerInputs[0] != nullptr) {
                        TraverseBackward(producerInputs[0], rootIndexOutcastOp, tensorGroup, visitedTensor,
                                         visitedOp, chainValid, opIndexMap);
                    }
                }
            }
        }
    }

    void TraverseForward(LogicalTensorPtr t,
                         Operation *rootIndexOutcastOp,
                         std::set<LogicalTensorPtr> &tensorGroup,
                         std::unordered_set<LogicalTensorPtr> &visitedTensor,
                         std::unordered_set<Operation *> &visitedOp,
                         bool &chainValid,
                         const std::unordered_map<Operation *, int> &opIndexMap) {
        if (!chainValid || t == nullptr) {
            return;
        }
        if (visitedTensor.insert(t).second) {
            tensorGroup.insert(t);
        }
        for (auto consumerOp : t->GetConsumers()) {
            if (consumerOp == nullptr) {
                continue;
            }
            if (!IsAllowedInplaceChainOpcode(consumerOp->GetOpcode())) {
                chainValid = false;
                return;
            }
            if (visitedOp.insert(consumerOp).second) {
                // 如果向后遍历到其他 INDEX_OUTCAST，将其从 indexOutcastOpIndices 中移除，避免之后重复遍历
                if (consumerOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST &&
                    consumerOp != rootIndexOutcastOp) {
                    auto it = opIndexMap.find(consumerOp);
                    if (it != opIndexMap.end()) {
                        indexOutcastOpIndices.erase(it->second);
                    }
                }
                // IndexOutcast / View / Reshape / Assemble 视为一个输入一个输出，只从其输出继续向后
                auto &consumerOutputs = consumerOp->GetOOperands();
                if (!consumerOutputs.empty() && consumerOutputs[0] != nullptr) {
                    TraverseForward(consumerOutputs[0], rootIndexOutcastOp, tensorGroup, visitedTensor,
                                    visitedOp, chainValid, opIndexMap);
                }
            }
        }
    }

    void DoAddTensorDataView(
            const std::shared_ptr<LogicalTensor> &tensor,
            const std::shared_ptr<LogicalTensorData> &dataView) {
        ASSERT(ControlFlowScene::FUNC_TENSOR_DATAVIEW_DUP, !tensorDataViewDict.count(tensor));
        tensorDataViewDict[tensor] = dataView;
    }
    void DoAddRawTensorDataView(
        const std::shared_ptr<RawTensor> &rawTensor, const std::shared_ptr<RawTensorData> &data) {
        rawTensorDataDict[rawTensor] = data;
    }
    void DoAddSpillRawTensor(
        const std::shared_ptr<LogicalTensor> &tensor, const std::shared_ptr<RawTensor> &rawtensor) {
        ASSERT(ControlFlowScene::FUNC_SPILL_RAW_TENSOR_DUP, !spillRawTensorDict.count(tensor));
        spillRawTensorDict[tensor] = rawtensor;
    }
    void DoAddCallopInOutDataView() {
        if (callop == nullptr) {
            return;
        }
        for (size_t i = 0; i < inoutDataPair->incastDataViewList.size(); i++) {
            callopDataViewTensorDict[inoutDataPair->incastDataViewList[i]] = callop->GetIOperands()[i];
        }
        for (size_t i = 0; i < inoutDataPair->outcastDataViewList.size(); i++) {
            callopDataViewTensorDict[inoutDataPair->outcastDataViewList[i]] = callop->GetOOperands()[i];
        }
    }
};

struct FunctionCaptureExecution {
    Function *func;
    std::shared_ptr<FunctionIODataPair> baseline;
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::unordered_map<std::string, ScalarImmediateType> loopSymbolDict;

    std::vector<std::shared_ptr<FunctionFrame>> frameList;

    std::shared_ptr<FunctionIODataPair> golden;

    FunctionCaptureExecution(Function *func_ = nullptr) : func(func_) {
        baseline = std::make_shared<FunctionIODataPair>();
        golden = std::make_shared<FunctionIODataPair>();
    }

    const std::vector<std::shared_ptr<FunctionFrame>> &GetFrameList() const { return frameList; }

    void CaptureFrom(
            const std::shared_ptr<FunctionIODataPair> &b,
            const std::unordered_map<std::string, ScalarImmediateType> &s) {
        FunctionIODataPair::CopyWithLinkRelationship(*baseline, *b);
        symbolDict = s;
    }

    void CaptureSymbolDictFrom(
            const std::unordered_map<std::string, ScalarImmediateType> &s) {
        symbolDict = s;
    }

    void CaptureGoldenFrom(
            const std::shared_ptr<FunctionIODataPair> &g) {
        FunctionIODataPair::CopyWithLinkRelationship(*golden, *g);
    }

    std::unordered_map<std::string, ScalarImmediateType> CaptureTo(
            std::shared_ptr<FunctionIODataPair> &c) const {
        FunctionIODataPair::CopyWithLinkRelationship(*c, *baseline);
        return symbolDict;
    }
};

struct FunctionControlFlowExecution {
    std::unordered_map<Function *, std::vector<std::shared_ptr<FunctionCaptureExecution>>> executionListDict;
};

constexpr int EXEC_DUMP_LEVEL_OPERATION = 1;
constexpr int EXEC_DUMP_LEVEL_TENSOR = 2;
const std::unordered_set<Opcode> MIX_PATH_OPS = {
    Opcode::OP_UB_COPY_L1,
    Opcode::OP_L0C_COPY_UB
};

enum class VerifyType { INVALID, TENSOR_GRAPH, PASS, EXECUTE_GRAPH };
enum class OpInfoCsvHeader {
    num = 0,
    passName,
    pathFuncMagicName,
    pathFuncMagic,
    pathFuncHash,
    loopInfo,
    rootFuncType,
    rootFuncGraphType,
    rootFuncID,
    rootFuncHash,
    funcType,
    funcGraphType,
    funcID,
    funcHash,
    rawTensorMagic,
    outputRawShape,
    outputDtype,
    outputFormat,
    outputSymbol,
    tensorMagic,
    tensorOffset,
    outputShape,
    outputValidShape,
    outputDynValidShape,
    callopMagic,
    callopRawMagic,
    opMagic,
    opCode,
    attrOffset,
    attrAtomic,
    ioflag,
    timeStamp,
    outputTensor,
    inputTensors,
    inputValidShape,
    COL_COUNT
};

enum class ProgrameInfoCsvHeader {
    num = 0,
    goldenPassName,
    passName,
    pathFuncMagicName,
    pathFuncMagic,
    pathFuncHash,
    loopInfo,
    ioflag,
    goldenRawMagic,
    outputRawMagic,
    outputRawShape,
    outputDtype,
    outputFormat,
    outputSymbol,
    outputShape,
    outputValidShape,
    aTimeStamp,
    goldenTensor,
    bTimeStamp,
    outputTensor,
    verifyResult,
    rtolAndAtol,
    failCnt,
    totalCnt,
    mre,
    mreTop8,
    mreTop1Permil,
    mae,
    maeTop8,
    maeTop1Permil,
    aMax,
    aMin,
    aAvg,
    aAavg,
    aZero,
    aInfnan,
    bMax,
    bMin,
    bAvg,
    bAavg,
    bZero,
    bInfnan,
    COL_COUNT
};

constexpr int32_t toIndex(OpInfoCsvHeader e) noexcept {
    return static_cast<int32_t>(e);
}

constexpr int32_t toIndex(ProgrameInfoCsvHeader e) noexcept {
    return static_cast<int32_t>(e);
}

struct FunctionInterpreter {
    FunctionInterpreter() : operationInterpreter(std::make_shared<OperationInterpreter>()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        timestamp << "_" << std::setw(6) << std::setfill('0') << us;    // 6 is the width
        dumpPath = config::GetVerifyOption<std::string>(KEY_PASS_VERIFY_SAVE_TENSOR_DIR);
        if (dumpPath.empty()) {
            dumpPath = config::LogTopFolder();
        }
        dumpPath = dumpPath + "/" + "verify_" + timestamp.str() + "/";
        CreateMultiLevelDir(dumpPath);

        std::string dumpOpFilePath = dumpPath + "verify_graph_data_metainfo.csv";
        std::string dumpProgrameFilePath = dumpPath + "verify_graph_result_brief.csv";
        std::string dumpErrorPath = dumpPath + "verify_graph_result_brief.log";
        execOpResultFile = fopen(dumpOpFilePath.c_str(), "w");
        execProgrameResultFile = fopen(dumpProgrameFilePath.c_str(), "w");
        execDumpErrorFile = fopen(dumpErrorPath.c_str(), "w");
        std::vector<std::string> OpcsvHeader = {"NO.", "PHASE_NAME", "PATH_FUNC:func_magicname", "PATH_FUNC:funcmagic", "PATH_FUNC:hash",
            "LOOP_INFO", "ROOT_FUNC:functype", "ROOT_FUNC:graphtype", "ROOT_FUNC:funcmagic", "ROOT_FUNC:hash", "FUNC:functype",
            "FUNC:graphtype", "FUNC:funcmagic", "FUNC:hash", ":rawmagic", ":rawshape", ":datatype", ":format", ":symbol", ":magic",
            ":offset", ":shape", ":validshape", "EVAL:dynvalidshape", "ROOT_CALL:opmagic", "ROOT_CALL:rawmagic", ":opmagic", ":opcode",
            "OP_ATTR_SYM_OFFSET", "OP_ATTR_ATOMIC", "OP_IO_FLAG", "TIMESTAMP", "FILENAME", "INPUT_FILENAMES", ":inputValidShape"};
        std::vector<std::string> ProgrameInfoCsvHeader = {"NO.", "A>PHASE_NAME", "B>PHASE_NAME", "PATH_FUNC:func_magicname",
            "PATH_FUNC:funcmagic", "PATH_FUNC:hash", "LOOP_INFO", "IO_FLAG", "A>:rawmagic", "B>:rawmagic", ":rawshape", ":datatype", ":format",
            ":symbol", ":shape", ":validshape", "A>TIMESTAMP", "A>FILENAME" ,"B>TIMESTAMP", "B>FILENAME", "AB>RESULT", "AB>rtol/atol",
            "AB>fail_cnt/warn_cnt/tol_cnt", "AB>total_cnt/zero_cnt/infnan_cnt", "AB>mre", "AB>mre_top8", "AB>mre_top1permil", "AB>mae", "AB>mae_top8",
            "AB>mae_top1permil", "A>max", "A>min", "A>avg", "A>aavg", "A>zero", "A>infnan", "B>max", "B>min", "B>avg", "B>aavg", "B>zero", "B>infnan"};
        WriteCsvRow(OpcsvHeader, opInfoRowNum, execOpResultFile);
        WriteCsvRow(ProgrameInfoCsvHeader, ProgrameRowNum, execProgrameResultFile);
    }

    ~FunctionInterpreter() {
        fclose(execOpResultFile);
        fclose(execProgrameResultFile);
        fclose(execDumpErrorFile);
    }

    Function *entry_;
    std::unordered_map<FunctionHash, Function *> calleeHashDict;
    std::unordered_set<int> outputSlotSet_;
    std::shared_ptr<OperationInterpreter> operationInterpreter;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotDataViewDict_;
    std::vector<std::shared_ptr<FunctionFrame>> *captureFrameList{nullptr};
    std::unordered_map<std::string, ScalarImmediateType> loopSymbolDict;
    std::unordered_map<std::pair<std::shared_ptr<LogicalTensor>, int32_t>, std::shared_ptr<LogicalTensorData>, PairHash> mixGlobalTensorDict;

    int execDumpLevel{0};

    std::string execDumpDir;
    std::string dumpPath;
    FILE *execDumpFile{nullptr};
    FILE *execOpResultFile{nullptr};
    FILE *execProgrameResultFile{nullptr};
    FILE *execDumpStyleFile{nullptr};
    FILE *execDumpErrorFile{nullptr};
    std::string execDumpFuncKey;
    std::string execDumpPassName;
    std::string execDumpFunPath;
    int pathFuncMagic;
    size_t pathFuncHash;
    std::vector<ElementDump> execDumpElementList;
    std::vector<std::shared_ptr<FunctionFrame>> execDumpStack;
    int frameCount{0};
    int opInfoRowNum{0};
    int ProgrameRowNum{0};

    std::map<std::string, uint64_t> opUsage;
    uint64_t dumpTensorUsage{0};
    uint64_t dumpOperationUsage{0};
    uint64_t totalTimeUsage{0};

    VerifyType verifyType{VerifyType::INVALID};
    int captureIndex{0};
    int passIndex{-1};

    std::vector<std::shared_ptr<LogicalTensorData>> &GetInputDataViewList() {
        return operationInterpreter->evaluateSymbol->GetInputDataViewList();
    }
    void UpdateInputDataViewList(size_t index, const std::shared_ptr<LogicalTensorData> &inputDataView) {
        operationInterpreter->evaluateSymbol->UpdateInputDataViewList(index,inputDataView);
    }
    void InitInputDataViewList(const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList) {
        operationInterpreter->evaluateSymbol->InitInputDataViewList(inputDataViewList);
    }
    void UpdateIODataPair(std::shared_ptr<FunctionIODataPair> &inoutDataPair) {
        operationInterpreter->evaluateSymbol->UpdateIODataPair(inoutDataPair);
    }
    const std::unordered_map<std::string, ScalarImmediateType> &GetSymbolDict() const {
        return operationInterpreter->evaluateSymbol->GetSymbolDict();
    }
    void UpdateSymbolDict(const std::string key, const ScalarImmediateType value) {
        operationInterpreter->evaluateSymbol->UpdateSymbolDict(key, value);
    }
    void SetSymbolDict(const std::unordered_map<std::string, ScalarImmediateType> &symbolDict) {
        operationInterpreter->evaluateSymbol->SetSymbolDict(symbolDict);
    }

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar &ss) {
        return operationInterpreter->EvaluateSymbolicScalar(ss);
    }
    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t> &offset, const std::vector<SymbolicScalar> &dynOffset,
            const std::vector<SymbolicScalar> &linearArgList = {}){
        return operationInterpreter->EvaluateOffset(offset, dynOffset, linearArgList);
    }
    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar> &dynValidShape,
            const std::vector<SymbolicScalar> &linearArgList = {}) {
        return operationInterpreter->EvaluateValidShape(dynValidShape, linearArgList);
    }
    void EvaluateDynParam(
        const std::map<std::string, DynParamInfo> &dynParamTable, const std::vector<SymbolicScalar> &linearArgList) {
        operationInterpreter->evaluateSymbol->EvaluateDynParam(dynParamTable, linearArgList);
    }

    size_t GetFrameSize() const { return execDumpStack.size(); }
    std::string GetFrameIndex(const std::shared_ptr<FunctionFrame> &frame) const {
        if (frame == nullptr) {
            return "null";
        } else {
            return std::to_string(frame->GetFrameIndex());
        }
    }
    std::shared_ptr<FunctionFrame> GetFrameCurr() const {
        if (execDumpStack.size() == 0) {
            return nullptr;
        } else {
            return execDumpStack.back();
        }
    }
    std::string GetFrameCurrIndex() const {
        return GetFrameIndex(GetFrameCurr());
    }

    LogicalTensorDataPtr FormatNZ2ND(LogicalTensorDataPtr &view) {
        auto out = LogicalTensorData::CreateEmpty(view->GetDataType(), view->GetShape(), view->GetValidShape(), view->GetData()->GetShape());
        calc::FormatNZ2ND(out, view);
        return out;
    }

    LogicalTensorDataPtr FormatND2NZ(LogicalTensorDataPtr &view) {
        auto out = LogicalTensorData::CreateEmpty(view->GetDataType(), view->GetShape(), view->GetValidShape(), view->GetData()->GetShape());
        calc::FormatND2NZ(out, view);
        return out;
    }

    void Initialize(
            Function *entry,
            const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList) {
        entry_ = entry;
        InitInputDataViewList(inputDataViewList);
    }

    Function *GetEntry() const { return entry_; }

    Function *GetCallee(const Operation *callop) {
        auto calleeHash = callop->GetCalleeHash();
        ASSERT(ControlFlowScene::INVALID_CALLEE_MAPPING, calleeHashDict.count(calleeHash));
        Function *callee = calleeHashDict.find(calleeHash)->second;
        return callee;
    }

    void UpdateHashDict(const std::unordered_map<FunctionHash, Function *> &hashDict) {
        for (auto &[hash, callee] : hashDict) {
            if (calleeHashDict.count(hash)) {
                ASSERT(ControlFlowScene::INVALID_CALLEE_MAPPING,
                       calleeHashDict.find(hash)->second == callee);
            } else {
                calleeHashDict[hash] = callee;
            }
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(FunctionFrame &frame,
        const std::shared_ptr<LogicalTensor> &tensor, DataType dtype,
        const std::shared_ptr<LogicalTensor> &inplaceTensor = nullptr) {
        std::vector<SymbolicScalar> linearArgList;
        if (frame.callopAttr != nullptr) {
            linearArgList = frame.callopAttr->GetLinearArgList();
        }
        std::vector<int64_t> offset = EvaluateOffset(tensor->GetOffset(), tensor->GetDynOffset(), linearArgList);
        auto validShape = EvaluateValidShape(tensor->GetDynValidShape(), linearArgList);
        auto rawShape = EvaluateValidShape(tensor->GetRawTensor()->GetDynRawShape());
        auto ret = frame.AllocateDataView(tensor, offset, validShape, rawShape, dtype, inplaceTensor);
        return ret;
    }
 
    std::shared_ptr<LogicalTensorData> AllocateDataView(FunctionFrame &frame,
        const std::shared_ptr<LogicalTensor> &tensor, const std::shared_ptr<LogicalTensor> &inplaceTensor = nullptr) {
        return AllocateDataView(frame, tensor, tensor->GetRawTensor()->GetDataType(), inplaceTensor);
    }

    void ExecuteOpCallLeaf(ExecuteOperationContext *ctx) {
        Function *callee = GetCallee(ctx->op);
        auto inoutDataPair =
            std::make_shared<FunctionIODataPair>(*ctx->ioperandDataViewList, *ctx->ooperandInplaceDataViewList);
        ExecuteFunctionFrame(callee, ctx->op, inoutDataPair);
    }

    int GetInplaceIndex(Operation *op, int pos) {
        struct {
            Opcode opcode;
            int oPos;
            int iPos;
        } inplaceInfo[] = {
            {Opcode::OP_INDEX_OUTCAST, 0, 2},
            {Opcode::OP_VIEW, 0, 0}
        };
        for (auto &info : inplaceInfo) {
            if (info.opcode == op->GetOpcode() && pos == info.oPos) {
                return info.iPos;
            }
        }
        if (op->HasAttribute(OpAttributeKey::inplaceIdx)) {
            ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, pos == 0);
            return op->GetIntAttribute(OpAttributeKey::inplaceIdx);
        }
        return -1;
    }

    bool IsViewInplace(const std::shared_ptr<LogicalTensor> &iOp, const std::shared_ptr<LogicalTensor> &oOp) {
        if (iOp->GetRawTensor()->GetRawMagic() == oOp->GetRawTensor()->GetRawMagic()) {
            return true;
        }
        return false;
    }

    void ExecuteInplaceOperation(FunctionFrame &frame, Operation &op, int oOperandIdx,
        const std::vector<std::shared_ptr<LogicalTensorData>> &iOpDataList,
        std::vector<std::shared_ptr<LogicalTensorData>> &oOpDataList) {
        auto oop = op.GetOOperands()[oOperandIdx];
        auto index = GetInplaceIndex(&op, oOperandIdx);
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, index != -1);
        auto iop = op.GetInputOperand(index);
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, iOpDataList[index] != nullptr);
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto opAttr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            if (opAttr == nullptr) {
                //viewType在Codegenpreproc后会走这个分支
                oOpDataList.emplace_back(AllocateDataView(frame, oop));
                return;
            }
            Offset iopOffsets = iOpDataList[index]->GetOffset();
            Offset viewOffsets = EvaluateOffset(opAttr->GetFromOffset(), opAttr->GetFromDynOffset());
            auto validShape = EvaluateValidShape(oop->GetDynValidShape(), (frame.callopAttr != nullptr) ? frame.callopAttr->GetLinearArgList() : std::vector<SymbolicScalar>{});
            auto rawShape = EvaluateValidShape(oop->GetRawTensor()->GetDynRawShape(), (frame.callopAttr != nullptr) ? frame.callopAttr->GetLinearArgList() : std::vector<SymbolicScalar>{});
            std::shared_ptr<LogicalTensorData> ret;
            if (IsViewInplace(iop, oop)) {
                ret = frame.AllocateDataView(oop, viewOffsets, validShape, rawShape, oop->GetRawTensor()->GetDataType(), iop);
            } else {
                ret = AllocateDataView(frame, oop);
            }
            oOpDataList.emplace_back(ret);
        } else {
            oOpDataList.emplace_back(AllocateDataView(frame, oop, iop));
        }
    }

    bool isConsumerAccMatmul(Operation *op) {
        for (auto cons : op->ConsumerOps()) {
            if (cons->GetOpcode() == Opcode::OP_A_MULACC_B ||
                cons->GetOpcode() == Opcode::OP_A_MULACC_BT) {
                return true;
            }
        }
        return false;
    }

    void ExecuteOperation(FunctionFrame &frame, Operation *op) {
        auto iOpDataList = frame.GetDataViewList(op->GetIOperands());
        for (size_t index = 0; index < iOpDataList.size(); index++) {
            if (iOpDataList[index] == nullptr) {
                auto iop = op->GetIOperands()[index];
                if (frame.callop != nullptr) {
                    VERIFY_LOGI("ExecuteOperation: iop %zu is null, try to find in mixGlobalTensorDict.", index);
                    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(frame.callop->GetOpAttribute());
                    iOpDataList[index] = mixGlobalTensorDict[{iop, callopAttr->wrapId}];
                    if (iOpDataList[index] != nullptr) {continue;}
                }
                ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN,
                       op->GetOpcode() == Opcode::OP_CALL);
                iOpDataList[index] = AllocateDataView(frame, iop);
            }
        }
        std::vector<std::shared_ptr<LogicalTensorData>> oOpDataList;
        for (size_t i = 0; i < op->GetOOperands().size(); i++) {
            auto oop = op->GetOOperands()[i];
            if (auto index = GetInplaceIndex(op, i); index != -1) {
                ExecuteInplaceOperation(frame, *op, i, iOpDataList, oOpDataList);
            } else {
                if (isConsumerAccMatmul(op)) {
                    auto dtype = oop->GetRawTensor()->GetDataType();
                    // mm output dtype promotion
                    if ((dtype == DataType::DT_FP16 || dtype == DataType::DT_BF16)) {
                        dtype = DataType::DT_FP32;
                    }
                    oOpDataList.push_back(AllocateDataView(frame, oop, dtype));
                } else {
                    auto ret = AllocateDataView(frame, oop);
                    if (frame.callop != nullptr && MIX_PATH_OPS.count(op->GetOpcode()) > 0) {
                        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(frame.callop->GetOpAttribute());
                        mixGlobalTensorDict[{oop, callopAttr->wrapId}] = ret;
                    }

                    oOpDataList.push_back(ret);
                }
            }
        }
        ExecuteOperationContext ctx = {&frame, {}, op, &iOpDataList, {}, &oOpDataList};

        if (op->GetOpcode() == Opcode::OP_CALL) {
            ExecuteOpCallLeaf(&ctx);
        } else {
            TimeStamp ts;
            operationInterpreter->ExecuteOperation(&ctx);
            opUsage[op->GetOpcodeStr()] += ts.Duration();

            auto *ooperandDumpList =
            ctx.ooperandInplaceDataViewList ? ctx.ooperandInplaceDataViewList : ctx.ooperandDataViewList;
            DumpOperationTensor(ctx.op, ctx.frame, ooperandDumpList, ctx.ioperandDataViewList);
            dumpOperationUsage += ts.Duration();
        }
    }

    void ExecuteHandleFunctionBegin(Function *func, std::shared_ptr<FunctionFrame> frame) {
        TimeStamp ts;
        execDumpStack.push_back(frame);
        DumpFunctionHead(func);
        if (frame->inoutDataPair != nullptr) {
            for (size_t k = 0; k < func->GetIncast().size(); k++) {
                auto rawMagic = func->GetIncast()[k]->GetRawTensor()->GetRawMagic();
                std::string fileName = "tensor_Incast_" + std::to_string(rawMagic) + ".data";
                DumpTensorBinary(frame->inoutDataPair->incastDataViewList[k], fileName);
                frame->tensorDataBinDict[func->GetIncast()[k]] = fileName;
            }
        }
        dumpTensorUsage += ts.Duration();
    }
    void ExecuteHandleFunctionEnd() { execDumpStack.pop_back(); }
    void ExecuteHandleOperationBegin(Operation *op) {
        execDumpStack.back()->UpdateCurrentOperation(op);
        TimeStamp ts;
        DumpOperation(op);
        dumpOperationUsage += ts.Duration();
    }
    void ExecuteHandleOperationEnd() {}

    std::shared_ptr<FunctionFrame> ExecuteFunctionFrame(
        Function *func, Operation *callop, std::shared_ptr<FunctionIODataPair> &inoutDataPair) {
        std::shared_ptr<CallOpAttribute> callopAttr;
        std::vector<SymbolicScalar> linearArgList;
        if (callop != nullptr) {
            callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            linearArgList = callopAttr->GetLinearArgList();
        }
        std::shared_ptr<FunctionFrame> frame =
            std::make_shared<FunctionFrame>(func, callop, callopAttr, inoutDataPair, frameCount++);
        captureFrameList->push_back(frame);
        frame->funcIndex = func->GetFuncMagic();
        frame->funcHash = func->GetFunctionHash().GetHash();
        frame->funcType = func->GetFunctionTypeStr();
        frame->funcGraphType = GetGraphTypeNameDict().Find(func->GetGraphType());
        frame->passIndex = passIndex;
        if (func->HasParent()) {
            frame->rootFuncIndex = func->Parent().GetFuncMagic();
            frame->rootFuncHash = func->Parent().GetFunctionHash().GetHash();
            frame->rootFuncType = func->Parent().GetFunctionTypeStr();
            frame->rootFuncGraphType = GetGraphTypeNameDict().Find(func->Parent().GetGraphType());
        }

        UpdateIODataPair(inoutDataPair);
        auto dynParamTable = func->GetDynParamTable();
        EvaluateDynParam(dynParamTable, linearArgList);

        ExecuteHandleFunctionBegin(func, frame);
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_PRINT && verifyType != VerifyType::TENSOR_GRAPH)
                continue;
            ExecuteHandleOperationBegin(&op);
            ExecuteOperation(*frame, &op);
            ExecuteHandleOperationEnd();
        }
        ExecuteHandleFunctionEnd();

        CopyInplaceOutcastToIncast(func, frame);

        EraseTensorDataView(func, *frame);
        return frame;
    }

    void CopyInplaceOutcastToIncast(Function *func, const std::shared_ptr<FunctionFrame> &frame) {
        // After function execution, if inplace chains exist, copy data from outcast back to incast.
        if (frame == nullptr || frame->inplaceTensorSetList.empty()) {
            return;
        }

        auto &incastList = func->GetIncast();
        auto &outcastList = func->GetOutcast();

        for (const auto &tensorGroup : frame->inplaceTensorSetList) {
            if (tensorGroup.size() < 2) {
                continue;
            }

            // Find incast tensor in this group.
            LogicalTensorPtr incastTensor = nullptr;
            bool hasIncastFromFunc = false;
            for (const auto &t : tensorGroup) {
                if (std::find(incastList.begin(), incastList.end(), t) != incastList.end()) {
                    incastTensor = t;
                    hasIncastFromFunc = true;
                    break;
                }
            }
            if (incastTensor == nullptr) {
                // Fallback: use first tensor in set if no explicit incast found.
                incastTensor = *tensorGroup.begin();
            }

            // Find outcast tensor in this group.
            LogicalTensorPtr outcastTensor = nullptr;
            bool hasOutcastFromFunc = false;
            for (const auto &t : tensorGroup) {
                if (std::find(outcastList.begin(), outcastList.end(), t) != outcastList.end()) {
                    outcastTensor = t;
                    hasOutcastFromFunc = true;
                    break;
                }
            }
            // Fallback: use last tensor in set if no explicit outcast found.
            if (outcastTensor == nullptr) {
                outcastTensor = *tensorGroup.rbegin();
            }

            // If this group has neither incast nor outcast belonging to current function IO,
            // it indicates that inplaceTensorSetList is inconsistent with function IO spec.
            ASSERT(ControlFlowScene::FUNC_INPLACE_GROUP_NO_FUNC_IO,
                   hasIncastFromFunc || hasOutcastFromFunc);

            auto incastView = frame->GetDataView(incastTensor);
            auto outcastView = frame->GetDataView(outcastTensor);
            if (incastView == nullptr || outcastView == nullptr) {
                continue;
            }

            auto incData = incastView->GetData();
            auto outData = outcastView->GetData();
            if (incData == nullptr || outData == nullptr || incData.get() == outData.get()) {
                continue;
            }

            ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
                   incData->GetDataType() == outData->GetDataType());
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE,
                   incData->GetDataSize() == outData->GetDataSize());
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE,
                   incData->size() == outData->size());

            std::copy(outData->data(), outData->data() + outData->size(), incData->data());
        }
    }

    void EraseTensorDataView(Function *func, FunctionFrame &frame) {
        for (auto it = frame.tensorDataViewDict.begin(); it != frame.tensorDataViewDict.end();) {
            auto incast = std::find(func->GetIncast().begin(), func->GetIncast().end(), it->first);
            if (incast != func->GetIncast().end()) {
                it++;
                continue;
            }
            auto outcast = std::find(func->GetOutcast().begin(), func->GetOutcast().end(), it->first);
            if (outcast != func->GetOutcast().end()) {
                it++;
                continue;
            }
            it = frame.tensorDataViewDict.erase(it);
        }
    }

    std::vector<std::shared_ptr<FunctionFrame>> ExecuteFunctionCapture(Function *func, Operation *callop, std::shared_ptr<FunctionIODataPair> &inoutDataPair) {
        std::vector<std::shared_ptr<FunctionFrame>> frameList;
        captureFrameList = &frameList;
        ExecuteFunctionFrame(func, callop, inoutDataPair);
        return frameList;
    }

    void ExecuteFunctionDynamic(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr, frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        std::vector<Operation *> callopList = func->GetCallopList();
        for (auto callop : callopList) {
            Function *callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    Operation *ExecuteFunctionLoopLookupSat(const std::shared_ptr<DynloopFunctionAttribute> &controlFlowExecution) {
        for (auto &path : controlFlowExecution->pathList) {
            bool sat = true;
            for (auto cond : path.pathCondList) {
                if (static_cast<bool>(EvaluateSymbolicScalar(cond.GetCond())) != cond.IsSat()) {
                    sat = false;
                    break;
                }
            }
            if (!sat) {
                continue;
            }
            return path.callop;
        }
        return nullptr;
    }

    void ExecuteFunctionLoop(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr, frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        auto loop = func->GetDynloopAttribute();
        ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
        ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
        ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
        if (begin == end) {
            VERIFY_EVENT("Function %s skip execute due to idx range = 0", func->GetMagicName().c_str());
        }
        for (ScalarImmediateType idx = begin; idx < end; idx += step) {
            UpdateSymbolDict(loop->IterSymbolName(), idx);
            loopSymbolDict[loop->IterSymbolName()] = idx;
            Operation *callop = ExecuteFunctionLoopLookupSat(loop);
            if (callop == nullptr) {
                continue;
            }
            Function *callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    void ExecuteControlFlow(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        if (func->GetFunctionType() != FunctionType::DYNAMIC && func->GetFunctionType() != FunctionType::DYNAMIC_LOOP_PATH) {
            func->SortOperations();
        }
        auto funcType = func->GetFunctionType();
        if (funcType == FunctionType::DYNAMIC) {
            ExecuteFunctionDynamic(func, controlFlowExecution);
        } else if (funcType == FunctionType::DYNAMIC_LOOP) {
            ExecuteFunctionLoop(func, controlFlowExecution);
        } else if (func->IsGraphType(GraphType::TENSOR_GRAPH)) {
            std::vector<Operation *> callopList = func->GetCallopList();
            if (callopList.size() != 0) {
                ExecuteFunctionDynamic(func, controlFlowExecution);
            } else {
                execDumpFunPath = func->GetMagicName();
                pathFuncMagic = func->GetFuncMagic();
                pathFuncHash = func->GetFunctionHash().GetHash();
                auto &incastSlot = func->GetSlotScope()->ioslot.incastSlot;
                auto &outcastSlot = func->GetSlotScope()->ioslot.outcastSlot;
                auto &partialSlot = func->GetSlotScope()->ioslot.partialUpdateOutcastList;

                auto getOutputSlot = [this](const std::vector<int> &slotList) {
                    for (auto &slot : slotList) {
                        if (this->outputSlotSet_.count(slot)) {
                            return slot;
                        }
                    }
                    return -1;
                };

                auto inoutDataPair = std::make_shared<FunctionIODataPair>();

                ASSERT(ControlFlowScene::FUNC_SLOT_IO_COUNT_MISMATCH,
                       func->GetIncast().size() == incastSlot.size());
                for (size_t i = 0; i < func->GetIncast().size(); i++) {
                    int slot = incastSlot[i][0];
                    ASSERT(ControlFlowScene::FUNC_SLOT_MISSING,
                           slotDataViewDict_.count(slot));
                    auto incastDataView = slotDataViewDict_[slot];
                    inoutDataPair->incastDataViewList.push_back(incastDataView);
                }

                ASSERT(ControlFlowScene::FUNC_SLOT_IO_COUNT_MISMATCH,
                       func->GetOutcast().size() == outcastSlot.size());
                for (size_t i = 0; i < func->GetOutcast().size(); i++) {
                    int outputSlot = getOutputSlot(outcastSlot[i]);
                    bool isPartialSlot = std::find(partialSlot.begin(), partialSlot.end(), i) != partialSlot.end();
                    std::shared_ptr<LogicalTensorData> outcastView;
                    if (outputSlot != -1) {
                        outcastView = slotDataViewDict_[outputSlot];
                    } else if (isPartialSlot && slotDataViewDict_[outcastSlot[i][0]]) {
                        outcastView = slotDataViewDict_[outcastSlot[i][0]];
                    } else {
                        auto outcast = func->GetOutcast()[i];
                        auto validShape = EvaluateValidShape(outcast->GetDynValidShape());
                        auto rawShape = EvaluateValidShape(outcast->GetRawTensor()->GetDynRawShape());
                        outcastView =
                            LogicalTensorData::CreateEmpty(outcast->Datatype(), outcast->GetShape(), validShape, rawShape);
                    }
                    for (auto &s : outcastSlot[i]) {
                        slotDataViewDict_[s] = outcastView;
                    }
                    inoutDataPair->outcastDataViewList.push_back(outcastView);
                }

                auto capture = std::make_shared<FunctionCaptureExecution>(func);

                capture->CaptureFrom(inoutDataPair, GetSymbolDict());
                capture->loopSymbolDict = loopSymbolDict;
                capture->frameList = ExecuteFunctionCapture(func, nullptr, inoutDataPair);
                capture->CaptureGoldenFrom(inoutDataPair);
                controlFlowExecution.executionListDict[func].emplace_back(capture);
            }
        } else {
            ASSERT(ControlFlowScene::FUNC_UNKNOWN_IO_TYPE, false);
        }
    }

    std::string DumpDataView(const std::shared_ptr<LogicalTensorData> &dataView);
    std::string DumpSymbolDict() const {
        std::ostringstream oss;
        for (auto &[name, value] : GetSymbolDict()) {
            oss << name << " = " << value << "\n";
        }
        return oss.str();
    }
    void DumpOperation(Operation *op);
    void DumpOperationTensor(
            Operation *op,
            FunctionFrame *frame,
            const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
        const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList);

private:
    void FillOperationBasicInfo(Operation *op, FunctionFrame *frame, std::vector<std::string> &opInfo);
    void FillOperationOffsetInfo(Operation *op, FunctionFrame *frame, 
                                  const std::vector<SymbolicScalar> &linearArgList,
                                  std::vector<std::string> &opInfo);
    void FillOperationInputInfo(Operation *op, FunctionFrame *frame,
                                const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList,
                                std::vector<std::string> &opInfo);
    void FillOperationOutputInfo(Operation *op, FunctionFrame *frame,
                                 const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
                                 const std::vector<SymbolicScalar> &linearArgList,
                                 int indent, std::vector<std::string> &opInfo);

public:
    void DumpTensorBinary(
            const std::shared_ptr<LogicalTensor> &tensor,
            const std::shared_ptr<LogicalTensorData> &dataView);
    void DumpTensorBinary(
            const std::shared_ptr<LogicalTensorData> &dataView,
            std::string dumpTensorFileName, bool isRaw = false);
    void DumpBinary(std::vector<int64_t> &shape, std::vector<int64_t> &stride, std::vector<int64_t> &offset, 
            FILE *fdata, uint8_t *data, size_t dtypeSize);
    void DumpTensorList(
            const std::string &name,
            const std::vector<std::shared_ptr<LogicalTensor>> *tensorList,
        const std::vector<std::shared_ptr<LogicalTensorData>> *dataViewList);
    std::shared_ptr<LogicalTensorData> LoadTensorBinary(
            const std::shared_ptr<LogicalTensor> &tensor,
            const std::string dumpTensorFileName);
    void DumpFunctionHead(Function *func);
    void DumpBegin();
    void DumpEnd();
    void DumpPassTensorDiff(
            const std::shared_ptr<FunctionCaptureExecution> &captureExecution,
        const std::shared_ptr<FunctionCaptureExecution> &captureGolden);
    std::string GetDumpFilePath(const std::string &lv0, const std::string &lv1, const std::string &filename);

    std::string GetDumpFrameDirName() const {
        std::string dirName = "frame_" + GetFrameCurrIndex();
        return dirName;
    }
    std::string GetDumpTensorListFileName(const std::string &name) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_" + name + ".html";
        return fileName;
    }
    std::string GetDumpOperationTensorFileName(Operation *op) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_operation_" + std::to_string(op->GetOpMagic()) + ".html";
        return fileName;
    }
    std::string GetDumpTensorFileName(const std::shared_ptr<LogicalTensor> &tensor) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_tensor_" + std::to_string(tensor->GetMagic()) + ".data";
        return fileName;
    }
     std::string GetDumpTensorFileName(const std::shared_ptr<LogicalTensor> &tensor, Operation *op, FunctionFrame *frame) const {
        auto callopMagic = (frame->callop != nullptr) ? std::to_string(frame->callop->GetOpMagic()) + "~" : "~";
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        auto ts =  tv.tv_sec * 1000000 + tv.tv_usec; // 1000000 is us per sec
 
        std::string fileName = std::to_string(frame->rootFuncIndex) + "~" + callopMagic  + GetLoopSymbolString(false) + "~" + std::to_string(frame->funcIndex) + "~" 
                        + std::to_string(op->GetOpMagic()) + "~" + op->GetOpcodeStr() + "~" + std::to_string(tensor->GetRawTensor()->GetRawMagic()) + "~" + 
                        std::to_string(tensor->GetMagic()) + "~" + std::to_string(ts) + ".data";
        return fileName;
    }
    std::string GetLoopSymbolString(bool withName=true) const {
        std::ostringstream loop;
        size_t loopCount = loopSymbolDict.size();
        size_t count = 0;
        for (auto &[name, value] : loopSymbolDict) {
            if (withName) {
                loop << name << "=" << value;
            } else {
                loop << value;
            }
            if(++count < loopCount) {
                loop << "@";
            } 
        }
        return loop.str();
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame> &frame, const std::shared_ptr<LogicalTensor> &tensor) const {
        std::string index = GetFrameIndex(frame);
        std::string magic = tensor != nullptr ? std::to_string(tensor->GetMagic()) : "null";
        std::string tensorId = "tensor_" + index + "_" + magic;
        return tensorId;
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame> &frame, Operation *op) const {
        std::shared_ptr<LogicalTensor> tensor = op->GetOOperands().size() != 0 ? op->GetOOperands()[0] : nullptr;
        return GetDumpTensorId(frame, tensor);
    }
    std::string GetDumpOperationId(const std::shared_ptr<FunctionFrame> &frame, Operation *op) const {
        std::string index = GetFrameIndex(frame);
        std::string magic = op != nullptr ? std::to_string(op->GetOpMagic()) : "null";
        std::string tensorId = "operation_" + index + "_" + magic;
        return tensorId;
    }

    void DumpSetLevelOperation() { execDumpLevel = EXEC_DUMP_LEVEL_OPERATION; }
    void DumpSetLevelTensor() { execDumpLevel = EXEC_DUMP_LEVEL_TENSOR; }

    void DumpReset() {
        execDumpLevel = 0;
        opUsage.clear();
        totalTimeUsage = 0;
        dumpTensorUsage = 0;
        dumpOperationUsage = 0;
    }

    static std::string ShapeToString(const std::vector<int64_t> &shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                oss << ",";
            }
            oss << std::to_string(shape[i]);
        }
        oss << "]";
        return oss.str();
    }
 
    void WriteCsvRow(std::vector<std::string>& row, int& rowNum, FILE* file) {
        if (rowNum > 0) {
            row[0] = std::to_string(rowNum);
        }
        rowNum += 1;
        std::string textLine = row[0];
        for (size_t i = 1; i < row.size(); ++i) {
            if (row[i].find(',') != std::string::npos) {
                textLine += ",\"" + row[i] + "\"";
            } else {
                textLine += "," + row[i];
            }
        }
        fprintf(file, "%s\n", textLine.c_str());
    }

    std::string DumpStatistics() const {
        std::stringstream ss;
        const int labelWidth = 24;
        uint64_t totalOpUsage = 0;
        for (auto &[opcode, time] : opUsage) {
            if (time) {
                totalOpUsage += time;
                ss << std::left << std::setw(labelWidth) << opcode << ": " << time << "\n";
            }
        }
        ss << std::left << std::setw(labelWidth) << "TotalTimeUsage" << ": " << totalTimeUsage << "\n";
        ss << std::left << std::setw(labelWidth) << "TotalOpUsage" << ": " << totalOpUsage << "\n";
        if (dumpTensorUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpTensorUsage:" << ": " << dumpTensorUsage << "\n";
        }
        if (dumpOperationUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpOperationUsage" << ": " << dumpOperationUsage << "\n";
        }
        return ss.str();
    }

    std::shared_ptr<FunctionCaptureExecution> ExecuteUnit(
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        auto unitCapture = std::make_shared<FunctionCaptureExecution>(func);
        auto symbolDict = capture->CaptureTo(unitCapture->baseline);
        SetSymbolDict(symbolDict);
        loopSymbolDict = capture->loopSymbolDict;

        Function *target = func;
        if (func->GetRootFunction()) {
            target = func->GetRootFunction();
        }
        unitCapture->CaptureGoldenFrom(unitCapture->baseline);
        unitCapture->CaptureSymbolDictFrom(capture->symbolDict);
        unitCapture->frameList = ExecuteFunctionCapture(target, nullptr, unitCapture->golden);

        DumpTensorList("Golden", &target->GetOutcast(), &capture->golden->outcastDataViewList);
        return unitCapture;
    }

    std::shared_ptr<FunctionControlFlowExecution> RunForControlFlow(
            const std::string &funcKey,
        const std::unordered_map<int, TileOpFormat> &slotTileOpFormatDict,
        const std::unordered_map<int, std::shared_ptr<LogicalTensorData>> &slotDataViewDict,
        const std::unordered_set<int> &outputSlotSet,
        const std::unordered_map<std::string, ScalarImmediateType> &controlFlowSymbolDict) {
        execDumpFuncKey = funcKey;
        std::shared_ptr<FunctionControlFlowExecution> execution = std::make_shared<FunctionControlFlowExecution>();

        SetSymbolDict(controlFlowSymbolDict);
        auto findInputIndex = [this](std::shared_ptr<LogicalTensorData> &inputDataView) -> int {
            for (size_t k = 0; k < this->GetInputDataViewList().size(); k++) {
                if (this->GetInputDataViewList()[k] == inputDataView) {
                    return k;
                }
            }
            return -1;
        };
        slotDataViewDict_ = slotDataViewDict;
        outputSlotSet_ = outputSlotSet;
        for (auto &[slot, tileOpFormat]: slotTileOpFormatDict) {
            if (tileOpFormat == TileOpFormat::TILEOP_NZ) {
                ASSERT(ControlFlowScene::FUNC_SLOT_MISSING,
                       slotDataViewDict_.count(slot));
                auto dataView = slotDataViewDict_[slot];
                auto inputIndex = findInputIndex(dataView);
                auto nzInputDataView = FormatNZ2ND(dataView);
                slotDataViewDict_[slot] = nzInputDataView;
                UpdateInputDataViewList(inputIndex, nzInputDataView);
            }
        }

        DumpBegin();
        TimeStamp ts;
        ExecuteControlFlow(entry_, *execution);
        for (auto &slot : outputSlotSet_) {
            if (slotTileOpFormatDict.count(slot) && slotTileOpFormatDict.at(slot) == TileOpFormat::TILEOP_NZ) {
                auto dataView = slotDataViewDict.find(slot)->second;
                slotDataViewDict_[slot] = FormatND2NZ(dataView);
            }
        }

        totalTimeUsage += ts.Duration();
        DumpEnd();
        return execution;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForPass(
            std::string &funcKey,
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        mixGlobalTensorDict.clear();
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        DumpEnd();
        TimeStamp ts1;
        DumpPassTensorDiff(unitCapture, capture);
        dumpTensorUsage += ts1.Duration();
        totalTimeUsage += ts.Duration();
        return unitCapture;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForExecuteGraph(
            const std::string &funcKey,
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        totalTimeUsage += ts.Duration();
        DumpEnd();
        return unitCapture;
    }
};

} // namespace npu::tile_fwk
