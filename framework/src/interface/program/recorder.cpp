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
 * \file recorder.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/utils/log.h"
#include "passes/pass_mgr/pass_manager.h"

namespace npu::tile_fwk {
const std::string PROGRAM_ENTRY_FUNCTION_NAME = "PROGRAM_ENTRY";

void static MergeAllFuncDupIocast(Function* func) {
    if (func == nullptr) {
        auto rootFunc = Program::GetInstance().GetFunctionByMagicName(PROGRAM_ENTRY_FUNCTION_NAME);
        if (rootFunc != nullptr) {
            auto calleeLists = rootFunc->GetCalleeFunctionList();
            for (auto callee : calleeLists) {
                MergeAllFuncDupIocast(callee);
            }
        }
        return;
    }

    FUNCTION_LOGI("Merge Duplicated Iocast for function: %s", func->GetMagicName().c_str());
    auto calleeLists = func->GetCalleeFunctionList();
    // leaf function has no duplicated tensor
    if (calleeLists.size() == 0) {
        return;
    }

    // 1.merge duplicated incast/outcast
    func->MergeFunctionDupIocast();
    // 2. remove useless view assemble op
    func->RemoveCallOpViewAssemble();
    if (config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false) &&
        func->IsGraphType(GraphType::TENSOR_GRAPH)) {
        func->DumpJsonFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + "_remove_dup.json");
        func->DumpFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + "_remove_dup.tifwkgr");
    }

    for (auto callee : calleeLists) {
        MergeAllFuncDupIocast(callee);
    }
}

void RecordFunc::RecordDynFuncInner(const std::vector<std::reference_wrapper<const Tensor>> &startArgsInputTensorList,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsOutputTensorList,
    const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>> &inplaceArgs) {
        CHECK(config::GetFunctionType() == FunctionType::DYNAMIC)
            << "Function graph type: " << GetFunctionTypeNameDict().Find(config::GetFunctionType());

#if ENABLE_HIDDENLOOP
        recordLoopFunc_ = std::make_unique<RecordLoopFunc>(
            funcName + "_loop",
            FunctionType::DYNAMIC_LOOP,
            funcName +"_unused_hidden_record_func_loop_idx",
            LoopRange(1)
        );
#endif

        Program::GetInstance().BeginFunction(funcName, config::GetFunctionType());

        std::shared_ptr<TensorSlotManager> manager = Program::GetInstance().GetTensorSlotManager();
        for (auto &param : startArgsInputTensorList) {
            manager->MarkInput(param.get());
        }
        for (auto &param : startArgsOutputTensorList) {
            manager->MarkOutput(param.get());
        }
        for (auto &param : inplaceArgs) {
            manager->MarkInplace(param.first.get(), param.second.get());
        }

        dynFunc_ = Program::GetInstance().GetCurrentFunction();
        dynFunc_->SetUnderDynamicFunction(true);
        dynFunc_->SetSourceLocation(SourceLocation::GetLocation());

        std::shared_ptr<DyndevFunctionAttribute> attr = std::make_shared<DyndevFunctionAttribute>();
        attr->startArgsInputTensorList = startArgsInputTensorList;
        attr->startArgsOutputTensorList = startArgsOutputTensorList;

        attr->startArgsInputLogicalTensorList.resize(startArgsInputTensorList.size());
        attr->startArgsOutputLogicalTensorList.resize(startArgsOutputTensorList.size());
        for (size_t k = 0; k < startArgsInputTensorList.size(); k++) {
            attr->startArgsInputLogicalTensorList[k] = startArgsInputTensorList[k].get().GetStorage(false);
        }
        for (size_t k = 0; k < startArgsOutputTensorList.size(); k++) {
            attr->startArgsOutputLogicalTensorList[k] = startArgsOutputTensorList[k].get().GetStorage(false);
        }

        dynFunc_->SetDyndevAttribute(attr);
        Program::GetInstance().SetCurrentDynamicFunction(dynFunc_);
}

RecordFunc::RecordFunc(const std::string &name) : funcName(FUNCTION_PREFIX + name) {
    ConfigManager::Instance().ResetLog();
    Program::GetInstance().BeginFunction(funcName, config::GetFunctionType());
}

RecordFunc::RecordFunc(const std::string &name,
    const std::vector<std::reference_wrapper<const Tensor>> &explicitOpArgs)
    : funcName(FUNCTION_PREFIX + name) {
    // RecordFunc start with TENSOR_GRAPH
    ConfigManager::Instance().ResetLog();
    if (config::GetFunctionType() == FunctionType::DYNAMIC) {
        RecordDynFuncInner(explicitOpArgs, {}, {});
    } else {
        Program::GetInstance().BeginFunction(funcName, config::GetFunctionType(), GraphType::TENSOR_GRAPH, explicitOpArgs);
    }
}

RecordFunc::RecordFunc(const std::string &name,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsInputTensorList,
    const std::vector<std::reference_wrapper<const Tensor>> &startArgsOutputTensorList,
    const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>> &inplaceArgs)
    : funcName(FUNCTION_PREFIX + name) {
    ConfigManager::Instance().ResetLog();
    RecordDynFuncInner(startArgsInputTensorList, startArgsOutputTensorList, inplaceArgs);
}

inline bool IsVerifyEnable() {
    return config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY);
}

void RecordFunc::EndFunction() {
    if (recordLoopFunc_) {
        recordLoopFunc_.reset();
    }

    if (IsVerifyEnable()) {
        FUNCTION_LOGI("FlowVerify has been enable.");
        config::SetRunDataOption(KEY_FLOW_VERIFY_PATH, config::GetAbsoluteTopFolder() + "/verify");
    }

    // might raise exception in EndFunction, force isEnd_ is always set
    Defer clean([this](){ isEnd_ = true;});
    (void)Program::GetInstance().EndFunction(funcName);
    if (dynFunc_) {
        Program::GetInstance().SetLastFunction(dynFunc_);
        if (dynFunc_->IsDyndev()) {
            Program::GetInstance().ClearEmptyHiddenFunction();
            dynFunc_->CleanRedundantOutCast();
            // Destructor GetTensorData small Tensor
            auto attr = dynFunc_->GetDyndevAttribute();
            attr->getTensorDataDescDict.clear();

            dynFunc_->ApplyLoopCallOrderGroup();
            if (config::GetVerifyOption<bool>(KEY_ENABLE_PASS_VERIFY)) {
                Program::GetInstance().VerifyTensorGraph();
            }
            MergeAllFuncDupIocast(nullptr);
            PassManager::Instance().RunPass(Program::GetInstance(),
                *Program::GetInstance().GetFunctionByMagicName(PROGRAM_ENTRY_FUNCTION_NAME), "FunctionUnroll");
            Program::GetInstance().UpdateCompileTask();
        }
        Program::GetInstance().SetCurrentDynamicFunction(nullptr);
        dynFunc_->SetUnderDynamicFunction(false);
    }
}

RecordFunc::Iterator RecordFunc::begin() {
    if (recordLoopFunc_) {
        return Iterator(*this, recordLoopFunc_->begin());
    }
    return Iterator(*this);
}

RecordFunc::IteratorEnd RecordFunc::end() {
    if (recordLoopFunc_) {
        return IteratorEnd(*this, recordLoopFunc_->end());
    }
    return IteratorEnd(*this);
}

RecordFunc::Iterator RecordFunc::Iterator::operator++() {
    if (!wrappedIter_.has_value()) {
        cur_ = 1;
        return *this;
    }
    ++(*wrappedIter_);
    return *this;
}

bool RecordFunc::Iterator::operator!=(const IteratorEnd &rhs) {
    if (!wrappedIter_.has_value()) {
        return cur_ != 1;
    }
    ASSERT(rhs.wrappedEnd.has_value()) << "Input param rhs has no value";
    bool result = *wrappedIter_ != *rhs.wrappedEnd;
    return result;
}

RecordLoopFunc::RecordLoopFunc(const std::string &name, FunctionType funcType, const std::string &iterName,
    const LoopRange &range, const std::set<int> &unrollList, bool submitBeforeLoop)
    : name_(FUNCTION_PREFIX + name),
        iterName_(iterName),
        loopRange_(std::make_shared<LoopRange>(range)),
        submitBeforeLoop_(submitBeforeLoop),
        funcType_(funcType) {
    CHECK(funcType == FunctionType::STATIC || funcType == FunctionType::DYNAMIC_LOOP)
        << "funcType: " << GetFunctionTypeNameDict().Find(funcType);
    Program::GetInstance().GetLoopStack().emplace_back(*this);

    GenDefaultUnrollTimes(unrollList);
    location_ = SourceLocation::GetLocation();
}

RecordLoopFunc::~RecordLoopFunc() {
    Program::GetInstance().GetLoopStack().pop_back();
}

bool RecordLoopFunc::IterationEnd() {
    auto result = Program::GetInstance().EndFunction(curPathFuncName_);
    auto pathFunc = std::get<0>(result);
    pathFunc->ApplyLoopCallOrderGroup();
    Program::GetInstance().GetTensorSlotManager()->Restore();
    auto isEnd = GetLoopAttr()->IterationEnd(CurUnrollTimes(), pathFunc, std::get<1>(result));
    if (isEnd) {
        endCount_ = 0;
    }
    return isEnd;
}

void RecordLoopFunc::BeginLoopFunction() {
    auto loopFuncName = name_ + "_Unroll" + std::to_string(CurUnrollTimes());
    Program::GetInstance().BeginFunction(loopFuncName, FunctionType::DYNAMIC_LOOP);
    currentLoopFunc_ = Program::GetInstance().GetCurrentFunction();
    CHECK(currentLoopFunc_->InsertLoopIdxNameList(iterName_)) << "Forbid duplicate name of loop idx. It names " << iterName_;
    auto currentStep = CurUnrollTimes() == 1 ? loopRange_->Step() : loopRange_->Step() * CurUnrollTimes();
    if (rangeOfEaceUnroll_.empty()) {
        auto newRangeEnd = (UnrollTimesSize() == 1 ? loopRange_->End() : loopRange_->End() / currentStep * currentStep);
        std::shared_ptr<LoopRange> newRange = std::make_shared<LoopRange>(loopRange_->Begin(), newRangeEnd, currentStep);
        rangeOfEaceUnroll_.push_back(newRange);
    } else {
        auto prevRange = rangeOfEaceUnroll_.back();
        auto newRangeEnd = (UnrollTimesSize() == 1 ? loopRange_->End() : prevRange->End() + (loopRange_->End() - prevRange->End()) / currentStep * currentStep);
        std::shared_ptr<LoopRange> newRange = std::make_shared<LoopRange>(prevRange->End(), newRangeEnd, currentStep);
        rangeOfEaceUnroll_.push_back(newRange);
    }
    auto range = rangeOfEaceUnroll_.back();
    range->End().AsIntermediateVariable();
    auto attr = std::make_shared<DynloopFunctionAttribute>(iterName_, *range, *loopRange_, submitBeforeLoop_);
    currentLoopFunc_->SetDynloopAttribute(attr);
    currentLoopFunc_->SetSourceLocation(location_);
}

void RecordLoopFunc::EndLoopFunction() {
    auto loopFuncName = name_ + "_Unroll" + std::to_string(CurUnrollTimes());
    Program::GetInstance().EndFunction(loopFuncName);
    currentLoopFunc_ = nullptr;
}

bool RecordLoopFunc::MatchUnrollTimes(int unrollTimes) {
    CHECK(unrollTimes > 0) << "unrollTimes[" << unrollTimes << "] must larger than zero!";
    auto &curRlf = Program::GetInstance().GetLoopStack().back().get();
    curRlf.customUnrollTimes_.emplace(unrollTimes);
    if (!curRlf.hasManualUnroll_) {
        curRlf.hasManualUnroll_ = true;
        curRlf.dryRun_ = true;
    }
    if (curRlf.dryRun_) {
        return false;
    }
    if (!curRlf.VisitedUnroll(unrollTimes)) {
        curRlf.VisitUnroll(unrollTimes);
    }
    ASSERT(curRlf.StillHaveUnrollTimes()) << "unrollTimes_ is empty.";
    if (curRlf.CurUnrollTimes() == unrollTimes) {
        return true;
    }

    // if cur unrollTimes is not user defined, use unrollTimes = 1 to auto merge loop
    if (curRlf.CurUnrollTimes() > 1 && !curRlf.CustomUnrollTimesMatched() && unrollTimes == 1) {
        return true;
    }
    return false;
}

RecordLoopFunc::Iterator RecordLoopFunc::Iterator::operator++() {
    if (rlf_.dryRun_) {
        return *this;
    }
    if (!rlf_.CustomUnrollTimesMatched()) {
        scalar_ = scalar_ + rlf_.LoopStep();
        cur_++;
    } else {
        ASSERT(cur_ == 0) << "The cur_ = " << cur_;
        scalar_ = scalar_ + rlf_.LoopStep() * rlf_.CurUnrollTimes();
        cur_ += rlf_.CurUnrollTimes();
    }

    rlf_.IterationNext();
    return *this;
}

bool RecordLoopFunc::Iterator::operator!=(const IteratorEnd &rhs) {
    (void)rhs;
    if (rlf_.dryRun_) {
        rlf_.dryRun_ = false;
        ASSERT(cur_ == 0) << "The cur_ = " << cur_;
        if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes())) {
            scalar_.AsLoopEnd(true);
        }
        return true;
    }
    ASSERT(rlf_.StillHaveUnrollTimes()) << "unrollTimes_ is empty.";
    if (cur_ < rlf_.CurUnrollTimes()) {
        if (cur_ == 0) {
            scalar_.AsLoopBegin(true);
            rlf_.IterationBegin();
            if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes())) {
                scalar_.AsLoopEnd(true);
            }
        }
        if (cur_ + 1 == rlf_.CurUnrollTimes()) {
            scalar_.AsLoopEnd(true);
        }
        return true;
    }
    ASSERT(cur_ == rlf_.CurUnrollTimes())
        <<  " cur_ = " << cur_ << ", rlf_.CurUnrollTimes() = " << rlf_.CurUnrollTimes();
    if (rlf_.IterationEnd()) {
        rlf_.EndLoopFunction();
        rlf_.NextUnrollTimes();
        if (!rlf_.StillHaveUnrollTimes()) {
            return false;
        }
    }
    ASSERT(rlf_.StillHaveUnrollTimes()) << "unrollTimes_ is empty.";
    cur_ = 0;
    scalar_ = originalScalar_;
    scalar_.AsLoopBegin(true);
    rlf_.IterationBegin();
    if (rlf_.IsCustomUnrollTimes(rlf_.CurUnrollTimes()) || cur_ + 1 == rlf_.CurUnrollTimes()) {
        scalar_.AsLoopEnd(true);
    }
    return true;
}

RecordLoopFunc::Iterator RecordLoopFunc::begin() {
    return {*this, SymbolicScalar(iterName_)};
}

RecordLoopFunc::IteratorEnd RecordLoopFunc::end() {
    return {*this, SymbolicScalar(iterName_)};
}

void RecordLoopFunc::IterationBegin() {
    if (currentLoopFunc_ == nullptr) {
        BeginLoopFunction();
    }
    curPathFuncName_ = name_ + "_Unroll" + std::to_string(CurUnrollTimes()) + GetLoopSuffix(endCount_++);
    Program::GetInstance().GetTensorSlotManager()->Checkpoint();
    Program::GetInstance().BeginFunction(curPathFuncName_, FunctionType::DYNAMIC_LOOP_PATH);
    auto loopPathFunc = Program::GetInstance().GetCurrentFunction();
    loopPathFunc->SetSourceLocation(location_);
    GetLoopAttr()->IterationBegin();
}

void RecordLoopFunc::IterationNext() {
    ASSERT(customUnrollTimes_.empty() || customUnrollTimes_.count(1) > 0) << "Must have unroll 1 if user defined custom unroll times.";
}

bool RecordLoopFunc::Condition(const SymbolicScalar &cond, const std::string &file, int line) {
    bool result = GetLoopAttr()->AppendCond(cond, file, line);
    FUNCTION_LOGI("[%s:%d]: %s", file.c_str(), line, result ? "true" : "false");
    return result;
}

void RecordLoopFunc::GenDefaultUnrollTimes(const std::set<int> &unrollList) {
    unrollTimes_.clear();
    visited_.clear();
    if (config::GetPlatformConfig("ONLY_MANUAL_UNROLL", false) || unrollList.empty()) {
        unrollTimes_.emplace(1);
        visited_.emplace(1);
    } else {
        for (auto n : unrollList) {
            unrollTimes_.emplace(n);
            visited_.emplace(n);
        }
    }
}

void RecordLoopFunc::VisitUnroll(int unrollTimes) {
    ASSERT(visited_.count(unrollTimes) == 0) << "unrollTimes[" << unrollTimes << "] already exists in visited.";
    ASSERT(unrollTimes_.count(unrollTimes) == 0) << "unrollTimes[" << unrollTimes << "] already exists..";
    visited_.emplace(unrollTimes);
    unrollTimes_.emplace(unrollTimes);
}

int RecordLoopFunc::CurUnrollTimes() const {
    ASSERT(StillHaveUnrollTimes()) << "unrollTimes_ is empty.";
    return *unrollTimes_.begin();
}

void RecordLoopFunc::NextUnrollTimes() {
    ASSERT(StillHaveUnrollTimes()) << "unrollTimes_ is empty.";
    unrollTimes_.erase(unrollTimes_.begin());
}

std::shared_ptr<DynloopFunctionAttribute> RecordLoopFunc::GetLoopAttr() {
    return currentLoopFunc_->GetDynloopAttribute();
}

const SymbolicScalar &RecordLoopFunc::LoopBegin() const { return loopRange_->Begin(); }
const SymbolicScalar &RecordLoopFunc::LoopStep() const { return loopRange_->Step(); }
const SymbolicScalar &RecordLoopFunc::LoopEnd() const { return loopRange_->End(); }

RecordIfBranch::operator bool() const {
    bool cond = Program::GetInstance().GetLoopStack().back().get().Condition(cond_, file_, line_);
    return cond;
}
} // namespace npu::tile_fwk