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
 * \file flow_verifier.cpp
 * \brief
 */

#include "flow_verifier.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

FlowVerifier::CompareResult FlowVerifier::VerifyResult(
        const std::shared_ptr<LogicalTensorData> &goldenDataView,
        const std::shared_ptr<LogicalTensorData> &outputDataView, float eps) {
    // tensor maybe padded during PadLocalBuffer Pass, tensor shape maybe changed, just check the valid data
    ASSERT(goldenDataView->GetValidShape() == outputDataView->GetValidShape());

    switch (goldenDataView->GetDataType()) {
        case DT_INT8: return CompareData<int8_t>(goldenDataView, outputDataView, eps);
        case DT_INT16: return CompareData<int16_t>(goldenDataView, outputDataView, eps);
        case DT_INT32: return CompareData<int32_t>(goldenDataView, outputDataView, eps);
        case DT_INT64: return CompareData<int64_t>(goldenDataView, outputDataView, eps);
        case DT_FP16: return CompareData<npu::tile_fwk::float16>(goldenDataView, outputDataView, eps);
        case DT_FP32: return CompareData<float>(goldenDataView, outputDataView, eps);
        case DT_BF16: return CompareData<npu::tile_fwk::bfloat16>(goldenDataView, outputDataView, eps);
        case DT_UINT8: return CompareData<uint8_t>(goldenDataView, outputDataView, eps);
        case DT_UINT16: return CompareData<uint16_t>(goldenDataView, outputDataView, eps);
        case DT_UINT32: return CompareData<uint32_t>(goldenDataView, outputDataView, eps);
        case DT_UINT64: return CompareData<uint64_t>(goldenDataView, outputDataView, eps);
        case DT_DOUBLE: return CompareData<double>(goldenDataView, outputDataView, eps);
        case DT_BOOL: return CompareData<uint8_t>(goldenDataView, outputDataView, eps);
        default: ASSERT(false); break;
    }
    return CompareResult();
}

bool FlowVerifier::VerifyResult(const std::string &key,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList, float eps) {
    ASSERT(goldenDataViewList.size() == outputDataViewList.size());
    for (size_t k = 0; k < goldenDataViewList.size(); k++) {
        auto &goldenView = goldenDataViewList[k];
        auto &outputView = outputDataViewList[k];
        if (goldenView == nullptr || outputView == nullptr) {
            continue;
        }
        auto result = VerifyResult(goldenView, outputView, eps);
        if (!result.Check()) {
            ALOG_ERROR(key, ":\n    Verify for ", goldenDataViewList.size(), " data view list index ", k, " result ", TTY_RED("FAILED"));
            ALOG_ERROR(key, result.Dump());
            return false;
        } else {
            ALOG_INFO(key, ": Verify for data ", k, " result ", TTY_GREEN("SUCCEED"));
        }
    }
    return true;
}

bool FlowVerifier::VerifyResult(const std::string &key,
    const std::vector<std::string> tensorNameList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &tensorDataViewList, float eps) {
    bool result = true;
    if (goldenDataViewList.size() == tensorDataViewList.size()) {
        for (size_t k = 0; k < tensorDataViewList.size(); k++) {
            if (!goldenDataViewList[k]){
                ALOG_EVENT(key, " Verify for ", goldenDataViewList.size(), " data view list index ", k, " result NO_COMPARE");
                continue;
            }
            std::string fileName = "tensor_" + tensorNameList[k] + ".data";
            functionInterpreter_->DumpTensorBinary(tensorDataViewList[k], fileName);
 
            std::vector<std::string> opInfo(toIndex(CsvCol::COL_COUNT));
            opInfo[toIndex(CsvCol::funcID)] = std::to_string(entry_->GetFuncMagic());
            opInfo[toIndex(CsvCol::verifyType)] = key;
 
            opInfo[toIndex(CsvCol::outputShape)] = functionInterpreter_->ShapeToString(tensorDataViewList[k]->GetShape());
            opInfo[toIndex(CsvCol::outputValidShape)] = functionInterpreter_->ShapeToString(tensorDataViewList[k]->GetValidShape());
            opInfo[toIndex(CsvCol::outputDtype)] = DataType2String(tensorDataViewList[k]->GetDataType()); 
            opInfo[toIndex(CsvCol::outputTensor)] = fileName;
            opInfo[toIndex(CsvCol::verifyResult)] = "PASS";
 
            auto tensorGraphResult = VerifyResult(goldenDataViewList_[k], tensorDataViewList[k], eps);
            if (!tensorGraphResult.Check()) {
                ALOG_ERROR(key, " Verify for ", goldenDataViewList.size(), " data view list index ", k, " result ", TTY_RED("FAILED"));
                opInfo[toIndex(CsvCol::verifyResult)] = "FAILED";
                result = false;
            } else {
                ALOG_EVENT(key, " Verify for ", goldenDataViewList.size(), " data view list index ", k, " result PASS");
            }
            auto res = tensorGraphResult.Dump();
            std::copy(res.begin(), res.end(), opInfo.begin() + toIndex(CsvCol::maxAbsDiff));
            functionInterpreter_->WriteCsvRow(opInfo);
        }
    } else {
        ALOG_EVENT(key, " Verify NO_COPARE");
    }
    return result;
}

void FlowVerifier::UpdateInterpreterCache() {
    auto &cache = Program::GetInstance().GetFunctionCache();
    std::unordered_map<FunctionHash, Function *> hashDict;
    cache.BuildHashDict(functionInterpreter_->GetEntry(), hashDict);
    functionInterpreter_->UpdateHashDict(hashDict);
}

void FlowVerifier::VerifyTensorGraph(Function *entry,
    const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::shared_ptr<TensorSlotManager> &slotManager) {
    entry_ = entry;
    inputDataViewList_ = inputDataViewList;
    outputDataViewList_ = outputDataViewList;
    goldenDataViewList_ = goldenDataViewList;

    ASSERT(calc::IsVerifyEnabled()) << "Verify not supported";
    auto attr = entry->GetDyndevAttribute();
    std::vector<int> inputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsInputTensorList);
    std::vector<int> outputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsOutputTensorList);

    std::unordered_map<int, TileOpFormat> slotTileOpFormatDict;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotDataViewDict;
    std::unordered_set<int> outputSlotSet;

    ASSERT(inputSlotList.size() == attr->startArgsInputTensorList.size());
    ASSERT(inputDataViewList.size() == inputSlotList.size());
    for (size_t i = 0; i < inputDataViewList.size(); i++) {
        auto inputTensor = attr->startArgsInputTensorList[i].get().GetStorage();
        if (inputTensor == nullptr) {
            continue;
        }
        auto tileop = inputTensor->Format();

        auto input = inputDataViewList[i];
        ASSERT(inputTensor->Datatype() == input->GetDataType());
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[inputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
        slotDataViewDict[inputSlotList[i]] = input;
    }
    ASSERT(outputDataViewList.size() == outputSlotList.size());
    for (size_t i = 0; i < outputDataViewList.size(); i++) {
        slotDataViewDict[outputSlotList[i]] = outputDataViewList[i];
        auto outputTensor = attr->startArgsOutputTensorList[i].get().GetStorage();
        auto tileop = outputTensor->Format();
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[outputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
    }
    if (outputDataViewList.size() == 0) {
        outputSlotSet.insert(inputSlotList.begin(), inputSlotList.end());
    } else {
        outputSlotSet.insert(outputSlotList.begin(), outputSlotList.end());
    }

    std::unordered_map<std::string, ScalarImmediateType> controlFlowSymbolDict;
    const std::vector<std::string> &inputNameList = slotManager->GetInputNameList();
    const std::vector<std::string> &outputNameList = slotManager->GetOutputNameList();
    size_t idx = 0;
    for (size_t i = 0; i < inputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(inputNameList[i])] = idx++;
    }
    for (size_t i = 0; i < outputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(outputNameList[i])] = idx++;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> inoutDataViewList = inputDataViewList_;
    inoutDataViewList.insert(inoutDataViewList.end(), outputDataViewList.begin(), outputDataViewList.end());
    functionInterpreter_ = std::make_shared<FunctionInterpreter>();
    functionInterpreter_->Initialize(entry, inoutDataViewList);
    functionInterpreter_->verifyType = VerifyType::TENSOR_GRAPH;
    UpdateInterpreterCache();

    if (config::GetVerifyOption<bool>(KEY_PASS_VERIFY_SAVE_TENSOR)) {
        functionInterpreter_->DumpSetLevelTensor();
    }

    auto tensorDir = config::LogTopFolder() + "/tensor";
    CreateMultiLevelDir(tensorDir);

    controlFlowExecution_ =
        functionInterpreter_->RunForControlFlow("tensor_graph", slotTileOpFormatDict, slotDataViewDict, outputSlotSet, controlFlowSymbolDict);

    functionInterpreter_->DumpReset();
    bool res = true;

    if (outputDataViewList.size() == 0){
        res = VerifyResult("tensor_graph", inputNameList, goldenDataViewList_, inputDataViewList_, static_cast<float>(1e-2));
    } else {
        res = VerifyResult("tensor_graph", outputNameList, goldenDataViewList_, outputDataViewList_, static_cast<float>(1e-2));
    }
    if (!res) {
        checkResult = false;
    }
}

template <typename T>
static std::string ToString(const T &val, size_t totalSize) {
    std::string data = std::to_string(val);
    if (totalSize < data.size()) {
        return data;
    } else {
        return std::string(totalSize - data.size(), '0') + data;
    }
}

void FlowVerifier::VerifyPass(Function *func, int passIndex, const std::string &passIdentifier) {
    functionInterpreter_->verifyType = VerifyType::PASS;
    UpdateInterpreterCache();
    if (controlFlowExecution_->executionListDict.count(func) == 0) {
        return;
    }

    std::vector<std::string> passFilter = config::GetVerifyOption<std::vector<std::string>>(KEY_PASS_VERIFY_FILTER);
    if (!passFilter.empty()) {
        auto it = std::find(passFilter.begin(), passFilter.end(), passIdentifier);
        if (it == passFilter.end()) {
            return;
        }
    }

    auto &captureList = controlFlowExecution_->executionListDict.find(func)->second;
    if (!lastCaptureExecution_.count(func)) {
        lastCaptureExecution_[func].resize(captureList.size());
    }

    if (config::GetVerifyOption<bool>(KEY_PASS_VERIFY_SAVE_TENSOR)) {
        functionInterpreter_->DumpSetLevelTensor();
    }
    for (size_t captureIndex = 0; captureIndex < captureList.size(); captureIndex++) {
        const std::string key = "function_" + func->GetMagicName() + ".pass_" + ToString(passIndex, 2) + "_" +
                                passIdentifier;
        ALOG_INFO(key, ": Verify");
        functionInterpreter_->captureIndex = captureIndex;

        std::shared_ptr<FunctionCaptureExecution> capture = nullptr;
        float eps = static_cast<float>(1e-3);
        capture = captureList[captureIndex];

        auto captureExecution = functionInterpreter_->RunForPass(key, func, capture);
        auto goldenDataViewList = capture->golden->outcastDataViewList;
        auto executeDataViewList = captureExecution->golden->outcastDataViewList;
        /* record it */
        lastCaptureExecution_[func][captureIndex] = captureExecution;

        for (size_t k = 0; k < executeDataViewList.size(); k++) {
            std::string fileName = "tensor_" + func->GetMagicName() + "~" + passIdentifier +
                    "~" + functionInterpreter_->GetLoopSymbolString() + "~" + "OUT_" + std::to_string(k) + ".data";
            functionInterpreter_->DumpTensorBinary(executeDataViewList[k], fileName);
            std::vector<std::string> opInfo(toIndex(CsvCol::COL_COUNT));
            opInfo[toIndex(CsvCol::funcID)] = std::to_string(func->GetFuncMagic());
            opInfo[toIndex(CsvCol::verifyType)] = key;
            opInfo[toIndex(CsvCol::loopInfo)] = functionInterpreter_->GetLoopSymbolString();

            opInfo[toIndex(CsvCol::outputShape)] = functionInterpreter_->ShapeToString(executeDataViewList[k]->GetShape());   
            opInfo[toIndex(CsvCol::outputValidShape)] = functionInterpreter_->ShapeToString(executeDataViewList[k]->GetValidShape());
            opInfo[toIndex(CsvCol::outputDtype)] = DataType2String(executeDataViewList[k]->GetDataType());
            opInfo[toIndex(CsvCol::outputTensor)] = fileName;
            opInfo[toIndex(CsvCol::verifyResult)] = "PASS";

            auto passResult = VerifyResult(goldenDataViewList[k], executeDataViewList[k], eps);
            if (!passResult.Check()) {
                ALOG_ERROR(key, ":\n    Verify for ", goldenDataViewList.size(), " data view list index ", k, " result ", TTY_RED("FAILED"));
                opInfo[toIndex(CsvCol::verifyResult)] = "FAILED";
                checkResult = false;
            } else {
                ALOG_EVENT(key, " Verify result PASS");
            }
            auto res = passResult.Dump();
            std::copy(res.begin(), res.end(), opInfo.begin() + toIndex(CsvCol::maxAbsDiff));

            functionInterpreter_->WriteCsvRow(opInfo);
        }
    }
    functionInterpreter_->DumpReset();
}

FlowVerifier &FlowVerifier::GetInstance() {
    static FlowVerifier flowVerifier;
    return flowVerifier;
}
}
