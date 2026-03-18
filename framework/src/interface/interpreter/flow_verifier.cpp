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
#include "tilefwk/pypto_fwk_log.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/verify_error.h"

namespace npu::tile_fwk {

FlowVerifier::CompareResult FlowVerifier::VerifyResult(
        const std::shared_ptr<LogicalTensorData> &goldenDataView,
        const std::shared_ptr<LogicalTensorData> &outputDataView, float rtol, float atol) {
    // tensor maybe padded during PadLocalBuffer Pass, tensor shape maybe changed, just check the valid data
    goldenDataView->UpdateValidShape(outputDataView->GetValidShape());
    ASSERT(VerifyResultScene::VERIFY_RESULT_SHAPE_DIFF,
           goldenDataView->GetValidShape() == outputDataView->GetValidShape());
    ASSERT(VerifyResultScene::VERIFY_RESULT_DTYPE_DIFF,
           goldenDataView->GetDataType() == outputDataView->GetDataType());
    switch (goldenDataView->GetDataType()) {
        case DT_INT8: return CompareData<int8_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT16: return CompareData<int16_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT32: return CompareData<int32_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT64: return CompareData<int64_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_FP16: return CompareData<npu::tile_fwk::float16, float>(goldenDataView, outputDataView, rtol, atol);
        case DT_FP32: return CompareData<float, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_BF16: return CompareData<npu::tile_fwk::bfloat16, float>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT8: return CompareData<uint8_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT16: return CompareData<uint16_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT32: return CompareData<uint32_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT64: return CompareData<uint64_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_DOUBLE: return CompareData<double, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_BOOL: return CompareData<uint8_t, double>(goldenDataView, outputDataView, rtol, atol);
        default:
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
            break;
    }
    return CompareResult();
}

bool FlowVerifier::VerifyResult(const std::string &key,
    const std::string tensorName,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &tensorDataViewList, float rtol, float atol) {
    bool result = true;
    if (goldenDataViewList.size() != tensorDataViewList.size()) {
        VERIFY_EVENT("%s Verify NO_COMPARE", key.c_str());
        return result;
    }
    for (size_t k = 0; k < tensorDataViewList.size(); k++) {
        if (!goldenDataViewList[k]){
            VERIFY_EVENT("%s Verify for %zu data view list index %zu result NO_COMPARE", key.c_str(), goldenDataViewList.size(), k);
            continue;
        }
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        auto ts = tv.tv_sec * 1000000 + tv.tv_usec;    // 1000000 is us per sec

        std::string fileName = tensorName + "~" + std::to_string(k) + "~" + std::to_string(ts) + ".data";
        functionInterpreter_->DumpTensorBinary(tensorDataViewList[k], fileName);

        std::vector<std::string> opInfo(toIndex(OpInfoCsvHeader::COL_COUNT));
        opInfo[toIndex(OpInfoCsvHeader::funcID)] = std::to_string(entry_->GetFuncMagic());
        opInfo[toIndex(OpInfoCsvHeader::verifyType)] = key;
        opInfo[toIndex(OpInfoCsvHeader::outputShape)] = functionInterpreter_->ShapeToString(tensorDataViewList[k]->GetShape());
        opInfo[toIndex(OpInfoCsvHeader::outputValidShape)] = functionInterpreter_->ShapeToString(tensorDataViewList[k]->GetValidShape());
        opInfo[toIndex(OpInfoCsvHeader::outputDtype)] = DataType2String(tensorDataViewList[k]->GetDataType());
        opInfo[toIndex(OpInfoCsvHeader::outputTensor)] = fileName;
        opInfo[toIndex(OpInfoCsvHeader::verifyResult)] = "PASS";

        auto tensorGraphResult = VerifyResult(goldenDataViewList[k], tensorDataViewList[k], rtol, atol);
        if (!tensorGraphResult.Check()) {
            VERIFY_LOGE_FULL_E(VerifyResultScene::VERIFY_RESULT_MISMATCH,
                               "%s Verify for %zu data view list index %zu result FAILED",
                               key.c_str(), goldenDataViewList.size(), k);
            opInfo[toIndex(OpInfoCsvHeader::verifyResult)] = "FAILED";
            result = false;
        } else {
            VERIFY_EVENT("%s Verify for %zu data view list index %zu result PASS", key.c_str(), goldenDataViewList.size(), k);
        }
        auto res = tensorGraphResult.Dump();
        std::copy(res.begin(), res.end(), opInfo.begin() + toIndex(OpInfoCsvHeader::maxAbsDiff));
        functionInterpreter_->WriteCsvRow(opInfo);
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

    ASSERT(VerifyEnableScene::VERIFY_NOT_ENABLE, calc::IsVerifyEnabled()) << "Verify not supported";
    auto attr = entry->GetDyndevAttribute();
    std::vector<int> inputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsInputTensorList);
    std::vector<int> outputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsOutputTensorList);

    std::unordered_map<int, TileOpFormat> slotTileOpFormatDict;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotDataViewDict;
    std::unordered_set<int> outputSlotSet;

    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC,
           inputSlotList.size() == attr->startArgsInputTensorList.size());
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC,
           inputDataViewList.size() == inputSlotList.size());
    for (size_t i = 0; i < inputDataViewList.size(); i++) {
        auto inputTensor = attr->startArgsInputTensorList[i].get().GetStorage();
        if (inputTensor == nullptr) {
            continue;
        }
        auto tileop = inputTensor->Format();

        auto input = inputDataViewList[i];
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
               inputTensor->Datatype() == input->GetDataType());
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[inputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
        slotDataViewDict[inputSlotList[i]] = input;
    }
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC,
           outputDataViewList.size() == outputSlotList.size());
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
    functionInterpreter_->execDumpPassName = "tensor_graph";
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

    std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
    float rtol = static_cast<float>(tolerance[0]);
    float atol = static_cast<float>(tolerance[1]);
    if (outputDataViewList.size() == 0){
        res = VerifyResult("tensor_graph", "tensor_graph", goldenDataViewList_, inputDataViewList_, rtol, atol);
    } else {
        res = VerifyResult("tensor_graph", "tensor_graph", goldenDataViewList_, outputDataViewList_, rtol, atol);
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
    functionInterpreter_->passIndex = passIndex;
    functionInterpreter_->execDumpPassName = "pass_" + ToString(passIndex, 2) + "_" + passIdentifier;
    functionInterpreter_->execDumpFunPath = "function_" + func->GetMagicName();
    UpdateInterpreterCache();
    if (controlFlowExecution_->executionListDict.count(func) == 0) {
        return;
    }

    std::vector<std::string> passFilter = config::GetVerifyOption<std::vector<std::string>>(KEY_PASS_VERIFY_FILTER);
    if (passFilter.empty()) {
 	         return;
 	}

 	if (std::find(passFilter.begin(), passFilter.end(), "all") == passFilter.end()) {
        auto it = std::find(passFilter.begin(), passFilter.end(), passIdentifier);
        if (it == passFilter.end()) {
            return;
        }
    }
    std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
    float rtol = static_cast<float>(tolerance[0]);
    float atol = static_cast<float>(tolerance[1]);

    auto &captureList = controlFlowExecution_->executionListDict.find(func)->second;

    if (config::GetVerifyOption<bool>(KEY_PASS_VERIFY_SAVE_TENSOR)) {
        functionInterpreter_->DumpSetLevelTensor();
    }
    for (size_t captureIndex = 0; captureIndex < captureList.size(); captureIndex++) {
        const std::string key = functionInterpreter_->execDumpFunPath + "_" + functionInterpreter_->execDumpPassName;
        VERIFY_LOGI("%s: Verify", key.c_str());
        functionInterpreter_->captureIndex = captureIndex;

        std::shared_ptr<FunctionCaptureExecution> capture = nullptr;
        capture = captureList[captureIndex];

        std::shared_ptr<FunctionCaptureExecution> captureExecution = nullptr;
        try {
            captureExecution = functionInterpreter_->RunForPass(functionInterpreter_->execDumpPassName, func, capture);
        } catch (std::exception &e) {
            VERIFY_LOGE_FULL_E(VerifyResultScene::VERIFY_RESULT_MISMATCH,
                        "VerifyPass failed for function %s, pass %s (passIndex: %d, captureIndex: %zu): %s",
                        func->GetMagicName().c_str(), passIdentifier.c_str(), passIndex, captureIndex, e.what());
            checkResult = false;
            continue;
        }

        auto goldenDataViewList = capture->golden->outcastDataViewList;
        auto executeDataViewList = captureExecution->golden->outcastDataViewList;

        std::string tensorName = "tensor~" + func->GetMagicName() + "~" + passIdentifier +
                    "~" + functionInterpreter_->GetLoopSymbolString(false);

        auto res = VerifyResult(key, tensorName, goldenDataViewList, executeDataViewList, rtol, atol);
        if (!res) {
            checkResult = false;
        }
    }
    functionInterpreter_->DumpReset();
}

FlowVerifier &FlowVerifier::GetInstance() {
    static FlowVerifier flowVerifier;
    return flowVerifier;
}
}
