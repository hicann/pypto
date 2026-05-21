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
 * \file cost_model.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "cost_model/simulation/cost_model_launcher.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

std::string ValidateDynamicFunctionAndIO(
    Function* func, const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs);
bool TryBuildDynamicCellMatchDesc(
    const DyndevFunctionAttribute::DynamicCellMatchLaunchMeta& launchMeta, Evaluator& eval,
    DevCellMatchTableDesc& patchedDesc);

static bool PrepareSingleDynamicCellMatchDescForCostModel(
    const DyndevFunctionAttribute::DynamicCellMatchLaunchMeta& meta, Evaluator& eval, DevAscendProgram* devProg)
{
    DevCellMatchTableDesc patchedDesc;
    bool ready = TryBuildDynamicCellMatchDesc(meta, eval, patchedDesc);
    if (!ready) {
        return false;
    }
    auto* dst = reinterpret_cast<uint8_t*>(devProg) + meta.descOffset;
    (void)memcpy_s(dst, sizeof(DevCellMatchTableDesc), &patchedDesc, sizeof(DevCellMatchTableDesc));
    return true;
}

static void PrepareDynamicCellMatchDescForCostModel(
    DyndevFunctionAttribute* attr, DevAscendProgram* devProg, const std::vector<DeviceTensorData>& inputs,
    const std::vector<DeviceTensorData>& outputs)
{
    if (attr == nullptr || devProg == nullptr || attr->dynamicCellMatchLaunchMetaList.empty()) {
        return;
    }
    Evaluator eval{attr->inputSymbolDict, inputs, outputs};
    std::vector<DyndevFunctionAttribute::DynamicCellMatchLaunchMeta> const& metas = attr->dynamicCellMatchLaunchMetaList;
    size_t unpreparedCount = 0;
    int firstUnpreparedSlot = -1;
    for (const auto& meta : metas) {
        if (!PrepareSingleDynamicCellMatchDescForCostModel(meta, eval, devProg)) {
            unpreparedCount++;
            if (firstUnpreparedSlot < 0) {
                firstUnpreparedSlot = meta.slotIndex;
            }
            continue;
        }
    }
    if (unpreparedCount != 0) {
        ASSERT(false)
            << "cost model dynamic cell match prepare failed, unpreparedCount=" << unpreparedCount
            << ", firstSlot=" << firstUnpreparedSlot;
    }
}

static std::string ValidateFunctionAndIO(
    Function* func, const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    return ValidateDynamicFunctionAndIO(func, inputs, outputs);
}

static void InitializeInputOutputData(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData =
            RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t*)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
}

static std::string InitInputOutputData(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    Function* func = Program::GetInstance().GetLastFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    InitializeInputOutputData(inputs, outputs);
    return "";
}

static void CopyTensorFromModel(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    auto& rawInputTensors = ProgramData::GetInstance().GetInputDataList();
    for (size_t i = 0; i < inputs.size(); i++) {
        StringUtils::DataCopy(
            (uint8_t*)inputs[i].GetAddr(), inputs[i].GetDataSize(), rawInputTensors[i]->data(),
            rawInputTensors[i]->GetDataSize());
    }

    auto& rawOutputTensors = ProgramData::GetInstance().GetOutputDataList();
    for (size_t i = 0; i < outputs.size(); i++) {
        StringUtils::DataCopy(
            (uint8_t*)outputs[i].GetAddr(), outputs[i].GetDataSize(), rawOutputTensors[i]->data(),
            rawOutputTensors[i]->GetDataSize());
    }
}

std::string CostModelRunOnceDataFromHost(
    const std::vector<DeviceTensorData>& inputs, const std::vector<DeviceTensorData>& outputs)
{
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    std::string initResult = InitInputOutputData(inputs, outputs);
    if (!initResult.empty()) {
        return initResult;
    }

    Function* func = Program::GetInstance().GetLastFunction();
    auto attr = func->GetDyndevAttribute();
    if (attr != nullptr && !attr->devProgBinary.empty()) {
        auto* devProg = reinterpret_cast<DevAscendProgram*>(attr->devProgBinary.data());
        Evaluator eval{attr->inputSymbolDict, inputs, outputs};
        PrepareDynamicCellMatchDescForCostModel(attr.get(), devProg, inputs, outputs);
        if (attr->maxDynamicAssembleOutcastMem.IsValid()) {
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(attr->maxDynamicAssembleOutcastMem);
        }
        if (attr->maxDynamicCellMatchTableMem.IsValid()) {
            devProg->memBudget.metadata.maxDynamicCellMatchTableMem = eval.Evaluate(attr->maxDynamicCellMatchTableMem);
            uint64_t totalDynamicCellMatchSlotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
            devProg->memBudget.metadata.dynamicCellMatch =
                totalDynamicCellMatchSlotNum * devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
        }
    }
    CostModelLauncher::CostModelRunOnce(func);
    CopyTensorFromModel(inputs, outputs);
    return "";
}

void BindCostModelRuntime(py::module& m) { m.def("CostModelRunOnceDataFromHost", &CostModelRunOnceDataFromHost); }
} // namespace pypto
