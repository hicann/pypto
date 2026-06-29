/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_tensor_format.cpp
 * \brief
 */

#include "infer_tensor_format.h"

#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/platform.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "InferTensorFormat"

namespace npu::tile_fwk {

// =============================================================================
// Op 分类
// =============================================================================

const std::unordered_set<Opcode> InferTensorFormat::kPassThroughOps = {
    Opcode::OP_VIEW, Opcode::OP_VIEW_TYPE, Opcode::OP_RESHAPE};

// =============================================================================
// Format 查询
// =============================================================================

TileOpFormat InferTensorFormat::GetRequiredInputFormat(Opcode opcode, const std::string& arch, size_t inputPos)
{
    const auto& formatList = OpcodeManager::Inst().GetSupportOpFormatList(opcode);
    auto archIt = formatList.find(arch);
    if (archIt == formatList.end() || archIt->second.empty()) {
        return TileOpFormat::TILEOP_ND;
    }
    const auto& inputFormats = archIt->second[0];
    if (inputPos >= inputFormats.size()) {
        return TileOpFormat::TILEOP_ND;
    }
    return inputFormats[inputPos];
}

TileOpFormat InferTensorFormat::GetOutputFormat(Opcode opcode, const std::string& arch, size_t outputPos)
{
    constexpr size_t kOutputFormatListIndex = 1; // formatList[0]=输入格式, [1]=输出格式
    const auto& formatList = OpcodeManager::Inst().GetSupportOpFormatList(opcode);
    auto archIt = formatList.find(arch);
    if (archIt == formatList.end() || archIt->second.size() <= kOutputFormatListIndex) {
        return TileOpFormat::TILEOP_ND;
    }
    const auto& outputFormats = archIt->second[kOutputFormatListIndex];
    if (outputPos >= outputFormats.size()) {
        return TileOpFormat::TILEOP_ND;
    }
    return outputFormats[outputPos];
}

int InferTensorFormat::GetOpGroupValue(Operation* op)
{
    if (op != nullptr && op->HasAttr(CONV_GROUPS_ATTR)) {
        return static_cast<int>(op->GetIntAttribute(CONV_GROUPS_ATTR));
    }
    return 1;
}

int InferTensorFormat::GetTransDataGroupValue(const std::shared_ptr<LogicalTensor>& srcTensor, 
        const std::shared_ptr<LogicalTensor>& fakeDstTensor, Operation* relatedOp)
{
    int group = GetOpGroupValue(relatedOp);
    if (group != 1 || srcTensor == nullptr) {
        return group;
    }
    for (auto* producer : srcTensor->GetProducers()) {
        group = GetOpGroupValue(producer);
        if (group != 1) {
            return group;
        }
    }
    if (srcTensor != fakeDstTensor){
        for (auto* consumer : fakeDstTensor->GetConsumers()) {
            group = GetOpGroupValue(consumer);
            if (group != 1) {
                return group;
            }
        }
    }
    return 1;
}

bool InferTensorFormat::IsSupportedTransData(TileOpFormat srcFormat, TileOpFormat targetFormat)
{
    if (srcFormat == targetFormat) {
        return true;
    }
    if (srcFormat == TileOpFormat::TILEOP_ND) {
        return targetFormat == TileOpFormat::TILEOP_NC1HWC0 ||
               targetFormat == TileOpFormat::TILEOP_FRACTAL_Z ||
               targetFormat == TileOpFormat::TILEOP_NDC1HWC0 ||
               targetFormat == TileOpFormat::TILEOP_FRACTAL_Z_3D;
    }
    return (srcFormat == TileOpFormat::TILEOP_NC1HWC0 || srcFormat == TileOpFormat::TILEOP_NDC1HWC0) &&
           targetFormat == TileOpFormat::TILEOP_ND;
}

bool InferTensorFormat::IsNdNzCompatibleFormat(TileOpFormat srcFormat, TileOpFormat targetFormat)
{
    return (srcFormat == TileOpFormat::TILEOP_ND && targetFormat == TileOpFormat::TILEOP_NZ) ||
           (srcFormat == TileOpFormat::TILEOP_NZ && targetFormat == TileOpFormat::TILEOP_ND);
}

bool InferTensorFormat::IsValidTileOpFormat(int64_t format)
{
    return format >= 0 && format < static_cast<int64_t>(TileOpFormat::TILEOP_FORMAT_NUM);
}

// =============================================================================
// 图操作
// =============================================================================

bool InferTensorFormat::IsFunctionOutcast(const Function& function, const std::shared_ptr<LogicalTensor>& tensor)
{
    for (const auto& outcast : function.GetOutcast()) {
        if (outcast == tensor) {
            return true;
        }
    }
    return false;
}

int InferTensorFormat::FindInputPosition(const Operation& op, const std::shared_ptr<LogicalTensor>& tensor)
{
    const auto& inputs = op.GetIOperands();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == tensor) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void InferTensorFormat::ApplyTransDataVecTile(const std::shared_ptr<LogicalTensor>& srcTensor, TileOpFormat targetFormat)
{
    int64_t c0 = srcTensor->Datatype() == DataType::DT_FP32 ? 8 : 16;
    TileOpFormat srcFormat = srcTensor->GetRawTensor()->format;
    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(DistributedErrorCode::INVALID_TILE_SHAPE, oriVectile.tile.back() % c0 == 0) << "The last dimension of `tile_shape` should be 32-byte aligned.";
    if (oriVectile.tile.size() == 3) {
        oriVectile.tile.insert(oriVectile.tile.begin() + 2, 1);
        TileShape::Current().SetVecTile(oriVectile);
    }
    if (srcFormat == TileOpFormat::TILEOP_NC1HWC0 && targetFormat == TileOpFormat::TILEOP_ND) {
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        tmpVectile.tile[1] = tmpVectile.tile[1] / c0;
        tmpVectile.tile.emplace_back(c0);
        TileShape::Current().SetVecTile(tmpVectile);
    } else if (srcFormat == TileOpFormat::TILEOP_NDC1HWC0 && targetFormat == TileOpFormat::TILEOP_ND) {
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        std::swap(tmpVectile.tile[1], tmpVectile.tile[2]);
        tmpVectile.tile[2] = tmpVectile.tile[2] / c0;
        tmpVectile.tile.emplace_back(c0);
        TileShape::Current().SetVecTile(tmpVectile);
    }
}

std::shared_ptr<LogicalTensor> InferTensorFormat::InsertTransDataOp(
    Function& function, const std::shared_ptr<LogicalTensor>& srcTensor, const std::shared_ptr<LogicalTensor>& fakeDstTensor,
    Operation* relatedOp, TileOpFormat targetFormat)
{
    int group_value = GetTransDataGroupValue(srcTensor, fakeDstTensor, relatedOp);
    ApplyTransDataVecTile(srcTensor, targetFormat);
    auto result = TransData(function, srcTensor, fakeDstTensor, targetFormat, group_value);
    result->GetRawTensor()->format = targetFormat;
    return result;
}

Status InferTensorFormat::EnsureTensorFormat(
    Function& function, std::shared_ptr<LogicalTensor>& tensor, Operation* relatedOp, TileOpFormat targetFormat)
{
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Null tensor when ensuring format %d for op [%s][%d].",
            static_cast<int>(targetFormat), relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
            relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    TileOpFormat current = tensor->Format();
    if (current == targetFormat || IsNdNzCompatibleFormat(current, targetFormat)) {
        return SUCCESS;
    }
    if (!IsSupportedTransData(current, targetFormat)) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Unsupported format conversion: src=%d dst=%d for op [%s][%d].",
            static_cast<int>(current), static_cast<int>(targetFormat),
            relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
            relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    std::shared_ptr<LogicalTensor> newTensor = nullptr;
    if (relatedOp->GetOpcode() == Opcode::OP_FAKE_TRANS){
        newTensor = InsertTransDataOp(function, tensor, relatedOp->GetOOperands()[0], relatedOp, targetFormat);
    } else {
        newTensor = InsertTransDataOp(function, tensor, tensor, relatedOp, targetFormat);
    }
    if (newTensor == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Insert TransData failed: tensor[%d] src=%d dst=%d for op [%s][%d].",
            tensor->GetMagic(), static_cast<int>(current), static_cast<int>(targetFormat),
            relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
            relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(
        Elements::Operation,
        "Inserted TransData: tensor[%d] (fmt=%d) -> tensor[%d] (fmt=%d) for op [%s][%d].",
        tensor->GetMagic(), static_cast<int>(current), newTensor->GetMagic(), static_cast<int>(newTensor->Format()),
        relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
        relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
    tensor = newTensor;
    return SUCCESS;
}

Status InferTensorFormat::GetFakeTransFormat(const Operation& op, const std::string& attrName, TileOpFormat& format)
{
    if (!op.HasAttr(attrName)) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "OP_FAKE_TRANS[%d] missing attribute [%s].",
            op.GetOpMagic(), attrName.c_str());
        return FAILED;
    }
    int64_t value = op.GetIntAttribute(attrName);
    if (!IsValidTileOpFormat(value)) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "OP_FAKE_TRANS[%d] attribute [%s] has invalid format value [%ld].",
            op.GetOpMagic(), attrName.c_str(), value);
        return FAILED;
    }
    format = static_cast<TileOpFormat>(value);
    return SUCCESS;
}

Status InferTensorFormat::ResolveFakeTransOp(
    Function& function, Operation& op, const std::shared_ptr<LogicalTensor>& inputTensor,
    std::unordered_map<int, bool>& visitedTensors, std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "OP_FAKE_TRANS[%d] expects 1 input and 1 output, got %zu inputs and %zu outputs.",
            op.GetOpMagic(), op.GetIOperands().size(), op.GetOOperands().size());
        return FAILED;
    }

    TileOpFormat fakeInFormat;
    TileOpFormat fakeOutFormat;
    if (GetFakeTransFormat(op, FAKE_TRANS_IN_FORMAT_ATTR, fakeInFormat) != SUCCESS ||
        GetFakeTransFormat(op, FAKE_TRANS_OUT_FORMAT_ATTR, fakeOutFormat) != SUCCESS) {
        return FAILED;
    }

    auto finalTensor = inputTensor;
    if (EnsureTensorFormat(function, finalTensor, &op, fakeInFormat) != SUCCESS ||
        EnsureTensorFormat(function, finalTensor, &op, fakeOutFormat) != SUCCESS) {
        return FAILED;
    }

    auto fakeOutput = op.GetOOperands()[0];
    auto consumers = fakeOutput->GetConsumers();
    for (auto* consumer : consumers) {
        if (consumer == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "Null consumer for OP_FAKE_TRANS[%d] output tensor [%d].",
                op.GetOpMagic(), fakeOutput->GetMagic());
            return FAILED;
        }
        consumer->ReplaceInput(finalTensor, fakeOutput);
    }

    op.SetAsDeleted();

    visitedTensors[finalTensor->GetMagic()] = true;
    worklist.push(finalTensor);
    return SUCCESS;
}

// =============================================================================
// 输出 format 推导
// =============================================================================

void InferTensorFormat::DetermineOutputFormat(
    const Function& function, const Operation& op, const std::string& arch,
    std::unordered_map<int, bool>& visitedTensors,
    std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    Opcode opcode = op.GetOpcode();

    for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
        auto output = op.GetOOperands()[i];
        TileOpFormat fmt;

        if (kPassThroughOps.count(opcode)) {
            fmt = op.GetIOperands()[0]->Format();
        } else if (opcode == Opcode::OP_ASSEMBLE) {
            fmt = IsFunctionOutcast(function, output) ? TileOpFormat::TILEOP_ND : op.GetIOperands()[0]->Format();
        } else {
            fmt = GetOutputFormat(opcode, arch, i);
        }

        output->GetRawTensor()->format = fmt;

        int magic = output->GetMagic();
        if (visitedTensors.find(magic) == visitedTensors.end()) {
            visitedTensors[magic] = true;
            worklist.push(output);
        }
    }
}

void InferTensorFormat::EnqueueTensorIfNeeded(
    const std::shared_ptr<LogicalTensor>& tensor, std::unordered_map<int, bool>& visitedTensors,
    std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    if (tensor == nullptr) {
        return;
    }
    int magic = tensor->GetMagic();
    if (visitedTensors.find(magic) == visitedTensors.end()) {
        visitedTensors[magic] = true;
        worklist.push(tensor);
    }
}

void InferTensorFormat::EnqueueFunctionInputs(
    const Function& function, std::unordered_map<int, bool>& visitedTensors,
    std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    for (const auto& tensor : function.GetIncast()) {
        EnqueueTensorIfNeeded(tensor, visitedTensors, worklist);
    }
}

TileOpFormat InferTensorFormat::ResolveRequiredInputFormat(
    const Function& function, const Operation& consumer, const std::shared_ptr<LogicalTensor>& tensor,
    const std::string& arch, int inputPos, std::unordered_set<int>& assembledOutputs)
{
    TileOpFormat required = GetRequiredInputFormat(consumer.GetOpcode(), arch, static_cast<size_t>(inputPos));
    if (consumer.GetOpcode() != Opcode::OP_ASSEMBLE || consumer.GetOOperands().empty()) {
        return required;
    }

    auto output = consumer.GetOOperands()[0];
    int outMagic = output->GetMagic();
    if (IsFunctionOutcast(function, output)) {
        return TileOpFormat::TILEOP_ND;
    }
    if (assembledOutputs.count(outMagic)) {
        return output->Format();
    }
    assembledOutputs.insert(outMagic);
    return tensor->Format();
}

Status InferTensorFormat::EnsureConsumerInputFormat(
    Function& function, Operation& consumer, const std::shared_ptr<LogicalTensor>& tensor,
    TileOpFormat required)
{
    auto actualTensor = tensor;
    if (actualTensor->Format() == required) {
        return SUCCESS;
    }
    if (EnsureTensorFormat(function, actualTensor, &consumer, required) != SUCCESS) {
        return FAILED;
    }
    if (actualTensor->GetMagic() != tensor->GetMagic()) {
        consumer.ReplaceInput(actualTensor, tensor);
    }
    return SUCCESS;
}

void InferTensorFormat::MarkConsumerInputProcessed(
    const Function& function, const Operation& consumer, const std::string& arch,
    std::unordered_map<int, int>& processedInputs, std::unordered_map<int, bool>& visitedTensors,
    std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    int opMagic = consumer.GetOpMagic();
    processedInputs[opMagic]++;
    if (static_cast<size_t>(processedInputs[opMagic]) == consumer.GetInputOperandSize()) {
        DetermineOutputFormat(function, consumer, arch, visitedTensors, worklist);
    }
}

Status InferTensorFormat::ProcessConsumerFormat(
    Function& function, Operation* consumer, const std::shared_ptr<LogicalTensor>& tensor,
    const std::string& arch, std::unordered_map<int, int>& processedInputs,
    std::unordered_map<int, bool>& visitedTensors, std::unordered_set<int>& assembledOutputs,
    std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    if (consumer == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Null consumer for tensor [%d].", tensor->GetMagic());
        return FAILED;
    }
    if (consumer->IsDeleted()) {
        return SUCCESS;
    }

    int pos = FindInputPosition(*consumer, tensor);
    if (pos < 0) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Tensor [%d] not in consumer [%d] inputs.",
            tensor->GetMagic(), consumer->GetOpMagic());
        return FAILED;
    }
    if (consumer->GetOpcode() == Opcode::OP_FAKE_TRANS) {
        return ResolveFakeTransOp(function, *consumer, tensor, visitedTensors, worklist);
    }

    TileOpFormat required = ResolveRequiredInputFormat(function, *consumer, tensor, arch, pos, assembledOutputs);
    if (EnsureConsumerInputFormat(function, *consumer, tensor, required) != SUCCESS) {
        return FAILED;
    }
    MarkConsumerInputProcessed(function, *consumer, arch, processedInputs, visitedTensors, worklist);
    return SUCCESS;
}

Status InferTensorFormat::ProcessTensorConsumers(
    Function& function, const std::shared_ptr<LogicalTensor>& tensor, const std::string& arch,
    std::unordered_map<int, int>& processedInputs, std::unordered_map<int, bool>& visitedTensors,
    std::unordered_set<int>& assembledOutputs, std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    auto consumers = tensor->GetConsumers();
    for (auto* consumer : consumers) {
        if (ProcessConsumerFormat(function, consumer, tensor, arch, processedInputs, visitedTensors,
            assembledOutputs, worklist) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

// =============================================================================
// 主算法: BFS 从 incast 沿 consumer 链推导 format
// =============================================================================

Status InferTensorFormat::DeriveFormats(Function& function)
{
    std::string arch = NPUArchToString(Platform::Instance().GetSoc().GetNPUArch());

    std::queue<std::shared_ptr<LogicalTensor>> worklist;
    std::unordered_map<int, bool> visitedTensors;
    std::unordered_map<int, int> processedInputs;
    std::unordered_set<int> assembledOutputs; // 已被首个 assemble 初始化的输出 tensor

    EnqueueFunctionInputs(function, visitedTensors, worklist);
    APASS_LOG_DEBUG_F(Elements::Tensor, "Initial worklist size: %zu (incast tensors).", worklist.size());

    while (!worklist.empty()) {
        auto tensor = worklist.front();
        worklist.pop();
        if (ProcessTensorConsumers(function, tensor, arch, processedInputs, visitedTensors, assembledOutputs,
            worklist) != SUCCESS) {
            return FAILED;
        }
    }

    return SUCCESS;
}

// =============================================================================
// Pass 入口
// =============================================================================

Status InferTensorFormat::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(
        Elements::Function, "Start InferTensorFormat for function [%s].", function.GetRawName().c_str());

    Status status = DeriveFormats(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(
            Elements::Function, "InferTensorFormat failed for function [%s].", function.GetRawName().c_str());
        return FAILED;
    }
    function.EraseOperations(true, false);

    APASS_LOG_INFO_F(
        Elements::Function, "End InferTensorFormat for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

} // namespace npu::tile_fwk
