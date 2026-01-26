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
 * \file infer_memory_conflict.cpp
 * \brief
 */

#include "infer_memory_conflict.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InferMemoryConflict"

namespace npu {
namespace tile_fwk {
namespace {
uint32_t GetPowerOfTwo(uint32_t cur) {
    uint32_t ret = 1;
    while (ret < cur) {
        ret <<= 1;
    }
    return ret;
}
}

Status InferMemoryConflict::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "Start InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    if (Init(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed.");
        return FAILED;
    }
    if (ForwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ForwardPropagation failed.");
        return FAILED;
    }
    if (BackwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BackwardPropagation failed.");
        return FAILED;
    }
    if (InsertCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertCopys failed.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW_TYPE) {
            auto output = op.GetOOperands()[0];
            auto outOp = *output->GetConsumers().begin();
            if (outOp == nullptr || outOp->GetOpcode() != Opcode::OP_REGISTER_COPY) {
                continue;
            }
            TileShape viewTypeTile;
            auto vecTypeTile = op.GetTileShape().GetVecTile();
            auto viewTypeIn = op.GetIOperands()[0];
            auto viewTypeOut = op.GetOOperands()[0];
            auto inType = viewTypeIn->tensor->datatype;
            auto outType = viewTypeOut->tensor->datatype;
            auto inEntry = viewTypeTable.find(inType);
            auto outEntry = viewTypeTable.find(outType);
            if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
                return FAILED;
            }
            if (inEntry->second < outEntry->second) {
                if (vecTypeTile.tile[vecTypeTile.tile.size()-1] % (outEntry->second / inEntry->second) != 0) {
                    APASS_LOG_ERROR_F(Elements::Operation, "vecTypeTile tile dim n is not even.");
                    return FAILED;
                }
                vecTypeTile.tile[vecTypeTile.tile.size()-1] /= (outEntry->second / inEntry->second);
            } else {
                vecTypeTile.tile[vecTypeTile.tile.size()-1] *= (inEntry->second / outEntry->second);
            }
            viewTypeTile.SetVecTile(vecTypeTile);
            outOp->UpdateTileShape(viewTypeTile);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

bool InferMemoryConflict::CheckConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor) {
    if (inTensor->Symbol() == outTensor->Symbol()) {
        return false;
    }
    if (inTensor->GetRawTensor()->memoryId == outTensor->GetRawTensor()->memoryId) {
        return false;
    }
    return true;
}

bool InferMemoryConflict::CheckRawShapeConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor) {
    int64_t inRawSize = 1;
    int64_t outRawSize = 1;
    Shape inShape = inTensor->GetRawTensor()->GetRawShape();
    Shape outShape = outTensor->GetRawTensor()->GetRawShape();
    auto inType = inTensor->tensor->datatype;
    auto outType = outTensor->tensor->datatype;
    auto inEntry = viewTypeTable.find(inType);
    auto outEntry = viewTypeTable.find(outType);
    auto consumerOp = *inTensor->GetConsumers().begin();
    if (consumerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
        if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
            return true;
        }
        if (inEntry->second > outEntry->second) {
            inRawSize *= (inEntry->second / outEntry->second);
        } else {
            outRawSize *= (outEntry->second / inEntry->second);
        }
    }
    for (size_t i = 0; i < inShape.size(); ++i) {
        if (inShape[i] < 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "inShape[%d] = %d, dynamic shape should trigger conflict", i, inShape[i]);
            return true;
        }
        inRawSize *= inShape[i];
    }
    for (size_t i = 0; i < outShape.size(); ++i) {
        if (outShape[i] < 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "outShape[%d] = %d, dynamic shape should trigger conflict", i, outShape[i]);
            return true;
        }
        outRawSize *= outShape[i];
    }
    if (inRawSize > 0 && outRawSize > 0 && inRawSize != outRawSize) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The raw size of input is %d, the raw size of output is %d", inRawSize, outRawSize);
        return true;
    }
    return false;
}

bool InferMemoryConflict::CheckTransmit(Operation &curOp) {
    LogicalTensorPtr curTensor;
    std::set<Opcode> NonCalcNode = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_RESHAPE, Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW_TYPE};
    bool transmit = (NonCalcNode.find(curOp.GetOpcode()) != NonCalcNode.end());
    if (curOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
        curTensor = *(curOp.GetIOperands().begin());
        for (const auto &producer : curTensor->GetProducers()) {
            if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                transmit = false;
            }
        }
    }
    return transmit;
}

bool InferMemoryConflict::IsValidTileShape(const Operation &op) const {
    auto input = op.GetIOperands().front();
    VecTile tileSize = op.GetTileShape().GetVecTile();
    if (input->GetShape().size() != tileSize.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has unequal input shape dims size and tile shape dims, input shape: %s, tile size: %s. %s", 
                            op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                            input->DumpType().c_str(), op.GetTileShape().ToString(TileType::VEC).c_str(), GetFormatBacktrace(op).c_str());
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "The size info of %s[%d]: input shape: %s, tile size: %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
            input->DumpType().c_str(), op.GetTileShape().ToString(TileType::VEC).c_str());
    return true;
}

// batch MatMul优化pattern，不插入register copy
bool InferMemoryConflict::MatchReshapePattern(const LogicalTensorPtr &reshapeIn,
                                            const LogicalTensorPtr &reshapeOut) {
    if (!reshapeIn || !reshapeOut) return false;
    
    const auto &inputShape = reshapeIn->GetShape();
    const auto &outputShape = reshapeOut->GetShape();
    const size_t inputDims = inputShape.size();
    const size_t outputDims = outputShape.size();
    
    // 常量定义
    constexpr size_t MIN_DIMENSIONS = 2;
    constexpr size_t MAX_DIMENSIONS = 4;
    constexpr size_t DIMENSIONS_2D = 2;
    constexpr size_t DIMENSIONS_3D = 3;
    constexpr size_t DIMENSIONS_4D = 4;
    
    if (inputDims < MIN_DIMENSIONS || outputDims < MIN_DIMENSIONS || inputDims > MAX_DIMENSIONS || outputDims > MAX_DIMENSIONS) return false;
    
    // 验证总元素数是否相等（reshape的基本要求）
    auto calculateTotalElements = [](const std::vector<int64_t>& shape) {
        int64_t total = 1;
        for (const auto& dim : shape) {
            total *= dim;
        }
        return total;
    };
    
    if (calculateTotalElements(inputShape) != calculateTotalElements(outputShape)) return false;
    
    // 编码维度对：输入维度在高位，输出维度在低位
    const uint32_t dimensionPair = (inputDims << 4) | outputDims;
    
    switch (dimensionPair) {
        // 4D转2D：[1, 1, H, W] -> [H, W]
        case (DIMENSIONS_4D << 4) | DIMENSIONS_2D: {
            return inputShape[0] == 1 && 
                   inputShape[1] == 1 &&
                   inputShape[2] == outputShape[0] &&
                   inputShape[3] == outputShape[1];
        }
        
        // 2D转4D：[H, W] -> [1, 1, H, W]
        case (DIMENSIONS_2D << 4) | DIMENSIONS_4D: {
            return outputShape[0] == 1 && 
                   outputShape[1] == 1 &&
                   inputShape[0] == outputShape[2] &&
                   inputShape[1] == outputShape[3];
        }
        
        // 3D转2D：[1, H, W] -> [H, W]
        case (DIMENSIONS_3D << 4) | DIMENSIONS_2D: {
            return inputShape[0] == 1 &&
                   inputShape[1] == outputShape[0] &&
                   inputShape[2] == outputShape[1];
        }
        
        // 2D转3D：[H, W] -> [1, H, W]
        case (DIMENSIONS_2D << 4) | DIMENSIONS_3D: {
            return outputShape[0] == 1 &&
                   inputShape[0] == outputShape[1] &&
                   inputShape[1] == outputShape[2];
        }
        
        default:
            return false;
    }
}

Status InferMemoryConflict::UpdateForwardTensor(Function &function, const LogicalTensorPtr &curTensor, Operation* consumer, std::queue<LogicalTensorPtr> &curTensors) {
    for (const auto &outputTensor : consumer->GetOOperands()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            auto reshapeInput = consumer->GetIOperands().front();
            bool isInplace = consumer->GetBoolAttribute(OP_ATTR_PREFIX + "isInplace");
            if (!MatchReshapePattern(reshapeInput, outputTensor) && !isInplace && CheckRawShapeConflict(memoryInfo[curTensor], outputTensor)) {
                preregcopys.insert(consumer);
                continue;
            }
        }
        if (memoryInfo.find(outputTensor) != memoryInfo.end() && function.IsFromOutCast(memoryInfo[outputTensor])) {
            if (CheckConflict(memoryInfo[curTensor], memoryInfo[outputTensor])) {
                preregcopys.insert(consumer);
            }
        } else {
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                int index = 2;
                memoryInfo[outputTensor] = memoryInfo[consumer->GetInputOperand(index)];
            } else {
                memoryInfo[outputTensor] = memoryInfo[curTensor];
            }
            curTensors.push(outputTensor);
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::UpdateBackwardTensor(const LogicalTensorPtr &curTensor, Operation* producer, std::queue<LogicalTensorPtr> &curTensors) {
    for (auto &inputTensor : producer->GetIOperands()) {
        int index = 2;
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && producer->GetIOperandIndex(inputTensor) != index) {
            continue;
        }
        auto reshapeOutput = producer->GetOOperands().front();
        if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
            bool isInplace = producer->GetBoolAttribute(OP_ATTR_PREFIX + "isInplace");
            if (!MatchReshapePattern(inputTensor, reshapeOutput) && !isInplace && CheckRawShapeConflict(inputTensor, memoryInfo[curTensor])) {
                postregcopys.insert(producer);
                continue;
            }
        }
        if (memoryInfo.find(inputTensor) != memoryInfo.end()) {
            if (CheckConflict(memoryInfo[curTensor], memoryInfo[inputTensor])) {
                if (producer->GetOpcode() == Opcode::OP_RESHAPE && !MatchReshapePattern(inputTensor, reshapeOutput)) {
                    postregcopys.insert(producer);
                } else {
                    preregcopys.insert(producer);
                }
            }
        } else {
            memoryInfo[inputTensor] = memoryInfo[curTensor];
            curTensors.push(inputTensor);
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::ForwardPropagation(Function &function) {
    std::queue<LogicalTensorPtr> curTensors;
    for (const auto &incast : function.GetIncast()) {
        curTensors.push(incast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto &consumer : curTensor->GetConsumers()) {
            if (!CheckTransmit(*consumer)) {
                continue;
            }
            int index = 2;
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && consumer->GetIOperandIndex(curTensor) != index) {
                continue;
            }
            if (UpdateForwardTensor(function, curTensor, consumer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "UpdateForwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::BackwardPropagation(Function &function) {
    std::queue<LogicalTensorPtr> curTensors;
    for (const auto &outcast : function.GetOutcast()) {
        curTensors.push(outcast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto &producer : curTensor->GetProducers()) {
            if (!CheckTransmit(*producer)) {
                continue;
            }
            if (UpdateBackwardTensor(curTensor, producer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "UpdateBackwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::SetDefaultShape(const LogicalTensorPtr &tensor, std::vector<int64_t> &defaultTile) {
    int64_t maximalTileSize = 16384;
    int64_t alignTailSize = 32;
    Shape shape = tensor->GetShape();
    size_t shapeDim = shape.size();
    int64_t curTile;
    defaultTile.clear();
    for (size_t i = 0; i < shape.size(); ++i) {
        defaultTile.emplace_back(1);
    }
    curTile = shape[shapeDim - 1] < alignTailSize ? alignTailSize : GetPowerOfTwo(shape[shapeDim - 1]);
    defaultTile[shapeDim - 1] = maximalTileSize < curTile ? maximalTileSize : curTile;
    for (int i = shapeDim - 2; i >= 0; --i) {
        maximalTileSize /= defaultTile[i + 1];
        curTile = GetPowerOfTwo(shape[i]);
        defaultTile[i] = maximalTileSize < curTile ? maximalTileSize : curTile;
        defaultTile[i] = defaultTile[i] == 0 ? 1 : defaultTile[i];
    }
    return SUCCESS;
}

TileShape InferMemoryConflict::ObtainTileShape(const std::unordered_set<Operation *> &origOps) {
    TileShape tile;
    if (origOps.empty()) {
        return tile;
    }
    TileShape base = (*origOps.begin())->GetTileShape();
    if (origOps.size() == 1) {
        return base;
    }
    for (const auto &origOp : origOps) {
        if (origOp->GetTileShape().GetVecTile().tile == base.GetVecTile().tile) {
            return tile;
        }
    }
    return base;
}

Status InferMemoryConflict::InferTileShape(Operation &op, const LogicalTensorPtr &tensor, TileShape parentTile, Shape &reshapeTile) {
    auto tileShapeSize = parentTile.GetVecTile().size();
    auto tensorSize = tensor->GetShape().size();
    if (tileShapeSize == 0 || tileShapeSize != tensorSize) {
        APASS_LOG_WARN_F(Elements::Operation, "Inserted op[%d]'s producer/consumer op has no tile shape.", op.GetOpMagic());
        TileShape tile;
        Shape defaultTile;
        if (!reshapeTile.empty() && reshapeTile.size() == tensorSize) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Derivate reshape tile shape.");
            defaultTile = reshapeTile;
        } else {
            if (SetDefaultShape(tensor, defaultTile) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SetDefaultShape failed.");
                return FAILED;
            }
        }
        tile.SetVecTile(defaultTile);
        op.UpdateTileShape(tile);
    } else {
        op.UpdateTileShape(parentTile);
    }
    if (!IsValidTileShape(op)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid tile size for %s[%d]. %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::ObtainReshapeTile(Operation &op, Shape &inTileShape, Shape &outTileShape) {
    if (op.GetOpcode() == Opcode::OP_RESHAPE) {
        //同时为空，证明对端不存在可用的tileshape
        if (inTileShape.empty() && outTileShape.empty()) {
            return SUCCESS;
        }
        Shape inShape = op.GetIOperands().front()->shape;
        Shape outShape = op.GetOOperands().front()->shape;
        DerivationTileShape derivationTileShapePass;
        if (derivationTileShapePass.DerivationReshapeTileShape(&op, inShape, outShape, inTileShape, outTileShape) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "DerivationReshapeTileShape failed. %s", GetFormatBacktrace(op).c_str());
            // 失败时返回空的tileshape，由InferTileShape自动推导
            return SUCCESS;
        }
    }
    return SUCCESS;
}

// 在OP_RESHAPE前面插入OP_REGISTER_COPY
Status InferMemoryConflict::InsertPrecededCopys(Function &function) {
    for (const auto op : preregcopys) {
        LogicalTensorPtr inputTensor = op->GetIOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(inputTensor->Datatype(), inputTensor->GetShape());
        Offset newOffset(inputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset, inputTensor->GetShape(), inputTensor->GetDynValidShape());
        auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {inputTensor}, {newTensor});
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert copy op [%d].", copyOp.GetOpMagic());
        Shape reshapeTile;
        if (ObtainReshapeTile(*op, reshapeTile, ObtainTileShape(op->ConsumerOps()).GetVecTile().tile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ObtainReshapeTile failed. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (InferTileShape(copyOp, inputTensor, ObtainTileShape(copyOp.ProducerOps()), reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InferTileShape failed. %s", GetFormatBacktrace(copyOp).c_str());
            return FAILED;
        }
        inputTensor->RemoveConsumer(op);
        op->ReplaceInput(newTensor, inputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertPostCopys(Function &function) {
    for (const auto op : postregcopys) {
        LogicalTensorPtr outputTensor = op->GetOOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(outputTensor->Datatype(), outputTensor->GetShape());
        Offset newOffset(outputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset, outputTensor->GetShape(), outputTensor->GetDynValidShape());
        auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {newTensor}, {outputTensor});
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert copy op [%d].", copyOp.GetOpMagic());
        Shape reshapeTile;
        if (ObtainReshapeTile(*op, ObtainTileShape(op->ProducerOps()).GetVecTile().tile, reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ObtainReshapeTile failed. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        if (InferTileShape(copyOp, outputTensor, ObtainTileShape(copyOp.ConsumerOps()), reshapeTile) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InferTileShape failed. %s", GetFormatBacktrace(copyOp).c_str());
            return FAILED;
        }
        outputTensor->RemoveConsumer(op);
        op->ReplaceOutput(newTensor, outputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertCopys(Function &function) {
    if (InsertPrecededCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertPrecededCopys failed.");
        return FAILED;
    }
    if (InsertPostCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertPostCopys failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::Init(Function &function) {
    for (auto &incast : function.GetIncast()) {
        memoryInfo[incast] = incast;
    }
    for (auto &outcast : function.GetOutcast()) {
        memoryInfo[outcast] = outcast;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
