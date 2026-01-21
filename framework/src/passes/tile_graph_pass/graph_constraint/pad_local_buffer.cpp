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
 * \file pad_local_buffer.cpp
 * \brief
 */

#include "pad_local_buffer.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PadLocalBuffer"

namespace npu::tile_fwk {
constexpr size_t MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t VECTOR_MIN_SHAPE_SIZE = 1;
constexpr size_t AXIS_COMBINE_MIN_SHAPE_SIZE = 2;
constexpr size_t TRANSPOSE_MIN_SHAPE_SIZE = 2;
constexpr size_t BROADCAST_OP_INPUT_SIZE = 2;
/* 记录不同matmul的padding值的index */
constexpr size_t CUBE_INPUT_SIZE = 4;
constexpr size_t MATMUL_AXIS_NUM = 2;
constexpr size_t MATRIX_AND_INDEX_NUM = 2;
constexpr size_t HIGH_INDEX = 0;
constexpr size_t LOW_INDEX = 1;
constexpr uint32_t LEFT_SHIFT32 = 32;
constexpr int64_t CUBE_PAD_VALUE = 16;
constexpr int64_t CUBE_PAD_INT8_VALUE = 32;
const std::vector<bool> AXIS_COMBINED = {true};
const std::vector<bool> BROADCAST_AXIS_COMBINED = {true, true};
const int64_t BRCB_SECOND_LAST_BASE = 8;
const size_t LAST_SECOND_AXIS = 2;
const std::string REDUCE_AXIS = OP_ATTR_PREFIX + "AXIS";
int64_t Pad(int64_t dim, int64_t padValue) {
    return (dim + padValue - 1) / padValue * padValue;
}

bool PadLocalBuffer::IsInputInt8(const Operation &op, const LogicalTensorPtr &in) const {
    std::vector<Opcode> cubeOps = {Opcode::OP_A_MUL_B, Opcode::OP_AT_MUL_B, Opcode::OP_A_MUL_BT, Opcode::OP_AT_MUL_BT, Opcode::OP_A_MULACC_B};

    if (in == nullptr || in->tensor == nullptr) {
        return false;
    }

    APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor", "Matmul Op %d is %s\n", op.opmagic, op.GetOpcodeStr().c_str());
    APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor", "####### %d data type is %s\n", in->magic, DataType2VectorRegStr(in->tensor->GetDataType()).c_str());

    bool matmulOp = std::find(cubeOps.begin(), cubeOps.end(), op.GetOpcode()) != cubeOps.end();
    bool opsInputInt8 = false;
    if (op.GetIOperands().size() > 0 && op.GetIOperands()[0] != nullptr && op.GetIOperands()[0]->tensor != nullptr) {
        opsInputInt8 = op.GetIOperands()[0]->tensor->GetDataType() == DataType::DT_INT8;
    }

    if (in->tensor->GetDataType() == DataType::DT_INT8 ||
        (matmulOp && opsInputInt8)) {
        // 检查op的输入数据类型是不是int8类型或者in的数据类型是否为int8
        // 包括matmul系列和GM->L1->L0系列
        return true;
    }

    if (op.GetOpcode() == Opcode::OP_COPY_OUT || op.GetOpcode() == Opcode::OP_L0C_TO_L1) {
        Operation *inProducerPtr = *in->GetProducers().begin();
        if (inProducerPtr != nullptr && inProducerPtr->GetIOperands().size() != 0 &&
            inProducerPtr->GetIOperands()[0] != nullptr && inProducerPtr->GetIOperands()[0]->tensor != nullptr) {
            // 检查in的前置op节点的输入是否为int8。
            // iOperands (dtype:int8) --> A_MULACC_B --> in (dtype:fp16/int32), iOperands (dtype:fp16/int32) --> COPY_OUT
            return inProducerPtr->GetIOperands()[0]->tensor->GetDataType() == DataType::DT_INT8;
        }
    }
    return false;
}

void PadLocalBuffer::PadMatmul(Operation &op, LogicalTensorPtr &in) {
    if (in == nullptr || in->tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "logical tensor pointer is null.");
        return;
    }
    if (in->shape.size() < MATMUL_MIN_SHAPE_SIZE) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Matmul Op %d %s input %d shape size is less than 2; Please check the input size. %s", op.opmagic, op.GetOpcodeStr().c_str(), in->magic, GetFormatBacktrace(op).c_str());
        return;
    }

    auto highIndex = in->shape.size() - 2; // matmul高轴
    auto lowIndex = in->shape.size() - 1;  // matmul低轴
    const auto &producers = in->GetProducers();
    const auto &consumers = in->GetConsumers();
    const bool isL1ConvertScene = !producers.empty() && !consumers.empty() && *producers.begin() != nullptr && *consumers.begin() != nullptr
        && ((*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE || (*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT
            || (*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT || (*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE);
    const bool isInt8Input = IsInputInt8(op, in);
    /*
    首先，可以通过in的数据类型是否为int8来判断是否要做32B对齐。
    再者，存在两种情况
    第一种：in (dtype:fp16/int32) -> iOperands (dtype:int8) -> Matmul系列(A_MUL_B, AT_MUL_B, A_MUL_BT, AT_MUL_BT)
    这种情况需要通过op.GetIOperands来判断输入是否为int8。
    第二种：iOperands (dtype:int8) -> Matmul系列(A_MUL_B, AT_MUL_B, A_MUL_BT, AT_MUL_BT, A_MULACC_B) -> in (dtype:fp16/int32) -> iOperands (dtype:fp16/int32) -> COPY_OUT
    这种情况是COPY_OUT需要根据in的producer的iOperands来进行判断，所以会需要获取到in的producer的iOperands的数据类型。
    */
    if (isL1ConvertScene) {
        /*
        输入带bias或fixpipe场景，切分tileShape为[1, N]，在L1_TO_BT和L1_TO_FIX_QUANT_PRE时，BT统一为FP32，BT BUFFER要求64B对齐，FixPipe为uint64，FB BUFFER为128B对齐，均要求N满足16元素对齐，否则会出现address misalign异常
        示例场景：biasShape = [1, 3208]，tileShapeN = [32, 128]，3208 % 32 = 8尾块非对齐
        另外，bias或fixpipe场景只做低维16元素对齐，高维保持不变
        Before:
        L1_TO_L0A --> L0A (shape:[24, 400]) ------------------------->    \
        L1_TO_BT --> bias_BT (shape:[1, 8]) -----> (address misalign)  A_MUL_B 
        L1_TO_L0B --> L0B (shape:[400, 16]) ------------------------->    /

        After:
        L1_TO_L0A --> L0A (shape:[32, 400])  -->   \
        L1_TO_BT --> bias_BT (shape:[1, 16]) --> A_MUL_B --> output(shape:[32, 16])
        L1_TO_L0B --> L0B (shape:[400, 16])  -->   /
        */
        in->shape[lowIndex] = Pad(in->shape[lowIndex], CUBE_PAD_VALUE);
    } else if (isInt8Input) {
        in->shape[highIndex] = Pad(in->shape[highIndex], CUBE_PAD_INT8_VALUE);
        in->shape[lowIndex] = Pad(in->shape[lowIndex], CUBE_PAD_INT8_VALUE);
    } else {
        in->shape[highIndex] = Pad(in->shape[highIndex], CUBE_PAD_VALUE);
        in->shape[lowIndex] = Pad(in->shape[lowIndex], CUBE_PAD_VALUE);
    }

    APASS_LOG_DEBUG_F(Elements::Tensor, "####### %d original shape is %s\n", in->magic, IntVecToStr(in->oriShape).c_str());
    APASS_LOG_DEBUG_F(Elements::Tensor, "####### %d #current shape is %s\n", in->magic, IntVecToStr(in->shape).c_str());
    if (in->tensor->rawshape.size() < MATMUL_MIN_SHAPE_SIZE) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Matmul Op %d %s input %d raw shape size is less than 2; Please check the input size.", op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }
    in->tensor->oriRawshape = in->tensor->rawshape;

    if (isL1ConvertScene) {
        in->tensor->rawshape[lowIndex] = Pad(in->tensor->oriRawshape[lowIndex], CUBE_PAD_VALUE);
    } else if (isInt8Input) {
        in->tensor->rawshape[highIndex] = Pad(in->tensor->oriRawshape[highIndex], CUBE_PAD_INT8_VALUE);
        in->tensor->rawshape[lowIndex] = Pad(in->tensor->oriRawshape[lowIndex], CUBE_PAD_INT8_VALUE);
    } else {
        in->tensor->rawshape[highIndex] = Pad(in->tensor->oriRawshape[highIndex], CUBE_PAD_VALUE);
        in->tensor->rawshape[lowIndex] = Pad(in->tensor->oriRawshape[lowIndex], CUBE_PAD_VALUE);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "####### %d %d set rawshape as %s\n", in->tensor->rawmagic, in->magic,
        IntVecToStr(in->tensor->rawshape).c_str());
}

size_t PadLocalBuffer::GetPaddingValue(LogicalTensorPtr &in) {
    auto bytes = BytesOf(in->Datatype());
    auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
    if (paddingIter == BLOCK_PADDING_DIM.end()) {
        return 1;
    }
    return paddingIter->second;
}

/* 1. 对于非BroadcastOp，默认做到Block对齐；
   2. 如果已经对齐到Block粒度，不做对齐---这里存在一个问题就是f16和fp32混用场景，可能对齐到一个block是不够的
   3. 对于broadcast op，如果shape小于一个Block的大小对齐到Block，否则做到两个输入之间的较大者的Block对齐。 */
void PadLocalBuffer::PadVector(Operation &op, LogicalTensorPtr &in, std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw,
    bool noPadding) {
    if (in->shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Vector Op %d %s input %d shape size is less than 2; Please check the input size. %s", op.opmagic, op.GetOpcodeStr().c_str(), in->magic, GetFormatBacktrace(op).c_str());
        return;
    }
    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    size_t paddingValue = GetPaddingValue(in);
    size_t lastIdx = in->shape.size() - 1;
    if (noPadding) {
        in->oriShape = in->shape;
        in->tensor->UpdateRawShape(in->shape);
        in->tensor->oriRawshape = in->tensor->rawshape;
        if (forceCombineAxis && paddingValue > 0 && in->tensor->rawshape[lastIdx - 1] % paddingValue != 0) {
            int64_t shapeAfterPad = Pad(in->tensor->rawshape[lastIdx - 1], paddingValue);
            in->tensor->rawshape[lastIdx - 1] = shapeAfterPad;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "Vector Op %d %s input %d, not handle unalign.", op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }
    in->oriShape = in->shape;
    int64_t lastDim = static_cast<int64_t>(in->shape[lastIdx]);
    if (calcType == OpCalcType::BROADCAST && broadcastLastAxis_.find(op.opmagic) != broadcastLastAxis_.end()) {
        lastDim = broadcastLastAxis_[op.opmagic];
    }
    int64_t shapeAfterPad = Pad(lastDim, paddingValue);
    in->shape[lastIdx] = shapeAfterPad;
    if (in->shape[lastIdx] != in->oriShape[lastIdx]) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s input has been changed\n", op.opmagic, op.GetOpcodeStr().c_str());
    }

    if (visitedRaw.count(in->tensor) == 0) {
        in->tensor->oriRawshape = in->tensor->rawshape;
        // shape已经对齐过，直接将rawShape对齐到shape；如果broadcast的输入是来自于view，那么整个链路上的非对齐shape都要按照
        // BROADCAST_LAST_AXIS来对齐，当前这样处理是有问题的
        in->tensor->rawshape[lastIdx] = Pad(in->tensor->oriRawshape[lastIdx], in->shape[lastIdx]);
        visitedRaw.emplace(in->tensor);
    }
}

bool PadLocalBuffer::IsExpandLastDim(const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
    if (axis == static_cast<int>(op.GetOOperands()[0]->shape.size() - 1)) {
        return true;
    }
    return false;
}

void PadLocalBuffer::TraverseCopyInConsumers(Function &function, Operation &consumer, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    bool allBrodOrElem = true;
    for (const auto &nextConsumer : function.FindConsumers(consumer)) {
        auto nextCalcType = OpcodeManager::Inst().GetOpCalcType(nextConsumer->GetOpcode());
        if (nextCalcType != OpCalcType::ELMWISE && nextCalcType != OpCalcType::BROADCAST) {
            allBrodOrElem = false;
            break;
        }
    }
    if (allBrodOrElem) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s consumers are all broadcast or elmwise op, output's last dim should not be padded.", consumer.opmagic, consumer.GetOpcodeStr().c_str());
        consumer.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
        TraverseAndSetAttr(consumer.GetOOperands()[0], function, visitedTensors);
    }
}

void PadLocalBuffer::TraverseBroadcast(Function &function, Operation &consumer, LogicalTensorPtr output, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    std::vector<bool> broadcastInputCombined(consumer.GetIOperands().size(), false);
    consumer.GetAttr(OpAttributeKey::inputCombineAxis, broadcastInputCombined);
    for (size_t index = 0; index < consumer.GetIOperands().size(); ++index) {
        if (consumer.GetIOperands()[index] == output) {
            broadcastInputCombined[index] = true;
        }
    }
    if (broadcastInputCombined.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
            "cannot find tensor %d in input of op %d %s; Please check the input tensor.", output->magic,
            consumer.opmagic, consumer.GetOpcodeStr().c_str());
        return;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s input's last dim should not be padded.", consumer.opmagic, consumer.GetOpcodeStr().c_str());
    consumer.SetAttr(OpAttributeKey::inputCombineAxis, std::move(broadcastInputCombined));
    if (broadcastInputCombined == BROADCAST_AXIS_COMBINED) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "op %d %s output's last dim should not be padded.", consumer.opmagic, consumer.GetOpcodeStr().c_str());
        consumer.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
        TraverseAndSetAttr(consumer.GetOOperands()[0], function, visitedTensors);
    }
}

void PadLocalBuffer::TraverseAndSetAttr(LogicalTensorPtr &output, Function &function, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    if (visitedTensors.count(output) != 0) {
        return;
    }
    visitedTensors.emplace(output);
    for (auto &consumer : output->GetConsumers()) {
        auto consCalcType = OpcodeManager::Inst().GetOpCalcType(consumer->GetOpcode());
        auto opcode = consumer->GetOpcode();
        // expandop为不padding的终止节点
        if ((opcode == Opcode::OP_EXPAND) && IsExpandLastDim(*consumer)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "expand op %d %s input's last dim should not be padded.", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
            continue;
        }
        if ((consCalcType == OpCalcType::ELMWISE) || (opcode == Opcode::OP_ASSEMBLE)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s is elmwise or assemble op, input's and output's last dim should not be padded.", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
            consumer->SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
            TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
            continue;
        }
        if (opcode == Opcode::OP_COPY_OUT) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s is copy out, input's last dim should not be padded.", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
            TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
            continue;
        }
        if (opcode == Opcode::OP_COPY_IN) {
            // CopyIn只有在CopyIn的后续算子仍然是VectorOp的情况下才做合轴
            TraverseCopyInConsumers(function, *consumer, visitedTensors);
            continue;
        }
        if (consCalcType == OpCalcType::BROADCAST) {
            TraverseBroadcast(function, *consumer, output, visitedTensors);
            continue;
        }
        if (consCalcType == OpCalcType::MOVE_OUT) {
            // 剩下来的move out，直接中断，仅在输入的地方支持尾轴非对齐
            APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s is move out, input's last dim should not be padded.", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
        }
    }
}

bool PadLocalBuffer::IsReduceLastDim(const Operation &op) {
    // 目前仅支持四类reduce，另外两个无reduceaxis属性，待确认
    if ((op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) ||
        (op.GetOpcode() == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE)) {
        int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
        if (axis == static_cast<int>(op.GetOOperands()[0]->shape.size() - 1)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s is reduce last dim\n", op.opmagic, op.GetOpcodeStr().c_str());
            return true;
        }
    }
    return false;
}

/*
z0必须32B对齐,尾轴reduce
      reduce
      [z0, 1]<-------->[z0, 1]  [z0, z1]
     /        \            \      /
  elementwise  copy_out    add_brc(break)
  [z0, 1]       [z0, 1]    [z0, pad(z1)]
    |              |
  expand(break)  copy_in
  [z0, pad(z1)]  [z0, 1]
                   |
                elementwise
                 [z0, 1]
*/
void PadLocalBuffer::ProcessReduce(Function &function, Operation &op) {
    // 轴的数量必须大于等于2， 并且倒数第二根轴为32B对齐， 否则无法命中优化pattern
    if ((op.GetOOperands()[0]->shape.size() >= AXIS_COMBINE_MIN_SHAPE_SIZE)) {
        auto out_bytes = BytesOf(op.oOperand[0]->Datatype());
        int paddingDim = 1;
        auto paddingIter = BLOCK_PADDING_DIM.find(out_bytes);
        if (paddingIter != BLOCK_PADDING_DIM.end()) {
            paddingDim = paddingIter->second;
        }
        if (!forceCombineAxis && paddingDim > 0 && op.oOperand[0]->shape[op.GetOOperands()[0]->shape.size() - AXIS_COMBINE_MIN_SHAPE_SIZE] % paddingDim != 0) {
            return;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "op %d %s is reduce, next to last dim is aligned\n", op.opmagic, op.GetOpcodeStr().c_str());
        std::vector<bool> reduceAxisCombined(op.GetOOperands().size(), false);
        reduceAxisCombined[0] = true;
        op.SetAttr(OpAttributeKey::outputCombineAxis, reduceAxisCombined);
        std::unordered_set<LogicalTensorPtr> visitedTensors;
        // dfs遍历打上input/output不做padding的相关属性
        TraverseAndSetAttr(op.GetOOperands()[0], function, visitedTensors);
    }
}

void PadLocalBuffer::ProcessBroadcast(Operation &op, size_t blockPadding) {
    int64_t maxLastAxis = 0;
    bool existLessBlock = false;
    for (const auto &in : op.iOperand) {
        if (in->shape.back() <= static_cast<int>(blockPadding)) {
            existLessBlock = true;
        }
        maxLastAxis = std::max(maxLastAxis, in->shape.back());
    }
    if (!existLessBlock) {
        broadcastLastAxis_[op.opmagic] = maxLastAxis;
    }
}


void PadLocalBuffer::ProcessCopyIn(Function &function, Operation &op) {
    // 轴的数量必须大于等于2，并且倒数第二根轴为32B对齐，否则无法命中pattern
    std::vector<bool> axisCombined(op.GetOOperands().size(), false);
    axisCombined[0] = true;
    op.SetAttr(OpAttributeKey::outputCombineAxis, axisCombined);
    std::unordered_set<LogicalTensorPtr> visitedTensors;
    // dfs遍历打上input/output不做padding的相关属性
    TraverseAndSetAttr(op.GetOOperands()[0], function, visitedTensors);
}

bool PadLocalBuffer::IsMatmul(const LogicalTensorPtr &tensor) const {
    if ((tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_FIX_QUANT_PRE) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_BT)) {
        return true;
    }
    return false;
}

bool PadLocalBuffer::IsVector(const LogicalTensorPtr &tensor) {
    if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        return true;
    }
    return false;
}

void PadLocalBuffer::DoPadding(Function &function) {
    std::unordered_set<std::shared_ptr<LogicalTensor>> visited;
    std::unordered_set<std::shared_ptr<RawTensor>> visitedRaw;
    for (auto &op : function.Operations()) {
        std::vector<bool> inputAxis;
        op.GetAttr(OpAttributeKey::inputCombineAxis, inputAxis);
        for (size_t i = 0; i < op.iOperand.size(); i++) {
            auto &in = op.iOperand[i];
            if (visited.count(in) != 0) {
                continue;
            }
            visited.emplace(in);
            if (IsMatmul(in)) {
                PadMatmul(op, in);
                continue;
            }
            if (!IsVector(in)) {
                continue;
            }
            if (in->tensor->GetRawDataSize() == 0) {
                continue;
            }
            if (function.paramConfigs_.combineAxis) {
                PadVectorForAxisCombine(op, in, visitedRaw);
            } else {
                bool noPadding = false;
                if ((inputAxis.size() > i) && inputAxis[i]) {
                    noPadding = true;
                }
                PadVector(op, in, visitedRaw, noPadding);
            }
        }
    }
}

// 对ub上transpose的特殊处理,其他类型的transpose不做处理
Status PadLocalBuffer::ProcessTranspose(Function &function) {
    for (const auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_TRANSPOSE_VNCHWCONV || op.GetIOperands()[0]->shape.size() < TRANSPOSE_MIN_SHAPE_SIZE) {
            continue;
        }
        auto transposeAxis = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
        if (transposeAxis.size() != TRANSPOSE_MIN_SHAPE_SIZE) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "transpose op %d %s's shape size %d is not two, skip.", op.opmagic, op.GetOpcodeStr().c_str(), transposeAxis.size());
            continue;
        }
        if (op.iOperand.size() <= 0 || op.oOperand.size() <= 0) {
            APASS_LOG_ERROR_F(Elements::Operation, "transpose op %d %s's input or output is empty. %s", op.opmagic, op.GetOpcodeStr().c_str(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto &inTensor = op.iOperand[0];
        auto &outTensor = op.oOperand[0];
        if (transposeAxis[0] == transposeAxis[1]) {
            APASS_LOG_ERROR_F(Elements::Tensor, "transpose op has the same transpose dims, not supported");
            return FAILED;
        }
        if ((transposeAxis[0] != static_cast<int32_t>(inTensor->shape.size() - 1)) && (transposeAxis[1] != static_cast<int32_t>(inTensor->shape.size() - 1))) {
            APASS_LOG_DEBUG_F(Elements::Tensor, "transpose op %d %s's transpose axis %d %d, not last dim transpose, skip.", op.opmagic, op.GetOpcodeStr().c_str(), transposeAxis[0], transposeAxis[1]);
            continue;
        }
        int32_t nonLastDimIdx = -1;
        int32_t lastDimIdx = inTensor->shape.size() - 1;
        if (transposeAxis[0] != static_cast<int32_t>(inTensor->shape.size() - 1)) {
            nonLastDimIdx = transposeAxis[0];
        } else {
            nonLastDimIdx = transposeAxis[1];
        }
        auto &inLastDim = inTensor->shape[lastDimIdx];
        auto &outFirstDim = outTensor->shape[nonLastDimIdx];
        if (inLastDim != outFirstDim) {
            APASS_LOG_DEBUG_F(Elements::Operation, "tune transpose output dim %d to %d.", inLastDim, outFirstDim);
            outTensor->shape[nonLastDimIdx] = inLastDim;
            outTensor->tensor->rawshape[nonLastDimIdx] = inLastDim;
        }
    }
    return SUCCESS;
}

inline bool IsCopyIn(Operation& op) {
    if (op.GetOpcode() != Opcode::OP_COPY_IN) {
        return false;
    }
    auto &outputTensor = op.GetOOperands()[0];
    if (outputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
        return false;
    }
    auto shape = outputTensor->GetShape();
    if ((shape.size() < AXIS_COMBINE_MIN_SHAPE_SIZE) || shape.back() != 1) {
        return false;
    }
    return true;
}

int64_t PadLocalBuffer::ProcessBroadcastForAxisCombine(LogicalTensorPtr &inTensor) {
    int dimSize = inTensor->GetShape().size();
    if (inTensor->shape.back() != 1) {
        return (dimSize - 1);
    }
    if (dimSize > 1) {
        return (dimSize - LAST_SECOND_AXIS);
    }
    return (dimSize - 1);
}

int64_t AlignedRawTensorIfNeed(LogicalTensorPtr &in, int64_t pos, const int64_t base) {
    if (in == nullptr || pos < 0 || pos >= static_cast<int64_t>(in->tensor->rawshape.size())) {
        return -1;
    }
    int64_t padDim = Pad(in->tensor->rawshape[pos], base);
    in->tensor->rawshape[pos] = padDim;
    return padDim;
}

void ProcessReduceForAxisCombine(Operation &op, LogicalTensorPtr &in, size_t paddingValue) {
    auto axis = op.GetIntAttribute(REDUCE_AXIS);
    int64_t shapeSize = static_cast<int64_t>(in->shape.size());
    int64_t lastIdx = shapeSize - 1;
    if (shapeSize == 1 || axis == shapeSize - 2) {
        AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
        return;
    }
    int64_t idx = lastIdx;
    bool isFound = false;
    for (; idx >= 0; --idx) {
        if (in->shape[idx] != 1) {
            isFound = true;
            break;
        }
    }
    if (!isFound) {
        idx = lastIdx;
    }
    int64_t padDim = AlignedRawTensorIfNeed(in, idx, paddingValue);
    if (op.GetOpcode() == Opcode::OP_ROWSUMLINE) {
        auto tempBuffer = op.GetOOperands()[1];
        size_t tempBufferLastIdx = tempBuffer->shape.size() - 1;
        tempBuffer->shape[tempBufferLastIdx] = padDim;
        tempBuffer->GetRawTensor()->rawshape[tempBufferLastIdx] = padDim;
    }
}

void PadLocalBuffer::PadVectorForAxisCombine(Operation &op, LogicalTensorPtr &in, std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw) {
    if (in->shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Vector Op %d %s input %d shape size is less than 2; Please check the input size. %s", op.opmagic, op.GetOpcodeStr().c_str(), in->magic, GetFormatBacktrace(op).c_str());
        return;
    }
    if (visitedRaw.count(in->tensor)) {
        return;
    }
    visitedRaw.emplace(in->tensor);
    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    size_t paddingValue = GetPaddingValue(in);
    size_t lastIdx = in->shape.size() - 1;
    in->oriShape = in->shape;
    in->tensor->oriRawshape = in->tensor->rawshape;
    auto producerOp = *(in->GetProducers().begin());
    if (producerOp != nullptr && producerOp->GetOpcode() == Opcode::OP_BRCB) {
        if (lastIdx == 0 && in->tensor->rawshape[lastIdx] != 1) {
            return;
        }
        AlignedRawTensorIfNeed(in, lastIdx - 1, BRCB_SECOND_LAST_BASE);
    }
    if (calcType == OpCalcType::REDUCE) {
        ProcessReduceForAxisCombine(op, in, paddingValue);
        return;
    }
    if (op.GetOpcode() == Opcode::OP_BRCB) {
        if (lastIdx == 0 && in->tensor->rawshape[lastIdx] != 1) {
            return;
        }
        AlignedRawTensorIfNeed(in, lastIdx - 1, BRCB_SECOND_LAST_BASE);
        for (auto &out : op.GetOOperands()) {
            AlignedRawTensorIfNeed(out, lastIdx - 1, BRCB_SECOND_LAST_BASE);
            AlignedRawTensorIfNeed(out, lastIdx, paddingValue);
            visitedRaw.emplace(out->tensor);
        }
    }
    if (calcType == OpCalcType::BROADCAST) {
        auto dimIdx = ProcessBroadcastForAxisCombine(in);
        AlignedRawTensorIfNeed(in, dimIdx, paddingValue);
        return;
    }
    if (calcType == OpCalcType::ELMWISE || calcType == OpCalcType::MOVE_IN || calcType == OpCalcType::MOVE_OUT ||
            (producerOp != nullptr && OpcodeManager::Inst().GetOpCalcType(producerOp->GetOpcode()) == OpCalcType::BROADCAST)) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
            return;
        }
        if (lastIdx > 0 && in->tensor->rawshape[lastIdx] == 1) {
            AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
            return;
        }
    }
    AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
}

Status PadLocalBuffer::RunOnFunction(Function &function) {
    combineAxis = function.paramConfigs_.combineAxis;
    forceCombineAxis = function.paramConfigs_.forceCombineAxis;
    if (combineAxis) {
        APASS_LOG_INFO_F(Elements::Operation, "======> Start PadLocalBuffer in COMBINE_AXIS mode.");
        DoPadding(function);
        APASS_LOG_INFO_F(Elements::Operation, "======> End PadLocalBuffer in COMBINE_AXIS mode.");
        return SUCCESS;
    }
    for (auto &op : function.Operations()) {
        auto calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
        // 尾轴Reduce且倒数第二根轴32B对齐的op起始的链路上的op不做padding，以节省UB空间
        if (IsReduceLastDim(op)) {
            ProcessReduce(function, op);
        }

        if (forceCombineAxis && IsCopyIn(op)) {
            ProcessCopyIn(function, op);
        }

        // Broadcast op设置最后一根轴的padding值
        if (calcType == OpCalcType::BROADCAST) {
            auto bytes = BytesOf(op.iOperand[0]->Datatype());
            auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
            if (paddingIter == BLOCK_PADDING_DIM.end()) {
                APASS_LOG_DEBUG_F(Elements::Operation, "broadcast op %d %s's datatype is not supported.", op.opmagic, op.GetOpcodeStr().c_str());
                continue;
            }
            ProcessBroadcast(op, paddingIter->second);
        }
    }
    DoPadding(function);
    if (processTranspose_) {
        if (ProcessTranspose(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessTranspose failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace
