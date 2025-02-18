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
 * \file split_large_fanout_tensor.cpp
 * \brief
 */

#include "split_large_fanout_tensor.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SplitLargeFanoutTensor"

namespace npu::tile_fwk {
Status SplitLargeFanoutTensor::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start SplitLargeFanoutTensor.");
    CollectLargeTensor(function);
    SplitLargeTensor(function);
    EraseRedundantAssembleOp(function);
    EraseRedundantViewOp(function);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate dead operation failed "
            "in general DeadOperation Eliminator; Please check abnormal unused operations and error messages (if any) above.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End SplitLargeFanoutTensor.");
    return SUCCESS;
}

// 求最大公约数
int64_t SplitLargeFanoutTensor::GCD(int64_t x, int64_t y) {
    int temp = 0;
    while (y != 0) {
        temp = x;
        x = y;
        y = temp % y;
    }
    return x;
}
// 求最小公倍数
Status SplitLargeFanoutTensor::LCM(int64_t x, int64_t y, int64_t &lcm) {
    auto gcd = GCD(x, y);
    if (gcd == 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "gcd is 0; gcd can't be 0.");
        return FAILED;
    } else {
        lcm = x * y / gcd;
        return SUCCESS;
    }
}

// 求两个shape的最小公倍数shape
Status SplitLargeFanoutTensor::CalLcmShape(const Shape &toShape, const Shape &fromShape, Shape &lcmShape) {
    if (toShape.size() != fromShape.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect shapes dim, toShape dim is %d, fromShape dim is %d; "
            "Please make sure they are the same.", toShape.size(), fromShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < toShape.size(); i++) {
        if(LCM(toShape[i], fromShape[i], lcmShape[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Shape's dim %d, %d and %d cal LCM failed; "
                "LCM is calculated to be zero, please check.", i, toShape[i], fromShape[i]);
            return FAILED;
        } else {
            APASS_LOG_INFO_F(Elements::Tensor, "Shape's dim %d, shape: %d and %d, LCM is %d.",
                i, toShape[i], fromShape[i], lcmShape[i]);
        }
    }
    return SUCCESS;
}

// 求两个shape的最大公约数shape
Status SplitLargeFanoutTensor::CalGcdShape(const Shape &toShape, const Shape &fromShape, Shape &lcmShape) {
    if (toShape.size() != fromShape.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Incorrect shapes dim, toShape dim is %d, fromShape dim is %d.",
            toShape.size(), fromShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < toShape.size(); i++) {
        lcmShape[i] = GCD(toShape[i], fromShape[i]);
        APASS_LOG_INFO_F(Elements::Tensor, "Shape's dim is %d, toShape is %d, fromShape is %d, GCD is %d.",
            i, toShape[i], fromShape[i], lcmShape[i]);
    }
    return SUCCESS;
}

// 递归函数, 根据maxsShape和stepsShape生成offset
void SplitLargeFanoutTensor::GenerateOffset(const Shape &maxs, const Shape &steps, Shape &current, std::vector<Shape> &result, size_t dim) {
    if (dim == maxs.size()) {
        // 生成一个offset
        Shape offset;
        for (size_t i = 0; i < current.size(); ++i) {
            offset.push_back(current[i]);
        }
        result.push_back(offset);
        return;
    }
    for (int val = 0; val < maxs[dim]; val += steps[dim]) {
        current[dim] = val;
        GenerateOffset(maxs, steps, current, result, dim + 1);
    }
}

// 收集BE_COVERED/PERFECTLY_MATCH lcmTile 的那些tile们, 分别更新到overlaps和dualOverlaps
void SplitLargeFanoutTensor::CollectOverlaps(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
    std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos,
    std::vector<std::pair<LogicalTensorPtr, Offset>> fromTensorInfos,
    LogicalTensors &overlaps, LogicalTensors &dualOverlaps) {
    // 创建一块lcmTile的镜像, 即: 与lcmTile的shape、offset相同, 但rawtensor是LargeTensor的一块等效tensor
    auto lcmTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, lcmTileOffset, lcmTileShape);
    for (const auto &toTensorInfo : toTensorInfos) {
        // 创建一块toTile的镜像, 即: 与toTile的shape、offset相同, 但rawtensor是LargeTensor的一块等效tensor
        auto toTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, toTensorInfo.second, toTensorInfo.first->shape);
        auto status = CalcOverlap(toTile, lcmTile, true);
        if (status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH) {
            overlaps.push_back(toTensorInfo.first);
        }
    }
    for (const auto &fromTensorInfo : fromTensorInfos) {
        // 创建一块fromTile的镜像, 即: 与fromTile的shape、offset相同, 但rawtensor是LargeTensor的一块等效tensor
        auto fromTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, fromTensorInfo.second, fromTensorInfo.first->shape);
        auto status = CalcOverlap(fromTile, lcmTile, true);
        if (status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH) {
            dualOverlaps.push_back(fromTensorInfo.first);
        }
    }
}

// 对于一对一、一对多场景创建新的AssembleOp和Tensor
void SplitLargeFanoutTensor::CreateOpFor1toM(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
    LogicalTensors overlaps, LogicalTensors dualOverlaps) {
    for (const auto &dualOverlap : dualOverlaps) {
        auto viewOp = *dualOverlap->GetProducers().begin();
        if (viewOp->GetIOperands().front()->tensor->rawmagic != largeTensor->tensor->rawmagic) {
            APASS_LOG_INFO_F(Elements::Tensor, "ViewOp[%d]'s input has been replaced, don't deal with this ViewOp.",
                viewOp->GetOpMagic());
        } else {
            auto newTensor = std::make_shared<LogicalTensor>(function, largeTensor->Datatype(),
                lcmTileShape, largeTensor->Format());
            auto overlap = overlaps[0];
            auto oldAssembleOp = *overlap->GetConsumers().begin();
            auto oldAssembleOpAttr = dynamic_cast<AssembleOpAttribute *>(oldAssembleOp->GetOpAttribute().get());
            Shape newAssembleOffset = oldAssembleOpAttr->GetToOffset();
            for (size_t j = 0; j < newAssembleOffset.size(); j++) {
                newAssembleOffset[j] -= lcmTileOffset[j];
            }
            auto newAssembleOp = AssembleOp{overlap->GetMemoryTypeOriginal(), newAssembleOffset, overlap, newTensor};
            GraphUtils::AddAssembleOperation(function, newAssembleOp);
            auto assembleOp = *newTensor->GetProducers().begin();
            APASS_LOG_INFO_F(Elements::Operation, "In one-to-multiple situation, create an AssembleOp[%d], input is a "
                "overlap[%d], output is a newTensor[%d].", assembleOp->GetOpMagic(), overlap->GetMagic(), newTensor->GetMagic());
            auto viewOpAttr = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
            Shape newViewOffset = viewOpAttr->GetFromOffset();
            for (size_t j = 0; j < newViewOffset.size(); j++) {
                newViewOffset[j] -= lcmTileOffset[j];
            }
            viewOpAttr->SetFromOffset(newViewOffset);
            GraphUtils::UpdateViewAttr(function, *viewOp);
            viewOp->ReplaceInput(newTensor, largeTensor);
            APASS_LOG_INFO_F(Elements::Operation, "In one-to-multiple situation, "
                "viewOp[%d]'s input[%d] has been replaced to newTensor[%d].", viewOp->GetOpMagic(), largeTensor->GetMagic(), newTensor->GetMagic());
        }
    }
}

// 对于多对一、多对多场景创建新的AssembleOp和Tensor
void SplitLargeFanoutTensor::CreateOpForMtoM(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
    LogicalTensors overlaps, LogicalTensors dualOverlaps) {
    auto newTensor = std::make_shared<LogicalTensor>(function, largeTensor->Datatype(),
        lcmTileShape, largeTensor->Format());
    for (const auto &overlap : overlaps) {
        // 由于tensor->assemble->largeTensor中assemble可以不唯一并指向其他tensor，或assemble位置为其他种类op(op_view)。所以需要找到largeTensor的生产者op
        Operation *oldAssembleOp = nullptr;
        for (const auto &consumerOp : overlap->GetConsumers()) {
            for (auto tensorPtr : consumerOp->GetOOperands()) {
                if (tensorPtr == largeTensor) {
                    oldAssembleOp = consumerOp;
                }
            }
        }
        if (oldAssembleOp == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "No valid assemble op found between tensor[%d] and tensor[%d], skip.",
                overlap->GetMagic(), largeTensor->GetMagic());
            continue;
        }

        auto oldAssembleOpAttr = dynamic_cast<AssembleOpAttribute *>(oldAssembleOp->GetOpAttribute().get());
        Shape newAssembleOffset = oldAssembleOpAttr->GetToOffset();
        for (size_t j = 0; j < newAssembleOffset.size(); j++) {
            newAssembleOffset[j] -= lcmTileOffset[j];
        }
        auto newAssembleOp = AssembleOp{overlap->GetMemoryTypeOriginal(), newAssembleOffset, overlap, newTensor};
        GraphUtils::AddAssembleOperation(function, newAssembleOp);
        auto assembleOp = *newTensor->GetProducers().begin();
        APASS_LOG_INFO_F(Elements::Operation, "In multiple-to-multiple situation, create an AssembleOp[%d], "
            "input is a overlap[%d], output is a newTensor[%d].", assembleOp->GetOpMagic(), overlap->GetMagic(), newTensor->GetMagic());
    }
    for (const auto &dualOverlap : dualOverlaps) {
        auto viewOp = *dualOverlap->GetProducers().begin();
        if (viewOp->GetIOperands().front()->tensor->rawmagic != largeTensor->tensor->rawmagic) {
            APASS_LOG_INFO_F(Elements::Operation, "ViewOp[%d]'s input has been replaced, don't deal with ViewOp.",
                viewOp->GetOpMagic());
        } else {
            auto viewOpAttr = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
            Shape newViewOffset = viewOpAttr->GetFromOffset();
            for (size_t j = 0; j < newViewOffset.size(); j++) {
                newViewOffset[j] -= lcmTileOffset[j];
            }
            viewOpAttr->SetFromOffset(newViewOffset);
            GraphUtils::UpdateViewAttr(function, *viewOp);
            viewOp->ReplaceInput(newTensor, largeTensor);
            APASS_LOG_INFO_F(Elements::Operation, "In multiple-to-multiple situation, viewOp[%d]'s input[%d] has been "
                "replaced to newTensor[%d].", viewOp->GetOpMagic(), largeTensor->GetMagic(), newTensor->GetMagic());
        }
    }
    // 进一步拆分, 未来通过旋钮的方式适时打开
    if (enableMoreSplit) {
        MoreSplit(function, largeTensor, overlaps, dualOverlaps);
    }
}

void SplitLargeFanoutTensor::MoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps, LogicalTensors dualOverlaps) {
    for (const auto &dualOverlap : dualOverlaps) {
        // 如果该dualOverlap已经被进一步拆分, 跳过(进一步拆分的特征是dualOverlap的生产者全是Assemble)
        bool isMoreSplit = true;
        for (const auto &producer : dualOverlap->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                isMoreSplit = false;
            }
        }
        if (isMoreSplit) {
            continue;
        }
        // 否则, dualOverlap的生产者全是View, 不存在其他场景
        auto toShape = overlaps.front()->shape;
        auto fromShape = dualOverlap->shape;
        Shape gcdShape(toShape.size(), 0);
        CalGcdShape(toShape, fromShape, gcdShape);
        std::vector<Shape> gcdTileOffsets;
        Shape current(gcdShape.size());
        GenerateOffset(dualOverlap->shape, gcdShape, current, gcdTileOffsets, 0);
        auto viewOp = *dualOverlap->GetProducers().begin();
        auto opAttr = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
        auto viewOpOffset = opAttr->GetFromOffset();
        // 断开viewOp--> tensor: 将tensor的生产者删除viewOp, 将viewOp的输出删除tensor
        dualOverlap->RemoveProducer(viewOp);
        viewOp->GetOOperands().erase(viewOp->GetOOperands().begin(), viewOp->GetOOperands().end());
        auto fromTensorInfos = fromInfoMap[largeTensor->tensor->rawmagic];
        for (const auto &fromTensorInfo : fromTensorInfos) {
            if (dualOverlap == fromTensorInfo.first) {
                viewOpOffset = fromTensorInfo.second;
            }
        }
        CreateOpForMoreSplit(function, largeTensor, overlaps, gcdShape, dualOverlap, gcdTileOffsets, viewOpOffset);
    }
}

void SplitLargeFanoutTensor::CreateOpForMoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps, Shape gcdShape, LogicalTensorPtr dualOverlap, std::vector<Shape> gcdTileOffsets, Offset viewOpOffset) {
    for (auto &gcdTileOffset : gcdTileOffsets) {
        auto newGcdTensor = std::make_shared<LogicalTensor>(function, largeTensor->Datatype(),
            gcdShape, largeTensor->Format());
        auto &newAssembleOp = function.AddOperation(Opcode::OP_ASSEMBLE, {newGcdTensor}, {dualOverlap});
        newAssembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(largeTensor->GetMemoryTypeOriginal(), gcdTileOffset));
        APASS_LOG_INFO_F(Elements::Operation, "For more split situation, create an AssembleOp[%d], input is a newGcdTensor[%d], "
            "output is a dualOverlap[%d].", newAssembleOp.GetOpMagic(), newGcdTensor->GetMagic(), dualOverlap->GetMagic());
        LogicalTensorPtr overlapGcdTile;
        Shape newViewOffset = gcdTileOffset;
        for (size_t j = 0; j < newViewOffset.size(); j++) {
            newViewOffset[j] += viewOpOffset[j];
        }
        Shape gcdTileOffsetForLarge = gcdTileOffset;
        for (size_t j = 0; j < gcdTileOffsetForLarge.size(); j++) {
            gcdTileOffsetForLarge[j] += viewOpOffset[j];
        }
        for (auto &overlap : overlaps) {
            auto gcdTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, gcdTileOffsetForLarge, gcdShape);
            auto oldAssembleOp = *overlap->GetConsumers().begin();
            auto oldopmagic = oldAssembleOp->opmagic;
            for (const auto &consumer : overlap->GetConsumers()) {
                if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    oldAssembleOp = consumer;
                    oldopmagic = oldAssembleOp->opmagic;
                    break;
                }
            }
            for (const auto &consumer : overlap->GetConsumers()) {
                if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE && oldopmagic > consumer->opmagic) {
                    oldAssembleOp = consumer;
                    oldopmagic = consumer->opmagic;
                }
            }
            auto oldAssembleOpAttr = dynamic_cast<AssembleOpAttribute *>(oldAssembleOp->GetOpAttribute().get());
            auto oldAssembleOffset = oldAssembleOpAttr->GetToOffset();
            auto toTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, oldAssembleOffset, overlap->shape);
            auto status = CalcOverlap(gcdTile, toTile, true);
            if (status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH) {
                overlapGcdTile = overlap;
                for (size_t j = 0; j < newViewOffset.size(); j++) {
                    newViewOffset[j] -= oldAssembleOffset[j];
                }
                auto &newViewOp = function.AddOperation(Opcode::OP_VIEW, {overlapGcdTile}, {newGcdTensor});
                newViewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(newViewOffset, overlap->GetMemoryTypeOriginal()));
                APASS_LOG_INFO_F(Elements::Operation, "For more split situation, create an ViewOp[%d], input is a "
                    "overlapGcdTile[%d], output is a newGcdTensor[%d].", newViewOp.GetOpMagic(), overlapGcdTile->GetMagic(), newGcdTensor->GetMagic());
            }
        }
    }
}

void SplitLargeFanoutTensor::CollectLargeTensorToInfo(const LogicalTensorPtr &largeTensor) {
    for (const auto &assembleOp : largeTensor->GetProducers()) {
        // 收集overlaps
        auto input = assembleOp->GetIOperands().front();
        if (toInfoMap.count(largeTensor->tensor->rawmagic) == 0) {
            toInfoMap.insert({largeTensor->tensor->rawmagic, {}});
        }
        auto opAttr = dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get());
        if (opAttr != nullptr) {
            toInfoMap[largeTensor->tensor->rawmagic].emplace_back(input, opAttr->GetToOffset());
        }
        // 收集overlaps的shape
        if (toShapes.count(largeTensor) == 0) {
            toShapes.insert({largeTensor, {}});
        }
        toShapes[largeTensor].insert(input->shape);
    }
}

void SplitLargeFanoutTensor::CollectLargeTensorFromInfo(const LogicalTensorPtr &largeTensor) {
    for (const auto &viewOp : largeTensor->GetConsumers()) {
        // 收集outputs
        auto output = viewOp->GetOOperands().front();
        if (fromInfoMap.count(largeTensor->tensor->rawmagic) == 0) {
            fromInfoMap.insert({largeTensor->tensor->rawmagic, {}});
        }
        auto opAttr = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
        if (opAttr != nullptr) {
            fromInfoMap[largeTensor->tensor->rawmagic].emplace_back(output, opAttr->GetFromOffset());
        }
        // 收集outputs的shape
        if (fromShapes.count(largeTensor) == 0) {
            fromShapes.insert({largeTensor, {}});
        }
        fromShapes[largeTensor].insert(output->shape);
    }
}

// 遍历所有的tensor, 对前序为Assemble后序为View的大Tensor进行拆分
void SplitLargeFanoutTensor::CollectLargeTensor(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "---> CollectLargeTensor.");
    auto &tensorMap = function.GetTensorMap().tensorMap_;
    for (const auto &tMap : tensorMap) {
        for (const auto &logicalTensor : tMap.second) {
            // 对于每个tensor, 寻找满足前序为Assemble且后序为View的LargeTensor
            auto producer = *logicalTensor->GetProducers().begin();
            auto consumer = *logicalTensor->GetConsumers().begin();
            if (producer == nullptr || consumer == nullptr) { break; }
            if (producer->GetOpcode() == Opcode::OP_ASSEMBLE && consumer->GetOpcode() == Opcode::OP_VIEW) {
                // 收集大Tensor, 形成Set{TensorPtr1, TensorPtr2, ...}
                largeTensors.insert(logicalTensor);
                CollectLargeTensorToInfo(logicalTensor);
                CollectLargeTensorFromInfo(logicalTensor);
                APASS_LOG_INFO_F(Elements::Tensor, "Large tensor magic is %d.", logicalTensor->GetMagic());
            }
        }
    }
}

bool SplitLargeFanoutTensor::IsBeCovered(Function &function, LogicalTensorPtr largeTensor,
    std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos) {
    for (const auto &toTensorInfo : toTensorInfos) {
        auto toTile = std::make_shared<LogicalTensor>(function, largeTensor->tensor, toTensorInfo.second, toTensorInfo.first->shape);
        auto status = CalcOverlap(toTile, largeTensor, true);
        if (!(status == OverlapStatus::BE_COVERED || status == OverlapStatus::PERFECTLY_MATCH)) {
            return false;
        }
    }
    return true;
}

bool SplitLargeFanoutTensor::HasDuplicateToTile(std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos) {
    std::map<Offset, int> countMap;
    for (const auto &toTensorInfo : toTensorInfos) {
        countMap[toTensorInfo.second]++;
    }
    for (const auto &pair : countMap) {
        if (pair.second > 1) {
            return true;
        }
    }
    return false;
}

// 遍历所有的大tensor, 对前后不同的tileShape计算lcmShape, 并尝试拆分
void SplitLargeFanoutTensor::SplitLargeTensor(Function &function) {
    for (const auto &largeTensor : largeTensors) {
        std::multiset<Shape, ShapeComparator> lcmShapes;
        // 验证Assemble成LargeTensor的tileTensor们需要包含于LargeTensor
        if (!IsBeCovered(function, largeTensor, toInfoMap[largeTensor->tensor->rawmagic])) {
            continue;
        }
        // 验证Assemble成LargeTensor的tileTensor们(的Offset)需要彼此不同
        if (HasDuplicateToTile(toInfoMap[largeTensor->tensor->rawmagic])) {
            continue;
        }
        for (const auto &toShape : toShapes[largeTensor]) {
            for (const auto &fromShape : fromShapes[largeTensor]) {
                Shape lcmShape(toShape.size(), 0);
                if(CalLcmShape(toShape, fromShape, lcmShape) != SUCCESS) {
                    APASS_LOG_INFO_F(Elements::Tensor, "Calculate LCM shape failed, don't cal LcmShape.");
                    continue;
                }
                // 当lcmTile的每个维度都大于等于largeTensor时, 仍会聚合到同样大小的Tensor, 因此不做处理
                bool unsplit = std::equal(lcmShape.begin(), lcmShape.end(), largeTensor->shape.begin(),
                    [](int lcmDim, int largeTensorDim) { return lcmDim >= largeTensorDim; });
                if (unsplit) {
                    APASS_LOG_INFO_F(Elements::Tensor, "Skip SplitLargeTensor for magic[%d] since shape to assemble (lcmShape) equals "
                        "or is larger than the largeTensor's shape.", largeTensor->GetMagic());
                    continue;
                }
                // 当lcmTile的shape小于largeTensor时, 开始尝试拆分
                APASS_LOG_INFO_F(Elements::Tensor, "Try to split, large tensor magic is %d.", largeTensor->GetMagic());
                lcmShapes.insert(lcmShape);
            }
        }
        for (const auto &lcmShape : lcmShapes) {
            TryToSplitLargeTensor(function, lcmShape, largeTensor);
        }
    }
}

void SplitLargeFanoutTensor::TryToSplitLargeTensor(Function &function, const Shape &lcmShape, const LogicalTensorPtr &largeTensor) {
    std::vector<Shape> lcmTileOffsets;
    Shape current(lcmShape.size());
    GenerateOffset(largeTensor->shape, lcmShape, current, lcmTileOffsets, 0);
    APASS_LOG_INFO_F(Elements::Operation, "LcmTile num: %d.", lcmTileOffsets.size());
    for (const auto &lcmTileOffset : lcmTileOffsets) {
        // 更新实际的lcmTileShape, 仅在尾块时会有变小的情况
        auto lcmTileShape = lcmShape;
        for (size_t i = 0; i < lcmShape.size(); i++) {
            if (lcmTileOffset[i] + lcmTileShape[i] > largeTensor->shape[i]) {
                lcmTileShape[i] = largeTensor->shape[i] - lcmTileOffset[i];
            }
        }
        LogicalTensors overlaps;
        LogicalTensors dualOverlaps;
        CollectOverlaps(function, largeTensor, lcmTileShape, lcmTileOffset, toInfoMap[largeTensor->tensor->rawmagic], fromInfoMap[largeTensor->tensor->rawmagic], overlaps, dualOverlaps);
        if (overlaps.size() == 0 || dualOverlaps.size() == 0) {
            APASS_LOG_INFO_F(Elements::Tensor, "Split large tensor miss, this lcmTile does NOT have both overlaps([%d]) "
                "and dualOverlaps([%d]) simultaneously.", overlaps.size(), dualOverlaps.size());
            continue;
        }
        
        auto multiply = [](const std::vector<int64_t>& vec) -> int64_t {
            return std::accumulate(
                vec.begin(), vec.end(), static_cast<int64_t>(1), [](int64_t a, int64_t b) { return a * b; });
        };
        int64_t overlapTotalArea = 0;
        for (const auto &overlap : overlaps) {
            overlapTotalArea += multiply(overlap->shape);
        }
        if (overlapTotalArea != multiply(lcmShape)) {
            APASS_LOG_INFO_F(Elements::Tensor, "Split large tensor miss, this lcmTile(shape %s, offset %s) of largeTensor %d is not filled up by all collected overlaps.",
                CommonUtils::ContainerToStr(lcmShape).c_str(), CommonUtils::ContainerToStr(lcmTileOffset).c_str(), largeTensor->GetMagic());
            continue;
        }
        APASS_LOG_INFO_F(Elements::Tensor, "Split large tensor hit, this lcmTile has [%d] overlaps and [%d] dualOverlaps.",
            overlaps.size(), dualOverlaps.size());
        // 对于是否有[多个tensor聚合到一个Tensor]的情况进行不同处理
        if (overlaps.size() == 1) {
            CreateOpFor1toM(function, largeTensor, lcmTileShape, lcmTileOffset, overlaps, dualOverlaps);
        } else {
            CreateOpForMtoM(function, largeTensor, lcmTileShape, lcmTileOffset, overlaps, dualOverlaps);
        }
    }
}

void SplitLargeFanoutTensor::RemoveOps(Function &function, std::vector<Operation *> &opList) const {
    for (const auto &op : opList) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (const auto op : opList) {
        APASS_LOG_INFO_F(Elements::Operation, "Remove %s[%d].", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        if (!op->IsDeleted()) {
            op->SetAsDeleted();
        }
    }
    function.EraseOperations(true);
}

void SplitLargeFanoutTensor::UpdateForRedundantAssemble(Operation &op) {
    auto output = op.oOperand.front();
    auto input = op.iOperand.front();
    auto consumersBackup = output->GetConsumers();
    for (const auto &childOp : consumersBackup) {
        childOp->ReplaceInput(input, output);
        if (childOp->GetOpcode() == Opcode::OP_VIEW) {
            auto tensorOffset = input->GetTensorOffset();
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(childOp->GetOpAttribute().get());
            auto viewOffset = viewOpAttribute->GetFromTensorOffset();
            auto newStaticOffset = TensorOffset::Add(viewOffset.offset_, tensorOffset.offset_);
            auto newDynOffset = TensorOffset::Add(viewOffset.dynOffset_, tensorOffset.dynOffset_);
            viewOpAttribute->SetFromOffset(newStaticOffset, newDynOffset);
            APASS_LOG_INFO_F(Elements::Tensor, "Update offset for OP_VIEW with opmagic %d.", childOp->GetOpMagic());
        }
    }
}

void SplitLargeFanoutTensor::EraseRedundantAssembleOp(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "---> Remove redundant Assemble op.");
    std::vector<Operation *> redundantCopyOuts;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto output = op.oOperand.front();
        auto input = op.iOperand.front();
        if ((input == nullptr) || (output == nullptr)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has nullptr input/output; "
                "Please ensure input and output are valid. %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                GetFormatBacktrace(op).c_str());
            continue;
        }
        if (!function.IsFromOutCast(output) && output->GetConsumers().empty()) {
            /* input --> Assemble --> output(非OCAST, 且没有consumer) */
            redundantCopyOuts.push_back(&op);
        }
        if (output->GetProducers().size() != 1 || output->GetConsumers().size() != 1) {
            continue;
        }
        auto consumerOp = *(output->GetConsumers().begin());
        // Assemble输入和输出的raw tensor大小不相等，意味着要做拷贝
        bool requireCopy = (input->tensor->GetRawShapeSize() != output->tensor->GetRawShapeSize());
        if (consumerOp->GetOpcode() == Opcode::OP_VIEW && !requireCopy) {
            /*
            Before: input --> Assmeble --> output --> View
            After:  input --> View
            因为input和output的raw shape相同，所以View上的offset不需要修改
            */
            redundantCopyOuts.push_back(&op);
            continue;
        }
        if (input->shape == output->shape && input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                    output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            /* 因为input和output raw shape size不同，但shape相同，因此删除前需要重新计算View的offset */
            UpdateForRedundantAssemble(op);
            redundantCopyOuts.push_back(&op);
        }
    }
    if (!redundantCopyOuts.empty()) {
        RemoveOps(function, redundantCopyOuts);
    }
}

void SplitLargeFanoutTensor::UpdateForRedundantView(Operation &op, Operation &consumer) {
    auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    auto newOffset = viewAttr->GetFromOffset();
    auto nextViewAttr = dynamic_cast<ViewOpAttribute *>(consumer.GetOpAttribute().get());
    auto viewDynShape = GetViewValidShape(viewAttr->GetToDynValidShape(), nextViewAttr->GetFromOffset(),
        nextViewAttr->GetFromDynOffset(), consumer.oOperand.front()->GetShape());
    nextViewAttr->SetToDynValidShape(viewDynShape);
    auto nextViewOffset = nextViewAttr->GetFromOffset();
    auto nextViewDynOffset = nextViewAttr->GetFromDynOffset();
    auto newDynOffset = viewAttr->GetFromDynOffset();
    auto ret = TensorOffset::Add(newOffset, newDynOffset, nextViewOffset, nextViewDynOffset);
    if (!ret.first.empty()) {
        newOffset = ret.first;
        newDynOffset = ret.second;
    }
    nextViewAttr->SetFromOffset(newOffset, newDynOffset);
    if (newDynOffset.size() == 0) {
        return;
    }
    auto consumerOOperand = consumer.oOperand.front();
    std::vector<SymbolicScalar> dynValidShape;
    if (!nextViewAttr->GetToDynValidShape().empty()) {
        dynValidShape = nextViewAttr->GetToDynValidShape();
    } else if (!consumerOOperand->GetDynValidShape().empty()) {
        dynValidShape = consumerOOperand->GetDynValidShape();
    } else {
        dynValidShape = SymbolicScalar::FromConcrete(consumerOOperand->GetShape());
    }
    if (nextViewAttr->GetToDynValidShape().empty()) {
        nextViewAttr->SetToDynValidShape(dynValidShape);
    }
    if (consumerOOperand->GetDynValidShape().empty()) {
        consumerOOperand->UpdateDynValidShape(dynValidShape);
    }
}

/*
before:
tensor -> View1 -> tensor1 -> View2 -> tesnor2

after:
tensor -> View2_new -> tensor2
*/
void SplitLargeFanoutTensor::EraseRedundantViewOp(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "---> Remove redundant View op.");
    std::vector<Operation *> redundantView;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        /*
        case1. split_large_fanout_tensor 在AssignmemType 之前，view op GetOpAttribute()->GetTo() == MemoryType::MEM_L1的view op一定是tile op展开时插入的，不能删；
        case2. 框架在tile 展开插入的view之前还插入了一个view，目前不删除，后续优化可以考虑删除。
        */
        bool isViewToL1 = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get())->GetTo() == MemoryType::MEM_L1;
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        auto consumers = op.oOperand.front()->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        bool allChildrenView = std::all_of(consumers.begin(), consumers.end(),
            [=](const Operation *opNext) {
                if (opNext->GetOpcode() != Opcode::OP_VIEW) {
                    return false;
                }
                auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(opNext->GetOpAttribute().get());
                bool isL1MultiLoad = (viewOpAttribute->GetTo() == MemoryType::MEM_L1);
                // 大包搬运场景下前端插入的view输入和输出shape相同
                auto inTensor = opNext->GetIOperands().front();
                auto outTensor = opNext->GetOOperands().front();
                isL1MultiLoad &= (inTensor->GetShape() == outTensor->GetShape());
                if(isViewToL1 || isL1MultiLoad) {
                    return false;
                }
                return true;
            });
        if (allChildrenView) {
            GraphUtils::UpdateViewAttr(function, op);
            for (auto &consumer : consumers) {
                UpdateForRedundantView(op, *consumer);
            }
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            APASS_LOG_DEBUG_F(Elements::Operation, "Found redundant view and remove it, "
                "opmagic: %d, to: %s. Input mem: %s, Output mem: %s.",
                op.GetOpMagic(),BriefMemoryTypeToString(viewAttr->GetTo()).c_str(),
                BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
            redundantView.push_back(&op);
        }
    }
    if (!redundantView.empty()) {
        RemoveOps(function, redundantView);
    }
}

} // namespace npu::tile_fwk
