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
 * \file reschedule_utils.cpp
 * \brief
 */

#include "reschedule_utils.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
bool RescheduleUtils::isAllocOp(Operation* op) {
    static std::unordered_set<Opcode> allocOpcodes = {
        Opcode::OP_L1_ALLOC,                    Opcode::OP_UB_ALLOC,
        Opcode::OP_L0A_ALLOC,                   Opcode::OP_L0B_ALLOC,
        Opcode::OP_L0C_ALLOC,                   Opcode::OP_BT_ALLOC,
        Opcode::OP_FIX_ALLOC,                   Opcode::OP_REG_ALLOC,
    };
    return allocOpcodes.find(op->GetOpcode()) != allocOpcodes.end();
}

// vector size == 2
// 这里in_graph和out_graph只考虑了数据依赖，需要确认控制依赖是不是也要加入到in_graph和out_graph中
std::vector<std::vector<std::vector<int>>> RescheduleUtils::GetInOutGraphs(
    const std::vector<Operation *> &opList, int functionmagic) {
    std::vector<std::vector<int>> inGraph;
    std::vector<std::vector<int>> outGraph;
    inGraph.resize(opList.size());
    outGraph.resize(opList.size());
    std::map<int, size_t> magic2Index;
    for (size_t i = 0; i < opList.size(); i++) {
        magic2Index[opList[i]->GetOpMagic()] = i;
    }
    if (functionmagic != -1) {
        for (size_t i = 0; i < opList.size(); i++) {
            // 收集节点输出输出关系，沿着一个方向，如果沿着输入输出都收集，需要去重
            for (auto &inTensor : opList[i]->GetIOperands()) {
                for (auto &producer : inTensor->GetProducers()) {
                    // 只收集了当前func内的节点关联关系
                    auto iter = magic2Index.find(producer->GetOpMagic());
                    if (iter == magic2Index.end()) {
                        continue;
                    }
                    inGraph[i].push_back(iter->second);
                    outGraph[iter->second].push_back(i);
                }
            }
        }
    } else {
        for (size_t i = 0; i < opList.size(); i++) {
            // 收集节点输出输出关系，沿着一个方向，如果沿着输入输出都收集，需要去重
            for (auto &inTensor : opList[i]->GetIOperands()) {
                for (auto &producer : inTensor->GetProducers()) {
                    // 只收集了当前func内的节点关联关系
                    inGraph[i].push_back(magic2Index[producer->GetOpMagic()]);
                    outGraph[magic2Index[producer->GetOpMagic()]].push_back(i);
                }
            }
        }
    }
    std::vector<std::vector<std::vector<int>>> inOutGraph{inGraph, outGraph};
    return inOutGraph;
}


PipeType RescheduleUtils::GetOpPipeType(const Operation* op) {
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        if (op->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            op->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            return PipeType::PIPE_MTE3;
        }
    }
    if (op->GetOpcode() == Opcode::OP_COPY_IN) {
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        auto dstMemType = attr->GetCopyInAttr().second;
        if (dstMemType == MemoryType::MEM_L1) {
            return PipeType::PIPE_MTE2;
        }
        if (dstMemType == MemoryType::MEM_UB) {
            return PipeType::PIPE_MTE2;
        }
    }
    if (op->GetOpcode() == Opcode::OP_COPY_OUT) {
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        auto srcMemType = attr->GetCopyOutAttr().first;
        if (srcMemType == MemoryType::MEM_L0C) {
            return PipeType::PIPE_FIX;
        }
        if (srcMemType == MemoryType::MEM_UB) {
            return PipeType::PIPE_MTE3;
        }
        if (srcMemType == MemoryType::MEM_L1) {
            return PipeType::PIPE_MTE3;
        } 
    }
    return opcfg.pipeIdStart_;
}

void GetConvOpAttrStr(std::stringstream& ss, const Operation* op) {
    ss << "<" << op->GetIntAttribute(ConvOpAttributeKey::cin) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::cout) << ",",
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingLeft) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingTop) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingRight) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::paddingBottom) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::strideh) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::stridew) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hposX) << ",",
    ss << op->GetIntAttribute(ConvOpAttributeKey::hsteP) << ",",
    ss << op->GetIntAttribute(ConvOpAttributeKey::wposX) << ",",
    ss << op->GetIntAttribute(ConvOpAttributeKey::wstep) << ",",
    ss << op->GetIntAttribute(ConvOpAttributeKey::hoffsetY) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::woffsetY) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::reluType) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::reluAlpha) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::clearFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasAccFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasEltFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::hasBiasFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::eltBrcbFlag) << ",";
    ss << op->GetIntAttribute(ConvOpAttributeKey::eltMode) << ">";
}

unsigned long RescheduleUtils::ComputeOperationHash(const Operation *op) {
    std::stringstream ss;
    ss << op->GetOpcodeStr();

    // 不需要cycle数校准的op。
    switch (op->GetOpcode()) {
        case Opcode::OP_L1_ALLOC:
        case Opcode::OP_UB_ALLOC:
        case Opcode::OP_L0A_ALLOC:
        case Opcode::OP_L0B_ALLOC:
        case Opcode::OP_L0C_ALLOC:
        case Opcode::OP_BT_ALLOC:
        case Opcode::OP_FIX_ALLOC:
        case Opcode::OP_SYNC_SRC:
        case Opcode::OP_SYNC_DST:
        case Opcode::OP_CV_SYNC_SRC:
        case Opcode::OP_CV_SYNC_DST:
        case Opcode::OP_PHASE1:
        case Opcode::OP_PHASE2:
        case Opcode::OP_BAR_V:
        case Opcode::OP_BAR_M:
            return 0;
        default:
            break;
    }

    switch (op->GetOpcode()) {
        case Opcode::OP_CONV:
        case Opcode::OP_CONV_ADD:
            GetConvOpAttrStr(ss, op);
            break;
        default:
            break;
    }

    for (const auto &inp : op->GetIOperands()) {
        ss << "[i(";
        for (auto s : inp->GetShape()) {
            ss << s << ",";
        }
        ss << ")" << DataType2String(inp->Datatype()) << "]";
    }
    for (const auto &inp : op->GetOOperands()) {
        ss << "[o(";
        for (auto s : inp->GetShape()) {
            ss << s << ",";
        }
        ss << ")" << DataType2String(inp->Datatype()) << "]";
    }

    std::hash<std::string> hasher;
    auto result = hasher(ss.str());

    return result;
}

void RescheduleUtils::FindOtherGraphOp(Operation *op, Function *funcPtr, std::unordered_set<Operation*> &otherGraphOp) {
    if (op->BelongTo() != funcPtr) {
        otherGraphOp.insert(op);
    }
}

void RescheduleUtils::ClearInputConsProd(Operation &op, Function *funcPtr, 
    std::unordered_set<LogicalTensorPtr> incastSet) {
    for (auto &inOperand : op.GetIOperands()) {
        if (incastSet.count(inOperand) == 0) {
            std::unordered_set<Operation*> otherGraphOp;
            for (auto prod : inOperand->GetProducers()) {
                FindOtherGraphOp(prod, funcPtr, otherGraphOp);
            }
            inOperand->GetProducers().clear();
            for (auto otherOp : otherGraphOp) {
                inOperand->AddProducer(otherOp);
            }
        }
        std::unordered_set<Operation*> otherGraphOps;
        for (auto cons : inOperand->GetConsumers()) {
            FindOtherGraphOp(cons, funcPtr, otherGraphOps);
        }
        inOperand->GetConsumers().clear();
        for (auto otherOp : otherGraphOps) {
            inOperand->AddConsumer(otherOp);
        }
    }
}

void RescheduleUtils::ClearOutputConsProd(Operation &op, Function *funcPtr,
    std::unordered_set<LogicalTensorPtr> outcastSet) {
    for (auto &outOperand : op.GetOOperands()) {
        std::unordered_set<Operation*> otherGraphOp;
        for (auto prod : outOperand->GetProducers()) {
            FindOtherGraphOp(prod, funcPtr, otherGraphOp);
        }
        outOperand->GetProducers().clear();
        for (auto otherOp : otherGraphOp) {
            outOperand->AddProducer(otherOp);
        }
        if (outcastSet.count(outOperand) == 0) {
            std::unordered_set<Operation*> otherGraphOps;
            for (auto cons : outOperand->GetConsumers()) {
                FindOtherGraphOp(cons, funcPtr, otherGraphOps);
            }
            outOperand->GetConsumers().clear();
            for (auto otherOp : otherGraphOps) {
                outOperand->AddConsumer(otherOp);
            }
        }
    }
}

void RescheduleUtils::UpdateTensorConsProd(Function *funcPtr) {
    std::unordered_map<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>> tensorMap;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> inverseMap;
    std::unordered_set<LogicalTensorPtr> incastSet;
    std::unordered_set<LogicalTensorPtr> outcastSet;
    for (auto &incast : funcPtr->inCasts_) {
        incastSet.emplace(incast);
    }
    for (auto &outcast : funcPtr->outCasts_) {
        outcastSet.emplace(outcast);
    }
    for (auto &op : funcPtr->Operations()) {
        ClearInputConsProd(op, funcPtr, incastSet);
        ClearOutputConsProd(op, funcPtr, outcastSet);
    }
    for (auto &op : funcPtr->Operations()) {
        for (auto &inOperand : op.GetIOperands()) {
            inOperand->AddConsumer(op);
            tensorMap[inOperand->GetRawTensor()->GetRawMagic()].insert(inOperand);
            inverseMap[inOperand->GetMagic()] = inOperand;
        }
    }
    for (auto &op : funcPtr->Operations()) {
        if (op.GetOpcodeStr().find("ALLOC") != std::string::npos) {
            op.oOperand.clear();
            continue;
        }
        for (auto &outOperand : op.GetOOperands()) {
            outOperand->AddProducer(op);
            tensorMap[outOperand->GetRawTensor()->GetRawMagic()].insert(outOperand);
            inverseMap[outOperand->GetMagic()] = outOperand;
        }
    }

    funcPtr->GetTensorMap().tensorMap_.swap(tensorMap);
    funcPtr->GetTensorMap().inverseMap_.swap(inverseMap);
}

void RescheduleUtils::PrintColorNode(Function &func) {
    std::map<int, std::vector<size_t>> colorNode;
    for (size_t i = 0; i < func.Operations().size(); i++) {
        auto &op = func.Operations()[i];
        auto color = op.GetSubgraphID();
        colorNode[color].push_back(i);
    }
    ALOG_DEBUG_F("After L1CopyInReuseMerge :");
    for (auto color : colorNode) {
        std::string colorInfo;
        colorInfo += "color " + std::to_string(color.first) + " : {";
        for (auto &opIdx : color.second) {
            colorInfo += "op " + std::to_string(func.Operations()[opIdx].GetOpMagic()) + "--" +
                         func.Operations()[opIdx].GetOpcodeStr() + ", ";
        }
        colorInfo += "}";
        ALOG_DEBUG_F("%s", colorInfo.c_str());
    }
}
} // namespace npu::tile_fwk
