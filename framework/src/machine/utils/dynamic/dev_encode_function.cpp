/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_encode_function.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode_function.h"

namespace npu::tile_fwk::dynamic {
namespace {
std::string DumpSymInt(const SymInt &s, const uint64_t *runtimeExpressionList) {
    std::ostringstream oss;
    if (s.IsExpression()) {
        if (runtimeExpressionList == nullptr) {
            oss << "?" << s.Value();
        } else {
            oss << runtimeExpressionList[s.Value()];
        }
    } else {
        oss << s.Value();
    }
        return oss.str();
}

std::string DumpSymIntList(const SymInt *s, int count, uint64_t *runtimeExpressionList) {
    std::ostringstream oss;
    oss << "<";
    for (int i = 0; i < count; i++) {
        oss << Delim(i != 0, ",") << DumpSymInt(s[i], runtimeExpressionList);
    }
    oss << ">";
    return oss.str();
}
}

std::string DevAscendFunction::DumpTensor(int tensorIndex) const {
    std::ostringstream oss;
    oss << "%" << tensorIndex << "@" << GetTensor(tensorIndex)->rawIndex;
    return oss.str();
}

std::string DevAscendFunction::DumpOperationAttr(int operationIndex, uint64_t *runtimeExpressionList,
                                                 bool dumpIndex) const {
    std::ostringstream oss;
    oss << SchemaGetCoa(operationIndex, runtimeExpressionList, dumpIndex).Dump();
    oss << " " << schema::pred(GetOperationDepGraphPredCount(operationIndex)).Dump();
    const DevLocalVector<int> &succList = GetOperationDepGraphSuccList(operationIndex);
    std::vector<schema::operation> succDataList;
    for (size_t j = 0; j < succList.size(); j++) {
        succDataList.push_back(At(succList, j));
    }
    oss << " " << schema::succ(succDataList).Dump();

    const DevLocalVector<int> &copyOutResolveCounterIndexList = GetOperationDepGraphCopyOutResolveSuccIndexList(operationIndex);
    std::vector<int64_t> outSuccIndexDataList;
    for (size_t j = 0; j < copyOutResolveCounterIndexList.size(); j++) {
        outSuccIndexDataList.push_back(At(copyOutResolveCounterIndexList, j));
    }
    oss << " " << schema::outSuccIndex(outSuccIndexDataList).Dump();
    oss << " " << "#outcastStitch{" << GetOperationOutcastStitchIndex(operationIndex) << "}";
    return oss.str();
}

std::string DevAscendFunction::DumpOperation(int operationIndex, int &totalAttrStartIdx,
                                             const std::vector<uintdevptr_t> &ooperandAddrList,
                                             const std::vector<uintdevptr_t> &ioperandAddrList,
                                             uint64_t *runtimeExpressionList) const {
    std::ostringstream oss;
    for (size_t j = 0; j < GetOperationOOperandSize(operationIndex); j++) {
        oss << Delim(j != 0, ",") << DumpTensor(GetOperationOOperandInfo(operationIndex, j).tensorIndex);
        if (j < ooperandAddrList.size()) {
            oss << AddressDescriptor::DumpAddress(ooperandAddrList[j]);
        }
    }
    oss << " = "
        << "!" << operationIndex << " ";
    for (size_t j = 0; j < GetOperationIOperandSize(operationIndex); j++) {
        oss << Delim(j != 0, ",") << DumpTensor(GetOperationIOperandInfo(operationIndex, j).tensorIndex);
        if (j < ioperandAddrList.size()) {
            oss << AddressDescriptor::DumpAddress(ioperandAddrList[j]);
        }
    }

    oss << " " << DumpOperationAttr(operationIndex, runtimeExpressionList, true);
    totalAttrStartIdx += static_cast<int>(GetOperationAttrSize(operationIndex));
    return oss.str();
}

std::string DevAscendFunction::DumpRawTensor(int rawIndex, uintdevptr_t addr) const {
    std::ostringstream oss;
    auto rawTensor = GetRawTensor(rawIndex);
    auto rawTensorDesc = GetRawTensorDesc(rawIndex);
    oss << rawTensor->DumpType() << " @" << rawIndex <<"&"<<rawTensor->linkedIncastId << " = ";
    oss << rawTensor->DumpAttr() << " ";
    oss << DevAscendRawTensor::DumpAttrDesc(rawTensorDesc);
    if (addr != 0) {
        oss << AddressDescriptor::DumpAddress(addr);
    }
    return oss.str();
}

std::string DevAscendFunction::DumpIncast(int incastIndex, const std::string &indent, uint64_t *runtimeExpressionList,
                                          const std::vector<uintdevptr_t> &slotAddrList) const {
    std::ostringstream oss;
    const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
    oss << "#incast:" << incastIndex << " = " << DumpTensor(incast.tensorIndex);
    for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
        int slot = At(incast.fromSlotList, j);
        oss << " <- #slot:" << slot;
        if (slot < static_cast<int>(slotAddrList.size())) {
            oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
        }
    }
    oss << "\n";
    oss << indent;
    oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(incast.cellMatchTableDesc);
    oss << " | #cellMatchStaticTable:" << incast.cellMatchStaticIncastTable.size();
    oss << "\n";

    oss << indent << " | #stitchPolicyFullCoverConsumerAllOpIdxList:[";
    for (size_t j = 0; j < incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); j++) {
        oss << Delim(j != 0, ",") << At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, j);
    }
    oss << "]\n";

    for (size_t j = 0; j < incast.consumerList.size(); j++) {
        auto &consumer = At(incast.consumerList, j);
        int consumerIdx = consumer.operationIdx;
        int operandIdx = consumer.operandIdx;
        int offsetAttrIdx = consumer.offsetAttrIdx;
        int shapeAttrIdx = consumer.shapeAttrIdx;
        oss << indent;
        oss << " | #consumerIdx:!" << consumerIdx;
        oss << " | #operandIdx:" << operandIdx;
        oss << " | #offsetAttrIdx:" << offsetAttrIdx;
        oss << " | #shapeAttrIdx:" << shapeAttrIdx;
        oss << " | #offsetAttr:" << DumpSymIntList(&GetOperationAttr(consumerIdx, offsetAttrIdx), incast.dim, runtimeExpressionList);
        oss << " | #shapeAttr:" << DumpSymIntList(&GetOperationAttr(consumerIdx, shapeAttrIdx), incast.dim, runtimeExpressionList);
        oss << "\n";
    }
    return oss.str();
}

std::string DevAscendFunction::DumpOutcast(int outcastIndex, const std::string &indent, uint64_t *runtimeExpressionList,
                                           const std::vector<uintdevptr_t> &slotAddrList) const {
    std::ostringstream oss;
    const DevAscendFunctionOutcast &outcast = GetOutcast(outcastIndex);
    auto dumpProducer = [this, &oss, &indent, &outcast, &runtimeExpressionList](const DevLocalVector<DevAscendFunctionCallOperandUse>& producerList) -> void {
        for (size_t j = 0; j < producerList.size(); j++) {
            auto &producer = At(producerList, j);
            int producerIdx = producer.operationIdx;
            int operandIdx = producer.operandIdx;
            int offsetAttrIdx = producer.offsetAttrIdx;
            int shapeAttrIdx = producer.shapeAttrIdx;
            oss << indent;
            oss << " | #producerIdx:!" << producerIdx;
            oss << " | #operandIdx:" << operandIdx;
            oss << " | #offsetAttrIdx:" << offsetAttrIdx;
            oss << " | #shapeAttrIdx:" << shapeAttrIdx;
            oss << " | #offsetAttr:" << DumpSymIntList(&GetOperationAttr(producerIdx, offsetAttrIdx), outcast.dim, runtimeExpressionList);
            oss << " | #shapeAttr:" << DumpSymIntList(&GetOperationAttr(producerIdx, shapeAttrIdx), outcast.dim, runtimeExpressionList);
            oss << "\n";
        }
    };

    oss << "#outcast:" << outcastIndex << " = " << DumpTensor(outcast.tensorIndex);
    for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
        int slot = At(outcast.toSlotList, j);
        oss << " -> #slot:" << slot;
        if (slot < static_cast<int>(slotAddrList.size())) {
            oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
        }
    }
    oss << "\n";
    oss << indent;
    oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(outcast.cellMatchTableDesc);
    oss << " | #cellMatchStaticTable:" << outcast.cellMatchStaticOutcastTable.size();
    oss << " | #cellMatchFullUpdateTable:" << outcast.cellMatchRuntimeFullUpdateTable.size();
    oss << "\n";

    oss << indent << " | #stitchPolicyFullCoverProducerList:[";
    for (size_t j = 0; j < outcast.stitchPolicyFullCoverProducerList.size(); j++) {
        oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverProducerList, j).operationIdx;
    }
    oss << "]\n";
    dumpProducer(outcast.stitchPolicyFullCoverProducerList);

    oss << indent << " | #stitchPolicyFullCoverProducerHubOpIdx:" << outcast.stitchPolicyFullCoverProducerHubOpIdx << "\n";
    oss << indent << " | #stitchPolicyFullCoverProducerAllOpIdxList:[";
    for (size_t j = 0; j < outcast.stitchPolicyFullCoverProducerAllOpIdxList.size(); j++) {
        oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverProducerAllOpIdxList, j);
    }
    oss << "]\n";
    dumpProducer(outcast.producerList);
    return oss.str();
}

std::string DevAscendFunction::Dump(int indent) const {
    std::string INDENT(indent, ' ');
    std::string INDENTINNER(indent + IDENT_SIZE, ' ');
    std::ostringstream oss;

    oss << INDENT << "DevFunction " << funcKey;
    oss << " " << schema::name(GetRawName()).Dump();
    oss << " " << schema::mem(rootInnerTensorWsMemoryRequirement).Dump();
    oss << " " << schema::memOut(exclusiveOutcastWsMemoryRequirement).Dump();
    oss << " {\n";
    for (size_t i = 0; i < GetRawTensorSize(); i++) {
        oss << INDENTINNER << DumpRawTensor(i) << "\n";
    }
    for (size_t i = 0; i < GetIncastSize(); i++) {
        oss << INDENTINNER << DumpIncast(i, INDENTINNER) << "\n";
    }
    for (size_t i = 0; i < GetOutcastSize(); i++) {
        oss << INDENTINNER << DumpOutcast(i, INDENTINNER) << "\n";
    }

    {
        oss << INDENTINNER << "#assembleSlotSize{" << GetRedaccAssembleSlotListSize() << "}\n";
        for (size_t j = 0; j < GetRedaccAssembleSlotListSize(); j++) {
            oss << INDENTINNER << "#assembleSlot_" << j << "{" << GetRedaccAssembleSlotList(j) << "}\n";
        }
    }
    oss << INDENTINNER << "#hubOpCount:" << hubOpCount_ << "\n";
    oss << INDENTINNER << "#zeropred:" << predInfo_.totalZeroPred << "\n";
    oss << INDENTINNER << "#zeropred-aiv:" << predInfo_.totalZeroPredAIV << "\n";
    oss << INDENTINNER << "#zeropred-aic:" << predInfo_.totalZeroPredAIC << "\n";
    oss << INDENTINNER << "#zeropred-aicpu:" << predInfo_.totalZeroPredAicpu << "\n";
    int totalAttrStartIdx = 0;
    for (size_t i = 0; i < GetOperationSize(); i++) {
        oss << INDENTINNER << DumpOperation(i, totalAttrStartIdx) << "\n";
    }
    oss << INDENT << "}";
    return oss.str();
}
}
