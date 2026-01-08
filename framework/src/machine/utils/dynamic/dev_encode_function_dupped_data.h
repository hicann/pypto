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
 * \file dev_encode_function_dupped_data.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {
constexpr int ARG_ATTR_TYPE = 4;
const uint32_t RAW_TENSOR_OFFSET_SIZE = 63;
const uint32_t RAW_TENSOR_DESC_PRE_SIZE = 8;

struct DevAscendFunctionDuppedData {
    DevAscendFunction *source_;
    DevAscendFunctionDuppedOperation operationList_;
    DevAscendFunctionDuppedVector incastList_;
    DevAscendFunctionDuppedVector outcastList_;
    DevAscendFunctionDuppedVector expressionList_;
    uintdevptr_t runtimeWorkspace_;
    RuntimeReuseInfo runtimeWsReuseInfo_;
    uintdevptr_t runtimeOutcastWorkspace_;
    uint8_t data_[0];
    /*
     *  Duplicated:
     *      predcount_t                                         predCountListData[];
     *  Allocated (& zero-ed):
     *      AddressDescriptor                                   incastAddressListData[];
     *      AddressDescriptor                                   outcastAddressListData[];
     *      uint64_t                                            expressionListData[];
     *      DevAscendFunctionDuppedStitchList                   stitchListData[];
     */
#define GET_DATA(type, data, base, index) ((reinterpret_cast<type *>(const_cast<uint8_t *>((data) + (base)))[index]))
    uint32_t GetOperationSize() const { return operationList_.size; }
    const predcount_t &GetOperationCurrPredCount(int index) const { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }
    predcount_t &GetOperationCurrPredCount(int index) { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }

    uint32_t GetStitchSize() const { return operationList_.stitchCount; }
    const DevAscendFunctionDuppedStitchList &GetStitch(int index) const { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }
    DevAscendFunctionDuppedStitchList &GetStitch(int index) { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }

    uint64_t GetExpressionSize() const { return expressionList_.size; }
    const uint64_t &GetExpression(int index) const { return GET_DATA(uint64_t, data_, expressionList_.base, index); }
    uint64_t &GetExpression(int index) { return GET_DATA(uint64_t, data_, expressionList_.base, index); }

    uint64_t *GetExpressionAddr() const {
        return &GET_DATA(uint64_t, data_, expressionList_.base, 0);
    }

    uint64_t GetIncastSize() const { return incastList_.size; }
    AddressDescriptor GetIncastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }
    AddressDescriptor &GetIncastAddress(int index) { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }

    uint64_t GetOutcastSize() const { return outcastList_.size; }
    AddressDescriptor GetOutcastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }
    AddressDescriptor &GetOutcastAddress(int index) { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return runtimeWsReuseInfo_; }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return runtimeWsReuseInfo_; }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace_; }
    uintdevptr_t &GetRuntimeWorkspace() { return runtimeWorkspace_; }

    uintdevptr_t GetRuntimeOutcastWorkspace() const { return runtimeOutcastWorkspace_; }
    uintdevptr_t &GetRuntimeOutcastWorkspace() { return runtimeOutcastWorkspace_; }

    DevAscendFunction *GetSource() const { return source_; }
    DevAscendFunction *&GetSource() { return source_; }

    const DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) const {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            if (!maybeNull && outcastStitchIndex == 0) {
                DEV_ERROR("GetOperationStitch: operation %d has invalid outcast stitch index 0", operationIndex);
            }
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }
    DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            if (!maybeNull && outcastStitchIndex == 0) {
                DEV_ERROR("GetOperationStitch: operation %d has invalid outcast stitch index 0", operationIndex);
            }
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }

    inline uint64_t GetIncastDataSize(int incastIndex) const {
        auto rawTensor = GetSource()->GetIncastRawTensor(incastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    inline uint64_t GetOutcastDataSize(int outcastIndex) const {
        auto rawTensor = GetSource()->GetOutcastRawTensor(outcastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    schema::range SchemaGetIncastRange(int arg) const {
        auto base = GetIncastAddress(arg).GetAddress();
        auto size = GetIncastDataSize(arg);
        return schema::Range(base, base + size);
    }
    schema::range SchemaGetOutcastRange(int arg) const {
        auto base = GetOutcastAddress(arg).GetAddress();
        auto size = GetOutcastDataSize(arg);
        return schema::Range(base, base + size);
    }

    schema::RActWorkspace SchemaGetWorkspace() const {
        auto workspaceBegin = GetRuntimeWorkspace();
        auto workspaceEnd = GetRuntimeWorkspace() + GetSource()->rootInnerTensorWsMemoryRequirement;
        return schema::RActWorkspace(schema::Range(workspaceBegin, workspaceEnd));
    }

    std::string Dump(int indent = 0) const {
        if (GetSource()->GetOperationSize() != GetOperationSize()) {
            DEV_ERROR("GetOperationSize mismatch: source=%zu, self=%u", GetSource()->GetOperationSize(), GetOperationSize());
        }
        DEV_ASSERT(GetSource()->GetOperationSize() == GetOperationSize());
        std::string INDENT(indent, ' ');
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');

        std::ostringstream oss;
        oss << INDENT << "DevFunctionDupped " << GetSource()->GetFuncKey() << " {\n";
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            oss << INDENTINNER << "#incast:" << incastIndex << " = " << GetIncastAddress(incastIndex).Dump() << "\n";
        }
        for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
            oss << INDENTINNER << "#outcast:" << outcastIndex << " = " << GetOutcastAddress(outcastIndex).Dump() << "\n";
        }
        for (size_t operationIndex = 0; operationIndex < GetOperationSize(); operationIndex++) {
            oss << INDENTINNER << "!" << operationIndex;
            oss << " #pred:" << GetSource()->GetOperationDepGraphPredCount(operationIndex);
            oss << " #succ:[";
            size_t succSize;
            auto succList = GetSource()->GetOperationDepGraphSuccAddr(operationIndex, succSize);
            for (size_t j = 0; j < succSize; j++) {
                oss << Delim(j != 0, ",") << "[" << j << "]=!" << succList[j];
            }
            oss << "]";
            oss << " #dynpred:" << GetOperationCurrPredCount(operationIndex);
            oss << " #dynsucc:" << GetOperationStitch(operationIndex).Dump();
            oss << "\n";
        }
        oss << INDENTINNER << "#expr:[";
        for (size_t exprIndex = 0; exprIndex < GetExpressionSize(); exprIndex++) {
            oss << Delim(exprIndex != 0, ",") << "[" << exprIndex << "]=" << GetExpression(exprIndex);
        }
        oss << "]";
        oss << INDENT << "}\n";
        return oss.str();
    }
};

struct DevAscendFunctionDupped {
    DevAscendFunctionDupped() = default;
    explicit DevAscendFunctionDupped(WsAllocation tinyAlloc) : dupTiny_(tinyAlloc) {}

    static DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func, WsAllocation tinyAlloc) {
        DevAscendFunctionDuppedData *dupData = tinyAlloc.As<DevAscendFunctionDuppedData>();
        DevAscendFunctionDuppedData *sourceData = func->GetDuppedData();
        (void)memcpy_s(reinterpret_cast<uint8_t *>(dupData),
            func->GetDuppedDataCopySize(),
            sourceData,
            func->GetDuppedDataCopySize());
        (void)memset_s(reinterpret_cast<uint8_t *>(dupData) + func->GetDuppedDataCopySize(),
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize(),
            0,
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize());
        dupData->GetSource() = func;

        DevAscendFunctionDupped dup(tinyAlloc);
        return dup;
    }

    void ReleaseDuppedMemory(WsMetadataAllocator &allocator) {
        (void)allocator;
    }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return DupData()->GetRuntimeReuseInfo(); }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return DupData()->GetRuntimeReuseInfo(); }

    uintdevptr_t RuntimeWorkspace() const { return DupData()->GetRuntimeWorkspace(); }
    uintdevptr_t &RuntimeWorkspace() { return DupData()->GetRuntimeWorkspace(); }

    uintdevptr_t RuntimeOutcastBase() const { return DupData()->GetRuntimeOutcastWorkspace(); }
    uintdevptr_t &RuntimeOutcastBase() { return DupData()->GetRuntimeOutcastWorkspace(); }

    const DevAscendFunction *GetSource() const { return DupData()->GetSource(); }
    DevAscendFunction *GetSource() { return DupData()->GetSource(); }

    inline const uint64_t &GetExpression(int arg) const { return DupData()->GetExpression(arg); };
    inline uint64_t &GetExpression(int arg) { return DupData()->GetExpression(arg); };
    inline uint64_t GetExpressionSize() const { return DupData()->GetExpressionSize(); }
    inline uint64_t *GetExpressionAddr() const { return DupData()->GetExpressionAddr(); }

    inline auto GetOperationSize() const { return DupData()->GetOperationSize(); }
    inline const predcount_t &GetOperationCurrPredCount(int arg) const { return DupData()->GetOperationCurrPredCount(arg); };
    inline predcount_t &GetOperationCurrPredCount(int arg) { return DupData()->GetOperationCurrPredCount(arg); };
    inline const auto &GetOperationStitch(int arg, bool maybeNull = true) const { return DupData()->GetOperationStitch(arg, maybeNull); };
    inline auto &GetOperationStitch(int arg, bool maybeNull = true) { return DupData()->GetOperationStitch(arg, maybeNull); };

    inline AddressDescriptor GetIncastAddress(int arg) const { return DupData()->GetIncastAddress(arg); };
    inline AddressDescriptor &GetIncastAddress(int arg) { return DupData()->GetIncastAddress(arg); };

    inline AddressDescriptor GetOutcastAddress(int arg) const { return DupData()->GetOutcastAddress(arg); };
    inline AddressDescriptor &GetOutcastAddress(int arg) { return DupData()->GetOutcastAddress(arg); };

    schema::expr SchemaGetExpressionTable() const {
        std::vector<schema::Int64Type> exprTable;
        uint64_t *exprAddr = GetExpressionAddr();
        uint64_t exprSize = GetExpressionSize();
        for (uint64_t i = 0; i < exprSize; i++) {
            exprTable.push_back(exprAddr[i]);
        }
        return schema::expr(exprTable);
    }

    inline uintdevptr_t GetRawTensorAddr(int rawIndex) const {
        uintdevptr_t addr = 0ULL;
        const DevAscendRawTensor *rawTensor = GetSource()->GetRawTensor(rawIndex);
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            AddressDescriptor incast = GetIncastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(!incast.IsNullAddress(),
                "Null incast: root [%s], rawIndex [%d], ioIndex [%d]",
                GetSource()->GetRawName(), rawIndex, rawTensor->ioIndex);
            addr = incast.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            AddressDescriptor outcast = GetOutcastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(!outcast.IsNullAddress(),
                "Null outcast: root [%s], rawIndex [%d], ioIndex [%d]",
                GetSource()->GetRawName(), rawIndex, rawTensor->ioIndex);
            addr = outcast.addr;
        } else {
            uintdevptr_t runtimeWorkspace = RuntimeWorkspace();
            DEV_ASSERT_MSG(runtimeWorkspace != 0,
                "Trying to access inner tensor addr with zero runtime workspace: root [%s], rawIndex [%d]",
                GetSource()->GetRawName(), rawIndex);
            addr = runtimeWorkspace + rawTensor->addrOffset;
        }
        return addr;
    }

    // for stitch
    inline void GetInTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationIOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetOutTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationOOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetFuncTensorOffsetAndShape(uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t shape[DEV_SHAPE_DIM_MAX], int dims,
        int operationIndex, int operandIndex, bool isIOperand = true) const {
        auto func = GetSource();
        GetTensorOffsetAndShape<false>(func, offset, shape, &GetExpression(0), dims, operationIndex, operandIndex, isIOperand);
    }

    std::string Dump(int indent = 0) const {
        return DupData()->Dump(indent);
    }

    inline int64_t GetValue(const SymInt *attrs, int idx) {
        return attrs[idx].IsExpression() ? funcData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
    }

    inline uint64_t GetRawTensorAddrEx(int idx) const {
        auto desc = funcData->rawTensorDesc[idx];
        if (desc.location == RAW_TENSOR_LOCATION_LOCAL)
            return funcData->workspaceAddr + desc.offsetOrIndex;
        else
            return funcData->rawTensorAddr[desc.offsetOrIndex] & ((1UL << RAW_TENSOR_OFFSET_SIZE) - 1);
    }

    std::string DumpDyn(int funcIdx, int operIdx, const DevCceBinary *cceBinary) {
        std::stringstream oss;
        auto func = GetSource();

        auto attrBase = reinterpret_cast<SymInt *>(&funcData->opAttrs[funcData->opAtrrOffsets[operIdx]]);
        auto funcIndex = attrBase[0].Value();
        oss << std::hex << " #funcKey " << func->funcKey << " #operIndex " << operIdx
            << " #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
            << " #coreType: " << cceBinary[funcIndex].coreType
            << " #taskID:" << MakeTaskID(funcIdx, operIdx) << "\n";

        auto dumpAttr = [this, &oss](auto attrs, auto &info) {
            int attrIndex = info.staticOffsetAttrBeginIndex;
            auto rawIndex = attrs[attrIndex - 1].Value();
            oss << rawIndex << "@" << GetRawTensorAddrEx(rawIndex) << ", ";
            int dim = info.GetDim();
            for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
                oss << GetValue(attrs, attrIndex + i) << ", ";
            }
        };

        int offset = 0;
        for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = static_cast<size_t>(offset); idx < func->GetOperationAttrSize(operIdx); idx++) {
            oss << GetValue(attrBase, idx) << ", ";
        }
        return oss.str();
    }

    void DumpTopo(std::ofstream &os, int seqNo, int funcIdx, const DevCceBinary *cceBinary) {
        auto func = GetSource();
        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            auto &cceInfo = cceBinary[func->GetOperationAttrCalleeIndex(opIdx)];
            os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << func->funcKey << "," << func->rootHash << ","
               <<func->GetOperationDebugOpmagic(opIdx) << "," << func->GetOperationAttrCalleeIndex(opIdx) << ","
               << cceInfo.funcHash << "," << cceInfo.coreType << "," << cceInfo.psgId << ",";
            auto &succList = func->GetOperationDepGraphSuccList(opIdx);
            for (size_t j = 0; j < succList.size(); j++) {
                os << "," << MakeTaskID(funcIdx, func->At(succList, j));
            }
            auto &stitch = GetOperationStitch(opIdx);
            stitch.ForEach([&os](uint32_t id) {
                os << "," << id;
            });
            os << "\n";
        }
    }

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(std::vector<std::string> &infos, uint32_t seqNo, uint32_t funcIdx) {
        // seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)
        auto *srcFunc = GetSource();

        auto dumpOperand = [&](const DevAscendOperationOperandInfo &operandInfo, size_t opIdx) {
            std::stringstream os;
            uint64_t rawIdx = srcFunc->GetTensor(operandInfo.tensorIndex)->rawIndex;
            auto *rawTensor = srcFunc->GetRawTensor(rawIdx);
            os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," <<
                rawTensor->rawMagic << "," <<
                GetRawTensorAddrEx(rawIdx) << "," <<
                BriefDataType2String(rawTensor->dataType) << "," <<
                BytesOf(rawTensor->dataType);

            uint32_t dimSize = rawTensor->GetDim();
            os << ",(";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    os << ",";
                }
                os << rawTensor->shape.At(i, GetExpressionAddr());
            }
            os << ")";

            os << "\n";
            infos.emplace_back(std::move(os).str());
        };

        for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
            for (size_t iopIdx = 0; iopIdx < srcFunc->GetOperationIOperandSize(opIdx); iopIdx++) {
                auto &iopInfo = srcFunc->GetOperationIOperandInfo(opIdx, iopIdx);
                dumpOperand(iopInfo, opIdx);
            }
            for (size_t oopIdx = 0; oopIdx < srcFunc->GetOperationOOperandSize(opIdx); oopIdx++) {
                auto &oopInfo = srcFunc->GetOperationOOperandInfo(opIdx, oopIdx);
                dumpOperand(oopInfo, opIdx);
            }
        }
    }
#endif // DEBUG_INFINITE_LIFETIME

    // Return result lines
    std::vector<std::string> DumpLeafs(uint32_t seqNo, uint32_t funcIdx) {
        std::vector<std::string> lines;
        std::stringstream oss;
        auto flushStream = [&] {
            lines.push_back(std::move(oss).str());
            oss.clear();
            oss.str("");
        };

        auto *srcFunc = GetSource();

        oss << "seqNo=" << seqNo << ", rootHash=" << srcFunc->rootHash;
        flushStream();

        auto dumpRawShape = [&](DevAscendRawTensor *rawTensor, uint32_t dimSize) {
            oss << "        rawShape=[";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << rawTensor->shape.At(i, GetExpressionAddr());
            }
            oss << "]";
            flushStream();
        };

        auto dumpOperandShape = [&](uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn) {
            uint64_t offset[DEV_SHAPE_DIM_MAX];
            uint64_t shape[DEV_SHAPE_DIM_MAX];
            GetFuncTensorOffsetAndShape(offset, shape, dimSize, opIdx, operandIdx, isIn);

            oss << "          offset=[";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << offset[i];
            }
            oss << "]";
            flushStream();

            oss << "           shape=[";
            isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << shape[i];
            }
            oss << "]";
            flushStream();
        };

        for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
            size_t iopNum = srcFunc->GetOperationIOperandSize(opIdx);
            size_t oopNum = srcFunc->GetOperationOOperandSize(opIdx);
            oss << "> taskId = " << MakeTaskID(funcIdx, opIdx) << ", opIdx=" << opIdx << ", #iop=" << iopNum << ", #oop=" << oopNum;
            flushStream();

            for (size_t iopIdx = 0; iopIdx < iopNum; iopIdx++) {
                uint64_t rawIdx = srcFunc->GetOperationIOperand(opIdx, iopIdx)->rawIndex;
                auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

                oss << "    iop [" << std::setw(IDENT_SIZE_THREE) << iopIdx << "]: rawMagic=" << rawTensor->rawMagic
                    << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
                flushStream();

                uint32_t dimSize = rawTensor->GetDim();
                dumpOperandShape(dimSize, opIdx, iopIdx, true);
                dumpRawShape(rawTensor, dimSize);
            }

            for (size_t oopIdx = 0; oopIdx < oopNum; oopIdx++) {
                uint64_t rawIdx = srcFunc->GetOperationOOperand(opIdx, oopIdx)->rawIndex;
                auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

                oss << "    oop [" << std::setw(IDENT_SIZE_THREE) << oopIdx << "]: rawMagic=" << rawTensor->rawMagic
                    << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
                flushStream();

                uint32_t dimSize = rawTensor->GetDim();
                dumpOperandShape(dimSize, opIdx, oopIdx, false);
                dumpRawShape(rawTensor, dimSize);
            }
        }

        return lines;
    }

    std::string DumpDyn(int funcIdx, const DevCceBinary *cceBinary) {
        std::stringstream oss;
        auto func = GetSource();

        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            oss << std::hex << "[" << opIdx << "] #predCnt:" << GetOperationCurrPredCount(opIdx);
            auto &succList = func->GetOperationDepGraphSuccList(opIdx);
            oss << " #succList: [";
            for (size_t j = 0; j < succList.size(); j++) {
                if (j != 0)
                    oss << ", ";
                oss << func->At(succList, j);
            }
            oss << ']';
            auto &stitch = GetOperationStitch(opIdx);
            if (!stitch.IsNull())
                oss << std::hex << " #stitch:" << stitch.Dump();
            oss << "\n";
        }

        auto dumpAttr = [this, &oss, func](const SymInt *attrs, const auto &info) {
            int attrOffset = info.staticOffsetAttrBeginIndex;
            auto rawIndex = attrs[attrOffset - 1].Value();
            oss << "@(rawidx:" << rawIndex << " attridx:" << (attrOffset - 1) << ")" << ", ";

            int dim = info.GetDim();
            auto rawTensor = func->GetRawTensor(rawIndex);
            if (rawIndex >= func->GetRawTensorSize()) {
                DEV_ERROR("Invalid rawIndex=%lu, exceeds raw tensor size=%lu", rawIndex, func->GetRawTensorSize());
            }
            if (dim != rawTensor->GetDim()) {
                DEV_ERROR("Dimension mismatch: info.dim=%d, rawTensor->dim=%d", dim, rawTensor->GetDim());
            }
            DEV_ASSERT(rawIndex < func->GetRawTensorSize());
            DEV_ASSERT(dim == rawTensor->GetDim());

            for (int d = 0; d < rawTensor->GetDim(); d++) {
                auto shapeIdx = attrOffset + d + rawTensor->GetDim() * 2;
                auto shape = static_cast<int64_t>(rawTensor->shape.At(d, funcData->exprTbl));
                auto actualShape = GetValue(attrs, shapeIdx);
                if (actualShape != shape) {
                    DEV_ERROR("Shape mismatch at dim %d: expacted=%ld, got=%ld", d, shape, actualShape);
                }
                DEV_ASSERT(actualShape == shape);
            }
            if (dim != rawTensor->GetDim()) {
                DEV_ERROR("Final dimension mismatch after shape validation: info.dim=%d, rawTensor->dim=%d", dim, rawTensor->GetDim());
            }
            DEV_ASSERT(dim == rawTensor->GetDim());
            for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
                oss << GetValue(attrs, attrOffset + i) << ", ";
            }
        };

        oss << " #funcKey: " << func->funcKey << " #gmStackBase: " << funcData->stackWorkSpaceAddr
            << " #stackSize: " << funcData->stackWorkSpaceSize << " #workspace: " << funcData->workspaceAddr << "\n";
        oss << "#funcData: [\n" << std::dec;
        for (size_t operIdx = 0; operIdx < func->GetOperationSize(); operIdx++) {
            auto attrBase = &func->GetOperationAttr(operIdx, 0);
            auto funcIndex = attrBase[0].Value();
            oss << "  [" << operIdx << "]  #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
                << " #funcIndex: " << funcIndex
                << " #taskID:" << MakeTaskID(funcIdx, operIdx) << " #opMagic: " << func->GetOperationDebugOpmagic(operIdx)
                << "\n";
            oss << "  #invokeAttrs : ";
            int offset = 0;
            for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                oss << " in:";
                dumpAttr(attrBase, opInfo);
            }
            for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                oss << " out:";
                dumpAttr(attrBase, opInfo);
            }
            oss << "\n other attr:";
            for (size_t idx = offset; idx < func->GetOperationAttrSize(operIdx); idx++) {
                oss << GetValue(attrBase, idx) << ", ";
            }
            oss << "\n";
        }
        oss << std::hex << "  #rawTensorAddrs: ";
        for (uint64_t i = 0; i < func->GetRawTensorDescSize(); i++) {
            if (i % RAW_TENSOR_DESC_PRE_SIZE == 0)
                oss << "\n   ";
            if (GetRawTensorAddrEx(i) != GetRawTensorAddr(i)) {
                DEV_ERROR("Tensor address mismatch at index %lu: addr=%lu, addrEx=%lu.", i, GetRawTensorAddr(i), GetRawTensorAddrEx(i));
            }
            DEV_ASSERT(GetRawTensorAddrEx(i) == GetRawTensorAddr(i));
            auto desc = funcData->rawTensorDesc[i];
            oss << GetRawTensorAddrEx(i) << "(location:" << desc.location << " offsetOrIdex: " << desc.offsetOrIndex << ")" << ", ";
        }
        oss << "\n]";
        return oss.str();
    }

    bool IsNull() const { return !dupTiny_; }
    void ResetNull() { dupTiny_.Invalidate(); }
    DynFuncData *GetFuncData() { return funcData; }
    void SetFuncData(DynFuncData *data) { funcData = data; }

    DevAscendFunctionDuppedData *DupDataForDynFuncData() { return DupData(); }
private:
    const DevAscendFunctionDuppedData *DupData() const { return dupTiny_.As<DevAscendFunctionDuppedData>(); }
    DevAscendFunctionDuppedData *DupData() { return dupTiny_.As<DevAscendFunctionDuppedData>(); }

private:
    DynFuncData *funcData{nullptr}; // used by aicore
    WsAllocation dupTiny_;
};
}