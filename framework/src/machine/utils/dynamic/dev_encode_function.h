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
 * \file dev_encode_function.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/dynamic/dev_encode_operation.h"

namespace npu::tile_fwk {
struct L2Info;
struct CceCodeInfo;
class RawTensor;
class LogicalTensor;
class Operation;
class Function;
class IncastOutcastLink;
class IncastOutcastSlot;
class SymbolicSymbolTable;
class SymbolicExpressionTable;
}

namespace npu::tile_fwk::dynamic {
constexpr int INVALID_INDEX = -1;

struct DevAscendFunctionPredInfo {
    uint64_t totalZeroPred;
    uint64_t totalZeroPredAIV;
    uint64_t totalZeroPredAIC;
    uint64_t totalZeroPredHub;
    uint64_t totalZeroPredAicpu;
};

struct EncodeDevAscendFunctionParam {
    /* The following are common parameter */
    std::unordered_map<uint64_t, int> calleeHashIndexDict;
    std::vector<CceCodeInfo> cceCodeInfoList;
    const SymbolicSymbolTable *symbolTable;
    const IncastOutcastLink *inoutLink;

    /* The following are per function parameter */
    const SymbolicExpressionTable *expressionTable;
    const IncastOutcastSlot *slot;
    Function *devRoot;
    std::vector<RuntimeSlotDesc> outcastDescList;
    std::vector<int> assembleSlotList;
};

struct DevAscendFunctionDuppedData;
struct DevAscendFunction {
    uint64_t rootHash;
    uint64_t funcKey;
    // source root function after duplication
    DevAscendFunction *sourceFunc{nullptr};

    // Fill base address after stitch
    uintdevptr_t runtimeWorkspace;
    // Base address of invoke entries
    uintdevptr_t opAttrs;

    int funcidx;

    int stackWorkSpaceSize;

    uint32_t getInputDataCount;
    uint32_t getTensorDataCount;
    uint64_t hubOpCount_{0};

    DevLocalVector<AddressDescriptor> incastAddressList;
    DevLocalVector<AddressDescriptor> outcastAddressList;

    DevLocalVector<uint64_t> expressionList;
#define allocateLastField expressionList

    DevAscendFunctionPredInfo predInfo_;
    uint64_t duppedDataAllocSize_;
    uint64_t duppedDataCopySize_;
    DevLocalVector<uint8_t> duppedData_;
// wrap for mix subgraph schedule
public:
    uint64_t wrapIdNum_{0};
    int *GetOpWrapListAddr() { return &At(opWrapList_, 0); }
    int *GetOpWrapTaskNumListAddr() { return &At(opWrapTaskNumList_, 0); }

    // total memory requirement of non-root-incast/outcast raw tensors
    uint64_t rootInnerTensorWsMemoryRequirement{0};
    uint64_t exclusiveOutcastWsMemoryRequirement{0};

private:
    DevLocalVector<int> opWrapList_;
    DevLocalVector<int> opWrapTaskNumList_;
    DevLocalVector<DevAscendRawTensor> rawTensorList_;
    DevLocalVector<DevRawTensorDesc> rawTensorDescList_;
    DevLocalVector<DevAscendTensor> tensorList_;
    DevLocalVector<int> noPredOpList_;
    DevLocalVector<int> noSuccOpList_;
    DevLocalVector<DevAscendOperation> operationList_;
    DevLocalVector<DevAscendOperationOperandInfo> operationOperandInfoList_;
    DevLocalVector<SymInt> operationAttrList_;
    DevLocalVector<int> opAttrOffsetList_;
    DevLocalVector<int> opCalleeList_;
    DevLocalVector<int> operationSuccList_;
    DevLocalVector<int> operationCopyOutResolveSuccIndexList_;

    DevLocalVector<DevAscendFunctionIncast> incastList;
    DevLocalVector<DevAscendFunctionOutcast> outcastList;
    DevLocalVector<int> slotList;
    DevLocalVector<int> redaccAssembleSlotList_;

    DevLocalVector<DevAscendFunctionCallOperandUse> useList;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList_;
    DevLocalVector<uint32_t> stitchPolicyFullCoverOpList_;

    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTableList;
    DevLocalVector<uint32_t> cellMatchStaticOutcastTableList;
    DevLocalVector<uint32_t> cellMatchStaticIncastTableList;
    DevLocalVector<char> rawName_;
#define sharedLastField rawName_
public:
    uint8_t data[0];
    /*
     *  Duplicated:
     *      AddressDescriptor                                   incastAddressListData;
     *      AddressDescriptor                                   outcastAddressListData;
     *      DevAscendOperationDynamicField                      opDynamicFieldListData[];
     *  Allocated:
     *      uint64_t                                            expressionListData[];
     *
     *  Shared:
     *      DevRawTensorDesc                              rawTensorDescListData[];
     *      DevAscendRawTensor                                  rawTensorListData[];
     *      DevAscendTensor                                     tensorListData[];
     *      int                                                 noPredOpListData[];
     *      int                                                 noSuccOpListData[];
     *      DevAscendOperation                                  operationListData[];
     *      DevAscendOperationOperandInfo                       operationOperandListData[];
     *      SymInt                                              operationAttrListData[];
     *      int                                                 operationSuccListData[];
     *      int                                                 operationCopyOutResolveSuccIndexData[];
     *      DevAscendFunctionIncast                             incastListData[];
     *      DevAscendFunctionOutcast                            outcastListData[];
     *      int                                                 slotListData[];
     *      int                                                 outputOutcastSlotList[];
     *      int                                                 assembleOutcastSlotList[];
     *      int                                                 offsetIdxListData[];
     *      int                                                 shapeIdxListData[];
     *      int                                                 producerConsumerListData[];
     *      char                                                rawNameData[];
     *      uint8_t                                             duppedData[];
     */

    template <typename T>
    const T &At(const DevLocalVector<T> &localvec, int index) const {
        return *reinterpret_cast<T *>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
    template <typename T>
    T &At(const DevLocalVector<T> &localvec, int index) {
        return *reinterpret_cast<T *>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
    template <typename T>
    const T &At(const DevRelocVector<T> &localvec, int index) const {
        return localvec[index];
    }
    template <typename T>
    T &At(DevRelocVector<T> &localvec, int index) {
        return localvec[index];
    }

private:
    std::string DumpTensor(int tensorIndex) const;
    std::string DumpOperationAttr(int operationIndex, uint64_t *runtimeExpressionList = nullptr, bool dumpIndex=false) const;
    std::string DumpOperation(int operationIndex, int &totalAttrStartIdx,
        const std::vector<uintdevptr_t> &ooperandAddrList = {}, const std::vector<uintdevptr_t> &ioperandAddrList = {},
        uint64_t *runtimeExpressionList = nullptr) const;
    std::string DumpRawTensor(int rawIndex, uintdevptr_t addr = 0) const;
    std::string DumpIncast(int incastIndex, const std::string &indent, uint64_t *runtimeExpressionList = nullptr,
        const std::vector<uintdevptr_t> &slotAddrList = {}) const;
    std::string DumpOutcast(int outcastIndex, const std::string &indent, uint64_t *runtimeExpressionList = nullptr,
        const std::vector<uintdevptr_t> &slotAddrList = {}) const;

public:
    std::string Dump(int indent = 0) const;

    bool HasValueDepend() const {
        return getInputDataCount + getTensorDataCount;
    }

    schema::coa SchemaGetCoa(int operationIndex, uint64_t *runtimeExpressionList = nullptr, bool dumpIndex = false) const {
        std::vector<schema::TextType> coaDataList;
        for (size_t j = 0; j < GetOperationAttrSize(operationIndex); j++) {
            const SymInt &s = GetOperationAttr(operationIndex, j);
            std::string textData;
            if (s.IsExpression()) {
                if (runtimeExpressionList != nullptr) {
                    textData = std::to_string(runtimeExpressionList[s.Value()]);
                } else {
                    textData = "?" + std::to_string(s.Value());
                }
            } else {
                textData = std::to_string(s.Value());
            }
            coaDataList.push_back(schema::TextType(textData));
        }
        return schema::coa(schema::coaType(coaDataList, dumpIndex));
    }

    template <typename T>
    uint64_t GetEndOffset(const DevLocalVector<T> &localvec) const {
        return localvec.End();
    }

    void Reloc(intptr_t /* shift */, bool /* relocShared */) {}

    int GetFuncKey() const { return funcKey; }

    const DevAscendFunction *GetSource() const { return sourceFunc; }
    DevAscendFunction *GetSource() { return sourceFunc; }

    int GetRootIndex() const { return funcKey; }

    const int &GetFuncidx() const { return funcidx; }
    int &GetFuncidx() { return funcidx; }

    const DevAscendFunctionPredInfo &GetPredInfo() const { return predInfo_; }
    uint64_t GetDuppedDataAllocSize() const { return duppedDataAllocSize_; }
    uint64_t GetDuppedDataCopySize() const { return duppedDataCopySize_; }
    DevAscendFunctionDuppedData *GetDuppedData() const { return reinterpret_cast<DevAscendFunctionDuppedData *>(const_cast<uint8_t*>(&At(duppedData_, 0))); }

    int32_t *GetOpAttrOffsetAddr() { return &At(opAttrOffsetList_, 0); }
    inline int32_t GetOpAttrOffsetSize() { return opAttrOffsetList_.size(); }
    int *GetCalleeIndexAddr() { return &At(opCalleeList_, 0); }

    uint64_t *GetExpressionAddr() { return &At(expressionList, 0); }
    uint64_t GetAllocateSize() const { return GetEndOffset(allocateLastField); }

    uint64_t GetSize() const { return GetEndOffset(sharedLastField); }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace; }
    uintdevptr_t &GetRuntimeWorkspace() { return runtimeWorkspace; }

    inline AddressDescriptor GetIncastAddress(int index) const { return At(incastAddressList, index); }
    inline AddressDescriptor &GetIncastAddress(int index) { return At(incastAddressList, index); }

    inline AddressDescriptor GetOutcastAddress(int index) const { return At(outcastAddressList, index); }
    inline AddressDescriptor &GetOutcastAddress(int index) { return At(outcastAddressList, index); }

    inline uint64_t GetExpression(int tableIndex) const { return At(expressionList, tableIndex); }
    inline uint64_t &GetExpression(int tableIndex) { return At(expressionList, tableIndex); }
    inline uint64_t GetExpressionSize() const { return expressionList.size(); }
    inline uint64_t GetRawTensorSize() const { return rawTensorList_.size(); }
    inline const DevAscendRawTensor *GetRawTensor(const DevAscendTensor *tensor) const {
        int rawTensorIndex = tensor->rawIndex;
        return &At(rawTensorList_, rawTensorIndex);
    }
    inline const DevAscendRawTensor *GetRawTensor(int rawIndex) const { return &At(rawTensorList_, rawIndex); }
    inline DevAscendRawTensor *GetRawTensor(int rawIndex) { return &At(rawTensorList_, rawIndex); }
    inline const DevRawTensorDesc *GetRawTensorDesc(int rawIndex) const { return &At(rawTensorDescList_, rawIndex); }
    inline DevRawTensorDesc *GetRawTensorDesc(int rawIndex) { return &At(rawTensorDescList_, rawIndex); }
    inline size_t GetRawTensorDescSize() const { return rawTensorDescList_.size(); }

    inline uint64_t GetTensorSize() const { return tensorList_.size(); }
    inline const DevAscendTensor *GetTensor(int index) const { return &At(tensorList_, index); }
    inline DevAscendTensor *GetTensor(int index) { return &At(tensorList_, index); }

    inline size_t GetNoPredOpSize() const { return noPredOpList_.size(); }
    inline int GetNoPredOpIdx(size_t idx) const { return At(noPredOpList_, idx); }

    inline size_t GetNoSuccOpSize() const { return noSuccOpList_.size(); }
    inline int GetNoSuccOpIdx(size_t idx) const { return At(noSuccOpList_, idx); }

    inline size_t GetOperationSize() const { return operationList_.size(); }
    inline uint32_t GetOperationOutcastStitchIndex(int operationIndex) const {
        return At(operationList_, operationIndex).outcastStitchIndex;
    }
    inline uint32_t GetOperationDebugOpmagic(int operationIndex) const {
        return At(operationList_, operationIndex).debugOpmagic;
    }
    inline size_t GetOperationIOperandSize(int operationIndex) const {
        return At(operationList_, operationIndex).ioperandList.size();
    }
    inline size_t GetOperationOOperandSize(int operationIndex) const {
        return At(operationList_, operationIndex).ooperandList.size();
    }

    inline const DevAscendOperationOperandInfo &GetOperationIOperandInfo(int operationIndex, int operandIndex) const {
        return At(At(operationList_, operationIndex).ioperandList, operandIndex);
    }
    inline const DevAscendTensor *GetOperationIOperand(int operationIndex, int operandIndex) const {
        int tensorIndex = GetOperationIOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo &GetOperationOOperandInfo(int operationIndex, int operandIndex) const {
        return At(At(operationList_, operationIndex).ooperandList, operandIndex);
    }
    inline const DevAscendTensor *GetOperationOOperand(int operationIndex, int operandIndex) const {
        int tensorIndex = GetOperationOOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo &GetOperationOperandInfo(
        int operationIndex, int operandIndex, bool isIOperand = true) const {
        if (isIOperand) {
            return GetOperationIOperandInfo(operationIndex, operandIndex);
        } else {
            return GetOperationOOperandInfo(operationIndex, operandIndex);
        }
    }

    inline size_t GetOperationAttrSize(int operationIndex) const {
        return At(operationList_, operationIndex).attrList.size();
    }
    inline const SymInt &GetOperationAttr(int operationIndex, int attrIndex) const {
        return At(At(operationList_, operationIndex).attrList, attrIndex);
    }
    inline int GetOperationAttrCalleeIndex(int operationIndex) const {
        return GetOperationAttr(operationIndex, 0).Value();
    }

    inline int GetOpAttrSize() { return operationAttrList_.size(); }

    inline void FillOpAttrs(DevCceBinary *cceInfo) {
        (void)cceInfo;
    }

    inline const uint32_t &GetOperationDepGraphPredCount(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphPredCount;
    }
    inline uint32_t &GetOperationDepGraphPredCount(int operationIndex) {
        return At(operationList_, operationIndex).depGraphPredCount;
    }

    inline const DevLocalVector<int> &GetOperationDepGraphSuccList(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphSuccList;
    }

    inline const DevLocalVector<int> &GetOperationDepGraphCopyOutResolveSuccIndexList(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
    }

    inline const int *GetOperationDepGraphSuccAddr(int operationIndex, size_t &size) const {
        auto &succList = At(operationList_, operationIndex).depGraphSuccList;
        size = succList.size();
        return &At(succList, 0);
    }

    inline const int *GetOperationDepGraphCopyOutResolveSuccIndexAddr(int operationIndex, size_t &size) const {
        auto &succIndexList = At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
        size = succIndexList.size();
        return &At(succIndexList, 0);
    }

    inline size_t GetIncastSize() const { return incastList.size(); }
    inline const struct DevAscendFunctionIncast &GetIncast(int index) const { return At(incastList, index); }
    inline struct DevAscendFunctionIncast &GetIncast(int index) { return At(incastList, index); }
    inline const DevAscendRawTensor *GetIncastRawTensor(int index) const {
        int tensorIndex = GetIncast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline size_t GetOutcastSize() const { return outcastList.size(); }
    inline const struct DevAscendFunctionOutcast &GetOutcast(int index) const { return At(outcastList, index); }
    inline struct DevAscendFunctionOutcast &GetOutcast(int index) { return At(outcastList, index); }

    inline size_t GetRedaccAssembleSlotListSize() const { return redaccAssembleSlotList_.size(); }
    inline const int &GetRedaccAssembleSlotList(int index) const { return At(redaccAssembleSlotList_, index); }
    inline int &GetRedaccAssembleSlotList(int index) { return At(redaccAssembleSlotList_, index); }

    int LookupIncastBySlotIndex(int slotIndex) const {
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
            for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
                int slot = At(incast.fromSlotList, fromIndex);
                if (slot == slotIndex) {
                    return static_cast<int>(incastIndex);
                }
            }
        }
        return INVALID_INDEX;
    }
    std::vector<int> LookupIncastBySlotIndexList(const std::vector<int> &slotIndexList) const {
        std::vector<int> resultList(slotIndexList.size());
        for (size_t i = 0; i < slotIndexList.size(); i++) {
            resultList[i] = LookupIncastBySlotIndex(slotIndexList[i]);
        }
        return resultList;
    }

    int LookupOutcastBySlotIndex(int slotIndex) const {
        for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
            const DevAscendFunctionOutcast &outcast = GetOutcast(outcastIndex);
            for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
                int slot = At(outcast.toSlotList, toIndex);
                if (slot == slotIndex) {
                    return static_cast<int>(outcastIndex);
                }
            }
        }
        return INVALID_INDEX;
    }
    std::vector<int> LookupOutcastBySlotIndexList(const std::vector<int> &slotIndexList) const {
        std::vector<int> resultList(slotIndexList.size());
        for (size_t i = 0; i < slotIndexList.size(); i++) {
            resultList[i] = LookupOutcastBySlotIndex(slotIndexList[i]);
        }
        return resultList;
    }

    std::vector<std::tuple<int, int, int>> LookupConnectionSlotIndexFrom(const DevAscendFunction *func) const {
        std::vector<std::tuple<int, int, int>> connectionList;
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
            for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
                int fromSlot = At(incast.fromSlotList, fromIndex);

                for (size_t outcastIndex = 0; outcastIndex < func->GetOutcastSize(); outcastIndex++) {
                    const DevAscendFunctionOutcast &outcast = func->GetOutcast(outcastIndex);
                    for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
                        int toSlot = func->At(outcast.toSlotList, toIndex);
                        if (fromSlot == toSlot) {
                            connectionList.push_back(std::tuple(outcastIndex, incastIndex, fromSlot));
                        }
                    }
                }
            }
        }
        return connectionList;
    }

    inline const DevAscendRawTensor *GetOutcastRawTensor(int index) const {
        int tensorIndex = GetOutcast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline void GetTensorOffset(uint64_t offset[DEV_SHAPE_DIM_MAX], const DevAscendRawTensor *rawTensor,
        const DevAscendOperationOperandInfo &operandInfo) const {
        const SymInt *offsetSymList = &At(operationAttrList_, operandInfo.staticOffsetAttrBeginIndex);
        for (int i = 0; i < rawTensor->GetDim(); i++) {
            offset[i] = offsetSymList[i].IsExpression() ? At(expressionList, offsetSymList[i].Value()) :
                                                          offsetSymList[i].Value();
        }
    }

    inline const SymInt *GetSymoffset(int offset) const { return &At(operationAttrList_, offset); }

    struct SymIntPair {
        const SymInt *offsetSymList;
        const SymInt *shapeSymList;
    };
    inline SymIntPair GetTensorOffsetShapeSymList(
            int operationIndex, int operandIndex, bool isIOperand = true) const {
        auto &operandInfo = GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
        const SymInt *offsetSymList = &GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        const SymInt *shapeSymList = &GetOperationAttr(operationIndex, operandInfo.staticShapeAttrBeginIndex);
        return SymIntPair{offsetSymList, shapeSymList};
    }

    inline const char *GetRawName() const { return &At(rawName_, 0); }

private:
    friend struct EncodeDevAscendFunctionInfo;

    void InitIncastOutcastAttr(
            uintdevptr_t &initOffset,
            const std::vector<std::shared_ptr<LogicalTensor>> &iList,
            const std::vector<std::shared_ptr<LogicalTensor>> &oList, bool fillContent);
    void InitOperationDynamicField(
            uintdevptr_t &initOffset,
            DevAscendFunctionPredInfo predInfo,
            uint32_t outcastStitchCount,
            const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
            const SymbolicExpressionTable *expressionTable,
            const OrderedSet<Operation *> &callList,
            const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
            const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
            const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,bool fillContent);
    void FillExclusiveOutcastSlotMark(const IncastOutcastLink *inoutLink, std::vector<bool>& isExclusiveOutcastSlotMarks);
    void InitRawTensorAndMemoryRequirement(
            uintdevptr_t &initOffset,
            const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
            const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
            const std::vector<EncodeRawTensorAttr> &rawAttrs,
            const EncodeDevAscendFunctionParam &param,
            const SymbolicExpressionTable *expressionTable,
            bool fillContent);

    void UpdateRawTensorDesc(const std::shared_ptr<RawTensor> &rawTensor, size_t i, size_t incastRawListSize,
        DevAscendRawTensor &encoded);

    void InitTensor(
            uintdevptr_t &initOffset,
            const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList, bool fillContent);

    void InitOperation(
            uintdevptr_t &initOffset,
            const SymbolicExpressionTable *expressionTable,
            const OrderedSet<Operation *> &callList,
            const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
            const std::unordered_map<Operation *, uint64_t> &callOpPredDict,
            const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,
            const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
            const std::vector<int32_t> &outcastStitchIndexList,
            const std::vector<int> &noPredOpList,
            const std::vector<int> &noSuccOpList,
            const std::unordered_map<Operation *, std::vector<int>> &copyOutResolveSuccIndexListDict,
            bool fillContent);
    void InitWrapInfo(uintdevptr_t &initOffset, const OrderedSet<Operation *> &callList, bool fillContent);
    void InitIncastOutcast(uintdevptr_t &initOffset, const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &incastOpAttrDict,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &outcastOpAttrDict,
        const EncodeDevAscendFunctionParam &param, const std::string &initRawName, bool fillContent);
};
}