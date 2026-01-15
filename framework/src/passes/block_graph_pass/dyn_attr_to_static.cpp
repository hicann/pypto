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
 * \file dyn_attr_to_static.cpp
 * \brief
 */

#include "passes/block_graph_pass/dyn_attr_to_static.h"

namespace npu {
namespace tile_fwk {

struct CoaInfo {
    CoaType macroType = CoaType::INVALID;
    int dim = -1;
    int base = -1;
    int idx = -1;

    static bool ParseParamOffset(const std::string &coaExpr, std::smatch &match) {
         static std::regex pattern("RUNTIME_COA_GET_PARAM_OFFSET\\((\\d+), (\\d+), (\\d+)\\)");
         return std::regex_search(coaExpr, match, pattern);
    }

    static bool ParseParamValidShape(const std::string &coaExpr, std::smatch &match) {
         static std::regex pattern("RUNTIME_COA_GET_PARAM_VALID_SHAPE\\((\\d+), (\\d+), (\\d+)\\)");
         return std::regex_search(coaExpr, match, pattern);
    }

    static bool ParseParam(const std::string &coaExpr, std::smatch &match) {
            static std::regex pattern("RUNTIME_COA_GET_PARAM\\((\\d+)\\)");
            return std::regex_search(coaExpr, match, pattern);
    }

    Status SToIParamShapeAndOffset(const std::smatch &match) {
        if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), dim) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert dim.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_TWO].str(), base) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert base.");
            return FAILED;
        }
        if (SToIWrapper(match[INPUT_PARAM_POS_THREE].str(), idx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert idx.");
            return FAILED;
        }
        return SUCCESS;
    }

    Status ParseCoaString(const std::string &coaExpr) {
        std::smatch match;
        if (ParseParamOffset(coaExpr, match)) {
            macroType = CoaType::PARAM_OFFSET;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ParseCoaString failed to convert indices,"
                    "CoaType::PARAM_OFFSET, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else if (ParseParamValidShape(coaExpr, match)) {
            macroType = CoaType::PARAM_VALID_SHAPE;
            if (SToIParamShapeAndOffset(match) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ParseCoaString failed to convert indices,"
                    "CoaType::PARAM_VALID_SHAPE, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else if (ParseParam(coaExpr, match)) {
            macroType = CoaType::PARAM;
            if (SToIWrapper(match[INPUT_PARAM_POS_ONE].str(), idx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ParseCoaString failed to convert indices,"
                    "CoaType::PARAM, input coaExpr %s.", coaExpr.c_str());
                return FAILED;
            }
        } else if (coaExpr.find(MAYBE_CONST_POSTFIX) != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation, "This coaExpr %s has been processed", coaExpr.c_str());
            return FAILED;
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "ParseCoaString input coaExpr %s is not recognized.", coaExpr.c_str());
            return FAILED;
        }
        return SUCCESS;
    }

    int CalculateCoaIndex() {
        if (macroType == CoaType::PARAM_OFFSET) {
            return ((base) + 1) + OFFSET_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return ((base) + 1) + VALID_SHAPE_INDEX_ORDER * (dim) + idx;
        } else if (macroType == CoaType::PARAM) {
            return idx;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "GetCoaFinalIdx Coa type is invalid.");
        return 0;
    }

    SymbolicScalar BuildMaybeConstCoa(int isConst, int attrValue) {
        if (macroType == CoaType::PARAM_OFFSET) {
            return MAYBE_CONST_COA_GetOffset(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM_VALID_SHAPE) {
            return MAYBE_CONST_COA_GetValidShape(isConst, attrValue, dim, base, idx);
        } else if (macroType == CoaType::PARAM) {
            return MAYBE_CONST_COA_GetParam(isConst, attrValue, idx);
        }
        APASS_LOG_ERROR_F(Elements::Operation, "BuildMaybeConstCoa Coa type is invalid.");
        return 0;
    }
};

struct IsConstMetric {
    int isConst = 1;
    int attrValue = -1;

    void MarkNotConst() {isConst = 0;}
    int GetIsConst() {return isConst;}
    int GetAttrValue() {return attrValue;}
    bool TryInitAndCheckEqual(int newValue) {
        if (attrValue == -1) {
            attrValue = newValue;
            return true;
        }

        if (newValue < 0 || newValue != attrValue) {
            isConst = 0;
            return false;
        }
        return true;
    }
};

Status SToIWrapper(const std::string str, int& result) {
    try {
        result = std::stoi(str);
        return SUCCESS;
    } catch (const std::exception &e) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert %s to int, error is %s.", str.c_str(), e.what());
    }
    return FAILED;
}

void DynAttrToStatic::RefSpecifiedValue(std::vector<SymbolicScalar> &oriList,
    std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const
{
    for (auto &value : oriList) {
        newList.push_back(std::reference_wrapper<SymbolicScalar>(value));
    }
}

void DynAttrToStatic::FilterSpecifiedValue(std::vector<OpImmediate> &oriList,
    std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const
{
    for (auto &value : oriList) {
        if (value.IsSpecified()) {
            newList.push_back(std::reference_wrapper<SymbolicScalar>(value.GetSpecifiedValue()));
        }
    }
}

std::vector<std::reference_wrapper<SymbolicScalar>> DynAttrToStatic::GetOpDynamicAttributeList(Operation &op) {
    std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList;
    auto opcode = op.GetOpcode();
    if(opcode == Opcode::OP_VIEW) {
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr != nullptr) {
            RefSpecifiedValue(viewAttr->GetFromDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(viewAttr->GetToDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    }

    if(opcode == Opcode::OP_ASSEMBLE) {
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        if (assembleAttr != nullptr) {
            RefSpecifiedValue(assembleAttr->GetToDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(assembleAttr->GetFromDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    } 

    const std::set<Opcode> specifiedOps = {Opcode::OP_VEC_DUP, Opcode::OP_EXPAND, Opcode::OP_RESHAPE};
    if (specifiedOps.count(opcode)) {
        auto &attrDict = op.GetAllAttr();
        auto it = attrDict.find(OpAttributeKey::dynScalar);
        if (it != attrDict.end()) {
            auto &value = *npu::tile_fwk::AnyCast<SymbolicScalar>(&it->second);
            dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(value));
        }
        return dynamicAttributeList;
    }

    if (OpcodeManager::Inst().IsCopyInOrOut(opcode)) {
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        if (copyAttr != nullptr) {
            FilterSpecifiedValue(copyAttr->GetToOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetToDynValidShape(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromDynValidShape(), dynamicAttributeList);
        }
    }
    return dynamicAttributeList;
}

Status DynAttrToStatic::GetCallee(const Operation &callop, Function *&callFunc) {
    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop.GetOpAttribute());
    callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
    if (callFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get callee function %s failed.", callopAttr->GetCalleeMagicName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status DynAttrToStatic::BuildLeafToCaller(Function *func) {
    if (func->IsFunctionTypeAndGraphType(
        {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            Function *nextFunc = nullptr;
            if (GetCallee(*callop, nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.%s",
                    func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic(), GetFormatBacktrace(callop).c_str());
                return FAILED;
            }
            if (BuildLeafToCaller(nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, nextFunc at %s failed",
                    func->GetRawName().c_str(), nextFunc->GetRawName().c_str());
                return FAILED;
            }
        }
        return SUCCESS;
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        Function *rootFunc = func->GetRootFunction();
        return BuildLeafToCaller(rootFunc);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            Function *leafFunc = nullptr;
            if (GetCallee(*callop, leafFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.%s",
                    func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic(), GetFormatBacktrace(callop).c_str());
                return FAILED;
            }
            leaf2Caller[leafFunc].push_back(callop);
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s entered unexpected function type %d.",
        func->GetRawName().c_str(), static_cast<int>(func->GetFunctionType()));
    return FAILED;
}

Status DynAttrToStatic::BuildNewCoa(
        std::reference_wrapper<SymbolicScalar>& dynScalar,
        std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim)
{
    // 1. 拆解dynScalar到对应的COA表达式
    std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynScalar);
    if (dynParamExpr.find(COA_PREFIX) != 1) { // dynParamExpr格式是"(RUNTIME_GET_COA_XXX"
        APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa skips non-COA dynamic expression.");
        return SUCCESS;
    }
    CoaInfo coaExpr;
    if (coaExpr.ParseCoaString(dynParamExpr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildNewCoa found unexpected COA expression at dynParamExpr.");
        return FAILED;
    }
    int coaIndex = coaExpr.CalculateCoaIndex();
    
    // 2. 遍历不同caller下的取值，确认是否是常数
    IsConstMetric scalarValue;
    for (auto& argList : callopArglistOneDim) {
        auto& callopAttr = argList[coaIndex];
        if (!callopAttr.IsImmediate()) {
            scalarValue.MarkNotConst();
            break;
        } else {
            if (!scalarValue.TryInitAndCheckEqual(callopAttr.Concrete())) {
                break;
            }
        }
    }

    // 3. 刷新新的COA宏
    APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa update dynScalar[%s] with isConst=%d, value=%d.",
        dynParamExpr.c_str(), scalarValue.isConst, scalarValue.attrValue);
    dynScalar.get() = coaExpr.BuildMaybeConstCoa(scalarValue.isConst, scalarValue.attrValue);
    return SUCCESS;
}

inline int GetCoaIndex(const DynParamInfo &paramInfo) {
    if (paramInfo.type == DynParamInfoType::VALID_SHAPE) {
        return ((paramInfo.tensorBaseAddrCoaIndex + 1) + VALID_SHAPE_INDEX_ORDER * paramInfo.dimSize + paramInfo.dimIndex);
    }
    if (paramInfo.type == DynParamInfoType::OFFSET) {
       return ((paramInfo.tensorBaseAddrCoaIndex + 1) + OFFSET_INDEX_ORDER * paramInfo.dimSize + paramInfo.dimIndex);
    }
    return paramInfo.dimIndex;
}

void ReplaceCommonSymbol(Function *leafFunc, std::vector<std::vector<SymbolicScalar>> &callopArglistOneDim) {
    VectorParamConsistencyChecker checker;
    for (auto& argList : callopArglistOneDim) {
        checker.RegisterCall(argList);
    }
    auto allRes1 = checker.GetAllConsistentIndexGroups();
    APASS_LOG_DEBUG_F(Elements::Operation, "Get all condicate params: %s.",checker.PrintIndexGroups(allRes1).c_str());
    std::map<size_t, size_t> index2GroupId;
    std::map<size_t, std::vector<size_t>> groupId2Index;
    for (size_t i = 0; i < allRes1.size(); i++) {
        for (size_t j = 0; j < allRes1[i].size(); j++) {
            index2GroupId[allRes1[i][j]] = i;
        }
    } 
    std::map<std::string, int> symbol2CoaIdx;
    for (const auto &dynParam : leafFunc->GetDynParamTable()) {
        if (dynParam.second.dim.IsValid()) {
            std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynParam.second.dim);
            if (dynParamExpr.find(COA_PREFIX) != 1) {
                continue;
            }
        }
        int coaIndex = GetCoaIndex(dynParam.second);
        symbol2CoaIdx.emplace(dynParam.first, coaIndex);
        APASS_LOG_DEBUG_F(Elements::Operation, "Need Replace symbols %s idx %d", dynParam.first.c_str(), coaIndex);
    }
    std::map<size_t, std::string> index2BaseSymbol;
    for (auto [symbolStr, coaIdx] : symbol2CoaIdx) {
        if (index2GroupId.find(coaIdx) != index2GroupId.end()) {
            if (index2BaseSymbol.find(index2GroupId[coaIdx]) == index2BaseSymbol.end()) {
                leafFunc->GetMutableDynParam(symbolStr).isBaseParam = true;
                index2BaseSymbol[index2GroupId[coaIdx]] = symbolStr;
                APASS_LOG_INFO_F(Elements::Operation, "Mark coaIndex[%d] groupId[%zu] symbolStr[%s] as baseParam",
                    coaIdx, index2GroupId[coaIdx], symbolStr.c_str());
            } else {
                leafFunc->GetMutableDynParam(symbolStr).replacedSymbol = index2BaseSymbol[index2GroupId[coaIdx]];
                APASS_LOG_INFO_F(Elements::Operation, "Replace coaIndex[%d] groupId[%zu] symbolStr[%s] with baseParam[%s]", 
                    coaIdx, index2GroupId[coaIdx], symbolStr.c_str(), index2BaseSymbol[index2GroupId[coaIdx]]);
            }
        }
    }
}

inline SymbolicScalar BuildMaybeConstCoa(int attrValue, const DynParamInfo &paramInfo) {
    if (paramInfo.type == DynParamInfoType::OFFSET) {
        return MAYBE_CONST_COA_GetOffset(1, attrValue, paramInfo.dimSize, paramInfo.tensorBaseAddrCoaIndex, paramInfo.dimIndex);
    }
    if (paramInfo.type == DynParamInfoType::VALID_SHAPE) {
        return MAYBE_CONST_COA_GetValidShape(1, attrValue,paramInfo.dimSize, paramInfo.tensorBaseAddrCoaIndex, paramInfo.dimIndex);
    }
    return MAYBE_CONST_COA_GetParam(1, attrValue, paramInfo.dimIndex);
}

void ReBuildConcreteParam(Function *leafFunc, std::vector<std::vector<SymbolicScalar>> &callopArglistOneDim) {
    std::map<std::string, int> concreteParamCoaIdx;
    for (auto &dynParam : leafFunc->GetDynParamTable()) {
        if (dynParam.second.dim.IsValid()) {
            std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynParam.second.dim);
            if (dynParamExpr.find(COA_PREFIX) != 1) {
                continue;
            }
        }
        if (!(dynParam.second.isBaseParam) && !(dynParam.second.replacedSymbol.empty())) {
            continue;
        }
        auto coaIdx = GetCoaIndex(dynParam.second);
        APASS_LOG_DEBUG_F(Elements::Operation, "Get concrete symbols %s idx %d", dynParam.first.c_str(), coaIdx);
        IsConstMetric scalarValue;
        auto isConstParam = [&callopArglistOneDim, &scalarValue](int argIdx) {
            for (auto& calleeArgs : callopArglistOneDim) {
                auto callopAttr = calleeArgs[argIdx];
                if (!callopAttr.IsImmediate()) {
                    return false;
                }
                if (!scalarValue.TryInitAndCheckEqual(callopAttr.Concrete())) {
                    return false; 
                }
            }
            return true;
        };
        if (!isConstParam(coaIdx)) {
            continue;
        }
        auto constParam = BuildMaybeConstCoa(scalarValue.attrValue, dynParam.second);
        leafFunc->GetMutableDynParam(dynParam.first).dim = constParam;
    }
}

Status DynAttrToStatic::TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList) {
    // 1. 为leafFunc拿到它所有caller的一维的callopArglistOneDim
    std::vector<std::vector<SymbolicScalar>> callopArglistOneDim;
    for (size_t i = 0; i < callList.size(); i++) {
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
        callopArglistOneDim.push_back(callop->GetLinearArgList());
    }

    // 2. 依次为leafFunc的所有op拿到所有动态attr，为每个动态attr刷新coa宏
    auto operationViewer = leafFunc->Operations(false);
    for (size_t j = 0; j < operationViewer.size(); j++) {
        auto &op = operationViewer[j];
        std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = GetOpDynamicAttributeList(op);
        for (auto dynScalar : dynScalarList) {
            if (BuildNewCoa(dynScalar, callopArglistOneDim) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "TryRemoveDynAttr failed to execute BuildNewCoa for op [%d][%s].",
                    op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
        }
    }

    // 3. 为dynParam的赋值刷新coa宏
    ReplaceCommonSymbol(leafFunc, callopArglistOneDim);
    ReBuildConcreteParam(leafFunc, callopArglistOneDim);
    return SUCCESS;
}


Status DynAttrToStatic::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "==============> Start DynAttrToStatic.");
    // 1. 遍历所有rootFunc，找到每个leaf的所有caller，生成leaf2Caller map
    if (BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    // 2. 遍历leaf2Caller，尝试为每个leaf消除动态attributes
    for (const auto& pair : leaf2Caller) {
        if (TryRemoveDynAttr(pair.first, pair.second) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to call TryRemoveDynAttr for leafFunc %s.", pair.first->GetRawName().c_str());
            return FAILED;
        }
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "==============> End DynAttrToStatic.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu