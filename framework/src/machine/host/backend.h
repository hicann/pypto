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
 * \file backend.h
 * \brief
 */

#pragma once
#include "interface/machine/host/machine_task.h"
#include "interface/cache/function_cache.h"
namespace npu::tile_fwk {
MachineTask *GenCode(
    MachineTask *task, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset, FunctionCache &cache,
    std::string &kernelPath);

struct Linker {
    SymbolicSymbolTable &symbolTable_;

    DyndevFunctionAttribute::FunctionGroup &funcGroup_;
    DyndevFunctionAttribute::ExpressionTableDictGroup &exprTableDictGroup_;

    Linker(SymbolicSymbolTable &symbolTable,
           DyndevFunctionAttribute::FunctionGroup &funcGroup,
           DyndevFunctionAttribute::ExpressionTableDictGroup &exprTableDictGroup)
        : symbolTable_(symbolTable), funcGroup_(funcGroup), exprTableDictGroup_(exprTableDictGroup) {}

    void AddSymbol(SymbolicScalar &ss) {
        symbolTable_.AddSymbol(ss);
    }

    const DyndevFunctionAttribute::ExpressionTableDictGroup &GetExpressionTableDictGroup() const { return exprTableDictGroup_; }
    DyndevFunctionAttribute::ExpressionTableDictGroup &GetExpressionTableDictGroup() { return exprTableDictGroup_; }

    static std::string GetTitle(Function *func) {
        std::string title = "name=" + func->GetRawName() + " hash=" + std::to_string(func->GetFunctionHash().GetHash());
        return title;
    }

    void AddPrimaryExpressionForLoopBes(Function*func, const SymbolicScalar &ss) {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.loopList.InsertAndGetIndex(func);
        std::string key = SymbolicExpressionTable::GetExprKeyLoopBes(funcKey);

        auto &exprTable = exprTableDictGroup_.loopBesDict[func];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForLoopPathCond(Function *func, const SymbolicScalar &ss) {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.loopPathList.InsertAndGetIndex(func);
        auto condKey = funcGroup_.loopPathCondList[func].InsertAndGetIndex(ss.Raw());
        std::string key = SymbolicExpressionTable::GetExprKeyLoopPathCond(funcKey, condKey);

        auto &exprTable = exprTableDictGroup_.loopPathCondDict[func][ss.Raw()];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForDevRootCoa(Function *func, const SymbolicScalar &ss) {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.devRootList.InsertAndGetIndex(func);
        std::string key = SymbolicExpressionTable::GetExprKeyDevRootCoa(funcKey);

        auto &exprTable = exprTableDictGroup_.devRootCoaDict[func];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void AddPrimaryExpressionForDevLeafOp(Function *func, Operation *op, const SymbolicScalar &ss) {
        AddSymbolFromExpression(ss);

        auto funcKey = funcGroup_.devLeafList.InsertAndGetIndex(func);
        auto opKey = funcGroup_.devLeafOpList[func].InsertAndGetIndex(op);
        std::string key = SymbolicExpressionTable::GetExprKeyDevLeafOp(funcKey, opKey);

        auto &exprTable = exprTableDictGroup_.devLeafOpDict[func][op];
        exprTable.AddPrimaryExpression(ss);
        exprTable.SetElementKeyOnce(key);
        exprTable.SetTitleOnce(GetTitle(func));
    }

    void SetMainBlockExpressionForDevRootCoa(Function *func, const SymbolicScalar &ss) {
 	    auto &exprTable = exprTableDictGroup_.devRootCoaDict[func];
 	    exprTable.mainBlockScalar_ = ss;
 	}

    SymbolicExpressionTable *LookupDevRootCoa(Function *func) {
        if (exprTableDictGroup_.devRootCoaDict.count(func)) {
            return &exprTableDictGroup_.devRootCoaDict[func];
        } else {
            return nullptr;
        }
    }

    SymbolicSymbolTable *GetSymbolTable() {
        return &symbolTable_;
    }

private:
    void AddSymbolFromExpression(const SymbolicScalar &ss) {
        symbolTable_.AddSymbolFromExpression(ss);
    }
};

// Force link library compiler as nothing depends on it.
void ForceLinkLibraryCompiler();

}
