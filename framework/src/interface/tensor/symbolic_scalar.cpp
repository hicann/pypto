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
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar.h"
#include <sys/mman.h>
#include <sstream>
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"

constexpr uint64_t IMMEDIATE = 0;
constexpr uint64_t SYMBOL = 1;
constexpr uint64_t EXPRESSION = 2;
constexpr int OPERAND_NUM = 2;
namespace npu::tile_fwk {

std::vector<uint8_t> CompileAndLoadSection(const std::string &code, const std::string &sourceFilePath,
    const std::string &gcc, const std::string &objcopy, const std::string &sectionName, bool needDump, const std::string &extraCflag) {
    if (needDump) {
        FILE *fsrc = fopen(sourceFilePath.c_str(), "w");
        fprintf(fsrc, "%s", code.c_str());
        fclose(fsrc);
    }

    std::string assembleFilePath = sourceFilePath + ".s";
    std::string objectFilePath = sourceFilePath + ".o";
    std::string binaryFilePath = sourceFilePath + ".bin";
    std::string LD_PRELOAD = "LD_PRELOAD= ";
    std::string includePath = GetCurrentSharedLibPath() + "/../include/tile_fwk";
    std::string cmdGcc = LD_PRELOAD + gcc + " -fPIC -O2 " + extraCflag +
        " -I" + includePath + " " +
        " -I" + GetCurrentSharedLibPath() + "/include/" +
        " -I" + includePath + "/tilefwk " +
        " -S " + sourceFilePath + " -o " + assembleFilePath;
    ALOG_INFO("[RunCmd] ", cmdGcc);
    ASSERT(system(cmdGcc.c_str()) == 0);

    std::string cmdAs = LD_PRELOAD + gcc + " -O2 -c " + assembleFilePath + " -o " + objectFilePath;
    ALOG_INFO("[RunCmd] ", cmdAs);
    ASSERT(system(cmdAs.c_str()) == 0);

    std::string cmdObjcopy = LD_PRELOAD + objcopy + " --dump-section " + sectionName + "=" + binaryFilePath + " " + objectFilePath;
    ALOG_INFO("[RunCmd] ", cmdObjcopy);
    ASSERT(system(cmdObjcopy.c_str()) == 0);

    FILE *fbin = fopen(binaryFilePath.c_str(), "rb");
    if (fbin == nullptr) {
        ALOG_FATAL("open binary file name failed");
        return {};
    }

    fseek(fbin, 0, SEEK_END);
    int size = ftell(fbin);
    fseek(fbin, 0, SEEK_SET);
    std::vector<uint8_t> binary(size);
    size_t readSize = fread(binary.data(), 1, size, fbin);
    if (readSize != static_cast<size_t>(size)) {
        fclose(fbin);
        return {};
    }
    fclose(fbin);
    return binary;
}

void SymbolicExpressionTable::SetElementKeyOnce(const std::string &key) {
    if (elementKey_.size() == 0) {
        elementKey_ = key;
    } else {
        ASSERT(elementKey_ == key);
    }
}

void SymbolicExpressionTable::SetTitleOnce(const std::string &title) {
    if (title_.size() == 0) {
        title_ = title;
    } else {
        ASSERT(title_ == title);
    }
}

std::string SymbolicExpressionTable::BuildExpression(const SymbolicScalar &ss) {
    return BuildExpression(ss.Raw());
}

std::string SymbolicExpressionTable::BuildExpression(const RawSymbolicScalarPtr &ss) {
    std::string expr = BuildExpressionByRaw(ss, {});
    return expr;
}

std::string SymbolicExpressionTable::BuildExpressionByRaw(const RawSymbolicScalarPtr &raw, const std::unordered_map<RawSymbolicScalarPtr, std::string> &exprDict) {
    if (exprDict.count(raw)) {
        return exprDict.find(raw)->second;
    }
    std::string result;
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            auto immediate = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw);
            result = std::to_string(immediate->Immediate());
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            auto symbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            if (CheckRuntimePrefix(symbol->Name())) {
                result = symbol->Name();
            } else if (CheckArgPrefix(symbol->Name())) {
                result = symbol->Name();
            } else {
                if (symbol->Name().rfind("sym_", 0) == 0)
                    result = symbol->Name();
                else
                    result = "VALUE_" + symbol->Name();
            }
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            RawSymbolicExpPtr expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
            result = BuildExpressionCode(expr, exprDict);
        } break;
        default: ASSERT(false); break;
    }
    return result;
}

std::string SymbolicExpressionTable::BuildExpressionCode(const RawSymbolicExpPtr &expr, const std::unordered_map<RawSymbolicScalarPtr, std::string> &exprDict) {
    std::ostringstream oss;
    oss << "(";
    if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
        oss << RawSymbolicExpression::GetSymbolicCalcOpcode(expr->Opcode());
        oss << BuildExpressionByRaw(expr->OperandList()[0], exprDict);
    } else if (SymbolicOpcode::T_BOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_BOP_END) {
        if (expr->Opcode() == SymbolicOpcode::T_BOP_MAX) {
            oss << "RUNTIME_Max(";
            oss << BuildExpressionByRaw(expr->OperandList()[0], exprDict);
            oss << ", ";
            oss << BuildExpressionByRaw(expr->OperandList()[1], exprDict);
            oss << ")";
        } else if (expr->Opcode() == SymbolicOpcode::T_BOP_MIN) {
            oss << "RUNTIME_Min(";
            oss << BuildExpressionByRaw(expr->OperandList()[0], exprDict);
            oss << ", ";
            oss << BuildExpressionByRaw(expr->OperandList()[1], exprDict);
            oss << ")";
        } else {
            for (size_t idx = 0; idx < expr->OperandList().size(); idx++) {
                if (idx != 0) {
                    oss << " " + RawSymbolicExpression::GetSymbolicCalcOpcode(expr->Opcode()) + " ";
                }
                oss << BuildExpressionByRaw(expr->OperandList()[idx], exprDict);
            }
        }
    } else if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
        std::string callee = BuildExpressionByRaw(expr->OperandList()[0], exprDict);
        if (CheckRuntimePrefix(callee)) {
            oss << callee;
        } else {
            oss << "((Call" << expr->OperandList().size() << "EntryType)" << callee << ")";
        }
        oss << "(";
        for (size_t idx = 1; idx < expr->OperandList().size(); idx++) {
            oss << (idx == 1 ? "" : ", ");
            oss << BuildExpressionByRaw(expr->OperandList()[idx], exprDict);
        }
        oss << ")";
    }
    oss << ")";
    return oss.str();
}

std::string SymbolicExpressionTable::BuildExpressionList() const {
    constexpr int INDENT = 0x20;
    std::ostringstream oss;
    std::unordered_map<RawSymbolicScalarPtr, std::string> exprDict;

    oss << "\n";
    oss << "/* Function info " << elementKey_ << ": " << title_ << " */\n";
    for (auto &expr : expressionSet) {
        int index = expressionSet.GetIndex(expr);
        std::string exprNameTempVarFlag = GetExprNameTempVarFlag(elementKey_, index);
        std::string exprNameTempVar = GetExprNameTempVar(elementKey_, index);
        std::string exprNameTempVarInit = GetExprNameTempVarInit(elementKey_, index);
        std::string exprNameCalc = GetExprNameCalc(elementKey_, index);
        std::string exprNameGet = GetExprNameUse(elementKey_, index);
        std::string calc = BuildExpressionByRaw(expr, exprDict);

        if (primaryExpressionSet.count(expr)) {
            oss << "\n";
            oss << "/* Full Expression: " << BuildExpressionByRaw(expr, {}) << " */" << "\n";
        }

        oss << "#define " << std::left << std::setw(INDENT) << exprNameTempVarFlag << 0 << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameTempVar << "tempVar_" << elementKey_ << "_" << index << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameCalc << calc << "\n";
        oss << "#if     " << exprNameTempVarFlag << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameTempVarInit << "int64_t " << exprNameTempVar << " = " << exprNameCalc << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameGet << exprNameTempVar << "\n";
        oss << "#else /*" << exprNameTempVarFlag << " */\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameTempVarInit << "\n";
        oss << "#define " << std::left << std::setw(INDENT) << exprNameGet << exprNameCalc << "\n";
        oss << "#endif/*" << exprNameTempVarFlag << " */\n";
        exprDict[expr] = exprNameGet;
    }
    return oss.str();
}

std::string SymbolicExpressionTable::BuildExpressionTempVarInit(int indent) {
    std::ostringstream oss;
    for (auto &expr : expressionSet) {
        int index = expressionSet.GetIndex(expr);
        std::string exprNameTempVarInit = GetExprNameTempVarInit(elementKey_, index);
        oss << std::setw(indent) << " " << exprNameTempVarInit << ";";
    }
    return oss.str();
}

ScalarImmediateType RawSymbolicScalar::GetImmediateValue() const {
    ASSERT(IsImmediate()) << "Mismatch immediate type: " << SymbolicScalarKind2Name(Kind());
    auto immediate = static_cast<const RawSymbolicImmediate *>(this);
    return immediate->Immediate();
}
const std::string &RawSymbolicScalar::GetSymbolName() const {
    ASSERT(IsSymbol()) << "Mismatch symbol type: " << SymbolicScalarKind2Name(Kind());
    auto symbol = static_cast<const RawSymbolicSymbol *>(this);
    return symbol->Name();
}
SymbolicOpcode RawSymbolicScalar::GetExpressionOpcode() const {
    ASSERT(IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression *>(this);
    return expression->Opcode();
}
const std::vector<RawSymbolicScalarPtr> &RawSymbolicScalar::GetExpressionOperandList() const {
    ASSERT(IsExpression()) << "Mismatch expression type: " << SymbolicScalarKind2Name(Kind());
    auto expression = static_cast<const RawSymbolicExpression *>(this);
    return expression->OperandList();
}

bool RawSymbolicScalar::IsExpressionCall(const std::string &calleeName) const {
    if (!IsExpression()) {
        return false;
    }
    if (GetExpressionOpcode() != SymbolicOpcode::T_MOP_CALL) {
        return false;
    }
    auto caller = GetExpressionOperandList()[0];
    if (!caller->IsSymbol()) {
        return false;
    }
    if (caller->GetSymbolName() != calleeName) {
        return false;
    }
    return true;
}

std::string RawSymbolicScalar::Dump() const {
    std::stringstream buf;
    DumpBuffer(buf);
    return buf.str();
}

static void DumpSymbolicScalar(const RawSymbolicScalarPtr &raw, Json &jarray) {
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            jarray.emplace_back(IMMEDIATE);
            auto immediate = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw);
            jarray.emplace_back(static_cast<uint64_t>(immediate->Immediate()));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            jarray.emplace_back(SYMBOL);
            auto symbol = std::dynamic_pointer_cast<RawSymbolicSymbol>(raw);
            jarray.emplace_back(symbol->Name());
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            jarray.emplace_back(EXPRESSION);
            RawSymbolicExpPtr expr = std::dynamic_pointer_cast<RawSymbolicExpression>(raw);
            jarray.emplace_back(static_cast<int32_t>(expr->Opcode()));
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                jarray.emplace_back(static_cast<int32_t>(expr->OperandList().size()));
            }
            for (auto &op : expr->OperandList()) {
                DumpSymbolicScalar(op, jarray);
            }
        } break;
        default: ASSERT(false); break;
    }
}

Json ToJson(const SymbolicScalar &sval) {
    Json jdata;
    DumpSymbolicScalar(sval.Raw(), jdata);
    return jdata;
}

static RawSymbolicScalarPtr LoadRawSymbolicScalar(const Json &symbolicJson, int &despos) {
    RawSymbolicScalarPtr raw;
    SymbolicScalarKind kind = static_cast<SymbolicScalarKind>(symbolicJson[despos++]);
    switch (kind) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            uint64_t immediateData = static_cast<uint64_t>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicImmediate>(immediateData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::string nameData = static_cast<std::string>(symbolicJson[despos++]);
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicSymbol>(nameData));
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            SymbolicOpcode opcode = static_cast<SymbolicOpcode>(symbolicJson[despos++]);
            std::vector<RawSymbolicScalarPtr> operandList;
            if (opcode == SymbolicOpcode::T_MOP_CALL) {
                int size = symbolicJson[despos++];
                for (int i = 0; i < size; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            } else {
                for (int i = 0; i < OPERAND_NUM; i++) {
                    operandList.push_back(LoadRawSymbolicScalar(symbolicJson, despos));
                }
            }
            raw = std::static_pointer_cast<RawSymbolicScalar>(std::make_shared<RawSymbolicExpression>(opcode, operandList));
        } break;
        default: break;
    }
    return raw;
}

SymbolicScalar LoadSymbolicScalar(const Json &jval) {
    int pos = 0;
    return SymbolicScalar(LoadRawSymbolicScalar(jval, pos));
}

void SymbolicScalar::AsIntermediateVariable() {
    raw_->AsIntermediateVariable();
}

bool SymbolicScalar::IsIntermediateVariable() const {
    return raw_->IsIntermediateVariable();
}

#define SYMBOLIC_SCALAR_DEFINE_UOP(name, uop, rawname)  \
    SymbolicScalar SymbolicScalar::name() const {       \
        auto raw = rawname(raw_);                       \
        if (ConcreteValid()) {                          \
            return SymbolicScalar(raw, uop Concrete()); \
        } else {                                        \
            return SymbolicScalar(raw);                 \
        }                                               \
    }
SYMBOLIC_SCALAR_DEFINE_UOP(Pos, +, RawSymbolicExpression::CreateUopPos)
SYMBOLIC_SCALAR_DEFINE_UOP(Neg, -, RawSymbolicExpression::CreateUopNeg)
SYMBOLIC_SCALAR_DEFINE_UOP(Not, !, RawSymbolicExpression::CreateUopNot)
#undef SYMBOLIC_SCALAR_DEFINE_UOP

#define SYMBOLIC_SCALAR_DEFINE_BOP(name, bop, rawname)                      \
    SymbolicScalar SymbolicScalar::name(const SymbolicScalar &sval) const { \
        auto raw = rawname(raw_, sval.raw_);                                \
        if (ConcreteValid() && sval.ConcreteValid()) {                      \
            return SymbolicScalar(raw, Concrete() bop sval.Concrete());     \
        } else {                                                            \
            return SymbolicScalar(raw);                                     \
        }                                                                   \
    }

SYMBOLIC_SCALAR_DEFINE_BOP(Add, +, RawSymbolicExpression::CreateBopAdd)
SYMBOLIC_SCALAR_DEFINE_BOP(Sub, -, RawSymbolicExpression::CreateBopSub)
SYMBOLIC_SCALAR_DEFINE_BOP(Mul, *, RawSymbolicExpression::CreateBopMul)
SYMBOLIC_SCALAR_DEFINE_BOP(Div, /, RawSymbolicExpression::CreateBopDiv)
SYMBOLIC_SCALAR_DEFINE_BOP(Mod, %, RawSymbolicExpression::CreateBopMod)
SYMBOLIC_SCALAR_DEFINE_BOP(Eq, ==, RawSymbolicExpression::CreateBopEq)
SYMBOLIC_SCALAR_DEFINE_BOP(Ne, !=, RawSymbolicExpression::CreateBopNe)
SYMBOLIC_SCALAR_DEFINE_BOP(Lt, <, RawSymbolicExpression::CreateBopLt)
SYMBOLIC_SCALAR_DEFINE_BOP(Le, <=, RawSymbolicExpression::CreateBopLe)
SYMBOLIC_SCALAR_DEFINE_BOP(Gt, >, RawSymbolicExpression::CreateBopGt)
SYMBOLIC_SCALAR_DEFINE_BOP(Ge, >=, RawSymbolicExpression::CreateBopGe)
#undef SYMBOLIC_SCALAR_DEFINE_BOP

static bool AllConcreteValid(const std::vector<SymbolicScalar> &slist) {
    for (auto &s : slist) {
        if (!s.ConcreteValid()) {
            return false;
        }
    }
    return true;
}

SymbolicScalar SymbolicScalar::operator()() const {
    auto raw = RawSymbolicExpression::CreateMopCall(raw_);
    if (ConcreteValid()) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(
    const SymbolicScalar &arg0, const SymbolicScalar &arg1, const SymbolicScalar &arg2) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2})) {
        return SymbolicScalar(
            raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(), arg2.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1,
    const SymbolicScalar &arg2, const SymbolicScalar &arg3) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2, arg3})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(),
                                       arg2.Concrete(), arg3.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}
SymbolicScalar SymbolicScalar::operator()(const SymbolicScalar &arg0, const SymbolicScalar &arg1,
    const SymbolicScalar &arg2, const SymbolicScalar &arg3, const SymbolicScalar &arg4) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_, arg0.raw_, arg1.raw_, arg2.raw_, arg3.raw_, arg4.raw_};
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this, arg0, arg1, arg2, arg3, arg4})) {
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall({Concrete(), arg0.Concrete(), arg1.Concrete(),
                                       arg2.Concrete(), arg3.Concrete()}));
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar SymbolicScalar::operator()(const std::vector<SymbolicScalar> &argList) const {
    std::vector<RawSymbolicScalarPtr> args = {raw_};
    for (auto &a : argList) {
        args.push_back(a.raw_);
    }
    auto raw = RawSymbolicExpression::CreateMopCall(args);
    if (AllConcreteValid({*this}) && AllConcreteValid(argList)) {
        std::vector<ScalarImmediateType> calcArgList = {Concrete()};
        for (auto &a : argList) {
            calcArgList.push_back(a.Concrete());
        }
        return SymbolicScalar(raw, RawSymbolicExpression::CalcMopCall(calcArgList));
    } else {
        return SymbolicScalar(raw);
    }
}

std::string SymbolicScalar::Dump() const {
    std::stringstream buf;
    if (raw_) {
        raw_->DumpBuffer(buf);
    }
    return buf.str();
}

bool SymbolicScalar::IsImmediate() const {
    return raw_ && raw_->IsImmediate();
}
bool SymbolicScalar::IsSymbol() const {
    return raw_ && raw_->IsSymbol();
}
bool SymbolicScalar::IsExpression() const {
    return raw_ && raw_->IsExpression();
}

SymbolicScalar SymbolicScalar::Min(const SymbolicScalar &sval) const {
    auto raw = RawSymbolicExpression::CreateBopMin(raw_, sval.raw_);
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(raw, std::min(Concrete(), sval.Concrete()));
    } else if (sval.Dump() == Dump()) {
        return sval;
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar SymbolicScalar::Max(const SymbolicScalar &sval) const {
    auto raw = RawSymbolicExpression::CreateBopMax(raw_, sval.raw_);
    if (ConcreteValid() && sval.ConcreteValid()) {
        return SymbolicScalar(raw, std::max(Concrete(), sval.Concrete()));
    } else if (sval.Dump() == Dump()) {
        return sval;
    } else {
        return SymbolicScalar(raw);
    }
}

SymbolicScalar SymbolicScalar::Ternary(const SymbolicScalar &sval1, const SymbolicScalar &sval2) const{
    std::string ternaryOpName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::TernaryOP);
    ternaryOpName = AddRuntimePrefix(ternaryOpName);
    SymbolicScalar ternaryOp(ternaryOpName);
    auto result = ternaryOp(raw_, sval1, sval2);
    return result;
}

SymbolicScalar::SymbolicScalar(int64_t value)
    : raw_(RawSymbolicImmediate::Create(value)), concreteValid_(true), concrete_(value) {}
SymbolicScalar::SymbolicScalar(const std::string &name) : raw_(RawSymbolicSymbol::Create(name)) {}
SymbolicScalar::SymbolicScalar(const std::string &name, int64_t value)
    : raw_(RawSymbolicSymbol::Create(name)), concreteValid_(true), concrete_(value) {}
SymbolicScalar::SymbolicScalar(RawSymbolicScalarPtr raw, int64_t concrete)
    : raw_(raw), concreteValid_(true), concrete_(concrete) {}
SymbolicScalar::SymbolicScalar(RawSymbolicScalarPtr raw) : raw_(raw) {
    if (raw_->IsImmediate()) {
        concreteValid_ = true;
        concrete_ = std::dynamic_pointer_cast<RawSymbolicImmediate>(raw)->Immediate();
    }
}

std::vector<int64_t> SymbolicScalar::Concrete(const std::vector<SymbolicScalar> &scalarList, int64_t defValue) {
    std::vector<int64_t> concreteList;
    for (auto &s : scalarList) {
        if (s.ConcreteValid()) {
            concreteList.push_back(s.Concrete());
        } else {
            concreteList.push_back(defValue);
        }
    }
    return concreteList;
}

std::vector<SymbolicScalar> SymbolicScalar::FromConcrete(const std::vector<int64_t> &values) {
    std::vector<SymbolicScalar> result;
    for (auto x : values) {
        result.push_back(SymbolicScalar(x));
    }
    return result;
}

static void LookupExpressionByOpcode(std::vector<RawSymbolicScalarPtr> &exprList, SymbolicOpcode opcode, const RawSymbolicScalarPtr &raw) {
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            if (raw->GetExpressionOpcode() == opcode) {
                exprList.emplace_back(raw);
            }
            for (auto &op : raw->GetExpressionOperandList()) {
                LookupExpressionByOpcode(exprList, opcode, op);
            }
        } break;
        default: ASSERT(false); break;
    }
}

std::vector<RawSymbolicScalarPtr> LookupExpressionByOpcode(const RawSymbolicScalarPtr &value, SymbolicOpcode opcode) {
    std::vector<RawSymbolicScalarPtr> exprList;
    LookupExpressionByOpcode(exprList, opcode, value);
    return exprList;
}

void RawSymbolicExpression::DumpBuffer(std::ostream &buffer) const {
    if (SymbolicOpcode::T_UOP_BEGIN <= opcode_ && opcode_ < SymbolicOpcode::T_UOP_END) {
        buffer << "(";
        buffer << GetSymbolicCalcOpcode(opcode_);
        operandList_[0]->DumpBuffer(buffer);
        buffer << ")";
    } else if (SymbolicOpcode::T_BOP_BEGIN <= opcode_ && opcode_ < SymbolicOpcode::T_BOP_END) {
        if (opcode_ == SymbolicOpcode::T_BOP_MAX) {
            buffer << "RUNTIME_Max(";
            operandList_[0]->DumpBuffer(buffer);
            buffer << ", ";
            operandList_[1]->DumpBuffer(buffer);
            buffer << ")";
        } else if (opcode_ == SymbolicOpcode::T_BOP_MIN) {
            buffer << "RUNTIME_Min(";
            operandList_[0]->DumpBuffer(buffer);
            buffer << ", ";
            operandList_[1]->DumpBuffer(buffer);
            buffer << ")";
        } else if (opcode_ == SymbolicOpcode::T_BOP_EQ) {
            buffer << "RUNTIME_Eq(";
            operandList_[0]->DumpBuffer(buffer);
            buffer << ", ";
            operandList_[1]->DumpBuffer(buffer);
            buffer << ")";
        } else if (opcode_ == SymbolicOpcode::T_BOP_NE) {
            buffer << "RUNTIME_Ne(";
            operandList_[0]->DumpBuffer(buffer);
            buffer << ", ";
            operandList_[1]->DumpBuffer(buffer);
            buffer << ")";
        } else {
            buffer << "(";
            for (size_t i = 0; i < operandList_.size(); i++) {
                if (i != 0) {
                    buffer << GetSymbolicCalcOpcode(opcode_);
                }
                operandList_[i]->DumpBuffer(buffer);
            }
            buffer << ")";
        }
    } else if (opcode_ == SymbolicOpcode::T_MOP_CALL) {
        operandList_[0]->DumpBuffer(buffer);
        buffer << "(";
        for (size_t i = 1; i < operandList_.size(); i++) {
            if (i != 1) {
                buffer << ",";
            }
            operandList_[i]->DumpBuffer(buffer);
        }
        buffer << ")";
    }
}

} // namespace npu::tile_fwk
