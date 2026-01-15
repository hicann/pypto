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
 * \file source_cpp.cpp
 * \brief Serialize to source format whose style is cplusplus
 */

#include "ir/serializer.h"
#include "tilefwk/error.h"
#include "ir/utils.h"
#include "interface/utils/common.h"
#include "interface/utils/string_utils.h"

using namespace npu::tile_fwk;

namespace pto {
namespace serializer {

static inline std::string Indent(int indent) {
    return std::string(indent * 4, ' ');
}

static void SerializeNodeHead(IRBuffer &buffer, const std::shared_ptr<SourceCppASTNode> &node, int indent) {
    if (node->GetName() != "") {
        if (node->GetName().find_first_of('#') != std::string::npos && node->GetArgList().size() == 0) {
            buffer << Indent(indent) << node->GetName();
        } else {
            buffer << Indent(indent) << node->GetName() << "(" << node->GetArgList() << ")";
        }
    }
}

static void SerializeNode(IRBuffer &buffer, const std::shared_ptr<SourceCppASTNode> &node, int indent) {
    SerializeNodeHead(buffer, node, indent);
    if (node->size() != 0) {
        if (node->GetName() != "") {
            buffer << " {\n";
        }
        for (size_t k = 0; k < node->size(); k++) {
            SerializeNode(buffer, node->at(k), indent + 1);
        }
        if (node->GetName() != "") {
            buffer << Indent(indent) << "}\n";
        }
    } else {
        buffer << "\n";
    }
}

const static std::unordered_map<char, SourceCppToken::TokenKind> punctuationDict = {
    {'{', SourceCppToken::BraceLeft},
    {'}', SourceCppToken::BraceRight},
    {'(', SourceCppToken::ParenthesisLeft},
    {')', SourceCppToken::ParenthesisRight},
    {',', SourceCppToken::Comma},
};
static int DeserializeTokenList(IRBuffer &buffer, std::vector<SourceCppToken> &tokenList) {
    char ch = 0;
    int count = buffer.Read(&ch, 1, true);
    while (count != 0) {
        if (std::isspace(ch)) {
            std::string space = buffer.ReadSpace();
            tokenList.emplace_back(SourceCppToken::Space, space);
        } else if (ch == '/') {
            char comment[2];
            count = buffer.Read(comment, 2, true);
            if (comment[1] == '/') {
                std::string commentLine = buffer.ReadLine();
                tokenList.emplace_back(SourceCppToken::CommentLine, commentLine);
            } else if (comment[1] == '*') {
                std::string commentBlock = buffer.ReadUntil("*/");
                tokenList.emplace_back(SourceCppToken::CommentBlock, commentBlock);
            } else {
                buffer.Read(&ch, 1);
                tokenList.emplace_back(SourceCppToken::Unknown, std::string(1, ch));
            }
        } else if (punctuationDict.count(ch)) {
            buffer.Read(&ch, 1);
            auto tokenKind = punctuationDict.find(ch)->second;
            tokenList.emplace_back(tokenKind, std::string(1, ch));
        } else if (std::isalpha(ch) || ch == '_') {
            std::string identifier = SourceReadIdentifier(buffer);
            tokenList.emplace_back(SourceCppToken::Identifier, identifier);
        } else if (std::isdigit(ch)) {
            std::string number = SourceReadNumber(buffer);
            tokenList.emplace_back(SourceCppToken::Number, number);
        } else if (ch == '#') {
            std::string preprocess = buffer.ReadLine();
            tokenList.emplace_back(SourceCppToken::Preprocess, preprocess);
        } else {
            break;
        }
        count = buffer.Read(&ch, 1, true);
    }
    return buffer.ReadSeek(0, IRBuffer::ReadSeekMode::Position);
}

static void DeserializeNode(IRBuffer &buffer, std::shared_ptr<SourceCppASTNode> &node) {
    (void)buffer;
    (void)node;
}

IRBuffer &operator<<(IRBuffer &buf, const std::shared_ptr<SourceCppASTNode> &node) {
    SerializeNode(buf, node, 0);
    return buf;
}

IRBuffer &operator>>(IRBuffer &buf, std::shared_ptr<SourceCppASTNode> &node) {
    DeserializeNode(buf, node);
    return buf;
}

class SerializeCommand {
public:
    SerializeCommand(const std::string &name) : runtimeName_(IR_SOURCE_CPP_PREFIX + name) {}

    template<typename ...TyArgs>
    SourceCppASTNodePtr operator()(TyArgs ...args) {
        std::vector<std::string> argumentList = SerializeUtils::List(args...);
        return std::make_shared<SourceCppASTNode>(runtimeName_, argumentList);
    }

    const std::string &GetRuntimeName() { return runtimeName_; }
private:
    std::string runtimeName_;
};

static SerializeCommand rtFunction(IR_SOURCE_CPP_FUNCTION);
static SerializeCommand rtOperation(IR_SOURCE_CPP_OPERATION);
static SerializeCommand rtOperationMacro(IR_SOURCE_CPP_OPERATION_MACRO);
static SerializeCommand rtDeclTypeTile(IR_SOURCE_CPP_DECL_TYPE_TILE);
static SerializeCommand rtDeclTypeTensor(IR_SOURCE_CPP_DECL_TYPE_TENSOR);
static SerializeCommand rtDeclValueScalar(IR_SOURCE_CPP_DECL_VALUE_SCALAR);
static SerializeCommand rtDeclValueTile(IR_SOURCE_CPP_DECL_VALUE_TILE);
static SerializeCommand rtDeclValueTensor(IR_SOURCE_CPP_DECL_VALUE_TENSOR);

static SerializeCommand rtInitValueTile(IR_SOURCE_CPP_INIT_VALUE_TILE);
static SerializeCommand rtInitValueTensor(IR_SOURCE_CPP_INIT_VALUE_TENSOR);
static SerializeCommand rtInitAddr(IR_SOURCE_CPP_INIT_ADDR);

static SerializeCommand rtStmtOp(IR_SOURCE_CPP_STMT_OP);
static SerializeCommand rtStmtIf(IR_SOURCE_CPP_STMT_IF);
static SerializeCommand rtStmtElse(IR_SOURCE_CPP_STMT_ELSE);
static SerializeCommand rtStmtFor(IR_SOURCE_CPP_STMT_FOR);
static SerializeCommand rtStmtYield(IR_SOURCE_CPP_STMT_YIELD);
static SerializeCommand rtStmtReturn(IR_SOURCE_CPP_STMT_RETURN);

static std::string SerializeValue(const ValuePtr &value) {
    std::string result;
    switch (value->GetValueKind()) {
        case ValueKind::Scalar: {
            ScalarValuePtr scalar = ObjectCast<ScalarValue>(value);
            if (scalar->GetScalarValueKind() == ScalarValueKind::Immediate) {
                auto immediate = scalar->GetImmediateValue();
                if (std::holds_alternative<bool>(immediate)) {
                    result = std::get<bool>(immediate) ? "true" : "false";
                } else if (std::holds_alternative<int>(immediate)) {
                    result = std::to_string(std::get<int>(immediate));
                } else if (std::holds_alternative<int64_t>(immediate)) {
                    result = std::to_string(std::get<int64_t>(immediate));
                } else if (std::holds_alternative<uint64_t>(immediate)) {
                    result = std::to_string(std::get<uint64_t>(immediate));
                } else if (std::holds_alternative<double>(immediate)) {
                    result = std::to_string(std::get<double>(immediate));
                } else {
                    ASSERT(false) << "Unknown value";
                }
            } else {
                result = scalar->GetName();
            }
            break;
        }
        case ValueKind::Tile: {
            result = value->GetName();
            break;
        }
        case ValueKind::Tensor: {
            result = value->GetName();
            break;
        }
        default: {
            ASSERT(false) << "Invalid value: " << static_cast<int>(value->GetValueKind());
        }
    }
    return result;
}

static bool SerializeIsSymbol(const ScalarValuePtr &ptr) {
    return !ptr->HasImmediateValue();
}

static std::vector<ScalarValuePtr> SerializeGetScalarDepend(const TileValuePtr &tile) {
    std::vector<ScalarValuePtr> argList;
    for (auto shape : tile->GetValidShape()) {
        if (SerializeIsSymbol(shape)) {
            argList.push_back(shape);
        }
    }
    return argList;
}

static std::vector<ScalarValuePtr> SerializeGetScalarDepend(const TensorValuePtr &tensor) {
    std::vector<ScalarValuePtr> argList;
    for (auto shape : tensor->GetShape()) {
        if (SerializeIsSymbol(shape)) {
            argList.push_back(shape);
        }
    }
    return argList;
}

static std::vector<std::string> SerializeGetScalarArgument(const TileValuePtr &tile) {
    std::vector<std::string> argList;
    for (auto shape : tile->GetValidShape()) {
        argList.push_back(SerializeValue(shape));
    }
    return argList;
}

static std::vector<std::string> SerializeGetScalarArgument(const TensorValuePtr &tensor) {
    std::vector<std::string> argList;
    for (auto shape : tensor->GetShape()) {
        argList.push_back(SerializeValue(shape));
    }
    return argList;
}

static std::string SerializeValueName(const ScalarValuePtr &scalar) {
    return scalar->GetName();
}
static std::string SerializeValueName(const TileValuePtr &tile) {
    return tile->GetName();
}
static std::string SerializeValueName(const TensorValuePtr &tensor) {
    return tensor->GetName();
}
static std::string SerializeTypeName(const TileValuePtr &tile) {
    return SerializeValueName(tile) + "Type";
}
static std::string SerializeTypeName(const TensorValuePtr &tensor) {
    return SerializeValueName(tensor) + "Type";
}
static std::string SerializeMemoryName(const TileValuePtr &tile) {
    auto memory = tile->GetMemory();
    return SerializeValueName(tile) + "Memory_S" + std::to_string(memory->GetAddr()) + "_E" + std::to_string(memory->GetAddr() + memory->GetSize());
}
static std::string SerializePrimType(const ScalarValuePtr &scalar) {
    return DTypeInfoOf(scalar->GetType()->GetDataType()).name;
}
static std::string SerializePrimType(const TileValuePtr &tile) {
    return DTypeInfoOf(tile->GetType()->GetDataType()).name;
}
static std::string SerializePrimType(const TensorValuePtr &tensor) {
    return DTypeInfoOf(tensor->GetType()->GetDataType()).name;
}
static std::string SerializeSpace(const TileValuePtr &tile) {
    return GetMemSpaceKindName(tile->GetMemory()->GetSpace());
}

struct SerializeContext {
    OrderedSet<ScalarValuePtr> scalarValueList;
    OrderedSet<TileValuePtr> tileValueList;
    OrderedSet<TensorValuePtr> tensorValueList;

    struct TileTensorDepend {
        std::vector<TileValuePtr> tileValueList;
        std::vector<TensorValuePtr> tensorValueList;
    };
    std::unordered_map<ScalarValuePtr, TileTensorDepend> dependDict;

    struct State {
        std::unordered_map<TileValuePtr, int> tilePredDict;
        std::unordered_map<TensorValuePtr, int> tensorPredDict;
    } state;
};

static void SerializeInitValue(SerializeContext &ctx, SourceCppASTNodePtr &stmtOpNode, const TileValuePtr &tile) {
    (void)ctx;
    auto initNode = rtInitValueTile(SerializeValueName(tile), SerializeTypeName(tile), SerializeMemoryName(tile), tile->GetShape().size(), SerializeGetScalarArgument(tile));
    stmtOpNode->push_back(initNode);
}

static void SerializeInitValue(SerializeContext &ctx, SourceCppASTNodePtr &stmtOpNode, const TensorValuePtr &tensor) {
    (void)ctx;
    auto initNode = rtInitValueTensor(SerializeValueName(tensor), SerializeTypeName(tensor), SerializePrimType(tensor), tensor->GetName() + "Addr", tensor->GetShape().size(), SerializeGetScalarArgument(tensor));
    stmtOpNode->push_back(initNode);
}

static bool SerializeOperationIsMacro(const ScalarValuePtr &scalar) {
    auto scalarName = SerializeValueName(scalar);
    return StringUtils::StartsWith(scalarName, IR_SOURCE_CPP_MACRO);
}

static bool SerializeOperationByMacro(const OperationPtr &op) {
    if (op->GetNumOutputOperand() != 1) {
        return false;
    }
    auto out = op->GetOutputOperand(0);
    if (out->GetValueKind() != ValueKind::Scalar) {
        return false;
    }

    if (!SerializeOperationIsMacro(ObjectCast<ScalarValue>(out))) {
        return false;
    }
    return true;
}

static void SerializeOperationAssign(SourceCppASTNodePtr &stmtOpNode, const std::vector<std::string> &argList) {
    auto opNode = rtOperation(argList);
    stmtOpNode->push_back(opNode);
}

static void SerializeOperationMacro(SourceCppASTNodePtr &stmtOpNode, const std::vector<std::string> &argList) {
    std::ostringstream oss;
    oss << "#define " << argList[1] << " " << rtOperationMacro.GetRuntimeName() << "(" << argList[0];
    for (size_t k = 2; k < argList.size(); k++) {
        oss << ", " << argList[k];
    }
    oss << ")";
    auto opNode = std::make_shared<SourceCppASTNode>(oss.str());
    stmtOpNode->push_back(opNode);
}

static void SerializeOperation(SerializeContext &ctx, SourceCppASTNodePtr &stmtOpNode, const OperationPtr &op) {
    std::vector<std::string> argList({GetOpcodeName(op->GetOpcode())});
    std::vector<TileValuePtr> readyTileList;
    std::vector<TensorValuePtr> readyTensorList;
    for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
        auto oop = op->GetOutputOperand(k);
        switch (oop->GetValueKind()) {
            case ValueKind::Scalar: {
                /* scalar is declared */
                ScalarValuePtr oopScalar = ObjectCast<ScalarValue>(oop);
                ASSERT(oopScalar->GetScalarValueKind() == ScalarValueKind::Symbolic) << "Output must be symbol";
                argList.push_back(oopScalar->GetName());

                if (ctx.dependDict.count(oopScalar)) {
                    for (auto tile : ctx.dependDict[oopScalar].tileValueList) {
                        ctx.state.tilePredDict[tile]--;
                        if (ctx.state.tilePredDict[tile] == 0) {
                            readyTileList.push_back(tile);
                        }
                    }
                    for (auto tensor : ctx.dependDict[oopScalar].tensorValueList) {
                        ctx.state.tensorPredDict[tensor]--;
                        if (ctx.state.tensorPredDict[tensor] == 0) {
                            readyTensorList.push_back(tensor);
                        }
                    }
                }

                break;
            }
            case ValueKind::Tile: {
                TileValuePtr oopTile = ObjectCast<TileValue>(oop);
                argList.push_back(oopTile->GetName());
                break;
            }
            case ValueKind::Tensor: {
                TensorValuePtr oopTensor = ObjectCast<TensorValue>(oop);
                argList.push_back(oopTensor->GetName());
                break;
            }
            default: {
                ASSERT(false) << "Invalid value: " << static_cast<int>(oop->GetValueKind());
                break;
            }
        }
    }
    for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
        auto iop = op->GetInputOperand(k);
        switch (iop->GetValueKind()) {
            case ValueKind::Scalar: {
                ScalarValuePtr iopScalar = ObjectCast<ScalarValue>(iop);
                argList.push_back(SerializeValue(iopScalar));
                break;
            }
            case ValueKind::Tile: {
                TileValuePtr iopTile = ObjectCast<TileValue>(iop);
                argList.push_back(iopTile->GetName());
                break;
            }
            case ValueKind::Tensor: {
                TensorValuePtr iopTensor = ObjectCast<TensorValue>(iop);
                argList.push_back(iopTensor->GetName());
                break;
            }
            default: {
                ASSERT(false) << "Invalid value: " << static_cast<int>(iop->GetValueKind());
                break;
            }
        }
    }

    std::vector<AttributeKeyValue> kvList = op->GetAttributeList();
    for (auto &kv : kvList) {
        const auto &value = kv.GetValue();
        if (std::holds_alternative<std::string>(value)) {
            argList.push_back(std::get<std::string>(value));
        } else if (std::holds_alternative<int64_t>(value)) {
            argList.push_back(std::to_string(std::get<int64_t>(value)));
        } else if (std::holds_alternative<bool>(value)) {
            argList.push_back(std::get<bool>(value) ? "true" : "false");
        } else {
            ASSERT(false) << "Unknown attribute: " << kv.GetName();
        }
    }

    if (SerializeOperationByMacro(op)) {
        SerializeOperationMacro(stmtOpNode, argList);
    } else {
        SerializeOperationAssign(stmtOpNode, argList);
    }

    for (auto readyTile : readyTileList) {
        SerializeInitValue(ctx, stmtOpNode, readyTile);
    }
    for (auto readyTensor : readyTensorList) {
        SerializeInitValue(ctx, stmtOpNode, readyTensor);
    }
}

static void SerializeStatement(SerializeContext &ctx, SourceCppASTNodePtr &parentNode, const StatementPtr &stmt, const StatementPtr &yieldTarget) {
    switch (stmt->GetKind()) {
        case StatementKind::Compound: {
            CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
            for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                SerializeStatement(ctx, parentNode, stmtCompound->GetStatement(k), yieldTarget);
            }
            break;
        }
        case StatementKind::Op: {
            OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
            auto stmtOpNode = rtStmtOp();
            parentNode->push_back(stmtOpNode);

            for (auto &op : stmtOp->Operations()) {
                SerializeOperation(ctx, stmtOpNode, op);
            }
            break;
        }
        case StatementKind::If: {
            IfStatementPtr stmtIf = ObjectCast<IfStatement>(stmt);
            auto cond = stmtIf->GetCondition();
            auto stmtIfNode = rtStmtIf(SerializeValue(cond));
            auto stmtElseNode = rtStmtElse();
            parentNode->push_back(stmtIfNode);
            parentNode->push_back(stmtElseNode);

            SerializeStatement(ctx, stmtIfNode, stmtIf->GetThenCompound(), stmt);
            SerializeStatement(ctx, stmtElseNode, stmtIf->GetElseCompound(), stmt);
            break;
        }
        case StatementKind::For: {
            ForStatementPtr stmtFor = ObjectCast<ForStatement>(stmt);
            auto var = stmtFor->GetIterationVar();
            auto begin = SerializeValue(stmtFor->GetStart());
            auto end = SerializeValue(stmtFor->GetEnd());
            auto step = SerializeValue(stmtFor->GetStep());
            for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                auto yieldDst = stmtFor->GetIterValue(k)->GetName();
                auto yieldSrc = SerializeValue(stmtFor->GetIterInitValue(k));
                auto stmtInitNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_BEGIN, yieldDst, yieldSrc);
                parentNode->push_back(stmtInitNode);
            }
            auto stmtForNode = rtStmtFor(var->GetName(), begin, end, step);
            parentNode->push_back(stmtForNode);

            SerializeStatement(ctx, stmtForNode, stmtFor->GetCompound(), stmt);
            for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                auto yieldDst = stmtFor->Results()[k]->GetName();
                auto yieldSrc = stmtFor->GetIterValue(k)->GetName();
                auto stmtYieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_ASSIGN, yieldDst, yieldSrc);
                parentNode->push_back(stmtYieldNode);
            }
            break;
        }
        case StatementKind::Yield: {
            YieldStatementPtr stmtYield = ObjectCast<YieldStatement>(stmt);
            switch (yieldTarget->GetKind()) {
                case StatementKind::If: {
                    IfStatementPtr yieldTargetIf = ObjectCast<IfStatement>(yieldTarget);
                    for (size_t k = 0; k < yieldTargetIf->Results().size(); k++) {
                        auto yieldDst = yieldTargetIf->Results()[k]->GetName();
                        auto yieldSrc = SerializeValue(stmtYield->Values()[k]);
                        auto yieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_IF_ASSIGN, yieldDst, yieldSrc);
                        parentNode->push_back(yieldNode);
                    }
                    break;
                }
                case StatementKind::For: {
                    ForStatementPtr yieldTargetFor = ObjectCast<ForStatement>(yieldTarget);
                    for (size_t k = 0; k < yieldTargetFor->Results().size(); k++) {
                        auto yieldDst = yieldTargetFor->GetIterValue(k)->GetName();
                        auto yieldSrc = SerializeValue(stmtYield->Values()[k]);
                        auto yieldNode = rtStmtYield(IR_SOURCE_CPP_STMT_YIELD_LOOP_ITER, yieldDst, yieldSrc);
                        parentNode->push_back(yieldNode);
                    }
                    break;
                }
                default: {
                    ASSERT(false) << "Unknown statement: " << static_cast<int>(stmt->GetKind());
                    break;
                }
            }
            break;
        }
        case StatementKind::Return: {
            ReturnStatementPtr stmtReturn = ObjectCast<ReturnStatement>(stmt);
            auto stmtReturnNode = rtStmtReturn();
            parentNode->push_back(stmtReturnNode);
            break;
        }
        default: {
            ASSERT(false) << "Unknown statement: " << static_cast<int>(stmt->GetKind());
            break;
        }
    }
}

static void SerializeFunctionFindValue(SerializeContext &ctx, const FunctionPtr &func) {
    struct Visit {
        static void VisitValue(const ValuePtr value, SerializeContext &ctx, bool skipScalar) {
            switch(value->GetValueKind()) {
                case ValueKind::Scalar: {
                    auto scalar = ObjectCast<ScalarValue>(value);
                    if (SerializeIsSymbol(scalar) && !skipScalar) {
                        ctx.scalarValueList.Insert(scalar);
                    }
                    break;
                }
                case ValueKind::Tile:
                    ctx.tileValueList.Insert(ObjectCast<TileValue>(value));
                    break;
                case ValueKind::Tensor:
                    ctx.tensorValueList.Insert(ObjectCast<TensorValue>(value));
                    break;
                default:
                    break;
            }
        }
        static void VisitFunc(const FunctionPtr &func, SerializeContext &ctx) {
            VisitStmt(func->GetCompound(), ctx);
        }
        static void VisitStmt(const StatementPtr &stmt, SerializeContext &ctx) {
            switch (stmt->GetKind()) {
                case StatementKind::Compound: {
                    CompoundStatementPtr stmtCompound = ObjectCast<CompoundStatement>(stmt);
                    for (size_t k = 0; k < stmtCompound->GetStatementsNum(); k++) {
                        VisitStmt(stmtCompound->GetStatement(k), ctx);
                    }
                    break;
                }
                case StatementKind::Op: {
                    OpStatementPtr stmtOp = ObjectCast<OpStatement>(stmt);
                    for (auto &op : stmtOp->Operations()) {
                        VisitOp(op, ctx);
                    }
                    break;
                }
                case StatementKind::If: {
                    IfStatementPtr stmtIf = ObjectCast<IfStatement>(stmt);
                    VisitStmt(stmtIf->GetThenCompound(), ctx);
                    VisitStmt(stmtIf->GetElseCompound(), ctx);
                    break;
                }
                case StatementKind::For: {
                    ForStatementPtr stmtFor = ObjectCast<ForStatement>(stmt);
                    VisitValue(stmtFor->GetIterationVar(), ctx, false);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->GetIterValue(k), ctx, false);
                    }
                    VisitStmt(stmtFor->GetCompound(), ctx);
                    for (size_t k = 0; k < stmtFor->Results().size(); k++) {
                        VisitValue(stmtFor->Results()[k], ctx, false);
                    }
                    break;
                }
                case StatementKind::Yield: {
                    break;
                }
                case StatementKind::Return: {
                    break;
                }
                default: {
                    break;
                }
            }

        }
        static void VisitOp(const OperationPtr &op, SerializeContext &ctx) {
            for (size_t k = 0; k < op->GetNumOutputOperand(); k++) {
                VisitValue(op->GetOutputOperand(k), ctx, false);
            }
            for (size_t k = 0; k < op->GetNumInputOperand(); k++) {
                VisitValue(op->GetInputOperand(k), ctx, true);
            }
        }
    };
    Visit::VisitFunc(func, ctx);

    std::vector<ValuePtr> arguments; // argument types with names stored in Value::name
    std::vector<ValuePtr> results;  // return types

    for (auto tile : ctx.tileValueList) {
        // Force Init 0
        ctx.state.tilePredDict[tile] += 0;
        for (auto value : SerializeGetScalarDepend(tile)) {
            ctx.dependDict[value].tileValueList.push_back(tile);
            ctx.state.tilePredDict[tile]++;
        }
    }
    for (auto tensor : ctx.tensorValueList) {
        // Force Init 0
        ctx.state.tensorPredDict[tensor] += 0;
        for (auto value : SerializeGetScalarDepend(tensor)) {
            ctx.dependDict[value].tensorValueList.push_back(tensor);
            ctx.state.tensorPredDict[tensor]++;
        }
    }
}

static void SerializeFunctionDeclTypeList(SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto &tile : ctx.tileValueList) {
        funcNode->push_back(rtDeclTypeTile(SerializeTypeName(tile), SerializePrimType(tile), SerializeSpace(tile), tile->GetShape().size(), tile->GetShape()));
    }
    for (auto &tensor : ctx.tensorValueList) {
        funcNode->push_back(rtDeclTypeTensor(SerializeTypeName(tensor), SerializePrimType(tensor), tensor->GetShape().size()));
    }
}

static void SerializeFunctionDeclValueList(const SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto scalar : ctx.scalarValueList) {
        if (!SerializeOperationIsMacro(scalar)) {
            funcNode->push_back(rtDeclValueScalar(SerializeValueName(scalar), SerializePrimType(scalar)));
        }
    }
    for (auto tile : ctx.tileValueList) {
        funcNode->push_back(rtDeclValueTile(SerializeValueName(tile), SerializeTypeName(tile)));
    }
    for (auto tensor : ctx.tensorValueList) {
        funcNode->push_back(rtDeclValueTensor(SerializeValueName(tensor), SerializeTypeName(tensor)));
    }
}

static void SerializeFunctionInitAddr(SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto tile : ctx.tileValueList) {
        auto mem = tile->GetMemory();
        funcNode->push_back(rtInitAddr(SerializeMemoryName(tile), SerializePrimType(tile), SerializeSpace(tile), mem->GetAddr(), mem->GetSize()));
    }
}

static void SerializeFunctionInitValue(SerializeContext &ctx, SourceCppASTNodePtr &funcNode) {
    for (auto tile : ctx.tileValueList) {
        if (ctx.state.tilePredDict[tile] == 0) {
            SerializeInitValue(ctx, funcNode, tile);
        }
    }
    for (auto tensor : ctx.tensorValueList) {
        if (ctx.state.tensorPredDict[tensor] == 0) {
            SerializeInitValue(ctx, funcNode, tensor);
        }
    }
}

static SourceCppASTNodePtr SerializeFunction(const FunctionPtr &func) {
    SourceCppASTNodePtr funcNode = rtFunction(func->GetName());
    SerializeContext ctx;
    SerializeFunctionFindValue(ctx, func);
    SerializeFunctionDeclTypeList(ctx, funcNode);
    SerializeFunctionDeclValueList(ctx, funcNode);
    SerializeFunctionInitAddr(ctx, funcNode);
    SerializeFunctionInitValue(ctx, funcNode);
    SerializeStatement(ctx, funcNode, func->GetCompound(), nullptr);
    return funcNode;
}

static SourceCppASTNodePtr SerializeProgram(const ProgramModulePtr &prog) {
    SourceCppASTNodePtr progNode = std::make_shared<SourceCppASTNode>(std::string(), std::vector<std::string>());
    progNode->push_back(std::make_shared<SourceCppASTNode>("#define __TILE_FWK_AICORE__ 1"));
    progNode->push_back(std::make_shared<SourceCppASTNode>("#include \"../kernel_aicpu/expression_0.h\""));
    progNode->push_back(std::make_shared<SourceCppASTNode>("#include \"TileOpImpl.h\""));
    for (auto func : prog->GetFunctions()) {
        progNode->push_back(SerializeFunction(func));
    }
    return progNode;
}

SourceCppASTNodePtr SourceCppIRSerializer::SerializeASTNode(const ProgramModulePtr &prog) {
    return SerializeProgram(prog);
}

void SourceCppIRSerializer::Tokenization(IRBuffer &buffer, std::vector<SourceCppToken> &tokenList) {
    DeserializeTokenList(buffer, tokenList);
}

void SourceCppIRSerializer::Serialize(IRBuffer &buffer, const ProgramModulePtr &prog) {
    auto progNode = SerializeASTNode(prog);
    buffer << progNode;
}

ProgramModulePtr SourceCppIRSerializer::Deserialize(IRBuffer &buffer) {
    (void)buffer;
    ASSERT(false);
    return nullptr;
}

} // namespace serializer
} // namespace pto
