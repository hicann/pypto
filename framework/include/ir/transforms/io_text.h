/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file io_text.h
 * \brief IR text I/O — dumper and loader following the declared IR Syntax grammar.
 *
 * IRDumperText produces the canonical text representation declared in the
 * Doxygen "IR Syntax" blocks of each IR node header.  IRLoaderText parses that
 * text back into IR nodes.  Together they provide full round-trip serialisation.
 */
#ifndef PYPTO_IR_TRANSFORMS_IO_TEXT_H_
#define PYPTO_IR_TRANSFORMS_IO_TEXT_H_

#include <sstream>
#include <string>
#include <string_view>

#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"

// ===========================================================================
// IR text grammar keywords and punctuation
// ===========================================================================
// ---- Word keywords ----
#define IR_KW_PROGRAM "program"
#define IR_KW_FUNCTION "function"
#define IR_KW_INCAST "incast"
#define IR_KW_OUTCAST "outcast"
#define IR_KW_IF "if"
#define IR_KW_THEN "then"
#define IR_KW_ELSE "else"
#define IR_KW_FOR "for"
#define IR_KW_INRANGE "inrange"
#define IR_KW_ITER "iter"
#define IR_KW_WHILE "while"
#define IR_KW_YIELD "yield"
#define IR_KW_RETURN "return"
#define IR_KW_SECTION "section"
#define IR_KW_EVAL "eval"
#define IR_KW_BREAK "break"
#define IR_KW_CONTINUE "continue"
#define IR_KW_TUPLE "tuple"
#define IR_KW_GETITEM "getitem"
#define IR_KW_SCALAR_EXPR "scalar_expr"
#define IR_KW_UNKNOWN "unknown"
#define IR_KW_TENSOR "tensor"
#define IR_KW_TILE "tile"
#define IR_KW_PTR "ptr"
#define IR_KW_TOKEN "token"
#define IR_KW_NONE "none"
#define IR_KW_NULL "null"
#define IR_KW_TENSOR_VIEW "tensor_view"
#define IR_KW_TILE_VIEW "tile_view"
#define IR_KW_MEMREF "memref"
#define IR_KW_MEMREF_TYPE "memref_type"
#define IR_KW_LOGICAL_TENSOR "logical_tensor"
#define IR_KW_HW_INFO "hw_info"
#define IR_KW_TRUE "true"
#define IR_KW_FALSE "false"
#define IR_KW_ENTRY "entry"
#define IR_KW_TYPE "type"
#define IR_KW_DIM "x"

// ---- Scalar binary operator names ----
#define IR_KW_SCALAR_BOP_ADD "add"
#define IR_KW_SCALAR_BOP_SUB "sub"
#define IR_KW_SCALAR_BOP_MUL "mul"
#define IR_KW_SCALAR_BOP_DIV "div"
#define IR_KW_SCALAR_BOP_MOD "mod"
#define IR_KW_SCALAR_BOP_FDIV "fdiv"
#define IR_KW_SCALAR_BOP_MIN "min"
#define IR_KW_SCALAR_BOP_MAX "max"
#define IR_KW_SCALAR_BOP_POW "pow"
#define IR_KW_SCALAR_BOP_EQ "eq"
#define IR_KW_SCALAR_BOP_NE "ne"
#define IR_KW_SCALAR_BOP_LT "lt"
#define IR_KW_SCALAR_BOP_LE "le"
#define IR_KW_SCALAR_BOP_GT "gt"
#define IR_KW_SCALAR_BOP_GE "ge"
#define IR_KW_SCALAR_BOP_LAND "land"
#define IR_KW_SCALAR_BOP_LOR "lor"
#define IR_KW_SCALAR_BOP_LXOR "lxor"
#define IR_KW_SCALAR_BOP_AND "and"
#define IR_KW_SCALAR_BOP_OR "or"
#define IR_KW_SCALAR_BOP_XOR "xor"
#define IR_KW_SCALAR_BOP_SHL "shl"
#define IR_KW_SCALAR_BOP_SHR "shr"

// ---- Scalar unary operator names ----
#define IR_KW_SCALAR_UOP_ABS "abs"
#define IR_KW_SCALAR_UOP_NEG "neg"
#define IR_KW_SCALAR_UOP_NOT "not"
#define IR_KW_SCALAR_UOP_INV "inv"
#define IR_KW_SCALAR_UOP_CAST "cast"

// ---- Punctuation ----
#define IR_PUN_VARNAME "%"
#define IR_PUN_LBRACE "{"
#define IR_PUN_RBRACE "}"
#define IR_PUN_LPAREN "("
#define IR_PUN_RPAREN ")"
#define IR_PUN_LT "<"
#define IR_PUN_GT ">"
#define IR_PUN_COMMA ","
#define IR_PUN_EQ "="
#define IR_PUN_SEMI ";"
#define IR_PUN_ATTRNAME "#"
#define IR_PUN_LBRACKET "["
#define IR_PUN_RBRACKET "]"

namespace pypto {
namespace ir {

enum class LexerTokenKind {
    // Word keywords
    KwProgram,
    KwFunction,
    KwIncast,
    KwOutcast,
    KwIf,
    KwThen,
    KwElse,
    KwFor,
    KwInrange,
    KwIter,
    KwWhile,
    KwYield,
    KwReturn,
    KwSection,
    KwEval,
    KwBreak,
    KwContinue,
    KwTuple,
    KwGetItem,
    KwScalarExpr,
    KwUnknown,
    KwTensor,
    KwTile,
    KwPtr,
    KwToken,
    KwNone,
    KwTensorView,
    KwTileView,
    KwMemref,
    KwMemrefType,
    KwLogicalTensor,
    KwHwInfo,
    KwTrue,
    KwFalse,
    KwEntry,
    KwType,
    KwDim,
    KwNull,
    // Scalar binary op keywords
    KwBOpAdd,
    KwBOpSub,
    KwBOpMul,
    KwBOpDiv,
    KwBOpMod,
    KwBOpFdiv,
    KwBOpMin,
    KwBOpMax,
    KwBOpPow,
    KwBOpEq,
    KwBOpNe,
    KwBOpLt,
    KwBOpLe,
    KwBOpGt,
    KwBOpGe,
    KwBOpLand,
    KwBOpLor,
    KwBOpLxor,
    KwBOpAnd,
    KwBOpOr,
    KwBOpXor,
    KwBOpShl,
    KwBOpShr,
    // Scalar unary op keywords
    KwUOpAbs,
    KwUOpNeg,
    KwUOpNot,
    KwUOpInv,
    KwUOpCast,
    // Punctuation
    PunVarName,
    PunAttrName,
    PunLBrace,
    PunRBrace,
    PunLParen,
    PunRParen,
    PunLt,
    PunGt,
    PunComma,
    PunEq,
    PunSemi,
    PunLBracket,
    PunRBracket,

    // Data Type
    KwTypeBool,
    KwTypeInt8,
    KwTypeInt16,
    KwTypeInt32,
    KwTypeInt64,
    KwTypeUint8,
    KwTypeUint16,
    KwTypeUint32,
    KwTypeUint64,
    KwTypeFp16,
    KwTypeFp32,
    KwTypeFp64,
    KwTypeBf16,
    KwTypeFp8e4m3fn,
    KwTypeFp8e5m2,
    KwTypeHf4,
    KwTypeHf8,
    KwTypeUnknown,
    KwTypeBegin = KwTypeBool,
    KwTypeEnd = KwTypeUnknown,

    // Valuable
    Ident,
    Int,
    Float,
    VarName,
    AttrName,
    Invalid,
};

static inline const npu::tile_fwk::BiMap<LexerTokenKind>& GetLexerTokenDict()
{
    static npu::tile_fwk::BiMap<LexerTokenKind> dict{{
        // Keywords
        {LexerTokenKind::KwProgram, IR_KW_PROGRAM},
        {LexerTokenKind::KwFunction, IR_KW_FUNCTION},
        {LexerTokenKind::KwIncast, IR_KW_INCAST},
        {LexerTokenKind::KwOutcast, IR_KW_OUTCAST},
        {LexerTokenKind::KwIf, IR_KW_IF},
        {LexerTokenKind::KwThen, IR_KW_THEN},
        {LexerTokenKind::KwElse, IR_KW_ELSE},
        {LexerTokenKind::KwFor, IR_KW_FOR},
        {LexerTokenKind::KwInrange, IR_KW_INRANGE},
        {LexerTokenKind::KwIter, IR_KW_ITER},
        {LexerTokenKind::KwWhile, IR_KW_WHILE},
        {LexerTokenKind::KwYield, IR_KW_YIELD},
        {LexerTokenKind::KwReturn, IR_KW_RETURN},
        {LexerTokenKind::KwSection, IR_KW_SECTION},
        {LexerTokenKind::KwEval, IR_KW_EVAL},
        {LexerTokenKind::KwBreak, IR_KW_BREAK},
        {LexerTokenKind::KwContinue, IR_KW_CONTINUE},
        {LexerTokenKind::KwTuple, IR_KW_TUPLE},
        {LexerTokenKind::KwGetItem, IR_KW_GETITEM},
        {LexerTokenKind::KwScalarExpr, IR_KW_SCALAR_EXPR},
        {LexerTokenKind::KwUnknown, IR_KW_UNKNOWN},
        {LexerTokenKind::KwTensor, IR_KW_TENSOR},
        {LexerTokenKind::KwTile, IR_KW_TILE},
        {LexerTokenKind::KwPtr, IR_KW_PTR},
        {LexerTokenKind::KwToken, IR_KW_TOKEN},
        {LexerTokenKind::KwNone, IR_KW_NONE},
        {LexerTokenKind::KwTensorView, IR_KW_TENSOR_VIEW},
        {LexerTokenKind::KwTileView, IR_KW_TILE_VIEW},
        {LexerTokenKind::KwMemref, IR_KW_MEMREF},
        {LexerTokenKind::KwMemrefType, IR_KW_MEMREF_TYPE},
        {LexerTokenKind::KwLogicalTensor, IR_KW_LOGICAL_TENSOR},
        {LexerTokenKind::KwHwInfo, IR_KW_HW_INFO},
        {LexerTokenKind::KwTrue, IR_KW_TRUE},
        {LexerTokenKind::KwFalse, IR_KW_FALSE},
        {LexerTokenKind::KwEntry, IR_KW_ENTRY},
        {LexerTokenKind::KwType, IR_KW_TYPE},
        {LexerTokenKind::KwDim, IR_KW_DIM},
        {LexerTokenKind::KwNull, IR_KW_NULL},
        // Scalar binary ops
        {LexerTokenKind::KwBOpAdd, IR_KW_SCALAR_BOP_ADD},
        {LexerTokenKind::KwBOpSub, IR_KW_SCALAR_BOP_SUB},
        {LexerTokenKind::KwBOpMul, IR_KW_SCALAR_BOP_MUL},
        {LexerTokenKind::KwBOpDiv, IR_KW_SCALAR_BOP_DIV},
        {LexerTokenKind::KwBOpMod, IR_KW_SCALAR_BOP_MOD},
        {LexerTokenKind::KwBOpFdiv, IR_KW_SCALAR_BOP_FDIV},
        {LexerTokenKind::KwBOpMin, IR_KW_SCALAR_BOP_MIN},
        {LexerTokenKind::KwBOpMax, IR_KW_SCALAR_BOP_MAX},
        {LexerTokenKind::KwBOpPow, IR_KW_SCALAR_BOP_POW},
        {LexerTokenKind::KwBOpEq, IR_KW_SCALAR_BOP_EQ},
        {LexerTokenKind::KwBOpNe, IR_KW_SCALAR_BOP_NE},
        {LexerTokenKind::KwBOpLt, IR_KW_SCALAR_BOP_LT},
        {LexerTokenKind::KwBOpLe, IR_KW_SCALAR_BOP_LE},
        {LexerTokenKind::KwBOpGt, IR_KW_SCALAR_BOP_GT},
        {LexerTokenKind::KwBOpGe, IR_KW_SCALAR_BOP_GE},
        {LexerTokenKind::KwBOpLand, IR_KW_SCALAR_BOP_LAND},
        {LexerTokenKind::KwBOpLor, IR_KW_SCALAR_BOP_LOR},
        {LexerTokenKind::KwBOpLxor, IR_KW_SCALAR_BOP_LXOR},
        {LexerTokenKind::KwBOpAnd, IR_KW_SCALAR_BOP_AND},
        {LexerTokenKind::KwBOpOr, IR_KW_SCALAR_BOP_OR},
        {LexerTokenKind::KwBOpXor, IR_KW_SCALAR_BOP_XOR},
        {LexerTokenKind::KwBOpShl, IR_KW_SCALAR_BOP_SHL},
        {LexerTokenKind::KwBOpShr, IR_KW_SCALAR_BOP_SHR},
        // Scalar unary ops
        {LexerTokenKind::KwUOpAbs, IR_KW_SCALAR_UOP_ABS},
        {LexerTokenKind::KwUOpNeg, IR_KW_SCALAR_UOP_NEG},
        {LexerTokenKind::KwUOpNot, IR_KW_SCALAR_UOP_NOT},
        {LexerTokenKind::KwUOpInv, IR_KW_SCALAR_UOP_INV},
        {LexerTokenKind::KwUOpCast, IR_KW_SCALAR_UOP_CAST},
        // Punctuation (VarName/AttrName handled as whole tokens, not here)
        {LexerTokenKind::PunLBrace, IR_PUN_LBRACE},
        {LexerTokenKind::PunRBrace, IR_PUN_RBRACE},
        {LexerTokenKind::PunLParen, IR_PUN_LPAREN},
        {LexerTokenKind::PunRParen, IR_PUN_RPAREN},
        {LexerTokenKind::PunLt, IR_PUN_LT},
        {LexerTokenKind::PunGt, IR_PUN_GT},
        {LexerTokenKind::PunComma, IR_PUN_COMMA},
        {LexerTokenKind::PunEq, IR_PUN_EQ},
        {LexerTokenKind::PunSemi, IR_PUN_SEMI},
        {LexerTokenKind::PunLBracket, IR_PUN_LBRACKET},
        {LexerTokenKind::PunRBracket, IR_PUN_RBRACKET},

        {LexerTokenKind::KwTypeBool, IR_KW_TYPE_BOOL},
        {LexerTokenKind::KwTypeInt8, IR_KW_TYPE_INT8},
        {LexerTokenKind::KwTypeInt16, IR_KW_TYPE_INT16},
        {LexerTokenKind::KwTypeInt32, IR_KW_TYPE_INT32},
        {LexerTokenKind::KwTypeInt64, IR_KW_TYPE_INT64},
        {LexerTokenKind::KwTypeUint8, IR_KW_TYPE_UINT8},
        {LexerTokenKind::KwTypeUint16, IR_KW_TYPE_UINT16},
        {LexerTokenKind::KwTypeUint32, IR_KW_TYPE_UINT32},
        {LexerTokenKind::KwTypeUint64, IR_KW_TYPE_UINT64},
        {LexerTokenKind::KwTypeFp16, IR_KW_TYPE_FP16},
        {LexerTokenKind::KwTypeFp32, IR_KW_TYPE_FP32},
        {LexerTokenKind::KwTypeFp64, IR_KW_TYPE_FP64},
        {LexerTokenKind::KwTypeBf16, IR_KW_TYPE_BF16},
        {LexerTokenKind::KwTypeFp8e4m3fn, IR_KW_TYPE_FP8E4M3FN},
        {LexerTokenKind::KwTypeFp8e5m2, IR_KW_TYPE_FP8E5M2},
        {LexerTokenKind::KwTypeHf4, IR_KW_TYPE_HF4},
        {LexerTokenKind::KwTypeHf8, IR_KW_TYPE_HF8},
        {LexerTokenKind::KwTypeUnknown, IR_KW_TYPE_UNKNOWN},
    }};
    return dict;
}

// ===========================================================================
// Dumper
// ===========================================================================
class IRDumperText : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    IRDumperText() = default;
    ~IRDumperText() override = default;

    static std::string Dump(const IRNodePtr& node);
    static std::string DumpType(const TypePtr& type);

protected:
    void VisitExpr_(const VarPtr& op) override;
    void VisitExpr_(const MemRefPtr& op) override;
    void VisitExpr_(const ConstIntPtr& op) override;
    void VisitExpr_(const ConstFloatPtr& op) override;
    void VisitExpr_(const ConstBoolPtr& op) override;
    void VisitExpr_(const CallPtr& op) override;
    void VisitExpr_(const MakeTuplePtr& op) override;
    void VisitExpr_(const GetItemExprPtr& op) override;
    void VisitExpr_(const ScalarExprPtr& op) override;

    void VisitBinaryExpr_(const BinaryExprPtr& op) override;
    void VisitUnaryExpr_(const UnaryExprPtr& op) override;

    void VisitType_(const UnknownTypePtr& op) override;
    void VisitType_(const ScalarTypePtr& op) override;
    void VisitType_(const TensorTypePtr& op) override;
    void VisitType_(const TileTypePtr& op) override;
    void VisitType_(const TupleTypePtr& op) override;
    void VisitType_(const MemRefTypePtr& op) override;
    void VisitType_(const PtrTypePtr& op) override;
    void VisitType_(const TokenTypePtr& op) override;
    void VisitType_(const NoneTypePtr& op) override;
    void VisitType_(const LogicalTensorTypePtr& op) override;

    void VisitStmt_(const AssignStmtPtr& op) override;
    void VisitStmt_(const SeqStmtsPtr& op) override;
    void VisitStmt_(const IfStmtPtr& op) override;
    void VisitStmt_(const YieldStmtPtr& op) override;
    void VisitStmt_(const ReturnStmtPtr& op) override;
    void VisitStmt_(const ForStmtPtr& op) override;
    void VisitStmt_(const WhileStmtPtr& op) override;
    void VisitStmt_(const SectionStmtPtr& op) override;
    void VisitStmt_(const EvalStmtPtr& op) override;
    void VisitStmt_(const BreakStmtPtr& op) override;
    void VisitStmt_(const ContinueStmtPtr& op) override;
    void VisitStmt_(const ScalarOpStmtPtr& op) override;
    void VisitStmt_(const TensorOpStmtPtr& op) override;
    void VisitStmt_(const StmtPtr& op) override;

    void VisitFunction(const FunctionPtr& func) override;
    void VisitProgram(const ProgramPtr& program) override;

private:
    std::ostringstream stream_;
    int indent_ = 0;

    std::string GetIndent() const { return std::string(static_cast<size_t>(indent_ * 4), ' '); }

    void PrintVarDef(const VarPtr& var);
    void PrintVarRef(const VarPtr& var);
    void PrintVarDefList(const std::vector<VarPtr>& varList, const std::vector<VarPtr>& tokenList = {});
    void PrintIterArgs(const std::vector<IterArgPtr>& iterArgs);

    void PrintType(const TypePtr& type);
    void PrintShape(const std::vector<ExprPtr>& shape);
    void PrintExprList(const std::vector<ExprPtr>& exprs);
    void PrintTokenList(const std::vector<VarPtr>& tokenList);
    void PrintAttrValue(const std::string& key, const std::any& value);
    void PrintAttr(const std::string& key, const std::any& value);
    template <typename T>
    void PrintAttr(const std::string& key, const T& value)
    {
        stream_ << " " << IR_PUN_ATTRNAME << key << IR_PUN_LPAREN << value << IR_PUN_RPAREN;
    }

    void PrintBody(const StmtPtr& body);

    static const char* BinaryOpName(const BinaryExprPtr& op);
    static const char* UnaryOpName(const UnaryExprPtr& op);
};

// ===========================================================================
// Loader
// ===========================================================================
class IRLoaderText {
public:
    IRLoaderText() = default;

    static ProgramPtr LoadProgram(const std::string& text);
    static FunctionPtr LoadFunction(const std::string& text);
    static StmtPtr LoadStmt(const std::string& text);
    static ExprPtr LoadExpr(const std::string& text);
    static TypePtr LoadType(const std::string& text);
};

// ===========================================================================
// Public free functions
// ===========================================================================
std::string TextDump(const IRNodePtr& node);
std::string TextDumpType(const TypePtr& type);

ProgramPtr TextLoadProgram(const std::string& text);
FunctionPtr TextLoadFunction(const std::string& text);
StmtPtr TextLoadStmt(const std::string& text);
ExprPtr TextLoadExpr(const std::string& text);
TypePtr TextLoadType(const std::string& text);

// ===========================================================================
// Operator BiMap dictionaries
// ===========================================================================
static inline const npu::tile_fwk::BiMap<ObjectKind>& GetBinaryOpDict()
{
    static npu::tile_fwk::BiMap<ObjectKind> dict{{
        {ObjectKind::Add, IR_KW_SCALAR_BOP_ADD},
        {ObjectKind::Sub, IR_KW_SCALAR_BOP_SUB},
        {ObjectKind::Mul, IR_KW_SCALAR_BOP_MUL},
        {ObjectKind::FloorDiv, IR_KW_SCALAR_BOP_DIV},
        {ObjectKind::FloorMod, IR_KW_SCALAR_BOP_MOD},
        {ObjectKind::FloatDiv, IR_KW_SCALAR_BOP_FDIV},
        {ObjectKind::Min, IR_KW_SCALAR_BOP_MIN},
        {ObjectKind::Max, IR_KW_SCALAR_BOP_MAX},
        {ObjectKind::Pow, IR_KW_SCALAR_BOP_POW},
        {ObjectKind::Eq, IR_KW_SCALAR_BOP_EQ},
        {ObjectKind::Ne, IR_KW_SCALAR_BOP_NE},
        {ObjectKind::Lt, IR_KW_SCALAR_BOP_LT},
        {ObjectKind::Le, IR_KW_SCALAR_BOP_LE},
        {ObjectKind::Gt, IR_KW_SCALAR_BOP_GT},
        {ObjectKind::Ge, IR_KW_SCALAR_BOP_GE},
        {ObjectKind::And, IR_KW_SCALAR_BOP_LAND},
        {ObjectKind::Or, IR_KW_SCALAR_BOP_LOR},
        {ObjectKind::Xor, IR_KW_SCALAR_BOP_LXOR},
        {ObjectKind::BitAnd, IR_KW_SCALAR_BOP_AND},
        {ObjectKind::BitOr, IR_KW_SCALAR_BOP_OR},
        {ObjectKind::BitXor, IR_KW_SCALAR_BOP_XOR},
        {ObjectKind::BitShiftLeft, IR_KW_SCALAR_BOP_SHL},
        {ObjectKind::BitShiftRight, IR_KW_SCALAR_BOP_SHR},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<ObjectKind>& GetUnaryOpDict()
{
    static npu::tile_fwk::BiMap<ObjectKind> dict{{
        {ObjectKind::Abs, IR_KW_SCALAR_UOP_ABS},
        {ObjectKind::Neg, IR_KW_SCALAR_UOP_NEG},
        {ObjectKind::Not, IR_KW_SCALAR_UOP_NOT},
        {ObjectKind::BitNot, IR_KW_SCALAR_UOP_INV},
        {ObjectKind::Cast, IR_KW_SCALAR_UOP_CAST},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TensorLayout>& GetTensorLayoutDict()
{
    static npu::tile_fwk::BiMap<TensorLayout> dict{{
        {TensorLayout::ND, "ND"},
        {TensorLayout::DN, "DN"},
        {TensorLayout::NZ, "NZ"},
        {TensorLayout::ZN, "ZN"},
        {TensorLayout::NN, "NN"},
        {TensorLayout::ZZ, "ZZ"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TileLayout>& GetTileLayoutDict()
{
    static npu::tile_fwk::BiMap<TileLayout> dict{{
        {TileLayout::none_box, "none_box"},
        {TileLayout::row_major, "row_major"},
        {TileLayout::col_major, "col_major"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<TilePad>& GetTilePadDict()
{
    static npu::tile_fwk::BiMap<TilePad> dict{{
        {TilePad::null, "null"},
        {TilePad::zero, "zero"},
        {TilePad::max, "max"},
        {TilePad::min, "min"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<CompactMode>& GetCompactModeDict()
{
    static npu::tile_fwk::BiMap<CompactMode> dict{{
        {CompactMode::null, "null"},
        {CompactMode::normal, "normal"},
        {CompactMode::row_plus_one, "row_plus_one"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<MemorySpace>& GetMemorySpaceDict()
{
    static npu::tile_fwk::BiMap<MemorySpace> dict{{
        {MemorySpace::DDR, "ddr"},
        {MemorySpace::Vec, "vec"},
        {MemorySpace::Mat, "mat"},
        {MemorySpace::Left, "left"},
        {MemorySpace::Right, "right"},
        {MemorySpace::Scaling, "scaling"},
        {MemorySpace::Acc, "acc"},
        {MemorySpace::Bias, "bias"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<SectionKind>& GetSectionKindDict()
{
    static npu::tile_fwk::BiMap<SectionKind> dict{{
        {SectionKind::Vector, "vector"},
        {SectionKind::Cube, "cube"},
        {SectionKind::VF, "vf"},
    }};
    return dict;
}

static inline const npu::tile_fwk::BiMap<FunctionType>& GetFunctionTypeDict()
{
    static npu::tile_fwk::BiMap<FunctionType> dict{{
        {FunctionType::OPAQUE, "opaque"},
        {FunctionType::ORCHESTRATION, "orchestration"},
        {FunctionType::IN_CORE, "in_core"},
        {FunctionType::HELPER, "helper"},
    }};
    return dict;
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_IO_TEXT_H_
