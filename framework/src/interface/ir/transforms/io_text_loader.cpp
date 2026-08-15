/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file loader_text.cpp
 * \brief IR text loader (parser) implementation.
 *
 * Recursive-descent parser that reads the text grammar produced by
 * IRDumperText and reconstructs IR nodes.
 */

#include "ir/transforms/io_text.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/span.h"
#include "ir/stmt.h"

namespace pypto {
namespace ir {

// ===========================================================================
// Tokenizer (regex-based)
// ===========================================================================
namespace {

struct LexerToken {
    LexerTokenKind kind = LexerTokenKind::Invalid;
    std::string text; // ident text, punct char, or string content (unquoted)
    int64_t intVal = 0;
    double floatVal = 0.0;
    size_t begin = 0; // byte offset of token start in source
    size_t end = 0;   // byte offset of token end in source
};

static bool IsKw(LexerTokenKind k) { return k >= LexerTokenKind::KwProgram && k <= LexerTokenKind::KwUOpCast; }

static bool IsDTypeKw(LexerTokenKind k) { return k >= LexerTokenKind::KwTypeBegin && k <= LexerTokenKind::KwTypeEnd; }

static bool IsIdentOrKw(LexerTokenKind k) { return k == LexerTokenKind::Ident || IsKw(k) || IsDTypeKw(k); }

/// Regex-based lexer.  Patterns are tried in priority order at the current
/// position using match_continuous, so the longest correct token always wins.
class Lexer {
public:
    explicit Lexer(const std::string& src) : src_(src)
    {
        using R = std::pair<LexerTokenKind, std::regex>;
        rules_ = {
            R{LexerTokenKind::VarName, std::regex(R"(%[A-Za-z_][A-Za-z0-9_.]*)")},
            R{LexerTokenKind::AttrName, std::regex(R"(#[A-Za-z_][A-Za-z0-9_.]*)")},
            R{LexerTokenKind::Float, std::regex(R"(-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+)")},
            R{LexerTokenKind::Int, std::regex(R"(-?\d+)")},
            R{LexerTokenKind::Ident, std::regex(R"([A-Za-z_][A-Za-z0-9_]*)")},
        };
        wsRe_ = std::regex(R"(\s+)");
    }

    std::optional<std::vector<LexerToken>> Tokenize()
    {
        std::vector<LexerToken> toks;
        auto it = src_.cbegin();
        const auto end = src_.cend();
        while (it != end) {
            // 1. Skip whitespace.
            std::smatch ws;
            if (std::regex_search(it, end, ws, wsRe_, std::regex_constants::match_continuous)) {
                it = ws[0].second;
                continue;
            }
            // 2. Try each token pattern in priority order.
            size_t startPos = static_cast<size_t>(it - src_.cbegin());
            bool matched = false;
            for (const auto& [kind, re] : rules_) {
                std::smatch m;
                if (std::regex_search(it, end, m, re, std::regex_constants::match_continuous)) {
                    size_t endPos = static_cast<size_t>(m[0].second - src_.cbegin());
                    toks.push_back(MakeToken(kind, m.str(), startPos, endPos));
                    it = m[0].second;
                    matched = true;
                    break;
                }
            }
            // 3. Try punctuation (single-char tokens).
            if (!matched) {
                LexerTokenKind punKind = ClassifyPunct(*it);
                if (punKind != LexerTokenKind::Invalid) {
                    std::string lexeme(1, *it);
                    size_t endPos = startPos + 1;
                    toks.push_back(LexerToken{punKind, lexeme, 0, 0.0, startPos, endPos});
                    ++it;
                    matched = true;
                }
            }
            if (!matched) {
                errorPos_ = startPos;
                return std::nullopt;
            }
        }
        toks.push_back(LexerToken{LexerTokenKind::Invalid, "", 0, 0.0, static_cast<size_t>(src_.size()),
                                  static_cast<size_t>(src_.size())});
        return toks;
    }

    size_t GetErrorPos() const { return errorPos_; }

private:
    const std::string& src_;
    std::vector<std::pair<LexerTokenKind, std::regex>> rules_;
    std::regex wsRe_;
    size_t errorPos_ = 0;

    static LexerTokenKind ClassifyKeyword(const std::string& s)
    {
        const auto& dict = GetLexerTokenDict();
        return dict.Count(s) ? dict.Find(s, LexerTokenKind::Ident) : LexerTokenKind::Ident;
    }

    static LexerTokenKind ClassifyPunct(char c)
    {
        std::string s(1, c);
        const auto& dict = GetLexerTokenDict();
        return dict.Count(s) ? dict.Find(s, LexerTokenKind::Invalid) : LexerTokenKind::Invalid;
    }

    static LexerToken MakeToken(LexerTokenKind kind, const std::string& lexeme, size_t begin, size_t end)
    {
        // Reclassify Ident as specific keyword if it matches.
        if (kind == LexerTokenKind::Ident) {
            kind = ClassifyKeyword(lexeme);
        }
        switch (kind) {
            case LexerTokenKind::Int: {
                int64_t val = std::stoll(lexeme);
                return LexerToken{kind, lexeme, val, 0.0, begin, end};
            }
            case LexerTokenKind::Float:
                return LexerToken{kind, lexeme, 0, std::stod(lexeme), begin, end};
            case LexerTokenKind::VarName:
            case LexerTokenKind::AttrName:
                return LexerToken{kind, lexeme.substr(1), 0, 0.0, begin, end};
            default:
                return LexerToken{kind, lexeme, 0, 0.0, begin, end};
        }
    }
};

} // namespace

// ===========================================================================
// Parser
// ===========================================================================
#define IR_TEXT_PARSER_RESULT "IRTextParserResult"
#define IR_TEXT_PARSER_SUCCESS "IRTextParserSuccess"

struct IRTextParserResult : public IRNode {
    IRTextParserResult() : IRNode(Span::Unknown()) {}
    ObjectKind GetKind() const override { return ObjectKind::Invalid; }
    std::string TypeName() const override { return IR_TEXT_PARSER_RESULT; }
};

struct IRTextParserSuccess : public IRTextParserResult {
    IRTextParserSuccess() : IRTextParserResult() {}
    std::string TypeName() const override { return IR_TEXT_PARSER_SUCCESS; }
};
using IRTextParserSuccessPtr = std::shared_ptr<const IRTextParserSuccess>;

template <typename T>
struct IRTextParserData : public IRTextParserResult {
    IRTextParserData() : IRTextParserResult() {}
    IRTextParserData(const T& d) : data(d) {}
    T data;
    std::string TypeName() const override { return typeid(T).name(); }
};
template <typename T>
using IRTextParserDataPtr = std::shared_ptr<IRTextParserData<T>>;

using IRTextParserStringPtr = IRTextParserDataPtr<std::string>;
using IRTextParserAttrValuePtr = IRTextParserDataPtr<std::any>;
using IRTextParserExprListPtr = IRTextParserDataPtr<std::vector<ExprPtr>>;
using IRTextParserIterArgListPtr = IRTextParserDataPtr<std::vector<IterArgPtr>>;
using IRTextParserAttrListPtr = IRTextParserDataPtr<std::vector<std::pair<std::string, std::any>>>;
using IRTextParserOptionalTensorViewPtr = IRTextParserDataPtr<std::optional<TensorView>>;
using IRTextParserOptionalTileViewPtr = IRTextParserDataPtr<std::optional<TileView>>;
using IRTextParserOptionalHardwareInfoPtr = IRTextParserDataPtr<std::optional<HardwareInfo>>;
using IRTextParserVarDefListPtr = IRTextParserDataPtr<std::vector<VarPtr>>;

#define MUST_MATCH(call)    \
    do {                    \
        if (!(call))        \
            return nullptr; \
    } while (0)

#define MUST_VALID(name, call) \
    auto name = (call);        \
    do {                       \
        if (!(name))           \
            return nullptr;    \
    } while (0)

class IRTextParser {
public:
    explicit IRTextParser(const std::string& text) : cur_(0)
    {
        succ_ = std::make_shared<IRTextParserSuccess>();
        Lexer lexer(text);
        auto result = lexer.Tokenize();
        if (result) {
            tokens_ = std::move(*result);
        } else {
            tokens_.push_back(LexerToken{LexerTokenKind::Invalid, "", 0, 0.0, 0, 0});
            errorMsg_ = "lex error at byte offset " + std::to_string(lexer.GetErrorPos());
            errorPos_ = lexer.GetErrorPos();
        }
    }

    friend class IRLoaderText;

    // ---- Helpers ----
    const LexerToken& Peek() const { return tokens_[cur_]; }
    const LexerToken& PeekAt(size_t offset) const { return tokens_[std::min(cur_ + offset, tokens_.size() - 1)]; }
    bool CheckToken(LexerTokenKind kind, bool consumeIfPass = false)
    {
        bool result = Peek().kind == kind;
        if (result) {
            if (consumeIfPass) {
                ++cur_;
            }
        }
        return result;
    }
    bool CheckVarName() const { return Peek().kind == LexerTokenKind::VarName; }
    bool CheckAttrName() const { return Peek().kind == LexerTokenKind::AttrName; }
    std::nullptr_t Error(const std::string& msg)
    {
        errorMsg_ = msg;
        errorPos_ = Peek().begin;
        return nullptr;
    }
    IRNodePtr ParseToken(LexerTokenKind kind)
    {
        if (!CheckToken(kind, true)) {
            return Error("expected token, got '" + Peek().text + "'");
        }
        return succ_;
    }
    IRTextParserStringPtr ParseIdent()
    {
        if (!IsIdentOrKw(Peek().kind)) {
            return Error("expected identifier, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserStringPtr ParseVarName()
    {
        if (!CheckVarName()) {
            return Error("expected variable name, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserStringPtr ParseAttrName()
    {
        if (!CheckAttrName()) {
            return Error("expected attribute name, got '" + Peek().text + "'");
        }
        auto result = std::make_shared<IRTextParserData<std::string>>();
        result->data = tokens_[cur_++].text;
        return result;
    }
    IRTextParserDataPtr<DataType> ParseDType()
    {
        if (!IsIdentOrKw(Peek().kind)) {
            return Error("expected dtype, got '" + Peek().text + "'");
        }
        const auto& dict = DataType::GetDataTypeCTypeStringDict();
        const std::string& s = Peek().text;
        if (!dict.Count(s)) {
            return Error("unknown C-type string: " + s);
        }
        auto result = std::make_shared<IRTextParserData<DataType>>();
        result->data = DataType(dict.Find(s, static_cast<uint8_t>(0)));
        ++cur_;
        return result;
    }
    IRTextParserExprListPtr ParseExprList(LexerTokenKind split, LexerTokenKind stop)
    {
        auto result = std::make_shared<IRTextParserData<std::vector<ExprPtr>>>();
        if (!CheckToken(stop)) {
            do {
                MUST_VALID(expr, ParseExpr());
                result->data.push_back(expr);
            } while (CheckToken(split, true));
        }
        return result;
    }
    IRTextParserIterArgListPtr ParseIterArgs()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<IterArgPtr>>>();
        MUST_MATCH(ParseToken(LexerTokenKind::KwIter));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLBrace));
        while (!CheckToken(LexerTokenKind::PunRBrace)) {
            MUST_VALID(type, ParseType());
            MUST_VALID(name, ParseVarName());
            MUST_MATCH(ParseToken(LexerTokenKind::PunEq));
            MUST_VALID(initVal, ParseExpr());
            MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
            auto ia = std::make_shared<IterArg>(std::make_shared<Var>(name->data, type, Span::Unknown()), initVal);
            symtab_[name->data] = ia->iterVar_;
            result->data.push_back(ia);
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRBrace));
        return result;
    }
    IRTextParserAttrListPtr ParseAttrList()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<std::pair<std::string, std::any>>>>();
        while (CheckAttrName()) {
            MUST_VALID(key, ParseAttrName());
            MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
            MUST_VALID(val, ParseAttrValueRaw());
            MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
            result->data.emplace_back(key->data, val->data);
        }
        return result;
    }

    // ---- Var definition:  type %name->data ----
    VarPtr ParseVarDef()
    {
        MUST_VALID(type, ParseType());
        MUST_VALID(name, ParseVarName());
        auto v = std::make_shared<Var>(name->data, type, Span::Unknown());
        symtab_[name->data] = v;
        return v;
    }

    // ---- Shape parsing:  dim0 x dim1 x ... ----
    IRTextParserExprListPtr ParseShape()
    {
        auto result = std::make_shared<IRTextParserData<std::vector<ExprPtr>>>();
        MUST_VALID(dim, ParseExpr());
        result->data.push_back(dim);
        while (CheckToken(LexerTokenKind::KwDim)) {
            ++cur_; // consume 'x'
            MUST_VALID(dim2, ParseExpr());
            result->data.push_back(dim2);
        }
        return result;
    }

    // ---- Attr value parsing (per-branch, dispatched by token) ----
    using ParseAttrValueFn = IRTextParserAttrValuePtr (IRTextParser::*)();

    IRTextParserAttrValuePtr ParseAttrValueInt()
    {
        const LexerToken& tok = Peek();
        ++cur_;
        std::any data = std::any(static_cast<int>(tok.intVal));
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueFloat()
    {
        const LexerToken& tok = Peek();
        ++cur_;
        std::any data = std::any(tok.floatVal);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueTrue()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTrue));
        std::any data = true;
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueFalse()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwFalse));
        std::any data = false;
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueIdent()
    {
        const LexerToken& tok = Peek();
        std::string s = tok.text;
        ++cur_;
        std::any data = std::any(s);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    IRTextParserAttrValuePtr ParseAttrValueList()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::PunLBracket));
        std::vector<int> vals;
        while (!CheckToken(LexerTokenKind::PunRBracket)) {
            if (Peek().kind == LexerTokenKind::Int) {
                vals.push_back(static_cast<int>(tokens_[cur_++].intVal));
            }
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRBracket));
        std::any data = std::any(vals);
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }

    IRTextParserAttrValuePtr ParseAttrValueRaw()
    {
        static const std::unordered_map<LexerTokenKind, ParseAttrValueFn> dispatch = {
            {LexerTokenKind::Int, &IRTextParser::ParseAttrValueInt},
            {LexerTokenKind::Float, &IRTextParser::ParseAttrValueFloat},
            {LexerTokenKind::Ident, &IRTextParser::ParseAttrValueIdent},
            {LexerTokenKind::KwTrue, &IRTextParser::ParseAttrValueTrue},
            {LexerTokenKind::KwFalse, &IRTextParser::ParseAttrValueFalse},
            {LexerTokenKind::PunLBracket, &IRTextParser::ParseAttrValueList},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            auto result = (this->*(it->second))();
            return result;
        }
        return Error("expected attr value");
    }

    ExprPtr MakeUnary(const std::string& opName, const ExprPtr& operand, const TypePtr& type)
    {
        DataType dt = DataType::INT64;
        if (auto st = std::dynamic_pointer_cast<const ScalarType>(type)) {
            dt = st->dtype_;
        }
        auto result = MakeUnaryOp(opName, operand, dt);
        if (!result) {
            return Error("Unknown unary op: " + opName);
        }
        return result;
    }

    ExprPtr MakeBinary(const std::string& opName, const ExprPtr& left, const ExprPtr& right)
    {
        auto result = MakeBinaryOp(opName, left, right);
        if (!result) {
            return Error("Unknown binary op: " + opName);
        }
        return result;
    }

    // ---- Operator helpers ----
    static bool IsUnaryOpName(const std::string& s)
    {
        return s == IR_KW_SCALAR_UOP_ABS || s == IR_KW_SCALAR_UOP_NEG || s == IR_KW_SCALAR_UOP_NOT ||
               s == IR_KW_SCALAR_UOP_INV || s == IR_KW_SCALAR_UOP_CAST;
    }

    // ---- Type parsing ----
    // ---- Type parsing (per-branch, dispatched by token) ----
    TypePtr ParseTypeUnknown()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwUnknown));
        return GetUnknownType();
    }
    TypePtr ParseTypeMemRefType()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwMemrefType));
        return GetMemRefType();
    }
    TypePtr ParseTypeToken()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwToken));
        return GetTokenType();
    }
    TypePtr ParseTypeNone()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwNone));
        return GetNoneType();
    }
    TypePtr ParseTypeLogicalTensor()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwLogicalTensor));
        return GetLogicalTensorType();
    }
    TypePtr ParseTypePtr()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwPtr));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));
        MUST_VALID(dt, ParseDType());
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<PtrType>(dt->data);
    }
    TypePtr ParseTypeTuple()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTuple));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));
        std::vector<TypePtr> types;
        if (!CheckToken(LexerTokenKind::PunGt)) {
            do {
                MUST_VALID(t, ParseType());
                types.push_back(t);
            } while (CheckToken(LexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<TupleType>(types);
    }
    TypePtr ParseTypeTensor()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTensor));
        return ParseTensorType();
    }
    TypePtr ParseTypeTile()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTile));
        return ParseTileType();
    }
    TypePtr ParseTypeScalar()
    {
        MUST_VALID(dt, ParseDType());
        return std::make_shared<ScalarType>(dt->data);
    }

    IRTextParserOptionalTensorViewPtr ParserTensorTypeTensorView()
    {
        // tensor_view<...> or tensor_view<>
        MUST_MATCH(ParseToken(LexerTokenKind::KwTensorView));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));

        std::optional<TensorView> tensorView;
        if (!CheckToken(LexerTokenKind::PunGt)) {
            MUST_VALID(validShape, ParseShape());
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(stride, ParseShape());
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(layoutStr, ParseIdent());
            TensorLayout layout = GetTensorLayoutDict().Find(layoutStr->data, TensorLayout::ND);
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(ptrExpr, ParseExpr());
            tensorView = TensorView(validShape->data, stride->data, layout, ptrExpr);
        }

        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalTensorViewPtr::element_type>(tensorView);
    }

    // ---- Type-specific parsers ----
    TypePtr ParseTensorType()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));
        MUST_VALID(shape, ParseShape());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(dtype, ParseDType());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(tensorView, ParserTensorTypeTensorView());
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<TensorType>(shape->data, dtype->data, std::nullopt, tensorView->data);
    }

    IRTextParserOptionalTileViewPtr ParseTileTypeTileView()
    {
        // tile_view<...> or tile_view<>
        MUST_MATCH(ParseToken(LexerTokenKind::KwTileView));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));

        std::optional<TileView> tileView;
        if (!CheckToken(LexerTokenKind::PunGt)) {
            auto validShape = ParseShape();
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            auto stride = ParseShape();
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(startOffset, ParseExpr());
            tileView = TileView(validShape->data, stride->data, startOffset);
        }

        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalTileViewPtr::element_type>(tileView);
    }

    IRTextParserOptionalHardwareInfoPtr ParseTileTypeHardwareInfo()
    {
        // hw_info<...> or hw_info<>
        MUST_MATCH(ParseToken(LexerTokenKind::KwHwInfo));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));

        std::optional<HardwareInfo> hwInfo;
        if (!CheckToken(LexerTokenKind::PunGt)) {
            MUST_VALID(blayoutStr, ParseIdent());
            TileLayout blayout = GetTileLayoutDict().Find(blayoutStr->data, TileLayout::row_major);
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(slayoutStr, ParseIdent());
            TileLayout slayout = GetTileLayoutDict().Find(slayoutStr->data, TileLayout::none_box);
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            uint64_t fractal = HardwareInfo::kDefaultFractal;
            if (Peek().kind == LexerTokenKind::Int) {
                fractal = static_cast<uint64_t>(tokens_[cur_].intVal);
                ++cur_;
            }
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(padStr, ParseIdent());
            TilePad pad = GetTilePadDict().Find(padStr->data, TilePad::null);
            MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
            MUST_VALID(compactStr, ParseIdent());
            CompactMode compact = GetCompactModeDict().Find(compactStr->data, CompactMode::null);
            hwInfo = HardwareInfo(blayout, slayout, fractal, pad, compact);
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<IRTextParserOptionalHardwareInfoPtr::element_type>(hwInfo);
    }

    TypePtr ParseTileType()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));
        MUST_VALID(shape, ParseShape());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(dtype, ParseDType());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(tileView, ParseTileTypeTileView());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(hwInfo, ParseTileTypeHardwareInfo());
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<TileType>(shape->data, dtype->data, std::nullopt, tileView->data, hwInfo->data);
    }

    TypePtr ParseType()
    {
        using ParseTypeFn = TypePtr (IRTextParser::*)();
        if (IsDTypeKw(Peek().kind)) {
            return ParseTypeScalar();
        }
        static const std::unordered_map<LexerTokenKind, ParseTypeFn> dispatch = {
            {LexerTokenKind::KwUnknown, &IRTextParser::ParseTypeUnknown},
            {LexerTokenKind::KwMemrefType, &IRTextParser::ParseTypeMemRefType},
            {LexerTokenKind::KwToken, &IRTextParser::ParseTypeToken},
            {LexerTokenKind::KwNone, &IRTextParser::ParseTypeNone},
            {LexerTokenKind::KwLogicalTensor, &IRTextParser::ParseTypeLogicalTensor},
            {LexerTokenKind::KwPtr, &IRTextParser::ParseTypePtr},
            {LexerTokenKind::KwTuple, &IRTextParser::ParseTypeTuple},
            {LexerTokenKind::KwTensor, &IRTextParser::ParseTypeTensor},
            {LexerTokenKind::KwTile, &IRTextParser::ParseTypeTile},
            {LexerTokenKind::Ident, &IRTextParser::ParseTypeScalar},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))();
        }
        return Error("expected type");
    }

    // ---- Expression parsing (per-branch, dispatched by token) ----
    ExprPtr ParseExprVarName()
    {
        std::string name = Peek().text;
        if (Peek().kind != LexerTokenKind::VarName) {
            return Error("expected variable name, got '" + name + "'");
        }
        ++cur_;
        auto it = symtab_.find(name);
        if (it != symtab_.end()) {
            return it->second;
        } else {
            return Error("Unknown var name: " + name);
        }
    }
    ExprPtr ParseExprInt()
    {
        auto val = Peek().intVal;
        ++cur_;
        return std::make_shared<ConstInt>(val, DataType::INT64, Span::Unknown());
    }
    ExprPtr ParseExprFloat()
    {
        auto val = Peek().floatVal;
        ++cur_;
        return std::make_shared<ConstFloat>(val, DataType::FP64, Span::Unknown());
    }
    ExprPtr ParseExprTrue()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTrue));
        return std::make_shared<ConstBool>(true, Span::Unknown());
    }
    ExprPtr ParseExprFalse()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwFalse));
        return std::make_shared<ConstBool>(false, Span::Unknown());
    }
    ExprPtr ParseExprTuple()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwTuple));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        MUST_VALID(elts, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
        return std::make_shared<MakeTuple>(elts->data, Span::Unknown());
    }
    ExprPtr ParseExprGetItem()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwGetItem));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        MUST_VALID(value, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(slice, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
        return std::make_shared<GetItemExpr>(value, slice, Span::Unknown());
    }
    ExprPtr ParseExprMemref()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwMemref));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLt));
        MUST_VALID(spaceStr, ParseIdent());
        MemorySpace space = StringToMemorySpace(spaceStr->data);
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        ExprPtr addr = ParseExpr();
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        uint64_t size = 0;
        if (Peek().kind == LexerTokenKind::Int) {
            size = static_cast<uint64_t>(tokens_[cur_++].intVal);
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunGt));
        return std::make_shared<MemRef>(space, addr, size, Span::Unknown());
    }
    ExprPtr ParseExprScalarExpr()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwScalarExpr));
        return std::make_shared<ScalarExpr>(DataType::INT64, Span::Unknown());
    }
    ExprPtr ParseExprCall()
    {
        std::string name = Peek().text;
        MUST_VALID(ident, ParseIdent());
        if (!CheckToken(LexerTokenKind::PunLParen)) {
            return Error("unexpected identifier: " + name);
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        std::vector<ExprPtr> args;
        std::vector<std::pair<std::string, std::any>> kwargs;
        if (!CheckToken(LexerTokenKind::PunRParen)) {
            do {
                if (Peek().kind == LexerTokenKind::Ident && PeekAt(1).kind == LexerTokenKind::PunEq) {
                    MUST_VALID(key, ParseIdent());
                    MUST_MATCH(ParseToken(LexerTokenKind::PunEq));
                    MUST_VALID(val, ParseAttrValueRaw());
                    kwargs.emplace_back(key->data, val->data);
                } else {
                    MUST_VALID(arg, ParseExpr());
                    args.push_back(arg);
                }
            } while (CheckToken(LexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
        return std::make_shared<Call>(name, args, kwargs, Span::Unknown());
    }
    ExprPtr ParseExprParen()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        const LexerToken& first = Peek();

        // Determine if this is unary: first token is an Ident that is a known unary op
        if (IsIdentOrKw(first.kind) && IsUnaryOpName(first.text)) {
            std::string opName = first.text;
            ++cur_;
            DataType castDtype = DataType::INT64;
            if (opName == IR_KW_SCALAR_UOP_CAST) {
                MUST_VALID(castDt, ParseDType());
                castDtype = castDt->data;
            }
            MUST_VALID(operand, ParseExpr());
            MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
            auto scalarType = std::make_shared<ScalarType>(castDtype);
            return MakeUnary(opName, operand, scalarType);
        } else {
            // Binary: expr op expr
            MUST_VALID(left, ParseExpr());
            MUST_VALID(op, ParseIdent());
            MUST_VALID(right, ParseExpr());
            MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
            return MakeBinary(op->data, left, right);
        }
    }

    ExprPtr ParseExpr()
    {
        using ParseExprFn = ExprPtr (IRTextParser::*)();
        static const std::unordered_map<LexerTokenKind, ParseExprFn> dispatch = {
            {LexerTokenKind::VarName, &IRTextParser::ParseExprVarName},
            {LexerTokenKind::Int, &IRTextParser::ParseExprInt},
            {LexerTokenKind::Float, &IRTextParser::ParseExprFloat},
            {LexerTokenKind::PunLParen, &IRTextParser::ParseExprParen},
            {LexerTokenKind::KwTrue, &IRTextParser::ParseExprTrue},
            {LexerTokenKind::KwFalse, &IRTextParser::ParseExprFalse},
            {LexerTokenKind::KwTuple, &IRTextParser::ParseExprTuple},
            {LexerTokenKind::KwGetItem, &IRTextParser::ParseExprGetItem},
            {LexerTokenKind::KwMemref, &IRTextParser::ParseExprMemref},
            {LexerTokenKind::KwScalarExpr, &IRTextParser::ParseExprScalarExpr},
            {LexerTokenKind::Ident, &IRTextParser::ParseExprCall},
        };
        auto it = dispatch.find(Peek().kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))();
        }
        return Error("unexpected token in expression: '" + Peek().text + "'");
    }

    // ---- Statement parsing ----

    IRTextParserVarDefListPtr ParseVarDefList(LexerTokenKind split, LexerTokenKind stop)
    {
        std::vector<VarPtr> varDefList;
        if (!CheckToken(stop)) {
            do {
                MUST_VALID(varDef, ParseVarDef());
                varDefList.push_back(varDef);
            } while (CheckToken(split, true));
        }
        return std::make_shared<IRTextParserVarDefListPtr::element_type>(varDefList);
    }

    StmtPtr ParseStmtSeq(const std::vector<VarPtr>& = {})
    {
        // Reads statements until '}' or end
        MUST_MATCH(ParseToken(LexerTokenKind::PunLBrace));
        std::vector<StmtPtr> stmts;
        while (!CheckToken(LexerTokenKind::PunRBrace) && Peek().kind != LexerTokenKind::Invalid) {
            stmts.push_back(ParseStmt());
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRBrace));
        return std::make_shared<SeqStmts>(stmts, Span::Unknown());
    }

    // ---- Individual statement parsers ----
    StmtPtr ParseStmtYield(const std::vector<VarPtr>&)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwYield));
        MUST_VALID(values, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<YieldStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtReturn(const std::vector<VarPtr>&)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwReturn));
        MUST_VALID(values, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<ReturnStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtIf(const std::vector<VarPtr>& defVarList)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwIf));
        ExprPtr cond = ParseExpr();
        MUST_MATCH(ParseToken(LexerTokenKind::KwThen));
        MUST_VALID(thenBody, ParseStmtSeq());
        MUST_MATCH(ParseToken(LexerTokenKind::KwElse));
        MUST_VALID(elseBody, ParseStmtSeq());
        return std::make_shared<IfStmt>(cond, thenBody, elseBody, defVarList, Span::Unknown());
    }

    StmtPtr ParseStmtFor(const std::vector<VarPtr>& defVarList)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwFor));
        // for %loopVar inrange start, stop, step [iter (...)] {body}
        MUST_VALID(loopVarName, ParseVarName());
        auto loopVar = std::make_shared<Var>(loopVarName->data, std::make_shared<ScalarType>(DataType::INT64),
                                             Span::Unknown());
        symtab_[loopVarName->data] = loopVar;
        MUST_MATCH(ParseToken(LexerTokenKind::KwInrange));
        MUST_VALID(start, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(stop, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunComma));
        MUST_VALID(step, ParseExpr());
        MUST_VALID(iterArgs, ParseIterArgs());
        MUST_VALID(attrs, ParseAttrList());
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs->data, body, defVarList, Span::Unknown(),
                                         attrs->data);
    }

    StmtPtr ParseStmtWhile(const std::vector<VarPtr>& defVarList)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwWhile));
        MUST_VALID(cond, ParseExpr());
        MUST_VALID(iterArgs, ParseIterArgs());
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<WhileStmt>(cond, iterArgs->data, body, defVarList, Span::Unknown());
    }

    StmtPtr ParseStmtSection(const std::vector<VarPtr>&)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwSection));
        MUST_VALID(sectionName, ParseIdent());
        SectionKind kind = StringToSectionKind(sectionName->data);
        MUST_VALID(body, ParseStmtSeq());
        return std::make_shared<SectionStmt>(kind, body, Span::Unknown());
    }

    StmtPtr ParseStmtEval(const std::vector<VarPtr>&)
    {
        MUST_VALID(expr, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<EvalStmt>(expr, Span::Unknown());
    }

    StmtPtr ParseStmtBreak(const std::vector<VarPtr>&)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwBreak));
        MUST_VALID(values, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<BreakStmt>(values->data, Span::Unknown());
    }

    StmtPtr ParseStmtContinue(const std::vector<VarPtr>&)
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwContinue));
        MUST_VALID(values, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunSemi));
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<ContinueStmt>(values->data, Span::Unknown());
    }

    // Assignment or op statement:
    //   type %result [= expr] ;
    //   type %result, type %token = opcode(args) [tokens(...)] [#attr(val)...] ;
    //   type %result, type %token = opcode(args) ;
    StmtPtr ParseStmtAssignOrOp(const std::vector<VarPtr>& defVarList)
    {
        MUST_VALID(opcode, ParseIdent());
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        MUST_VALID(args, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));

        std::vector<VarPtr> tokenList;
        if (CheckToken(LexerTokenKind::KwToken, true)) {
            MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
            MUST_VALID(tokens, ParseExprList(LexerTokenKind::PunComma, LexerTokenKind::PunRParen));
            MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
            for (const auto& token : tokens->data) {
                tokenList.push_back(std::dynamic_pointer_cast<const Var>(token));
            }
        }
        MUST_VALID(attrs, ParseAttrList());
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));

        // Trailing token-typed var def is the result token, the rest are results.
        std::vector<VarPtr> resultList = defVarList;
        VarPtr resultToken = nullptr;
        if (resultList.size() > 1 && std::dynamic_pointer_cast<const TokenType>(resultList.back()->GetType())) {
            resultToken = resultList.back();
            resultList.pop_back();
        }

        // ScalarOpStmt has single result and no tokens/attrs
        if (tokenList.empty() && attrs->data.empty() && resultList.size() == 1) {
            return std::make_shared<ScalarOpStmt>(resultList.front(), resultToken, opcode->data, args->data,
                                                  Span::Unknown());
        } else {
            return std::make_shared<TensorOpStmt>(resultList, resultToken, opcode->data, args->data, tokenList,
                                                  attrs->data, Span::Unknown());
        }
    }

    // Assignment: type %result = expr;
    StmtPtr ParseStmtAssign(const std::vector<VarPtr>& defVarList)
    {
        MUST_VALID(value, ParseExpr());
        MUST_MATCH(ParseToken(LexerTokenKind::PunSemi));
        return std::make_shared<AssignStmt>(defVarList.front(), value, Span::Unknown());
    }

    using ParseStmtFn = StmtPtr (IRTextParser::*)(const std::vector<VarPtr>&);

    StmtPtr ParseStmtVarDef()
    {
        MUST_VALID(varDefList, ParseVarDefList(LexerTokenKind::PunComma, LexerTokenKind::PunEq));
        MUST_MATCH(ParseToken(LexerTokenKind::PunEq));

        static const std::unordered_map<LexerTokenKind, ParseStmtFn> dispatch = {
            {LexerTokenKind::KwIf, &IRTextParser::ParseStmtIf},
            {LexerTokenKind::KwFor, &IRTextParser::ParseStmtFor},
            {LexerTokenKind::KwWhile, &IRTextParser::ParseStmtWhile},
            {LexerTokenKind::Ident, &IRTextParser::ParseStmtAssignOrOp},
            {LexerTokenKind::PunLParen, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::Int, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::Float, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::VarName, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwTrue, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwFalse, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwTuple, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwGetItem, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwMemref, &IRTextParser::ParseStmtAssign},
            {LexerTokenKind::KwScalarExpr, &IRTextParser::ParseStmtAssign},
        };
        const LexerToken& tok = Peek();
        auto it = dispatch.find(tok.kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))(varDefList->data);
        } else {
            return Error("Unknown var def: " + tok.text);
        }
    }

    // ---- Statement dispatch ----
    StmtPtr ParseStmt()
    {
        static const std::unordered_map<LexerTokenKind, ParseStmtFn> dispatch = {
            {LexerTokenKind::PunLBrace, &IRTextParser::ParseStmtSeq},
            {LexerTokenKind::KwYield, &IRTextParser::ParseStmtYield},
            {LexerTokenKind::KwReturn, &IRTextParser::ParseStmtReturn},
            {LexerTokenKind::KwSection, &IRTextParser::ParseStmtSection},
            {LexerTokenKind::KwBreak, &IRTextParser::ParseStmtBreak},
            {LexerTokenKind::KwContinue, &IRTextParser::ParseStmtContinue},
            {LexerTokenKind::KwIf, &IRTextParser::ParseStmtIf},
            {LexerTokenKind::KwFor, &IRTextParser::ParseStmtFor},
            {LexerTokenKind::KwWhile, &IRTextParser::ParseStmtWhile},

            {LexerTokenKind::PunLParen, &IRTextParser::ParseStmtEval},
            {LexerTokenKind::KwTrue, &IRTextParser::ParseStmtEval},
            {LexerTokenKind::KwFalse, &IRTextParser::ParseStmtEval},
            {LexerTokenKind::Int, &IRTextParser::ParseStmtEval},
            {LexerTokenKind::Float, &IRTextParser::ParseStmtEval},
            {LexerTokenKind::Ident, &IRTextParser::ParseStmtEval},
        };
        const LexerToken& tok = Peek();
        auto it = dispatch.find(tok.kind);
        if (it != dispatch.end()) {
            return (this->*(it->second))({});
        }

        static const std::unordered_set<LexerTokenKind> typeKeywords = {
            LexerTokenKind::KwTensor,        LexerTokenKind::KwTile,          LexerTokenKind::KwTuple,
            LexerTokenKind::KwPtr,           LexerTokenKind::KwToken,         LexerTokenKind::KwNone,
            LexerTokenKind::KwMemrefType,    LexerTokenKind::KwLogicalTensor, LexerTokenKind::KwTypeBool,
            LexerTokenKind::KwTypeInt8,      LexerTokenKind::KwTypeInt16,     LexerTokenKind::KwTypeInt32,
            LexerTokenKind::KwTypeInt64,     LexerTokenKind::KwTypeUint8,     LexerTokenKind::KwTypeUint16,
            LexerTokenKind::KwTypeUint32,    LexerTokenKind::KwTypeUint64,    LexerTokenKind::KwTypeFp16,
            LexerTokenKind::KwTypeFp32,      LexerTokenKind::KwTypeFp64,      LexerTokenKind::KwTypeBf16,
            LexerTokenKind::KwTypeFp8e4m3fn, LexerTokenKind::KwTypeFp8e5m2,   LexerTokenKind::KwTypeHf4,
            LexerTokenKind::KwTypeHf8,       LexerTokenKind::KwTypeUnknown,
        };
        if (typeKeywords.count(tok.kind)) {
            return ParseStmtVarDef();
        } else {
            return Error("unknown statement");
        }
    }

    // ---- Function parsing ----
    FunctionPtr ParseFunction()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwFunction));
        MUST_VALID(name, ParseIdent());

        MUST_MATCH(ParseToken(LexerTokenKind::KwIncast));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        MUST_VALID(params, ParseVarDefList(LexerTokenKind::PunComma, LexerTokenKind::PunRParen));
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));

        MUST_MATCH(ParseToken(LexerTokenKind::KwOutcast));
        MUST_MATCH(ParseToken(LexerTokenKind::PunLParen));
        std::vector<TypePtr> retTypes;
        if (!CheckToken(LexerTokenKind::PunRParen)) {
            do {
                MUST_VALID(t, ParseType());
                retTypes.push_back(t);
            } while (CheckToken(LexerTokenKind::PunComma, true));
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRParen));
        MUST_VALID(attrs, ParseAttrList());
        MUST_VALID(body, ParseStmtSeq());

        FunctionType funcType = FunctionType::OPAQUE;
        bool isEntry = false;
        for (const auto& [key, value] : attrs->data) {
            if (key == IR_KW_TYPE) {
                const auto& typeStr = AnyCast<std::string>(value);
                try {
                    funcType = StringToFunctionType(typeStr);
                } catch (const std::invalid_argument&) {
                    return Error("unknown function type: " + typeStr);
                }
            } else if (key == IR_KW_ENTRY) {
                isEntry = AnyCast<bool>(value);
            }
        }
        return std::make_shared<Function>(name->data, params->data, retTypes, body, Span::Unknown(), funcType, isEntry);
    }

    // ---- Program parsing ----
    ProgramPtr ParseProgram()
    {
        MUST_MATCH(ParseToken(LexerTokenKind::KwProgram));
        MUST_MATCH(ParseAttrList());
        MUST_MATCH(ParseToken(LexerTokenKind::PunLBrace));

        std::vector<FunctionPtr> funcs;
        while (!CheckToken(LexerTokenKind::PunRBrace)) {
            MUST_VALID(func, ParseFunction());
            funcs.push_back(func);
        }
        MUST_MATCH(ParseToken(LexerTokenKind::PunRBrace));

        auto prog = std::make_shared<Program>(funcs, IR_KW_PROGRAM, Span::Unknown());
        return prog;
    }

private:
    // ---- Member variables ----
    std::vector<LexerToken> tokens_;
    size_t cur_;
    std::unordered_map<std::string, VarPtr> symtab_;
    IRTextParserSuccessPtr succ_;
    std::string errorMsg_;
    size_t errorPos_ = 0;
};

// ===========================================================================
// Public API
// ===========================================================================
ProgramPtr IRLoaderText::LoadProgram(const std::string& text)
{
    IRTextParser parser(text);
    return parser.ParseProgram();
}

FunctionPtr IRLoaderText::LoadFunction(const std::string& text)
{
    IRTextParser parser(text);
    return parser.ParseFunction();
}

StmtPtr IRLoaderText::LoadStmt(const std::string& text)
{
    IRTextParser parser(text);
    if (parser.Peek().kind == LexerTokenKind::PunLBrace) {
        return parser.ParseStmtSeq();
    }
    return parser.ParseStmt();
}

ExprPtr IRLoaderText::LoadExpr(const std::string& text)
{
    IRTextParser parser(text);
    return parser.ParseExpr();
}

TypePtr IRLoaderText::LoadType(const std::string& text)
{
    IRTextParser parser(text);
    return parser.ParseType();
}

ProgramPtr TextLoadProgram(const std::string& text) { return IRLoaderText::LoadProgram(text); }
FunctionPtr TextLoadFunction(const std::string& text) { return IRLoaderText::LoadFunction(text); }
StmtPtr TextLoadStmt(const std::string& text) { return IRLoaderText::LoadStmt(text); }
ExprPtr TextLoadExpr(const std::string& text) { return IRLoaderText::LoadExpr(text); }
TypePtr TextLoadType(const std::string& text) { return IRLoaderText::LoadType(text); }

} // namespace ir
} // namespace pypto
