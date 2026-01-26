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
 * \file codegen_symbol.h
 * \brief
 */

#pragma once

#include <map>
#include <tuple>
#include <cstdint>
#include <string>

#include "interface/utils/log.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/error.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
const std::string TILE_TENSOR = "TileTensor";
const std::string LAYOUT = "Layout";
const std::string SCOPE_NAMESPACE = "Hardware";
const std::string DIM = "Dim";
const std::string COORD = "Coord";
using BufferType = enum OperandType;
using AllocKey = std::tuple<BufferType, int64_t /*RangeStart*/, int64_t /*RangeEnd*/>;

struct ShapeInLoop {
    size_t loopDepth{0};
    std::vector<int64_t> originShape;
    std::vector<int64_t> rawShape;
    std::vector<SymbolicScalar> dynamicValidShape;
};

inline std::string GetLayoutType(BufferType bufType, int dim, bool isStatic) {
    std::string prefix = bufType == BUF_DDR ? "Dyn" : isStatic ? "Static" : "Local";
    std::ostringstream ss;
    ss << prefix << LAYOUT << dim << DIM;
    return ss.str();
}

// e.g.
// UBTileTensorFP32Dim2 ubTile_0((__ubuf__ float*)UB_S0_E16384, DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1),
// Stride<int, int>(64, 1)));
struct TileTensor {
    bool isStatic;
    int magic; // tensor magic numbuer
    int dim;
    DataType dtype;
    BufferType bufType;
    std::string bufVar;
    std::string usingType;
    std::string tensorName;         // e.g. "ubTile_0"
    std::vector<std::string> shape; // valid shape
    std::vector<std::string> stride;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> localBufOffset;
    ShapeInLoop shapeInLoop;

    bool operator==(const TileTensor &other) const {
        return dim == other.dim && bufVar == other.bufVar && shape == other.shape && dtype == other.dtype &&
               localBufOffset == other.localBufOffset && rawShape == other.rawShape;
    }

    /*  e.g.
        ((__ubuf__ float*)UB_S0_E16384,
        Layout2Dim(Shape2Dim<int, int>(sym_18_dim_0, sym_18_dim_1), Stride2Dim<int, int>(64, 1)));
    */
    std::string GenInitParam() const {
        std::ostringstream oss;
        std::vector<std::string> params;
        // ddr: e.g. (__gm__ float*)GET_PARAM_ADDR(...)
        // local: e.g. (uint64_t)UB_S0_E16384
        oss << "(";
        if (bufType == BUF_DDR) {
            oss << OPERAND_TYPE_TO_ADDR_TYPE.at(bufType) << " " << DataType2CCEStr(dtype) << "*)" << bufVar;
        } else {
            // cast local buffer pointer to uint64_t to adapt TileTensor mode
            oss << "uint64_t)";
            int64_t linearOffset = CalcLinearOffset(rawShape, localBufOffset);
            if (linearOffset != 0) {
                // append linear offset, e.g. UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)((float *)UB_S0_E4096 + 32))
                oss << "((" << DataType2CCEStr(dtype) << " *)" << bufVar << " + " << linearOffset << ")";
            } else {
                oss << bufVar;
            }
        }

        if (isStatic && bufType != BUF_DDR) {
            return "(" + oss.str() + ")";
        }
        params.emplace_back(oss.str());
        oss.str("");
        // ddr: e.g. DynLayout2Dim(Shape2Dim<int, int>(sym_18_dim_0, sym_18_dim_1), Stride2Dim<int, int>(64, 1)));
        // local: e.g. Shape2Dim(sym_18_dim_0, sym_18_dim_1));
        if (bufType == BUF_DDR) {
            oss << GetLayoutType(bufType, dim, isStatic);
        }
        oss << "(" << GenShapeParam();
        if (bufType == BUF_DDR) {
            oss << ", " << GenStrideParam();
        }
        oss << ")";
        params.emplace_back(oss.str());
        return WrapParamByParentheses(params);
    }

    std::string ToString() const {
        std::ostringstream oss;
        oss << usingType << " " << tensorName << GenInitParam() << STMT_END;
        return oss.str();
    }

private:
    std::string GenLayoutParam(const std::string &paramName, const std::vector<std::string> &paramValue) const {
        std::ostringstream oss;
        oss << paramName << dim << DIM;
        oss << WrapParamByParentheses(paramValue);
        return oss.str();
    }
    std::string GenShapeParam() const { return GenLayoutParam("Shape", shape); }
    std::string GenStrideParam() const { return GenLayoutParam("Stride", stride); }
};

struct TileTensorHash {
    std::size_t operator()(const TileTensor &t) const noexcept {
        std::size_t seed = 0;
        HashCombine(seed, t.dim);
        HashCombine(seed, t.bufVar);
        HashCombine(seed, ToUnderlying(t.dtype));
        for (const auto &s : t.shape) {
            HashCombine(seed, s);
        }
        for (const auto &s : t.rawShape) {
            HashCombine(seed, s);
        }
        for (const auto &s : t.localBufOffset) {
            HashCombine(seed, s);
        }
        return seed;
    };
};

struct TileTensorUsing {
    DataType dtype;
    BufferType bufType;
    int dim;
    std::vector<int64_t> originShape; // only used for static shape
    std::vector<int64_t> rawShape;
    bool isStatic;

    bool operator==(const TileTensorUsing &other) const {
        bool baseCompare = dtype == other.dtype && bufType == other.bufType && rawShape == other.rawShape;
        return isStatic ? baseCompare && originShape == other.originShape : baseCompare;
    }

    std::string GenName() const {
        std::ostringstream oss;
        // e.g. "UBTileTensorFP32Dim2_0"
        oss << BUFFER_TYPE_TO_PREFIX.at(bufType) << TILE_TENSOR << BriefDataType2String(dtype) << DIM << dim << "_"
            << IdGen<IdType::CG_USING_NAME>::Inst().NewId();
        return oss.str();
    }

    // dynamic shape: e.g. "TileTensor<__gm__ float, DynLayout4Dim, Hardware::GM>"
    // static shape: e.g. "TileTensor<float, LocalLayout4Dim<16, 16>, Hardware::UB>"
    std::string ToString() const {
        std::ostringstream ss;
        ss << TILE_TENSOR << "<";
        if (bufType == BUF_DDR) {
            ss << GetAddrTypeByOperandType(bufType) << " ";
        }
        ss << DataType2CCEStr(dtype) << ", ";
        ss << GetLayoutType(bufType, dim, isStatic);
        if (bufType != BUF_DDR) {
            ss << GetLayoutParams();
        }
        ss << ", " << SCOPE_NAMESPACE << "::" << BUFFER_TYPE_TO_PREFIX.at(bufType) << ">;\n";
        return ss.str();
    }

private:
    constexpr static int SHAPE_KIND = 2; // origin shape; raw shape
    std::string GetLayoutParams() const {
        std::vector<int64_t> params;
        params.reserve(dim * SHAPE_KIND);
        if (isStatic) {
            params.insert(params.end(), originShape.begin(), originShape.end());
        }
        params.insert(params.end(), rawShape.begin(), rawShape.end());
        return WrapParamByAngleBrackets(params);
    }
};

class SymbolManager {
public:
    SymbolManager() = default;
    virtual ~SymbolManager() = default;

    using AllocRecord = std::pair<uint64_t /*AllocaAddr*/, unsigned /*AllocaSize*/>;

    virtual std::string QueryVariableName(const AllocKey &key);
    std::string QueryVariableNameTileTensor(const AllocKey &key);
    std::string QueryVarNameByTensorMagic(int magic, bool isTileTensor = false);
    SymbolManager(SymbolManager &other) = delete;

    void operator=(const SymbolManager &other) = delete;

    bool BindAddrWithVariableName(
        const AllocKey &key, const std::string &varName, const std::string &varNameTileTensor);

    void AddToTensorMap(int magicNum, const std::shared_ptr<LogicalTensor> &tensor) {
        auto res = tensorMap_.insert({magicNum, tensor});
        if (!res.second) {
            ASSERT(tensor == tensorMap_[magicNum])
                << "!!! ERROR !!! tensor magic : " << magicNum
                << " is conflicted!!!\ninsert tensor key: " << FormatAllocKey(CreateAllocKey(tensor))
                << "\ntensor dump info -- " << tensor->Dump()
                << "\nexisted tensor key: " << FormatAllocKey(CreateAllocKey(tensorMap_[magicNum]))
                << "\ntensor dump info -- " << tensorMap_[magicNum]->Dump();
        }
    }

    static std::string FormatAllocKey(const AllocKey &key);

    std::string AddTileTensorUsing(const TileTensorUsing &tileTensorUsing);
    std::string AddTileTensor(const TileTensor &tileTensor);
    std::vector<TileTensor> QueryTileTensorByMagic(int magic);
    std::vector<TileTensor> QueryTileTensorInLoopByMagic(int magic);
    void InsertTensorNameInLoopToFullDim(const std::string &tensorName, const std::string &fullDimTensorName);
    std::string QueryTileTensorFullDimByTensorInLoop(const std::string &tensorName);
    // To be compatible with GM Tensor in Static Function Type like same ddr magic number with different parmaIdx &
    // 'GMStackBase' e.g. ((__gm__ GMTensorInfo*)param + 1), ((__gm__ GMTensorInfo*)param + 2)
    std::string QueryTileTensorByBufVarName(const std::string &bufVarName);

    std::string GenUsingList();
    std::string GenTileTensorDefList();

    void OutForLoop() {
        tileTensorByMagicInLoop_.clear();
        tensorNameInLoopToFullDim_.clear();
    }

private:
    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magicNum) const;
    AllocKey CreateAllocKey(const std::shared_ptr<LogicalTensor> &tensor) const;
    AllocKey CreateAllocKey(int tensorMagicNum) const;
    std::string FindUsingName(const TileTensorUsing &tileTensorUsing) const;

    // <AllocKey, buffer variable name>
    std::map<AllocKey, std::string> key2VariableName_;
    // <AllocKey, buffer variable name of TileTensor mode>
    std::map<AllocKey, std::string> key2VariableNameTileTensor_;
    // <tensor magic, LogicalTensor>
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorMap_;
    // <TileTensor, tensorName>
    std::unordered_map<TileTensor, std::string, TileTensorHash> tileTensor_;
    // When use forcing axis merging feature under TileTensor mode,
    // we may encounter a situation where Tensors with the same magic ID have different Shapes.
    // <tensor magic, TileTensor>
    std::multimap<int, TileTensor> tileTensorByMagic_;
    std::multimap<int, TileTensor> tileTensorByMagicInLoop_;
    // <tensorName in for loop, tensorName with full dim out of loop>
    // both key and value are from same tile operation
    std::unordered_map<std::string, std::string> tensorNameInLoopToFullDim_;
    // <using type, TileTensorUsing>
    std::unordered_map<std::string, TileTensorUsing> tileTensorUsing_;
};
} // namespace npu::tile_fwk
