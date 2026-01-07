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
 * \file test_type.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "ir/type.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/value.h"
#include "ir/utils.h"

namespace pto {

TEST(IRTEST, TestScalarType) {
    // 测试 ScalarType 的基本功能
    // 测试不同的数据类型
    auto int32Type = std::make_shared<ScalarType>(DataType::INT32);
    auto fp32Type = std::make_shared<ScalarType>(DataType::FP32);
    auto fp64Type = std::make_shared<ScalarType>(DataType::FP64);
    auto int64Type = std::make_shared<ScalarType>(DataType::INT64);
    auto boolType = std::make_shared<ScalarType>(DataType::BOOL);
    auto int8Type = std::make_shared<ScalarType>(DataType::INT8);
    auto int16Type = std::make_shared<ScalarType>(DataType::INT16);
    auto fp16Type = std::make_shared<ScalarType>(DataType::FP16);

    // 测试 GetDataType()
    ASSERT_EQ(int32Type->GetDataType(), DataType::INT32);
    ASSERT_EQ(fp32Type->GetDataType(), DataType::FP32);
    ASSERT_EQ(fp64Type->GetDataType(), DataType::FP64);
    ASSERT_EQ(int64Type->GetDataType(), DataType::INT64);
    ASSERT_EQ(boolType->GetDataType(), DataType::BOOL);

    // 测试 GetDataTypeSize() - 静态方法
    ASSERT_EQ(Type::GetDataTypeSize(DataType::INT32), 4);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::FP32), 4);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::FP64), 8);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::INT64), 8);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::BOOL), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::INT8), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::INT16), 2);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::FP16), 2);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::UINT8), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::UINT32), 4);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::UINT64), 8);

    // 测试 GetDataTypeSize() - 实例方法
    ASSERT_EQ(int32Type->GetDataTypeSize(), 4);
    ASSERT_EQ(fp32Type->GetDataTypeSize(), 4);
    ASSERT_EQ(fp64Type->GetDataTypeSize(), 8);
    ASSERT_EQ(int64Type->GetDataTypeSize(), 8);
    ASSERT_EQ(boolType->GetDataTypeSize(), 1);
    ASSERT_EQ(int8Type->GetDataTypeSize(), 1);
    ASSERT_EQ(int16Type->GetDataTypeSize(), 2);
    ASSERT_EQ(fp16Type->GetDataTypeSize(), 2);

    // 测试 GetTypeSize() - ScalarType 应该等于 GetDataTypeSize()
    ASSERT_EQ(int32Type->GetTypeSize(), 4);
    ASSERT_EQ(fp32Type->GetTypeSize(), 4);
    ASSERT_EQ(fp64Type->GetTypeSize(), 8);
    ASSERT_EQ(int64Type->GetTypeSize(), 8);
    ASSERT_EQ(boolType->GetTypeSize(), 1);

    // 测试 Print()
    std::ostringstream oss1;
    int32Type->Print(oss1);
    ASSERT_FALSE(oss1.str().empty());

    std::ostringstream oss2;
    fp32Type->Print(oss2);
    ASSERT_FALSE(oss2.str().empty());

    std::ostringstream oss3;
    fp64Type->Print(oss3);
    ASSERT_FALSE(oss3.str().empty());
}

TEST(IRTEST, TestTileType) {
    // 测试 TileType 的基本功能
    // 测试不同形状和数据类型的 TileType
    std::vector<size_t> shape1D = {128};
    std::vector<size_t> shape2D = {16, 32};
    std::vector<size_t> shape3D = {4, 8, 16};
    std::vector<size_t> shape4D = {2, 4, 8, 16};

    auto tile1D = std::make_shared<TileType>(DataType::FP32, shape1D);
    auto tile2D = std::make_shared<TileType>(DataType::FP32, shape2D);
    auto tile3D = std::make_shared<TileType>(DataType::INT32, shape3D);
    auto tile4D = std::make_shared<TileType>(DataType::FP64, shape4D);

    // 测试 GetDataType()
    ASSERT_EQ(tile1D->GetDataType(), DataType::FP32);
    ASSERT_EQ(tile2D->GetDataType(), DataType::FP32);
    ASSERT_EQ(tile3D->GetDataType(), DataType::INT32);
    ASSERT_EQ(tile4D->GetDataType(), DataType::FP64);

    // 测试 GetShape()
    const auto& shape1D_ref = tile1D->GetShape();
    ASSERT_EQ(shape1D_ref.size(), 1);
    ASSERT_EQ(shape1D_ref[0], 128);

    const auto& shape2D_ref = tile2D->GetShape();
    ASSERT_EQ(shape2D_ref.size(), 2);
    ASSERT_EQ(shape2D_ref[0], 16);
    ASSERT_EQ(shape2D_ref[1], 32);

    const auto& shape3D_ref = tile3D->GetShape();
    ASSERT_EQ(shape3D_ref.size(), 3);
    ASSERT_EQ(shape3D_ref[0], 4);
    ASSERT_EQ(shape3D_ref[1], 8);
    ASSERT_EQ(shape3D_ref[2], 16);

    const auto& shape4D_ref = tile4D->GetShape();
    ASSERT_EQ(shape4D_ref.size(), 4);
    ASSERT_EQ(shape4D_ref[0], 2);
    ASSERT_EQ(shape4D_ref[1], 4);
    ASSERT_EQ(shape4D_ref[2], 8);
    ASSERT_EQ(shape4D_ref[3], 16);

    // 测试 GetDataTypeSize()
    ASSERT_EQ(tile1D->GetDataTypeSize(), 4);  // FP32 = 4 bytes
    ASSERT_EQ(tile3D->GetDataTypeSize(), 4);  // INT32 = 4 bytes
    ASSERT_EQ(tile4D->GetDataTypeSize(), 8);  // FP64 = 8 bytes

    // 测试 GetTypeSize() - 应该等于元素大小 * 元素总数
    // tile1D: FP32 (4 bytes) * 128 = 512 bytes
    ASSERT_EQ(tile1D->GetTypeSize(), 4 * 128);

    // tile2D: FP32 (4 bytes) * 16 * 32 = 2048 bytes
    ASSERT_EQ(tile2D->GetTypeSize(), 4 * 16 * 32);

    // tile3D: INT32 (4 bytes) * 4 * 8 * 16 = 2048 bytes
    ASSERT_EQ(tile3D->GetTypeSize(), 4 * 4 * 8 * 16);

    // tile4D: FP64 (8 bytes) * 2 * 4 * 8 * 16 = 8192 bytes
    ASSERT_EQ(tile4D->GetTypeSize(), 8 * 2 * 4 * 8 * 16);

    // 测试 Print()
    std::ostringstream oss1;
    tile1D->Print(oss1);
    ASSERT_FALSE(oss1.str().empty());
    // 应该包含形状信息
    ASSERT_NE(oss1.str().find("128"), std::string::npos);

    std::ostringstream oss2;
    tile2D->Print(oss2);
    ASSERT_FALSE(oss2.str().empty());
    ASSERT_NE(oss2.str().find("16"), std::string::npos);
    ASSERT_NE(oss2.str().find("32"), std::string::npos);

    // 测试空形状的情况
    std::vector<size_t> emptyShape = {};
    auto emptyTile = std::make_shared<TileType>(DataType::FP32, emptyShape);
    ASSERT_EQ(emptyTile->GetShape().size(), 0);
    ASSERT_EQ(emptyTile->GetTypeSize(), 4);  // 只有一个元素
}

TEST(IRTEST, TestTensorType) {
    // 测试 TensorType 的基本功能
    auto tensorFP32 = std::make_shared<TensorType>(DataType::FP32);
    auto tensorFP64 = std::make_shared<TensorType>(DataType::FP64);
    auto tensorINT32 = std::make_shared<TensorType>(DataType::INT32);
    auto tensorINT64 = std::make_shared<TensorType>(DataType::INT64);
    auto tensorBOOL = std::make_shared<TensorType>(DataType::BOOL);
    auto tensorFP16 = std::make_shared<TensorType>(DataType::FP16);

    // 测试 GetDataType()
    ASSERT_EQ(tensorFP32->GetDataType(), DataType::FP32);
    ASSERT_EQ(tensorFP64->GetDataType(), DataType::FP64);
    ASSERT_EQ(tensorINT32->GetDataType(), DataType::INT32);
    ASSERT_EQ(tensorINT64->GetDataType(), DataType::INT64);
    ASSERT_EQ(tensorBOOL->GetDataType(), DataType::BOOL);
    ASSERT_EQ(tensorFP16->GetDataType(), DataType::FP16);

    // 测试 GetDataTypeSize()
    ASSERT_EQ(tensorFP32->GetDataTypeSize(), 4);
    ASSERT_EQ(tensorFP64->GetDataTypeSize(), 8);
    ASSERT_EQ(tensorINT32->GetDataTypeSize(), 4);
    ASSERT_EQ(tensorINT64->GetDataTypeSize(), 8);
    ASSERT_EQ(tensorBOOL->GetDataTypeSize(), 1);
    ASSERT_EQ(tensorFP16->GetDataTypeSize(), 2);

    // 测试 GetTypeSize() - TensorType 应该等于 GetDataTypeSize()
    ASSERT_EQ(tensorFP32->GetTypeSize(), 4);
    ASSERT_EQ(tensorFP64->GetTypeSize(), 8);
    ASSERT_EQ(tensorINT32->GetTypeSize(), 4);
    ASSERT_EQ(tensorINT64->GetTypeSize(), 8);
    ASSERT_EQ(tensorBOOL->GetTypeSize(), 1);
    ASSERT_EQ(tensorFP16->GetTypeSize(), 2);

    // 测试 Print()
    std::ostringstream oss1;
    tensorFP32->Print(oss1);
    ASSERT_FALSE(oss1.str().empty());

    std::ostringstream oss2;
    tensorFP64->Print(oss2);
    ASSERT_FALSE(oss2.str().empty());

    std::ostringstream oss3;
    tensorINT32->Print(oss3);
    ASSERT_FALSE(oss3.str().empty());
}

TEST(IRTEST, TestTypePolymorphism) {
    // 测试类型系统的多态性
    // 使用基类指针指向不同的派生类
    TypePtr scalarType = std::make_shared<ScalarType>(DataType::FP32);
    TypePtr tileType = std::make_shared<TileType>(DataType::FP32, std::vector<size_t>{16, 32});
    TypePtr tensorType = std::make_shared<TensorType>(DataType::FP32);

    // 测试多态调用 GetDataType()
    ASSERT_EQ(scalarType->GetDataType(), DataType::FP32);
    ASSERT_EQ(tileType->GetDataType(), DataType::FP32);
    ASSERT_EQ(tensorType->GetDataType(), DataType::FP32);

    // 测试多态调用 GetDataTypeSize()
    ASSERT_EQ(scalarType->GetDataTypeSize(), 4);
    ASSERT_EQ(tileType->GetDataTypeSize(), 4);
    ASSERT_EQ(tensorType->GetDataTypeSize(), 4);

    // 测试多态调用 GetTypeSize()
    ASSERT_EQ(scalarType->GetTypeSize(), 4);
    ASSERT_EQ(tileType->GetTypeSize(), 4 * 16 * 32);  // FP32 * 16 * 32
    ASSERT_EQ(tensorType->GetTypeSize(), 4);

    // 测试多态调用 Print()
    std::ostringstream oss1, oss2, oss3;
    scalarType->Print(oss1);
    tileType->Print(oss2);
    tensorType->Print(oss3);

    ASSERT_FALSE(oss1.str().empty());
    ASSERT_FALSE(oss2.str().empty());
    ASSERT_FALSE(oss3.str().empty());
}

TEST(IRTEST, TestTypeEdgeCases) {
    // 测试边界情况
    // 测试 UNKNOWN 和 BOTTOM 类型
    auto unknownScalar = std::make_shared<ScalarType>(DataType::UNKNOWN);
    auto bottomScalar = std::make_shared<ScalarType>(DataType::BOTTOM);

    ASSERT_EQ(unknownScalar->GetDataType(), DataType::UNKNOWN);
    ASSERT_EQ(bottomScalar->GetDataType(), DataType::BOTTOM);
    ASSERT_EQ(unknownScalar->GetDataTypeSize(), 0);
    ASSERT_EQ(bottomScalar->GetDataTypeSize(), 0);
    ASSERT_EQ(unknownScalar->GetTypeSize(), 0);
    ASSERT_EQ(bottomScalar->GetTypeSize(), 0);

    // 测试特殊数据类型的大小
    ASSERT_EQ(Type::GetDataTypeSize(DataType::INT4), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::HF4), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::FP8), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::HF8), 1);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::BF16), 2);
    ASSERT_EQ(Type::GetDataTypeSize(DataType::UINT16), 2);

    // 测试大形状的 TileType
    std::vector<size_t> largeShape = {1024, 1024};
    auto largeTile = std::make_shared<TileType>(DataType::FP32, largeShape);
    ASSERT_EQ(largeTile->GetTypeSize(), 4 * 1024 * 1024);  // 4MB

    // 测试单元素 TileType
    std::vector<size_t> singleElement = {1};
    auto singleTile = std::make_shared<TileType>(DataType::FP32, singleElement);
    ASSERT_EQ(singleTile->GetTypeSize(), 4);
    ASSERT_EQ(singleTile->GetShape()[0], 1);
}

TEST(IRTEST, TestTypeAllDataTypes) {
    // 测试所有数据类型的 ScalarType
    std::vector<DataType> allTypes = {
        DataType::BOOL,
        DataType::INT4, DataType::INT8, DataType::INT16, DataType::INT32, DataType::INT64,
        DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64,
        DataType::FP8, DataType::FP16, DataType::BF16, DataType::FP32, DataType::FP64,
        DataType::HF4, DataType::HF8,
        DataType::UNKNOWN, DataType::BOTTOM
    };

    for (DataType dt : allTypes) {
        auto scalarType = std::make_shared<ScalarType>(dt);
        ASSERT_EQ(scalarType->GetDataType(), dt);

        // 测试 Print 不会崩溃
        std::ostringstream oss;
        scalarType->Print(oss);
        ASSERT_FALSE(oss.str().empty());
    }

    // 测试常用数据类型的 TensorType
    std::vector<DataType> commonTypes = {
        DataType::INT32, DataType::INT64,
        DataType::FP32, DataType::FP64,
        DataType::BOOL
    };

    for (DataType dt : commonTypes) {
        auto tensorType = std::make_shared<TensorType>(dt);
        ASSERT_EQ(tensorType->GetDataType(), dt);

        std::ostringstream oss;
        tensorType->Print(oss);
        ASSERT_FALSE(oss.str().empty());
    }
}

TEST(IRTEST, TestStringToValueType) {
    // 测试 StringToValueType 函数的基本功能
    // 测试所有支持的数据类型字符串（完整名称）
    ASSERT_EQ(StringToValueType("bool"), DataType::BOOL);
    ASSERT_EQ(StringToValueType("int4"), DataType::INT4);
    ASSERT_EQ(StringToValueType("int8"), DataType::INT8);
    ASSERT_EQ(StringToValueType("int16"), DataType::INT16);
    ASSERT_EQ(StringToValueType("int32"), DataType::INT32);
    ASSERT_EQ(StringToValueType("int64"), DataType::INT64);
    ASSERT_EQ(StringToValueType("uint8"), DataType::UINT8);
    ASSERT_EQ(StringToValueType("uint16"), DataType::UINT16);
    ASSERT_EQ(StringToValueType("uint32"), DataType::UINT32);
    ASSERT_EQ(StringToValueType("uint64"), DataType::UINT64);
    ASSERT_EQ(StringToValueType("fp8"), DataType::FP8);
    ASSERT_EQ(StringToValueType("fp16"), DataType::FP16);
    ASSERT_EQ(StringToValueType("bf16"), DataType::BF16);
    ASSERT_EQ(StringToValueType("fp32"), DataType::FP32);
    ASSERT_EQ(StringToValueType("fp64"), DataType::FP64);
    ASSERT_EQ(StringToValueType("hf4"), DataType::HF4);
    ASSERT_EQ(StringToValueType("hf8"), DataType::HF8);
    ASSERT_EQ(StringToValueType("bottom"), DataType::BOTTOM);
    ASSERT_EQ(StringToValueType("unknown"), DataType::UNKNOWN);

    // 测试别名（简短形式）
    ASSERT_EQ(StringToValueType("i8"), DataType::INT8);
    ASSERT_EQ(StringToValueType("i16"), DataType::INT16);
    ASSERT_EQ(StringToValueType("i32"), DataType::INT32);
    ASSERT_EQ(StringToValueType("i64"), DataType::INT64);
    ASSERT_EQ(StringToValueType("u8"), DataType::UINT8);
    ASSERT_EQ(StringToValueType("u16"), DataType::UINT16);
    ASSERT_EQ(StringToValueType("u32"), DataType::UINT32);
    ASSERT_EQ(StringToValueType("u64"), DataType::UINT64);
    ASSERT_EQ(StringToValueType("f16"), DataType::FP16);
    ASSERT_EQ(StringToValueType("f32"), DataType::FP32);
    ASSERT_EQ(StringToValueType("f64"), DataType::FP64);

    // 测试未知字符串（应该返回默认值 INT32）
    ASSERT_EQ(StringToValueType("invalid_type"), DataType::UNKNOWN);
    ASSERT_EQ(StringToValueType(""), DataType::UNKNOWN);
    ASSERT_EQ(StringToValueType("xyz"), DataType::UNKNOWN);
    ASSERT_EQ(StringToValueType("float"), DataType::UNKNOWN);
    ASSERT_EQ(StringToValueType("double"), DataType::UNKNOWN);

    // 测试大小写敏感性（函数是大小写敏感的，不匹配的字符串返回默认值 INT32）
    ASSERT_EQ(StringToValueType("INT32"), DataType::UNKNOWN);  // 不匹配，返回默认值 INT32
    ASSERT_EQ(StringToValueType("Int32"), DataType::UNKNOWN);  // 不匹配，返回默认值 INT32
    ASSERT_EQ(StringToValueType("FP32"), DataType::UNKNOWN);   // 不匹配，返回默认值 INT32

    // 测试与 DataTypeToString 的往返转换
    std::vector<DataType> testTypes = {
        DataType::BOOL, DataType::INT8, DataType::INT32, DataType::INT64,
        DataType::UINT8, DataType::UINT32, DataType::UINT64,
        DataType::FP16, DataType::FP32, DataType::FP64,
        DataType::BF16, DataType::FP8, DataType::HF4, DataType::HF8
    };

    for (DataType dt : testTypes) {
        std::string typeStr = DataTypeToString(dt);
        DataType converted = StringToValueType(typeStr);
        ASSERT_EQ(converted, dt) << "Failed to convert " << typeStr << " back to DataType";
    }
}

TEST(IRTEST, TestTypeCompleteProgram) {
    // ===== 创建一个完整的程序，只使用 Tile 和 Scalar 操作，不涉及 Tensor =====
    auto module = std::make_shared<ProgramModule>("test_type_program");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // ===== 函数签名 =====
    FunctionSignature sig;

    // 输入：直接使用 Tile 和 Scalar，不使用 Tensor
    // 输入 Tile: tile<[16, 32], fp32>
    auto inputTile = std::make_shared<TileValue>(std::vector<size_t>{16, 32}, DataType::FP32, "input_tile");
    // 输入 Scalar: scalar<fp32>
    auto scale = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    // 输出：Tile tile<[16, 32], fp32>
    auto result = std::make_shared<TileValue>(std::vector<size_t>{16, 32}, DataType::FP32, "output_tile");

    sig.arguments = { inputTile, scale, result };
    sig.results.push_back(std::make_shared<ScalarValue>(DataType::FP64));

    // ===== 创建函数 =====
    auto func = builder.CreateFunction("test_type_complete", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // 进入函数体作用域
    builder.EnterFunctionBody(ctx, func);

    // 创建常量 Scalar
    auto constant2 = builder.CreateConst(ctx, 2.0, "const_2");
    auto constant3 = builder.CreateConst(ctx, 3.0, "const_3");

    // Tile 乘法操作：tile_mul = mul(input_tile, scale)
    // 当两个操作数都是 Tile 时，输出也是 Tile，会使用 tile.mul
    auto tileMul = builder.CreateTile(ctx, std::vector<size_t>{16, 32}, DataType::FP32, "tile_mul");
    auto mulOp = builder.CreateBinaryOp(Opcode::OP_MUL, inputTile, scale, tileMul);
    builder.Emit(ctx, mulOp);

    // Tile 加法操作：tile_add = add(tile_mul, constant2)
    // Tile + Scalar -> Tile，会使用 tile.add
    auto tileAdd = builder.CreateTile(ctx, std::vector<size_t>{16, 32}, DataType::FP32, "tile_add");
    auto addOp = builder.CreateBinaryOp(Opcode::OP_ADD, tileMul, constant2, tileAdd);
    builder.Emit(ctx, addOp);

    // Tile 减法操作：tile_sub = sub(tile_add, constant3)
    auto tileSub = builder.CreateTile(ctx, std::vector<size_t>{16, 32}, DataType::FP32, "tile_sub");
    auto subOp = builder.CreateBinaryOp(Opcode::OP_SUB, tileAdd, constant3, tileSub);
    builder.Emit(ctx, subOp);

    // Tile 除法操作：tile_div = div(tile_sub, scale)
    auto tileDiv = builder.CreateTile(ctx, std::vector<size_t>{16, 32}, DataType::FP32, "output_tile");
    auto divOp = builder.CreateBinaryOp(Opcode::OP_DIV, tileSub, scale, tileDiv);
    builder.Emit(ctx, divOp);

    // Scalar 操作：计算两个 scalar 的和
    auto scalar1 = builder.CreateConst(ctx, 10.5, "scalar1");
    auto scalar2 = builder.CreateConst(ctx, 5.2, "scalar2");

    // Scalar 加法：scalar_add = add(scalar1, scalar2)
    auto scalarAdd = builder.CreateScalar(ctx, DataType::FP64, "scalar_add");
    auto scalarAddOp = builder.CreateBinaryScalarOp(Opcode::OP_SCALAR_ADD, scalar1, scalar2, scalarAdd);
    builder.Emit(ctx, scalarAddOp);

    // Scalar 乘法：scalar_mul = mul(scalar_add, constant2)
    auto scalarMul = builder.CreateScalar(ctx, DataType::FP64, "scalar_mul");
    auto scalarMulOp = builder.CreateBinaryScalarOp(Opcode::OP_SCALAR_MUL, scalarAdd, constant2, scalarMul);
    builder.Emit(ctx, scalarMulOp);

    // 创建返回语句，返回 tile 和 scalar
    builder.CreateReturn(ctx, { scalarMul });

    // 验证构建器状态
    ASSERT_EQ(ctx.func, func);
    ASSERT_EQ(ctx.compound, func->GetCompound());
    ASSERT_NE(ctx.activeOpStmt, nullptr);

    // 验证值类型
    ASSERT_EQ(tileMul->GetValueKind(), ValueKind::Tile);
    ASSERT_EQ(tileAdd->GetValueKind(), ValueKind::Tile);
    ASSERT_EQ(scalarAdd->GetValueKind(), ValueKind::Scalar);
    ASSERT_EQ(scalarMul->GetValueKind(), ValueKind::Scalar);

    ctx.PopScope();
    
    // 验证离开作用域后的状态
    ASSERT_EQ(ctx.func, nullptr);
    ASSERT_EQ(ctx.compound, nullptr);
    ASSERT_EQ(ctx.activeOpStmt, nullptr);

    // 设置模块属性
    module->Attributes()["arch"] = "\"PTOv2\"";
    module->Attributes()["tile_default"] = "{ M=16, N=16, K=16 }";
    module->Attributes()["enable_debug"] = "true";
    module->Attributes()["test_type"] = "\"tile_scalar_only\"";

    // 打印完整的 IR
    std::cout << "========== Complete Type Test Program IR (Tile & Scalar Only) ==========" << std::endl;
    std::cout << *module << std::endl;
    std::cout << "=======================================================================" << std::endl;
}

} // namespace pto

