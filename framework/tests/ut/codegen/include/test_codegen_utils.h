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
 * \file test_codegen_utils.h
 * \brief Unit test for codegen.
 */

#ifndef TEST_CODEGEN_UTILS_H
#define TEST_CODEGEN_UTILS_H

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
const constexpr int DummyFuncMagic = 1;
struct LogicalTensorInfo {
    LogicalTensorInfo(Function &func, DataType dataType, MemoryType memoryType, const std::vector<int64_t> &tShape)
        : function(func), dType(dataType), memType(memoryType), shape(tShape) {};
    LogicalTensorInfo(
        Function &func, DataType dataType, MemoryType memoryType, const std::vector<int64_t> &tShape, std::string tName)
        : function(func), dType(dataType), memType(memoryType), shape(tShape), tensorName(std::move(tName)) {};
    LogicalTensorInfo(Function &func, DataType dataType, MemoryType memoryType, const std::vector<int64_t> &tShape,
        const std::vector<SymbolicScalar> &dynShape)
        : function(func), dType(dataType), memType(memoryType), shape(tShape), dynValidShape(dynShape) {};
    LogicalTensorInfo(Function &func, DataType dataType, MemoryType memoryType, const std::vector<int64_t> &tShape,
        int magicVal, const std::vector<SymbolicScalar> &dynShape)
        : function(func),
          dType(dataType),
          memType(memoryType),
          shape(tShape),
          magic(magicVal),
          dynValidShape(dynShape) {};
    LogicalTensorInfo(Function &func, DataType dataType, MemoryType memoryType, const std::vector<int64_t> &tShape,
        std::string tName, int magicVal, const std::vector<SymbolicScalar> &dynShape)
        : function(func),
          dType(dataType),
          memType(memoryType),
          shape(tShape),
          tensorName(std::move(tName)),
          magic(magicVal),
          dynValidShape(dynShape) {};

    Function &function;
    DataType dType;
    MemoryType memType;
    const std::vector<int64_t> &shape;
    const std::string tensorName;
    int magic = -1;
    std::vector<SymbolicScalar> dynValidShape;
};

std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo &info);

std::string GetResultFromCpp(const Function &function);

void CheckStringExist(const std::string &expect, const std::string &result);

} // namespace npu::tile_fwk

#endif // TEST_CODEGEN_UTILS_H