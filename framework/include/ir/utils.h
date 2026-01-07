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
 * \file utils.h
 * \brief
 */

#pragma once

#include <map>
#include <ostream>
#include <string>

namespace pto {

// Forward declaration
enum class DataType;
enum class ObjectType;

// ID generator for different object types.
// Each object type maintains its own independent ID counter.
class IDGen {
public:
    // Get the next ID for the given object type.
    static int NextID(ObjectType type);

    // Reset the ID counter for a specific type.
    static void Reset(ObjectType type);

    // Reset all ID counters.
    static void ResetAll();

private:
    static std::map<ObjectType, int> counters_;
};

// Utility to help printing indentation.
void PrintIndent(std::ostream& os, int indent);

// Helper function to convert DataType enum to string name.
std::string DataTypeToString(DataType type);

// Helper function to convert string name to DataType enum.
DataType StringToValueType(const std::string& name);

} // namespace pto

