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
 * \file object.h
 * \brief
 */

#pragma once

#include "ir/utils.h"

namespace pto {
    
// Object type categories for ID generation.
enum class ObjectType {
    Program,
    Function,
    Statement,
    Operation,
    Value,
    Memory,
};

// Simple key/value attribute bag used across IR nodes.
using AttributeMap = std::map<std::string, std::string>;

// Base class for all IR objects.
class Object {
public:
    explicit Object(ObjectType type, const std::string &name = "")
        : id_(IDGen::NextID(type)), name_(name) {}
    virtual ~Object() = default;

    int GetID() const { return id_; }
    const std::string& GetName() const { return name_; }
    void SetName(std::string name) { name_ = name; }
    
    // Get the name with prefix for display/printing
    // - Program and Function: add @ prefix
    // - Value: add % prefix
    // - Other types: no prefix
    std::string GetPrefixedName() const {
        if (name_.empty()) {
            return name_;
        }
        
        ObjectType type = GetObjectType();
        char prefix = '\0';
        if (type == ObjectType::Program || type == ObjectType::Function) {
            prefix = '@';
        } else if (type == ObjectType::Value) {
            prefix = '%';
        }
        
        if (prefix != '\0') {
            return std::string(1, prefix) + name_;
        }
        
        return name_;
    }

    // Each derived class must specify its object type.
    virtual ObjectType GetObjectType() const = 0;

    AttributeMap& Attributes() { return attributes_; }
    const AttributeMap& Attributes() const { return attributes_; }

protected:
    int id_;
    std::string name_;
    AttributeMap attributes_;
};

}