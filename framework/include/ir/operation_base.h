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
 * \file operation_base.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>

#include "value.h"
#include "opcode.h"

namespace pto {

using AttributeValue = std::variant<std::string, int64_t, bool>;
class AttributeKeyValue {
public:
    explicit AttributeKeyValue(const std::string &name, const std::string &value) : name_(name), value_(value) {}
    explicit AttributeKeyValue(const std::string &name, const int &value) : name_(name), value_(int64_t(value)) {}
    explicit AttributeKeyValue(const std::string &name, const int64_t &value) : name_(name), value_(value) {}
    explicit AttributeKeyValue(const std::string &name, const bool &value) : name_(name), value_(value) {}
    AttributeKeyValue(const AttributeKeyValue &) = default;

    const std::string &GetName() const { return name_; }
    const AttributeValue &GetValue() const { return value_; }
private:
    std::string name_;
    AttributeValue value_;
};

// Base class for operations inside statements.
class Operation : public Object {
public:
    /// Default: invalid / placeholder op
    Operation();
    explicit Operation(Opcode opcode);
    Operation(Opcode opcode, std::string name);

    /// Full construction
    Operation(Opcode opcode,
              std::vector<ValuePtr> ioprands,
              std::vector<ValuePtr> ooprands,
              std::string name = "");

    ~Operation() override = default;

    ObjectType GetObjectType() const override { return ObjectType::Operation; }
    // ---- OpCode ----
    Opcode GetOpcode() const { return opcode_; }

    // ---- IOprands ----
    void AppendInput(const ValuePtr& value) {
        ioperands_.push_back(value);
    }

    size_t GetNumInputOperand() const { return ioperands_.size(); }

    ValuePtr GetInputOperand(size_t idx) const {
        return ioperands_.at(idx);
    }

    void SetInputOperand(size_t idx, const ValuePtr& value) {
        ioperands_.at(idx) = value;
    }

    size_t GetNumInputScalarOperand() const { return iScalarIndex_; }
    ScalarValuePtr GetInputScalarOperand(size_t idx) const { return std::static_pointer_cast<ScalarValue>(GetInputOperand(iScalarIndex_ + idx)); }
    void SetInputScalarOperand(size_t idx, ScalarValuePtr ptr) { SetInputOperand(iScalarIndex_ + idx, ptr); }

    // ---- OOprands ----
    void AppendOutput(const ValuePtr& value) {
        ooperands_.push_back(value);
    }

    size_t GetNumOutputOperand() const { return ooperands_.size(); }

    ValuePtr GetOutputOperand(size_t idx) const {
        return ooperands_.at(idx);
    }

    void SetOutputOperand(size_t idx, const ValuePtr& value) {
        ooperands_.at(idx) = value;
    }

    std::vector<ValuePtr> &GetIOperands() { return ioperands_; }
    std::vector<ValuePtr> &GetOOperands() { return ooperands_; }

    size_t GetNumOutputScalarOperand() const { return oScalarIndex_; }
    ScalarValuePtr GetOutputScalarOperand(size_t idx) const { return std::static_pointer_cast<ScalarValue>(GetOutputOperand(oScalarIndex_ + idx)); }
    void SetOutputScalarOperand(size_t idx, ScalarValuePtr ptr) { SetOutputOperand(oScalarIndex_ + idx, ptr); }

    // Pretty-print with the given indentation (in spaces).
    void Print(std::ostream& os, int indent = 0) const;

    virtual std::vector<AttributeKeyValue> GetAttributeList() const { return {}; }

protected:
    std::vector<ValuePtr> ioperands_;
    std::vector<ValuePtr> ooperands_;
    Opcode opcode_;
    ssize_t iScalarIndex_{-1};
    ssize_t oScalarIndex_{-1};
};

using OperationPtr = std::shared_ptr<Operation>;

class ScalarBaseOp : public Operation {
public:
    ScalarBaseOp(
            Opcode opcode,
            const std::vector<ScalarValuePtr> &inOperandList,
            const std::vector<ScalarValuePtr> &outOperandList)
        : Operation(opcode, ValueUtils::Join(inOperandList), ValueUtils::Join(outOperandList)) {}

    ScalarValuePtr GetInOperand(size_t index) const;
    ScalarValuePtr GetOutOperand(size_t index) const;
};
using ScalarBaseOpPtr = std::shared_ptr<ScalarBaseOp>;

class UnaryScalarBaseOp : public ScalarBaseOp {
protected:
    UnaryScalarBaseOp(Opcode opcode, ScalarValuePtr in, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({in}), std::vector<ScalarValuePtr>({out})) {}
};

class BinaryScalarBaseOp : public ScalarBaseOp {
protected:
    BinaryScalarBaseOp(Opcode opcode, ScalarValuePtr lhs, ScalarValuePtr rhs, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({lhs, rhs}), std::vector<ScalarValuePtr>({out})) {}
};

class CondScalarBaseOp : public ScalarBaseOp {
protected:
    CondScalarBaseOp(Opcode opcode, ScalarValuePtr cond, ScalarValuePtr sat, ScalarValuePtr unsat, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({cond, sat, unsat}), std::vector<ScalarValuePtr>({out})) {}
};

class Call1ScalarBaseOp : public ScalarBaseOp {
protected:
    Call1ScalarBaseOp(Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({arg0}), std::vector<ScalarValuePtr>({out})) {}
};

class Call2ScalarBaseOp : public ScalarBaseOp {
protected:
    Call2ScalarBaseOp(Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({arg0, arg1}), std::vector<ScalarValuePtr>({out})) {}
};

class Call3ScalarBaseOp : public ScalarBaseOp {
protected:
    Call3ScalarBaseOp(Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({arg0, arg1, arg2}), std::vector<ScalarValuePtr>({out})) {}
};

class Call4ScalarBaseOp : public ScalarBaseOp {
protected:
    Call4ScalarBaseOp(Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2, ScalarValuePtr arg3, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({arg0, arg1, arg2, arg3}), std::vector<ScalarValuePtr>({out})) {}
};

class Call5ScalarBaseOp : public ScalarBaseOp {
protected:
    Call5ScalarBaseOp(Opcode opcode, ScalarValuePtr arg0, ScalarValuePtr arg1, ScalarValuePtr arg2, ScalarValuePtr arg3, ScalarValuePtr arg4, ScalarValuePtr out)
        : ScalarBaseOp(opcode, std::vector<ScalarValuePtr>({arg0, arg1, arg2, arg3, arg4}), std::vector<ScalarValuePtr>({out})) {}
};

} // namespace pto
