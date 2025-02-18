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
 * \file expected_value.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

class RawExpectedOperator {
public:
    RawExpectedOperator(Opcode opcode, const std::vector<int64_t> &attrs)
        : opcode_(opcode), attrs_(attrs), hash_(CalculateHash()) {}

    Opcode GetOpcode() const { return opcode_; }
    const std::vector<int64_t> &GetAttrs() const { return attrs_; }
    std::size_t GetHash() const { return hash_; }

private:
    std::size_t CalculateHash() const;

    Opcode opcode_;
    std::vector<int64_t> attrs_;
    std::size_t hash_;
};

class ExpectedOperator {
public:
    ExpectedOperator() = default;
    ExpectedOperator(Opcode opcode, const std::vector<int64_t> &attrs)
        : ptr_(std::make_shared<RawExpectedOperator>(opcode, attrs)) {}

    const RawExpectedOperator *Get() const { return ptr_.get(); }
    const RawExpectedOperator *operator->() const { return Get(); }

    bool operator==(const ExpectedOperator &rhs) const {
        return Get() == rhs.Get() || (Get()->GetOpcode() == rhs->GetOpcode() && Get()->GetAttrs() == rhs->GetAttrs());
    }

private:
    std::shared_ptr<RawExpectedOperator> ptr_;
};

class RawExpectedInputValue;
class RawExpectedOperationValue;
class RawExpectedExtractValue;
class RawExpectedInsertValue;
class RawExpectedResultofValue;
class RawExpectedValue {
public:
    enum class ValueKind {
        T_EXPECTED_INPUT,
        T_EXPECTED_OPERATION,
        T_EXPECTED_INSERT,
        T_EXPECTED_EXTRACT,
        T_EXPECTED_RESULTOF,
    };
    explicit RawExpectedValue(ValueKind kind) : kind_(kind) {}

    std::size_t GetHash() const { return hash_; }
    ValueKind Kind() const { return kind_; }
    virtual ~RawExpectedValue() = default;
    virtual std::size_t CalculateHash() const = 0;
protected:
    ValueKind kind_;
    std::size_t hash_{0};
};

struct RawExpectedInsertValueElement;
class ExpectedValue {
public:
    ExpectedValue() = default;
    explicit ExpectedValue(const std::shared_ptr<RawExpectedValue> &ptr) : ptr_(ptr) {}
    ExpectedValue(const std::vector<int64_t> &shape, DataType dataType, const std::string &name);
    ExpectedValue(ExpectedOperator oper, const std::vector<ExpectedValue> &operands);
    ExpectedValue(const ExpectedValue &source, const std::vector<int64_t> &sourceShape, const std::vector<int64_t> &resultOffset, const std::vector<int64_t> &resultShape);
    ExpectedValue(const std::vector<int64_t> &shape, const std::vector<RawExpectedInsertValueElement> &elements);
    ExpectedValue(const ExpectedValue &resultof, int index);

    const RawExpectedValue *Get() const { return ptr_.get(); }
    const RawExpectedValue *operator->() const { return Get(); }
    bool IsNull() const { return ptr_ == nullptr; }

    std::shared_ptr<RawExpectedInputValue> CastInputValue() const;
    std::shared_ptr<RawExpectedOperationValue> CastOperationValue() const;
    std::shared_ptr<RawExpectedExtractValue> CastExtractValue() const;
    std::shared_ptr<RawExpectedInsertValue> CastInsertValue() const;
    std::shared_ptr<RawExpectedResultofValue> CastResultofValue() const;

    bool IsInputValue() const { return ptr_->Kind() == RawExpectedValue::ValueKind::T_EXPECTED_INPUT; }
    bool IsOperationValue() const { return ptr_->Kind() == RawExpectedValue::ValueKind::T_EXPECTED_OPERATION; }
    bool IsExtractValue() const { return ptr_->Kind() == RawExpectedValue::ValueKind::T_EXPECTED_EXTRACT; }
    bool IsInsertValue() const { return ptr_->Kind() == RawExpectedValue::ValueKind::T_EXPECTED_INSERT; }
    bool IsResultofValue() const { return ptr_->Kind() == RawExpectedValue::ValueKind::T_EXPECTED_RESULTOF; }

    bool operator==(const ExpectedValue &rhs) const;
private:
    std::shared_ptr<RawExpectedValue> ptr_;
};

class RawExpectedInputValue : public RawExpectedValue {
public:
    RawExpectedInputValue(const std::vector<int64_t> &shape, DataType dataType, const std::string &name)
        : RawExpectedValue(ValueKind::T_EXPECTED_INPUT), shape_(shape), dataType_(dataType), name_(name) {
        hash_ = CalculateHash();
    }

    const std::vector<int64_t> &GetShape() const { return shape_; }
    DataType GetDataType() const { return dataType_; }
    const std::string &GetName() const { return name_; }

    bool operator==(const RawExpectedInputValue &rhs) const {
        return shape_ == rhs.shape_ && dataType_ == rhs.dataType_ && name_ == rhs.name_;
    }

    std::size_t CalculateHash() const override;
private:
    std::vector<int64_t> shape_;
    DataType dataType_;
    std::string name_;
};

class RawExpectedOperationValue : public RawExpectedValue {
public:
    RawExpectedOperationValue(ExpectedOperator oper, const std::vector<ExpectedValue> &operands)
        : RawExpectedValue(ValueKind::T_EXPECTED_OPERATION), oper_(oper), operands_(operands) {
        hash_ = CalculateHash();
    }

    const ExpectedOperator &GetOperator() const { return oper_; }
    const std::vector<ExpectedValue> &GetOperands() const { return operands_; }

    bool operator==(const RawExpectedOperationValue &rhs) const {
        return oper_ == rhs.oper_ && operands_ == rhs.operands_;
    }

    std::size_t CalculateHash() const override;

private:
    ExpectedOperator oper_;
    std::vector<ExpectedValue> operands_;
};

class RawExpectedExtractValue : public RawExpectedValue {
public:
    RawExpectedExtractValue(const ExpectedValue &source, const std::vector<int64_t> &sourceShape, const std::vector<int64_t> &resultOffset, const std::vector<int64_t> &resultShape)
        : RawExpectedValue(ValueKind::T_EXPECTED_EXTRACT), source_(source), sourceShape_(sourceShape), resultOffset_(resultOffset), resultShape_(resultShape) {
        hash_ = CalculateHash();
    }

    const ExpectedValue &GetSource() const { return source_; }
    const Shape &GetSourceShape() const { return sourceShape_; }
    const Offset &GetResultOffset() const { return resultOffset_; }
    const Shape &GetResultShape() const { return resultShape_; }

    bool operator==(const RawExpectedExtractValue &rhs) const {
        return source_ == rhs.source_ && sourceShape_ == rhs.sourceShape_ && resultOffset_ == rhs.resultOffset_ && resultShape_ == rhs.resultShape_;
    }

    std::size_t CalculateHash() const override;
private:
    ExpectedValue source_;
    Shape sourceShape_;
    Offset resultOffset_;
    Shape resultShape_;
};

struct RawExpectedInsertValueElement {
    RawExpectedInsertValueElement(const std::vector<int64_t> &offsetIn, const std::vector<int64_t> &shapeIn,
        const ExpectedValue &sourceIn)
        : offset(offsetIn), shape(shapeIn), source(sourceIn) {}

    bool operator<(const RawExpectedInsertValueElement &rhs) const {
        return offset < rhs.offset;
    }

    bool operator==(const RawExpectedInsertValueElement &rhs) const {
        return offset == rhs.offset;
    }
public:
    Offset offset;
    Shape shape;
    ExpectedValue source;
};
class RawExpectedInsertValue : public RawExpectedValue {
public:
    RawExpectedInsertValue(const std::vector<int64_t> &shape, const std::vector<RawExpectedInsertValueElement> &elements)
        : RawExpectedValue(ValueKind::T_EXPECTED_INSERT), shape_(shape), elements_(elements) {
        std::sort(elements_.begin(), elements_.end());
        hash_ = CalculateHash();
    }

    const std::vector<int64_t> &GetShape() { return shape_; }
    const std::vector<RawExpectedInsertValueElement> &GetElements() { return elements_; }

    bool operator==(const RawExpectedInsertValue &rhs) const {
        return shape_ == rhs.shape_ && elements_ == rhs.elements_;
    }

    std::size_t CalculateHash() const override;
private:
    std::vector<int64_t> shape_;
    std::vector<RawExpectedInsertValueElement> elements_;
};

class RawExpectedResultofValue : public RawExpectedValue {
public:
    RawExpectedResultofValue(const ExpectedValue &resultof, int index)
        : RawExpectedValue(ValueKind::T_EXPECTED_RESULTOF), resultof_(resultof), index_(index) {
        hash_ = CalculateHash();
    }

    const ExpectedValue &GetResultof() const { return resultof_; }
    int GetIndex() const { return index_; }

    bool operator==(const RawExpectedResultofValue &rhs) const {
        return resultof_ == rhs.resultof_ && index_ == rhs.index_;
    }

    std::size_t CalculateHash() const override;

private:
    ExpectedValue resultof_;
    int index_;
};

class ListExpectedValue {
public:
    ListExpectedValue() {}
    explicit ListExpectedValue(const std::vector<ExpectedValue> &elements)
        : elements_(elements), hash_(CalculateHash()) {}

    const std::vector<ExpectedValue> &GetElements() const { return elements_; }
    std::size_t GetHash() const { return hash_; }

    bool operator==(const ListExpectedValue &call) const {
        return elements_ == call.elements_;
    }

private:
    std::size_t CalculateHash() const;

    std::vector<ExpectedValue> elements_;
    std::size_t hash_{0};
};

} // namespace npu::tile_fwk

template <>
struct std::hash<npu::tile_fwk::ExpectedOperator> {
    std::size_t operator()(const npu::tile_fwk::ExpectedOperator &o) const { return o->GetHash(); }
};

template <>
struct std::hash<npu::tile_fwk::ExpectedValue> {
    std::size_t operator()(const npu::tile_fwk::ExpectedValue &v) const { return v->GetHash(); }
};

template <>
struct std::hash<npu::tile_fwk::ListExpectedValue> {
    std::size_t operator()(const npu::tile_fwk::ListExpectedValue &v) const { return v.GetHash(); }
};

namespace npu::tile_fwk {

    struct CallExpectedValue {
        std::unordered_map<std::shared_ptr<LogicalTensor>, ExpectedValue> expectedTensorDict;
        ListExpectedValue expectedOutcast;
        std::string debugTrace;

        std::vector<ExpectedValue> GetExpectedValueList(const std::vector<std::shared_ptr<LogicalTensor>> &tensorList) const {
            std::vector<ExpectedValue> result;
            for (auto &tensor : tensorList) {
                if (expectedTensorDict.count(tensor)) {
                    result.push_back(expectedTensorDict.find(tensor)->second);
                } else {
                    result.push_back(ExpectedValue());
                }
            }
            return result;
        }

        void InsertExpectedValueList(const std::vector<std::shared_ptr<LogicalTensor>> &tensorList,
                                     const std::vector<ExpectedValue> &tensorExpectedValueList) {
            ASSERT(tensorList.size() == tensorExpectedValueList.size());
            for (size_t i = 0; i < tensorList.size(); i++) {
                expectedTensorDict[tensorList[i]] = tensorExpectedValueList[i];
            }
        }
    };

    struct FunctionExpectedValue {
        std::shared_ptr<CallExpectedValue> local;
        std::unordered_map<ListExpectedValue, std::shared_ptr<CallExpectedValue>> callDict;
    };

    class ExpectedValueBuilder {
    public:
        ExpectedValueBuilder() = default;

        ExpectedOperator CreateOperator(const Operation &op);

        ExpectedValue CreateValue(const std::vector<int64_t> &shape, DataType type, const std::string &name);
        ExpectedValue CreateValue(const Operation &op, const std::vector<ExpectedValue> &values);
        ExpectedValue CreateValue(const ExpectedValue &resultof, int index);

        ExpectedValue CreateIncast(const std::shared_ptr<LogicalTensor> &tensor);
        std::vector<ExpectedValue> CreateOperationOOperands(const Operation &op, const std::vector<ExpectedValue> &ioperandExpectedValueList);
        std::vector<ExpectedValue> CreateOperationInsertOOperands(const Operation &op, const std::vector<ExpectedValue> &ioperandExpectedValueList, const std::vector<ExpectedValue> &ooperandExpectedValueList);
        ListExpectedValue CreateList(const std::vector<ExpectedValue> &elements);

        std::shared_ptr<CallExpectedValue> CreateCall(Function *func, const std::vector<ExpectedValue> &incasts = {}, const std::string &debugTracePrefix = "");
        FunctionExpectedValue &GetFunction(Function *func) {
            return funcDict_[func];
        }
    private:
        ExpectedValue ValueLookup(ExpectedValue &v);

        std::unordered_set<ExpectedOperator> operatorSet_;
        std::unordered_set<ExpectedValue> valueSet_;
        std::unordered_map<Function *, FunctionExpectedValue> funcDict_;
    public:
        static ExpectedValueBuilder &GetBuilder() {
            static ExpectedValueBuilder builder;
            return builder;
        }
    };

    inline ExpectedValueBuilder &GlobalExpectedValueBuilder() {
        return ExpectedValueBuilder::GetBuilder();
    }

    ExpectedValue CreateIncastExpectedValue(const std::shared_ptr<LogicalTensor> &tensor);

    std::vector<ExpectedValue> CreateOperationOOperandExpectedValue(const Operation &op, const std::vector<ExpectedValue> &ioperandList);

    ListExpectedValue CreateListExpectedValue(const std::vector<ExpectedValue> &listExpectedValue);

} // namespace npu::tile_fwk
