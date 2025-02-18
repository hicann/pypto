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
 * \file expected_value.cpp
 * \brief
 */

#include "expected_value.h"

namespace npu::tile_fwk {

static inline
std::size_t Digest(const HashBuffer &hashBuffer) {
    auto hash = hashBuffer.Digest();
    return hash;
}

static inline
bool CheckAllZero(const std::vector<int64_t> &vec) {
    return std::all_of(vec.begin(), vec.end(), [](int v){ return v == 0; });
}

static inline
bool CheckSameShape(const std::shared_ptr<LogicalTensor> &lhs, const std::shared_ptr<LogicalTensor> &rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return lhs->GetShape() == rhs->GetShape();
}

std::size_t RawExpectedOperator::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(opcode_), attrs_);
    return Digest(hashBuffer);;
}

template<typename T>
static bool Equal(const std::shared_ptr<RawExpectedValue> &lhsRaw, const std::shared_ptr<RawExpectedValue> &rhsRaw) {
    auto lhs = std::static_pointer_cast<T>(lhsRaw);
    auto rhs = std::static_pointer_cast<T>(rhsRaw);
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return *lhs == *rhs;
}

bool ExpectedValue::operator==(const ExpectedValue &rhs) const {
    if (Get() == rhs.Get()) {
        return true;
    }
    if (Get() == nullptr || rhs.Get() == nullptr) {
        return false;
    }
    if (Get()->Kind() != rhs.Get()->Kind()) {
        return false;
    }

    switch (Get()->Kind()) {
    case RawExpectedValue::ValueKind::T_EXPECTED_INPUT:
        return Equal<RawExpectedInputValue>(ptr_, rhs.ptr_);
    case RawExpectedValue::ValueKind::T_EXPECTED_OPERATION:
        return Equal<RawExpectedOperationValue>(ptr_, rhs.ptr_);
    case RawExpectedValue::ValueKind::T_EXPECTED_EXTRACT:
        return Equal<RawExpectedExtractValue>(ptr_, rhs.ptr_);
    case RawExpectedValue::ValueKind::T_EXPECTED_INSERT:
        return Equal<RawExpectedInsertValue>(ptr_, rhs.ptr_);
    case RawExpectedValue::ValueKind::T_EXPECTED_RESULTOF:
        return Equal<RawExpectedResultofValue>(ptr_, rhs.ptr_);
    default:
        ASSERT(false);
        break;
    }
    return true;
}

ExpectedValue::ExpectedValue(const std::vector<int64_t> &shape, DataType dataType, const std::string &name)
    : ExpectedValue(std::make_shared<RawExpectedInputValue>(shape, dataType, name)) {}
ExpectedValue::ExpectedValue(ExpectedOperator oper, const std::vector<ExpectedValue> &operands)
    : ExpectedValue(std::make_shared<RawExpectedOperationValue>(oper, operands)) {}
ExpectedValue::ExpectedValue(const ExpectedValue &source, const std::vector<int64_t> &sourceShape, const std::vector<int64_t> &resultOffset, const std::vector<int64_t> &resultShape)
    : ExpectedValue(std::make_shared<RawExpectedExtractValue>(source, sourceShape, resultOffset, resultShape)) {}
ExpectedValue::ExpectedValue(const std::vector<int64_t> &shape, const std::vector<RawExpectedInsertValueElement> &elements)
    : ExpectedValue(std::make_shared<RawExpectedInsertValue>(shape, elements)) {}
ExpectedValue::ExpectedValue(const ExpectedValue &resultof, int index)
    : ExpectedValue(std::make_shared<RawExpectedResultofValue>(resultof, index)) {}

std::shared_ptr<RawExpectedInputValue> ExpectedValue::CastInputValue() const {
    return std::static_pointer_cast<RawExpectedInputValue>(ptr_);
}
std::shared_ptr<RawExpectedOperationValue> ExpectedValue::CastOperationValue() const {
    return std::static_pointer_cast<RawExpectedOperationValue>(ptr_);
}
std::shared_ptr<RawExpectedExtractValue> ExpectedValue::CastExtractValue() const {
    return std::static_pointer_cast<RawExpectedExtractValue>(ptr_);
}
std::shared_ptr<RawExpectedInsertValue> ExpectedValue::CastInsertValue() const {
    return std::static_pointer_cast<RawExpectedInsertValue>(ptr_);
}
std::shared_ptr<RawExpectedResultofValue> ExpectedValue::CastResultofValue() const {
    return std::static_pointer_cast<RawExpectedResultofValue>(ptr_);
}

std::size_t RawExpectedInputValue::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(kind_), shape_, static_cast<char32_t>(dataType_), name_);
    auto hash = hashBuffer.Digest();
    return hash;
}

std::size_t RawExpectedOperationValue::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(kind_), oper_->GetHash());
    for (auto &op : operands_) {
        hashBuffer.Update(op->GetHash());
    }
    return Digest(hashBuffer);;
}

std::size_t RawExpectedExtractValue::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(kind_), source_->GetHash());
    hashBuffer.Append(sourceShape_);
    hashBuffer.Append(resultOffset_);
    hashBuffer.Append(resultShape_);
    return Digest(hashBuffer);;
}

std::size_t RawExpectedInsertValue::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(kind_), shape_);
    for (auto &e : elements_) {
        hashBuffer.Append(e.offset);
        hashBuffer.Append(e.shape);
        hashBuffer.Append(e.source->GetHash());
    }
    return Digest(hashBuffer);;
}

std::size_t RawExpectedResultofValue::CalculateHash() const {
    HashBuffer hashBuffer(static_cast<char32_t>(kind_), resultof_->GetHash(), index_);
    return Digest(hashBuffer);;
}

std::size_t ListExpectedValue::CalculateHash() const {
    HashBuffer hashBuffer;
    for (auto &arg : elements_) {
        hashBuffer.Update(arg->GetHash());
    }
    return Digest(hashBuffer);;
}

ExpectedValue ExpectedValueBuilder::ValueLookup(ExpectedValue &v) {
    if (!valueSet_.count(v)) {
        valueSet_.insert(v);
    } else {
        v = *valueSet_.find(v);
    }
    return v;
}

ExpectedOperator ExpectedValueBuilder::CreateOperator(const Operation &op) {
    HashBuffer hashBuffer;
    std::vector<int64_t> attrs(hashBuffer.begin(), hashBuffer.end());
    ExpectedOperator operatorExpectedValue(op.GetOpcode(), attrs);
    return operatorExpectedValue;
}

ExpectedValue ExpectedValueBuilder::CreateValue(const std::vector<int64_t> &shape, DataType type, const std::string &name) {
    ExpectedValue v(shape, type, name);

    v = ValueLookup(v);
    return v;
}

static std::function<std::vector<ExpectedValue>(ExpectedValueBuilder *builder, const Operation &operation, const std::vector<ExpectedValue> &values)>
    g_normalizers[static_cast<int>(Opcode::OP_UNKNOWN)];

#define DEFINE_NORMALIZER(opcode, builder, values)                                              \
    static std::vector<ExpectedValue> Normalizer_##opcode(                                      \
        ExpectedValueBuilder *builder, const Operation &operation, const std::vector<ExpectedValue> &values);               \
    static struct NormalizerRegister_##opcode {                                                 \
        NormalizerRegister_##opcode() { g_normalizers[static_cast<int>(Opcode::opcode)] = Normalizer_##opcode; } \
    } normalizerRegister_##opcode;                                                              \
    static std::vector<ExpectedValue> Normalizer_##opcode(                                      \
        [[maybe_unused]]ExpectedValueBuilder *builder, [[maybe_unused]]const Operation &operation, \
        const std::vector<ExpectedValue> &values)

DEFINE_NORMALIZER(OP_ADD, builder, values) {
    std::vector<ExpectedValue> valuesNorm;
    for (auto &v : values) {
        if (v.Get() == nullptr) {
            valuesNorm.push_back(v);
            continue;
        }
        if (v.IsInputValue() || v.IsResultofValue() || v.IsInsertValue() || v.IsExtractValue()) {
            valuesNorm.push_back(v);
        } else {
            ASSERT(v.IsOperationValue());
            if (v.CastOperationValue()->GetOperator()->GetOpcode() == Opcode::OP_ADD) {
                for (auto &velt : v.CastOperationValue()->GetOperands()) {
                    valuesNorm.push_back(velt);
                }
            }else {
                valuesNorm.push_back(v);
            }
        }
    }
    std::sort(valuesNorm.begin(), valuesNorm.end(),
        [](const ExpectedValue &lhs, const ExpectedValue &rhs) { return lhs->GetHash() < rhs->GetHash(); });
    return valuesNorm;
}

static std::function<ExpectedValue(ExpectedValueBuilder *builder, const Operation &operation, const std::vector<ExpectedValue> &values)>
    g_evaluators[static_cast<int>(Opcode::OP_UNKNOWN)];

#define DEFINE_EVALUATOR(opcode, builder, values)                                               \
    static ExpectedValue Evaluator_##opcode(                                            \
        ExpectedValueBuilder *builder, const Operation &operation, const std::vector<ExpectedValue> &values);               \
    static struct EvaluatorRegister_##opcode {                                                       \
        EvaluatorRegister_##opcode() { g_evaluators[static_cast<int>(Opcode::opcode)] = Evaluator_##opcode; }   \
    } evaluatorRegister_##opcode;                                                                    \
    static ExpectedValue Evaluator_##opcode(                                            \
        [[maybe_unused]]ExpectedValueBuilder *builder, [[maybe_unused]]const Operation &operation, \
        const std::vector<ExpectedValue> &values)

DEFINE_EVALUATOR(OP_VIEW, builder, values) {
    if (operation.GetOOperands().size() == 0) {
        return {};
    }

    ASSERT(operation.GetIOperands().size() == 1);
    ASSERT(operation.GetOOperands().size() == 1);

    std::shared_ptr<ViewOpAttribute> op = std::static_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (CheckAllZero(op->GetFromOffset()) && CheckSameShape(operation.GetIOperands()[0], operation.GetOOperands()[0])) {
        return values[0];
    }
    std::vector<int64_t> resultOffset = op->GetFromOffset();
    ExpectedValue view(values[0], operation.GetIOperands()[0]->GetShape(), resultOffset, operation.GetOOperands()[0]->GetShape());
    return view;
}

DEFINE_EVALUATOR(OP_CONVERT, builder, values) {
    if (operation.GetOOperands().size() == 0) {
        return {};
    }

    ASSERT(operation.GetIOperands().size() == 1);
    ASSERT(operation.GetOOperands().size() == 1);
    ASSERT(CheckSameShape(operation.GetIOperands()[0], operation.GetOOperands()[0]));
    return values[0];
}

ExpectedValue ExpectedValueBuilder::CreateValue(const Operation &operation, const std::vector<ExpectedValue> &values) {
    ASSERT(operation.GetOpcode() != Opcode::OP_CALL);

    std::vector<ExpectedValue> normalizedValues;
    auto norm = g_normalizers[static_cast<int>(operation.GetOpcode())];
    if (norm != nullptr) {
        normalizedValues = norm(this, operation, values);
    }
    if (normalizedValues.size() == 0) {
        /* Failed to norm, use default */
        normalizedValues = values;
    }

    ExpectedValue v;
    auto eval = g_evaluators[static_cast<int>(operation.GetOpcode())];
    if (eval != nullptr) {
        v = eval(this, operation, normalizedValues);
    }
    if (v.Get() == nullptr) {
        /* Failed to eval, use default */
        ExpectedOperator op = CreateOperator(operation);
        v = ExpectedValue(op, normalizedValues);
    }

    v = ValueLookup(v);
    return v;
}

ExpectedValue ExpectedValueBuilder::CreateValue(const ExpectedValue &resultof, int index)
{
    ExpectedValue v(resultof, index);

    v = ValueLookup(v);
    return v;
}

ExpectedValue ExpectedValueBuilder::CreateIncast(const std::shared_ptr<LogicalTensor> &incast)
{
    ASSERT(incast->nodetype == NodeType::INCAST);
    ASSERT(incast->tensor->GetRawShape() == incast->GetShape());
    ExpectedValue incastExpectedValue(incast->GetShape(), incast->tensor->datatype, incast->tensor->symbol);
    return incastExpectedValue;
}

std::vector<ExpectedValue> ExpectedValueBuilder::CreateOperationOOperands(const Operation &op, const std::vector<ExpectedValue> &ioperandExpectedValueList)
{
    ExpectedValue operationExpectedValue = CreateValue(op, ioperandExpectedValueList);

    std::vector<ExpectedValue> ooperandExpectedValueList;
    if (op.GetOOperands().size() == 1) {
        ooperandExpectedValueList.push_back(operationExpectedValue);
    } else if (op.GetOOperands().size() > 1) {
        for (size_t index = 0; index < op.GetOOperands().size(); index++) {
            ExpectedValue ooperandExpectedValue(operationExpectedValue, index);
            ooperandExpectedValueList.push_back(ooperandExpectedValue);
        }
    }
    return ooperandExpectedValueList;
}

std::vector<ExpectedValue> ExpectedValueBuilder::CreateOperationInsertOOperands(const Operation &op, const std::vector<ExpectedValue> &ioperandExpectedValueList, const std::vector<ExpectedValue> &ooperandExpectedValueList)
{
    ASSERT(op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_OUT);
    ASSERT(op.GetIOperands().size() == 1);
    ASSERT(op.GetOOperands().size() == 1);
    ASSERT(ioperandExpectedValueList.size() == 1);
    ASSERT(ooperandExpectedValueList.size() == 1);

    std::shared_ptr<AssembleOpAttribute> opattr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (CheckAllZero(opattr->GetToOffset()) && CheckSameShape(op.GetIOperands()[0], op.GetOOperands()[0])) {
        return {ioperandExpectedValueList[0]};
    }

    RawExpectedInsertValueElement element(opattr->GetToOffset(), op.GetIOperands()[0]->GetShape(), ioperandExpectedValueList[0]);

    std::vector<RawExpectedInsertValueElement> elementList = {element};

    if (!ooperandExpectedValueList[0].IsNull()) {
        ASSERT(ooperandExpectedValueList[0].IsInsertValue());
        auto insert = ooperandExpectedValueList[0].CastInsertValue();
        elementList.insert(elementList.end(), insert->GetElements().begin(), insert->GetElements().end());
    }
    ExpectedValue assemble(op.GetOOperands()[0]->GetShape(), elementList);
    return {assemble};
}

ListExpectedValue ExpectedValueBuilder::CreateList(const std::vector<ExpectedValue> &elements) {
    return ListExpectedValue(elements);
}

std::shared_ptr<CallExpectedValue> ExpectedValueBuilder::CreateCall(Function *func, const std::vector<ExpectedValue> &incastExpectedValueList, const std::string &debugTracePrefix) {
    std::shared_ptr<CallExpectedValue> result = std::make_shared<CallExpectedValue>();
    std::ostringstream debugTraceStream;

    ASSERT(incastExpectedValueList.size() == static_cast<size_t>(0) ||
           incastExpectedValueList.size() == func->GetIncast().size());
    for (size_t index = 0; index < func->GetIncast().size(); index++) {
        std::shared_ptr<LogicalTensor> incast = func->GetIncast()[index];
        ExpectedValue local = incastExpectedValueList.size() == 0 ? CreateIncast(incast) : incastExpectedValueList[index];
        result->expectedTensorDict[incast] = local;

        if (debugTracePrefix != "") {
            debugTraceStream << debugTracePrefix << incast->Dump() << ": " << local->GetHash() << "\n";
        }
    }

    for (auto &op : func->Operations()) {
        std::vector<ExpectedValue> ioperandExpectedValueList = result->GetExpectedValueList(op.GetIOperands());

        std::vector<ExpectedValue> ooperandExpectedValueList;
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_OUT) {
            ooperandExpectedValueList = result->GetExpectedValueList(op.GetOOperands());
            ooperandExpectedValueList = CreateOperationInsertOOperands(op, ioperandExpectedValueList, ooperandExpectedValueList);
        } else {
            ooperandExpectedValueList = CreateOperationOOperands(op, ioperandExpectedValueList);
        }

        result->InsertExpectedValueList(op.GetOOperands(), ooperandExpectedValueList);

        if (debugTracePrefix != "") {
            debugTraceStream << debugTracePrefix << op.Dump();
            debugTraceStream << debugTracePrefix <<
                [&](){
                    std::string ohash;
                    for (auto &v : ooperandExpectedValueList) {
                        ohash += " " + std::to_string(v->GetHash());
                    }
                    return ohash;
                }() << " <- " <<
                [&](){
                    std::string ihash;
                    for (auto &v : ioperandExpectedValueList) {
                        ihash += " " + std::to_string(v->GetHash());
                    }
                    return ihash;
                }();
        }
    }

    std::vector<ExpectedValue> outcastExpectedValue = result->GetExpectedValueList(func->GetOutcast());
    result->expectedOutcast = CreateList(outcastExpectedValue);

    if (debugTracePrefix != "") {
        result->debugTrace = debugTraceStream.str();
    }
    return result;
}

ExpectedValue CreateIncastExpectedValue(const std::shared_ptr<LogicalTensor> &tensor) {
    return GlobalExpectedValueBuilder().CreateIncast(tensor);
}

std::vector<ExpectedValue> CreateOperationOOperandExpectedValue(const Operation &op, const std::vector<ExpectedValue> &ioperandList) {
    return GlobalExpectedValueBuilder().CreateOperationOOperands(op, ioperandList);
}

ListExpectedValue CreateListExpectedValue(const std::vector<ExpectedValue> &listExpectedValue) {
    return GlobalExpectedValueBuilder().CreateList(listExpectedValue);
}

} // namespace npu::tile_fwk
