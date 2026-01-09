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

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <unordered_set>

#define MAP_SIZE_( a0,  a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9, a10, a11, a12, a13, a14, a15, \
                  a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, v, ...) v
#define MAP_SIZE(...) MAP_SIZE_(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
                                             16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1)

#define MAP_1(head, body, tail, a0)        head(a0)
#define MAP_2(head, body, tail, a0, ...)   head(a0) MAP_1(tail, tail, tail, __VA_ARGS__)
#define MAP_3(head, body, tail, a0, ...)   head(a0) MAP_2(body, body, tail, __VA_ARGS__)
#define MAP_4(head, body, tail, a0, ...)   head(a0) MAP_3(body, body, tail, __VA_ARGS__)
#define MAP_5(head, body, tail, a0, ...)   head(a0) MAP_4(body, body, tail, __VA_ARGS__)
#define MAP_6(head, body, tail, a0, ...)   head(a0) MAP_5(body, body, tail, __VA_ARGS__)
#define MAP_7(head, body, tail, a0, ...)   head(a0) MAP_6(body, body, tail, __VA_ARGS__)
#define MAP_8(head, body, tail, a0, ...)   head(a0) MAP_7(body, body, tail, __VA_ARGS__)
#define MAP_9(head, body, tail, a0, ...)   head(a0) MAP_8(body, body, tail, __VA_ARGS__)
#define MAP_10(head, body, tail, a0, ...)  head(a0) MAP_9(body, body, tail, __VA_ARGS__)
#define MAP_11(head, body, tail, a0, ...)  head(a0) MAP_10(body, body, tail, __VA_ARGS__)
#define MAP_12(head, body, tail, a0, ...)  head(a0) MAP_11(body, body, tail, __VA_ARGS__)
#define MAP_13(head, body, tail, a0, ...)  head(a0) MAP_12(body, body, tail, __VA_ARGS__)
#define MAP_14(head, body, tail, a0, ...)  head(a0) MAP_13(body, body, tail, __VA_ARGS__)
#define MAP_15(head, body, tail, a0, ...)  head(a0) MAP_14(body, body, tail, __VA_ARGS__)
#define MAP_16(head, body, tail, a0, ...)  head(a0) MAP_15(body, body, tail, __VA_ARGS__)
#define MAP_17(head, body, tail, a0, ...)  head(a0) MAP_16(body, body, tail, __VA_ARGS__)
#define MAP_18(head, body, tail, a0, ...)  head(a0) MAP_17(body, body, tail, __VA_ARGS__)
#define MAP_19(head, body, tail, a0, ...)  head(a0) MAP_18(body, body, tail, __VA_ARGS__)
#define MAP_20(head, body, tail, a0, ...)  head(a0) MAP_19(body, body, tail, __VA_ARGS__)
#define MAP_21(head, body, tail, a0, ...)  head(a0) MAP_20(body, body, tail, __VA_ARGS__)
#define MAP_22(head, body, tail, a0, ...)  head(a0) MAP_21(body, body, tail, __VA_ARGS__)
#define MAP_23(head, body, tail, a0, ...)  head(a0) MAP_22(body, body, tail, __VA_ARGS__)
#define MAP_24(head, body, tail, a0, ...)  head(a0) MAP_23(body, body, tail, __VA_ARGS__)
#define MAP_25(head, body, tail, a0, ...)  head(a0) MAP_24(body, body, tail, __VA_ARGS__)
#define MAP_26(head, body, tail, a0, ...)  head(a0) MAP_25(body, body, tail, __VA_ARGS__)
#define MAP_27(head, body, tail, a0, ...)  head(a0) MAP_26(body, body, tail, __VA_ARGS__)
#define MAP_28(head, body, tail, a0, ...)  head(a0) MAP_27(body, body, tail, __VA_ARGS__)
#define MAP_29(head, body, tail, a0, ...)  head(a0) MAP_28(body, body, tail, __VA_ARGS__)
#define MAP_30(head, body, tail, a0, ...)  head(a0) MAP_29(body, body, tail, __VA_ARGS__)
#define MAP_31(head, body, tail, a0, ...)  head(a0) MAP_30(body, body, tail, __VA_ARGS__)

#define MAP_CONCAT_(a, b) a##b
#define MAP_CONCAT(a, b) MAP_CONCAT_(a, b)
#define MAP_HEAD_BODY_TAIL(head, body, tail, ...) MAP_CONCAT(MAP_, MAP_SIZE(__VA_ARGS__))(head, body, tail, __VA_ARGS__)
#define MAP(fn, ...) MAP_HEAD_BODY_TAIL(fn, fn, fn, __VA_ARGS__)
#define MAP3(fn, ...) MAP_HEAD_BODY_TAIL(fn##_HEAD, fn##_BODY, fn##_TAIL, __VA_ARGS__)

#define DEFOP_OPCODE(name, inherit, opcode, ...) DEFOP_OPCODE_##opcode,
#define DEFOP_OPCODE_OPCODE(...) __VA_ARGS__

#define DEFOP_OPCODE_DICT(name, inherit, opcode, ...) DEFOP_OPCODE_DICT_##opcode
#define DEFOP_OPCODE_DICT_OPCODE(...) MAP(DEFOP_OPCODE_DICT_NAME, __VA_ARGS__)
#define DEFOP_OPCODE_DICT_NAME(n) {Opcode::n, #n},

#define DEFOP_OPCODE_PYENUM(name, inherit, opcode, ...) DEFOP_OPCODE_PYENUM_##opcode
#define DEFOP_OPCODE_PYENUM_OPCODE(...) MAP(DEFOP_OPCODE_PYENUM_NAME, __VA_ARGS__)
#define DEFOP_OPCODE_PYENUM_NAME(n) .value(#n, Opcode::n)

#define DEFOP_CLASS_Scalar const ScalarValuePtr &
#define DEFOP_CLASS_Tile const TileValuePtr &
#define DEFOP_CLASS_ScalarList const std::vector<ScalarValuePtr> &

#define DEFOP_CLASS_CONSTRUCT_PARAM_INPUT(type, name) , DEFOP_CLASS_##type name
#define DEFOP_CLASS_CONSTRUCT_PARAM_OUTPUT(type, name) , DEFOP_CLASS_##type name
#define DEFOP_CLASS_CONSTRUCT_PARAM_ATTR(type, name, defaultValue) , const type &name = defaultValue

#define DEFOP_CLASS_CONSTRUCT_INHERIT_INPUT(type, name) , name
#define DEFOP_CLASS_CONSTRUCT_INHERIT_OUTPUT(type, name) , name
#define DEFOP_CLASS_CONSTRUCT_INHERIT_ATTR(type, name, defaultValue)

#define DEFOP_CLASS_CONSTRUCT_ATTR_INPUT(type, name)
#define DEFOP_CLASS_CONSTRUCT_ATTR_OUTPUT(type, name)
#define DEFOP_CLASS_CONSTRUCT_ATTR_ATTR(type, name, defaultValue) , attr##name##_(name)

#define DEFOP_CLASS_INPUT_FIELD_INDEX(name) static_cast<int>(InputOperandIndex::index##name)
#define DEFOP_CLASS_OUTPUT_FIELD_INDEX(name) static_cast<int>(OutputOperandIndex::index##name)

#define DEFOP_CLASS_FILL_INDEX(indexBaseRef, indexRef, indexCount) \
    do { \
        indexRef = indexBaseRef; \
        indexBaseRef += indexCount; \
    } while (0)
#define DEFOP_CLASS_GET_OPERAND_COUNT_Scalar(name) 1
#define DEFOP_CLASS_GET_OPERAND_COUNT_Tile(name) 1
#define DEFOP_CLASS_GET_OPERAND_COUNT_ScalarList(name) ((name).size())

#define DEFOP_CLASS_CONSTRUCT_INPUT_INDEX_INPUT(type, name) DEFOP_CLASS_FILL_INDEX( \
    inputIndexBase, inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(name)], DEFOP_CLASS_GET_OPERAND_COUNT_##type(name));
#define DEFOP_CLASS_CONSTRUCT_INPUT_INDEX_OUTPUT(type, name)
#define DEFOP_CLASS_CONSTRUCT_INPUT_INDEX_ATTR(type, name, defaultAttr)

#define DEFOP_CLASS_CONSTRUCT_OUTPUT_INDEX_INPUT(type, name)
#define DEFOP_CLASS_CONSTRUCT_OUTPUT_INDEX_OUTPUT(type, name) DEFOP_CLASS_FILL_INDEX( \
    outputIndexBase, outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(name)], DEFOP_CLASS_GET_OPERAND_COUNT_##type(name));
#define DEFOP_CLASS_CONSTRUCT_OUTPUT_INDEX_ATTR(type, name, defaultAttr)

#define DEFOP_CLASS_INPUT_OPERAND_INDEX_INPUT(type, name) index##name,
#define DEFOP_CLASS_INPUT_OPERAND_INDEX_OUTPUT(type, name)
#define DEFOP_CLASS_INPUT_OPERAND_INDEX_ATTR(type, name, defaultAttr)

#define DEFOP_CLASS_OUTPUT_OPERAND_INDEX_INPUT(type, name)
#define DEFOP_CLASS_OUTPUT_OPERAND_INDEX_OUTPUT(type, name) index##name,
#define DEFOP_CLASS_OUTPUT_OPERAND_INDEX_ATTR(type, name, defaultAttr)

#define DEFOP_CLASS_OPERAND_INPUT_INDEX(name, n) (inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(name)] + (n))
#define DEFOP_CLASS_OPERAND_OUTPUT_INDEX(name, n) (outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(name)] + (n))

#define DEFOP_CLASS_OPERAND_TYPE(name, type, getIndex, getOperand, setOperand) \
    type##Ptr Get##name() const { \
        return std::static_pointer_cast<type>(getOperand(getIndex(name, 0))); \
    } \
    void Set##name(const type##Ptr &value) { \
        return setOperand(getIndex(name, 0), std::static_pointer_cast<Value>(value)); \
    }
#define DEFOP_CLASS_OPERAND_TYPE_LIST(name, type, getIndex, getOperand, setOperand, count) \
    size_t Get##name##Size() const { \
        return (count); \
    } \
    type##Ptr Get##name(size_t index) const { \
        return std::static_pointer_cast<type>(getOperand(getIndex(name, index))); \
    } \
    void Set##name(size_t index, const type##Ptr &value) { \
        return setOperand(getIndex(name, index), std::static_pointer_cast<Value>(value)); \
    }

#define DEFOP_CLASS_OPERAND_INPUT_Scalar(name) DEFOP_CLASS_OPERAND_TYPE(name, ScalarValue, DEFOP_CLASS_OPERAND_INPUT_INDEX, GetInputOperand, SetInputOperand)
#define DEFOP_CLASS_OPERAND_INPUT_Tile(name) DEFOP_CLASS_OPERAND_TYPE(name, TileValue, DEFOP_CLASS_OPERAND_INPUT_INDEX, GetInputOperand, SetInputOperand)
#define DEFOP_CLASS_OPERAND_INPUT_ScalarList(name) DEFOP_CLASS_OPERAND_TYPE_LIST(name, ScalarValue, DEFOP_CLASS_OPERAND_INPUT_INDEX, GetInputOperand, SetInputOperand, \
        inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(name) + 1] - inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(name)])

#define DEFOP_CLASS_OPERAND_OUTPUT_Scalar(name) DEFOP_CLASS_OPERAND_TYPE(name, ScalarValue, DEFOP_CLASS_OPERAND_OUTPUT_INDEX, GetOutputOperand, SetOutputOperand)
#define DEFOP_CLASS_OPERAND_OUTPUT_Tile(name) DEFOP_CLASS_OPERAND_TYPE(name, TileValue, DEFOP_CLASS_OPERAND_OUTPUT_INDEX, GetOutputOperand, SetOutputOperand)
#define DEFOP_CLASS_OPERAND_OUTPUT_ScalarList(name) DEFOP_CLASS_OPERAND_TYPE_LIST(name, ScalarValue, DEFOP_CLASS_OPERAND_OUTPUT_INDEX, GetOutputOperand, SetOutputOperand, \
        outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(name) + 1] - outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(name)])

#define DEFOP_CLASS_OPERAND_INPUT(type, name) DEFOP_CLASS_OPERAND_INPUT_##type(name)
#define DEFOP_CLASS_OPERAND_OUTPUT(type, name) DEFOP_CLASS_OPERAND_OUTPUT_##type(name)
#define DEFOP_CLASS_OPERAND_ATTR(type, name, defaultAttr)

#define DEFOP_CLASS_ATTR_INPUT(type, name)
#define DEFOP_CLASS_ATTR_OUTPUT(type, name)
#define DEFOP_CLASS_ATTR_ATTR(type, name, defaultAttr) \
    private: \
        type attr##name##_; \
    public: \
        type Get##name() const { return attr##name##_; } \
        void Set##name(type &&value) { attr##name##_ = value; }

#define DEFOP_CLASS_INHERIT(name) name
#define DEFOP_CLASS_PREFIX_CONSTRUCT_PARAM(n) DEFOP_CLASS_CONSTRUCT_PARAM_##n
#define DEFOP_CLASS_PREFIX_CONSTRUCT_INHERIT(n) DEFOP_CLASS_CONSTRUCT_INHERIT_##n
#define DEFOP_CLASS_PREFIX_CONSTRUCT_ATTR(n) DEFOP_CLASS_CONSTRUCT_ATTR_##n
#define DEFOP_CLASS_PREFIX_CONSTRUCT_INPUT_INDEX(n) DEFOP_CLASS_CONSTRUCT_INPUT_INDEX_##n
#define DEFOP_CLASS_PREFIX_CONSTRUCT_OUTPUT_INDEX(n) DEFOP_CLASS_CONSTRUCT_OUTPUT_INDEX_##n
#define DEFOP_CLASS_PREFIX_INPUT_OPERAND_INDEX(n) DEFOP_CLASS_INPUT_OPERAND_INDEX_##n
#define DEFOP_CLASS_PREFIX_OUTPUT_OPERAND_INDEX(n) DEFOP_CLASS_OUTPUT_OPERAND_INDEX_##n
#define DEFOP_CLASS_PREFIX_OPERAND(n) DEFOP_CLASS_OPERAND_##n
#define DEFOP_CLASS_PREFIX_ATTR(n) DEFOP_CLASS_ATTR_##n

#define DEFOP_CLASS(name, inherit, opcode, ...) \
    class name : public DEFOP_CLASS_##inherit { \
    public: \
        name(Opcode opc MAP(DEFOP_CLASS_PREFIX_CONSTRUCT_PARAM, __VA_ARGS__)) \
            : DEFOP_CLASS_##inherit( \
                opc MAP(DEFOP_CLASS_PREFIX_CONSTRUCT_INHERIT, __VA_ARGS__)) \
              MAP(DEFOP_CLASS_PREFIX_CONSTRUCT_ATTR, __VA_ARGS__) { \
            int inputIndexBase = 0; \
            MAP(DEFOP_CLASS_PREFIX_CONSTRUCT_INPUT_INDEX, __VA_ARGS__) \
            inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(DefopMax)] = inputIndexBase; \
            int outputIndexBase = 0; \
            MAP(DEFOP_CLASS_PREFIX_CONSTRUCT_OUTPUT_INDEX, __VA_ARGS__) \
            outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(DefopMax)] = outputIndexBase; \
        } \
    private: \
        enum class InputOperandIndex : int { \
            MAP(DEFOP_CLASS_PREFIX_INPUT_OPERAND_INDEX, __VA_ARGS__) \
            indexDefopMax, \
        }; \
        enum class OutputOperandIndex : int { \
            MAP(DEFOP_CLASS_PREFIX_OUTPUT_OPERAND_INDEX, __VA_ARGS__) \
            indexDefopMax, \
        }; \
        MAP(DEFOP_CLASS_PREFIX_OPERAND, __VA_ARGS__) \
    private: \
        MAP(DEFOP_CLASS_PREFIX_ATTR, __VA_ARGS__) \
        int inputIndexList_[DEFOP_CLASS_INPUT_FIELD_INDEX(DefopMax) + 1]; \
        int outputIndexList_[DEFOP_CLASS_OUTPUT_FIELD_INDEX(DefopMax) + 1]; \
    }; \
    using name##Ptr = std::shared_ptr<name>;

#define DEFOP_IRBUILDER_CONSTRUCT_INHERIT_INPUT(type, name) , name
#define DEFOP_IRBUILDER_CONSTRUCT_INHERIT_OUTPUT(type, name) , name
#define DEFOP_IRBUILDER_CONSTRUCT_INHERIT_ATTR(type, name, defaultValue) , name

#define DEFOP_IRBUILDER_PREFIX_CONSTRUCT_PARAM DEFOP_CLASS_PREFIX_CONSTRUCT_PARAM
#define DEFOP_IRBUILDER_PREFIX_CONSTRUCT_INHERIT(n) DEFOP_IRBUILDER_CONSTRUCT_INHERIT_##n

#define DEFOP_IRBUILDER(name, inherit, opcode, ...) \
    name##Ptr Create##name(Opcode opc MAP(DEFOP_IRBUILDER_PREFIX_CONSTRUCT_PARAM, __VA_ARGS__)) { \
        return std::make_shared<name>(opc  MAP(DEFOP_IRBUILDER_PREFIX_CONSTRUCT_INHERIT, __VA_ARGS__)); \
    }
