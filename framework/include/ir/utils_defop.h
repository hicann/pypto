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

#define MAP_1(fn, a0)        fn(a0)
#define MAP_2(fn, a0, ...)   fn(a0) MAP_1(fn, __VA_ARGS__)
#define MAP_3(fn, a0, ...)   fn(a0) MAP_2(fn, __VA_ARGS__)
#define MAP_4(fn, a0, ...)   fn(a0) MAP_3(fn, __VA_ARGS__)
#define MAP_5(fn, a0, ...)   fn(a0) MAP_4(fn, __VA_ARGS__)
#define MAP_6(fn, a0, ...)   fn(a0) MAP_5(fn, __VA_ARGS__)
#define MAP_7(fn, a0, ...)   fn(a0) MAP_6(fn, __VA_ARGS__)
#define MAP_8(fn, a0, ...)   fn(a0) MAP_7(fn, __VA_ARGS__)
#define MAP_9(fn, a0, ...)   fn(a0) MAP_8(fn, __VA_ARGS__)
#define MAP_10(fn, a0, ...)  fn(a0) MAP_9(fn, __VA_ARGS__)
#define MAP_11(fn, a0, ...)  fn(a0) MAP_10(fn, __VA_ARGS__)
#define MAP_12(fn, a0, ...)  fn(a0) MAP_11(fn, __VA_ARGS__)
#define MAP_13(fn, a0, ...)  fn(a0) MAP_12(fn, __VA_ARGS__)
#define MAP_14(fn, a0, ...)  fn(a0) MAP_13(fn, __VA_ARGS__)
#define MAP_15(fn, a0, ...)  fn(a0) MAP_14(fn, __VA_ARGS__)
#define MAP_16(fn, a0, ...)  fn(a0) MAP_15(fn, __VA_ARGS__)
#define MAP_17(fn, a0, ...)  fn(a0) MAP_16(fn, __VA_ARGS__)
#define MAP_18(fn, a0, ...)  fn(a0) MAP_17(fn, __VA_ARGS__)
#define MAP_19(fn, a0, ...)  fn(a0) MAP_18(fn, __VA_ARGS__)
#define MAP_20(fn, a0, ...)  fn(a0) MAP_19(fn, __VA_ARGS__)
#define MAP_21(fn, a0, ...)  fn(a0) MAP_20(fn, __VA_ARGS__)
#define MAP_22(fn, a0, ...)  fn(a0) MAP_21(fn, __VA_ARGS__)
#define MAP_23(fn, a0, ...)  fn(a0) MAP_22(fn, __VA_ARGS__)
#define MAP_24(fn, a0, ...)  fn(a0) MAP_23(fn, __VA_ARGS__)
#define MAP_25(fn, a0, ...)  fn(a0) MAP_24(fn, __VA_ARGS__)
#define MAP_26(fn, a0, ...)  fn(a0) MAP_25(fn, __VA_ARGS__)
#define MAP_27(fn, a0, ...)  fn(a0) MAP_26(fn, __VA_ARGS__)
#define MAP_28(fn, a0, ...)  fn(a0) MAP_27(fn, __VA_ARGS__)
#define MAP_29(fn, a0, ...)  fn(a0) MAP_28(fn, __VA_ARGS__)
#define MAP_30(fn, a0, ...)  fn(a0) MAP_29(fn, __VA_ARGS__)
#define MAP_31(fn, a0, ...)  fn(a0) MAP_30(fn, __VA_ARGS__)

#define MAP_CONCAT_(a, b) a##b
#define MAP_CONCAT(a, b) MAP_CONCAT_(a, b)
#define MAP(fn, ...) MAP_CONCAT(MAP_, MAP_SIZE(__VA_ARGS__))(fn, __VA_ARGS__)

#define DEFOP_OPCODE(name, inherit, opcode, ...) DEFOP_OPCODE_##opcode,
#define DEFOP_OPCODE_OPCODE(...) __VA_ARGS__

#define DEFOP_OPCODE_DICT(name, inherit, opcode, ...) DEFOP_OPCODE_DICT_##opcode
#define DEFOP_OPCODE_DICT_OPCODE(...) MAP(DEFOP_OPCODE_DICT_NAME, __VA_ARGS__)
#define DEFOP_OPCODE_DICT_NAME(n) {Opcode::n, #n},

#define DEFOP_CLASS_INHERIT(name) name
#define DEFOP_CLASS_ATTR(attr) DEFOP_CLASS_GET_SET_##attr
#define DEFOP_CLASS_GET_SET_
#define DEFOP_CLASS_GET_SET_ATTR(type, name) \
    private: \
        type attr##name##_; \
    public: \
        type Get##name() const { return attr##name##_; } \
        void Set##name(type &&value) { attr##name##_ = value; }

#define DEFOP_CLASS(name, inherit, opcode, ...) \
    class name : public DEFOP_CLASS_##inherit { \
    public: \
        template<typename ...TyArgs> \
        name(TyArgs && ...args) : DEFOP_CLASS_##inherit(args...) {} \
    private: \
        MAP(DEFOP_CLASS_ATTR, __VA_ARGS__) \
    }; \
    using name##Ptr = std::shared_ptr<name>;

#define DEFOP_IRBUILDER(name, inherit, opcode, ...) \
    template<typename ...TyArgs> \
    name##Ptr Create##name(TyArgs && ...args) { return std::make_shared<name>(args...); }
