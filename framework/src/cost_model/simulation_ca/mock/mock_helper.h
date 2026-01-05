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
 * \file mock_helper.h
 * \brief
 */

#ifndef __MOCK_HELPER_H__
#define __MOCK_HELPER_H__

#include <iostream>
#include <type_traits>

#include "mock_types.h"

using namespace std;

namespace MockHelper {
    void print_args() {
        std::cout << std::endl;
    }

    template<typename T, typename... Args>
    void print_args(T first, Args&&... args) {
        if constexpr (std::is_pointer_v<T>) {
            std::cout << reinterpret_cast<std::uintptr_t>(first) << " ";
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            std::cout << (int)first << " ";
        } else {
            std::cout << first << " ";
        }
        print_args(std::forward<Args>(args)...); 
    }

    void print_typed_args(int num)
    {
    }

    template<typename T, typename... Args>
    void print_typed_args(int num, T first, Args&&... args) {
        if constexpr (std::is_same_v<T, uint32_t*> || std::is_same_v<T, int*>
            || std::is_same_v<T, uint32_t> || std::is_same_v<T, int>) {
            std::cout << "int32_t ";
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float*>) {
            std::cout << "float ";
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, half*>) {
            std::cout << "half ";
        } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, bfloat16_t*>) {
            std::cout << "bfloat16_t ";
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int8_t*>) {
            std::cout << "int8_t ";
        } else if constexpr (std::is_same_v<T, ub_addr8_t> || std::is_same_v<T, ub_addr8_t*>) {
            std::cout << "ub_addr8_t ";
        } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int64_t*>) {
            std::cout << "int64_t ";
        } else {
            throw std::invalid_argument(std::string("bad type: ") + typeid(T).name());
        }

        if constexpr (std::is_pointer_v<T>) {
            std::cout << reinterpret_cast<std::uintptr_t>(first) << " ";
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            std::cout << (int)first << " ";
        } else {
            std::cout << first << " ";
        }

        num--;
        if (num) {
            print_typed_args(num, std::forward<Args>(args)...);
        } else {
            print_args(std::forward<Args>(args)...); 
        }
    }

    template<typename T, typename... Args>
    void print_inst_typed_arg_1(T op, Args&&... args) {
        std::cout << op << " ";
        print_typed_args(1, std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    void print_inst_typed_arg_2(T op, Args&&... args) {
        int arg_num = 2;
        std::cout << op << " ";
        print_typed_args(arg_num, std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    void print_inst_typed_arg_3(T op, Args&&... args) {
        int arg_num = 3;
        std::cout << op << " ";
        print_typed_args(arg_num, std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    void print_inst(T op, Args&&... args) {
        std::cout << op << " ";
        print_args(std::forward<Args>(args)...);
    }
}

/* macro used to define 3 type-specific argumants instructions. e.g. vconv_f322f16r<half, float>(...)  */
#define COSTMODEL_MOCK_INST_TYPED_ARG_3(inst) \
    template <typename... Args> \
    void inst(Args&&... args) { \
    MockHelper::print_inst_typed_arg_3(#inst, std::forward<Args>(args)...); \
}

/* macro used to define 2 type-specific argumants instructions. e.g. vconv_f322f16r<half, float>(...)  */
#define COSTMODEL_MOCK_INST_TYPED_ARG_2(inst) \
    template <typename... Args> \
    void inst(Args&&... args) { \
    MockHelper::print_inst_typed_arg_2(#inst, std::forward<Args>(args)...); \
}

/* macro used to define 1 type-specific argumant instructions. e.g. vadd<float>(...)  */
#define COSTMODEL_MOCK_INST_TYPED_ARG_1(inst) \
    template <typename... Args> \
    void inst(Args&&... args) { \
    MockHelper::print_inst_typed_arg_1(#inst, std::forward<Args>(args)...); \
}

/* macro used to define typeless instructions. e.g. pipe_barrier(...)  */
#define COSTMODEL_MOCK_INST(inst) \
    template <typename... Args> \
    void inst(Args&&... args) { \
    MockHelper::print_inst(#inst, std::forward<Args>(args)...); \
}

#endif
