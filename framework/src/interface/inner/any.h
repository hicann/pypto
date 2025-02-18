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
 * \file any.h
 * \brief
 */

#pragma once

#include <typeinfo>
#include <type_traits>
#include <utility>

namespace npu::tile_fwk {
class Any {
private:
    struct Base {
        virtual ~Base() = default;
        virtual Base *Clone() const = 0;
        virtual const std::type_info &Type() const noexcept = 0;
    };

    template <typename T>
    struct Derived : Base {
        T value_;

        template <typename U>
        Derived(U &&value) : value_(std::forward<U>(value)) {}

        Base *Clone() const override { return new Derived(value_); }
        const std::type_info &Type() const noexcept override { return typeid(T); }
    };

    Base *ptr_ = nullptr;

public:
    Any() noexcept = default;

    Any(const Any &other) : ptr_(other.ptr_ ? other.ptr_->Clone() : nullptr) {}

    Any(Any &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    template <typename T, typename = std::enable_if_t<!std::is_same<std::decay_t<T>, Any>::value>>
    Any(T &&value) : ptr_(new Derived<std::decay_t<T>>(std::forward<T>(value))) {}

    ~Any() { delete ptr_; }

    Any &operator=(const Any &other) {
        if (this != &other) {
            Any(other).Swap(*this);
        }
        return *this;
    }

    Any &operator=(Any &&other) noexcept {
        if (this != &other) {
            other.Swap(*this);
        }
        return *this;
    }

    template <typename T, typename = std::enable_if_t<!std::is_same<std::decay_t<T>, Any>::value>>
    Any &operator=(T &&value) {
        Any(std::forward<T>(value)).Swap(*this);
        return *this;
    }

    void Reset() noexcept {
        delete ptr_;
        ptr_ = nullptr;
    }

    void Swap(Any &other) noexcept { std::swap(ptr_, other.ptr_); }

    bool HasValue() const noexcept { return ptr_ != nullptr; }

    const std::type_info &Type() const noexcept { return ptr_ ? ptr_->Type() : typeid(void); }

    template <typename T>
    friend T *AnyCast(Any *any) noexcept;

    template <typename T>
    friend const T *AnyCast(const Any *any) noexcept;
};

template <typename T>
T *AnyCast(Any *any) noexcept {
    if (!any || any->Type() != typeid(T)) {
        return nullptr;
    }
    return &static_cast<Any::Derived<T> *>(any->ptr_)->value_;
}

template <typename T>
const T *AnyCast(const Any *any) noexcept {
    if (!any || any->Type() != typeid(T)) {
        return nullptr;
    }

    return &static_cast<const Any::Derived<T> *>(any->ptr_)->value_;
}

template <typename T>
T AnyCast(const Any &any) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto p = AnyCast<U>(&any);
    if (!p) {
        throw std::bad_cast();
    }

    return *p;
}

template <typename T>
T AnyCast(Any &any) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto p = AnyCast<U>(&any);
    if (!p) {
        throw std::bad_cast();
    }

    return *p;
}

template <typename T>
T AnyCast(Any &&any) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto p = AnyCast<U>(&any);
    if (!p) {
        throw std::bad_cast();
    }

    return std::move(*p);
}
} // namespace npu::tile_fwk
