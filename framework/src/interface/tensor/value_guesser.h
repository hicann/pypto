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
 * \file value_guesser.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <tilefwk/symbolic_scalar.h>

#include "tilefwk/error.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
class ValueGuesser {
public:
   ValueGuesser() = default;
   explicit ValueGuesser(NotLessThan value)
       : calculated_(true), maybeInfinitesimal_(false), minimumValue_(value.AsInt64()) {}
   explicit ValueGuesser(NotGreaterThan value)
       : calculated_(true), maybeInfinity_(false), maximumValue_(value.AsInt64()) {}
   explicit ValueGuesser(NotLessThan l, NotGreaterThan r)
       : calculated_(true),
         maybeInfinitesimal_(false),
         maybeInfinity_(false),
         minimumValue_(l.AsInt64()),
         maximumValue_(r.AsInt64()) {}

   static ValueGuesser False() { return ValueGuesser(NotLessThan(0), NotGreaterThan(0)); } // 永远不成立
   static ValueGuesser True() { return ValueGuesser(NotLessThan(1), NotGreaterThan(1)); }  // 永远成立
   static ValueGuesser TrueOrFalse() { return ValueGuesser(NotLessThan(0), NotGreaterThan(1)); }
   static ValueGuesser Any() {
       ValueGuesser result;
       result.calculated_ = true;
       result.maybeInfinitesimal_ = true;
       result.maybeInfinity_ = true;
       return result;
   }

   [[nodiscard]] bool IsCalculated() const { return calculated_; }

   [[nodiscard]] bool IsImmediate() const {
       ASSERT(IsCalculated());
       return !maybeInfinitesimal_ && !maybeInfinity_ && minimumValue_ == maximumValue_;
   }

   [[nodiscard]] bool IsTrue() const {
       ASSERT(IsCalculated());
       return IsImmediate() && minimumValue_ == 1;
   }

   [[nodiscard]] bool IsFalse() const {
       ASSERT(IsCalculated());
       return IsImmediate() && minimumValue_ == 0;
   }

   [[nodiscard]] ValueGuesser operator-() const {
       ASSERT(IsCalculated());
       ValueGuesser result(NotLessThan(0), NotGreaterThan(0));
       result.maybeInfinitesimal_ = maybeInfinity_;
       result.maybeInfinity_ = maybeInfinitesimal_;
       result.minimumValue_ = maximumValue_;
       result.maximumValue_ = minimumValue_;
       return result;
   }

   [[nodiscard]] ValueGuesser operator+() const {
       return *this;
   }

   [[nodiscard]] ValueGuesser operator!() const {
       ASSERT(IsCalculated());
       if (IsImmediate() && minimumValue_ == 0) {
           return True();
       }
       if ((!maybeInfinity_ && maximumValue_ < 0) || (!maybeInfinitesimal_ && minimumValue_ > 0)) {
           return False();
       }
       return TrueOrFalse();
   }

   ValueGuesser operator<(const ValueGuesser &other) const {
       ASSERT(IsCalculated() && other.IsCalculated());
       if (!maybeInfinity_ && !other.maybeInfinitesimal_ && maximumValue_ < other.minimumValue_) {
           return True();
       }
       if (!maybeInfinitesimal_ && !other.maybeInfinity_ && other.maximumValue_ <= minimumValue_) {
           return False();
       }
       return TrueOrFalse();
   }

   ValueGuesser operator<=(const ValueGuesser &other) const {
       ASSERT(IsCalculated() && other.IsCalculated());
       if (!maybeInfinity_ && !other.maybeInfinitesimal_ && maximumValue_ <= other.minimumValue_) {
           return True();
       }
       if (!maybeInfinitesimal_ && !other.maybeInfinity_ && other.maximumValue_ < minimumValue_) {
           return False();
       }
       return TrueOrFalse();
   }

   ValueGuesser operator>(const ValueGuesser &other) const {
       return other < *this;
   }

   ValueGuesser operator>=(const ValueGuesser &other) const {
       return other <= *this;
   }

   ValueGuesser operator==(const ValueGuesser &other) const {
       ASSERT(IsCalculated() && other.IsCalculated());
       if (IsImmediate()) {
           if (other.IsImmediate()) {
               return minimumValue_ == other.minimumValue_ ? True() : False();
           }

           // IsImmediate() && !other.IsImmediate()
           if (!other.maybeInfinitesimal_ && other.minimumValue_ > minimumValue_) {
               return False();
           }
           if (!other.maybeInfinity_ && other.minimumValue_ < minimumValue_) {
               return False();
           }
           return TrueOrFalse();
       }

       // !IsImmediate() && other.IsImmediate()
       if (other.IsImmediate()) {
           if (!maybeInfinitesimal_ && minimumValue_ > other.minimumValue_) {
               return False();
           }
           if (!maybeInfinity_ && minimumValue_ < other.minimumValue_) {
               return False();
           }
           return TrueOrFalse();
       }

       // !IsImmediate() && !other.IsImmediate()
       if (!maybeInfinity_ && !other.maybeInfinitesimal_ && maximumValue_ < other.minimumValue_) {
           return False();
       }
       if (!other.maybeInfinity_ && !maybeInfinitesimal_ && other.maximumValue_ < minimumValue_) {
           return False();
       }
       return TrueOrFalse();
   }

   ValueGuesser operator!=(const ValueGuesser &other) const {
       return !(*this == other);
   }

   ValueGuesser operator+(const ValueGuesser &other) const {
       ASSERT(IsCalculated() && other.IsCalculated());
       ValueGuesser result(NotLessThan(0), NotGreaterThan(0));
       if (maybeInfinitesimal_ || other.maybeInfinitesimal_) {
           result.maybeInfinitesimal_ = true;
       } else {
           result.minimumValue_ = minimumValue_ + other.minimumValue_;
       }
       if (maybeInfinity_ || other.maybeInfinity_) {
           result.maybeInfinity_ = true;
       } else {
           result.maximumValue_ = maximumValue_ + other.maximumValue_;
       }
       return result;
   }

   ValueGuesser operator-(const ValueGuesser &other) const {
       return *this + (-other);
   }

   ValueGuesser operator*(const ValueGuesser &other) const {
       if (!other.IsImmediate()) {
           return Any();
       }
       if (other.minimumValue_ == 0) {
           return ValueGuesser(NotLessThan(0), NotGreaterThan(0));
       }
       ValueGuesser result = *this;
       if (!result.maybeInfinitesimal_) {
           result.minimumValue_ *= other.minimumValue_;
       }
       if (!result.maybeInfinity_) {
           result.maximumValue_ *= other.minimumValue_;
       }
       if (other.minimumValue_ < 0) {
           std::swap(result.minimumValue_, result.maximumValue_);
           std::swap(result.maybeInfinity_, result.maybeInfinitesimal_);
       }
       return result;
   }

   ValueGuesser operator/(const ValueGuesser &other) const {
       if (!other.IsImmediate()) {
           return Any();
       }
       ASSERT(other.minimumValue_ != 0);
       ValueGuesser result = *this;
       if (!result.maybeInfinitesimal_) {
           result.minimumValue_ /= other.minimumValue_;
       }
       if (!result.maybeInfinity_) {
           result.maximumValue_ /= other.minimumValue_;
       }
       if (other.minimumValue_ < 0) {
           std::swap(result.minimumValue_, result.maximumValue_);
           std::swap(result.maybeInfinity_, result.maybeInfinitesimal_);
       }
       return result;
   }

   ValueGuesser operator%(const ValueGuesser &other) const {
       (void)other;
       ALOG_WARN("unsupported guess with % now");
       return {};
   }

   friend ValueGuesser Min(const ValueGuesser &a, const ValueGuesser &b) {
       ASSERT(a.IsCalculated() && b.IsCalculated());
       ValueGuesser result(NotLessThan(0), NotGreaterThan(0));
       if (a.maybeInfinitesimal_ || b.maybeInfinitesimal_) {
           result.maybeInfinitesimal_ = true;
       } else {
           result.minimumValue_ = std::min(a.minimumValue_, b.maximumValue_);
       }
       if (a.maybeInfinity_ && b.maybeInfinity_) {
           result.maybeInfinity_ = true;
       } else if (a.maybeInfinity_) {
           result.maximumValue_ = b.maximumValue_;
       } else if (b.maybeInfinity_) {
           result.maximumValue_ = a.maximumValue_;
       } else {
           result.maximumValue_ = std::min(a.maximumValue_, b.maximumValue_);
       }
       return result;
   }

   friend ValueGuesser Max(const ValueGuesser &a, const ValueGuesser &b) {
       ASSERT(a.IsCalculated() && b.IsCalculated());
       ValueGuesser result(NotLessThan(0), NotGreaterThan(0));
       if (a.maybeInfinity_ || b.maybeInfinity_) {
           result.maybeInfinity_ = true;
       } else {
           result.maximumValue_ = std::max(a.maximumValue_, b.maximumValue_);
       }
       if (a.maybeInfinitesimal_ && b.maybeInfinitesimal_) {
           result.maybeInfinitesimal_ = true;
       } else if (a.maybeInfinitesimal_) {
           result.minimumValue_ = b.minimumValue_;
       } else if (b.maybeInfinitesimal_) {
           result.minimumValue_ = a.minimumValue_;
       } else {
           result.minimumValue_ = std::max(a.minimumValue_, b.minimumValue_);
       }
       return result;
   }

private:
   bool calculated_{false};
   bool maybeInfinitesimal_{true};
   bool maybeInfinity_{true};
   int64_t minimumValue_{0};
   int64_t maximumValue_{0};
};
} // namespace npu::tile_fwk
