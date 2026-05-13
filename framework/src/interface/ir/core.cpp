/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/core.h"

#include <sstream>
#include <string>
#include <unordered_set>
#include <mutex>
#include <atomic>

namespace pypto {
namespace ir {

class SpanImpl {
public:
    const std::string filename_; ///< Source filename
    const int beginLine_;        ///< Beginning line number (1-indexed)
    const int beginColumn_;      ///< Beginning column number (1-indexed)
    const int endLine_;          ///< Ending line number (1-indexed), -1 means unknown
    const int endColumn_;        ///< Ending column number (1-indexed), -1 means unknown

    static SpanImpl* Create(const std::string& filename, int beginLine, int beginColumn, int endLine, int endColumn);

    void Put();
    void Get() { refcnt++; }

private:
    std::atomic_int refcnt{1};
    SpanImpl(const std::string& filename, int beginLine, int beginColumn, int endLine, int endColumn)
        : filename_(filename),
          beginLine_(beginLine),
          beginColumn_(beginColumn),
          endLine_(endLine),
          endColumn_(endColumn)
    {}
};

struct SpanImplHash {
    std::size_t operator()(SpanImpl* span) const
    {
        auto combine = [](std::size_t seed, std::size_t hash) -> std::size_t {
            return seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };
        std::size_t h = std::hash<std::string>()(span->filename_);
        h = combine(h, span->beginLine_);
        h = combine(h, span->beginColumn_);
        h = combine(h, span->endLine_);
        h = combine(h, span->endColumn_);
        return h;
    }
};

struct SpanImplEqual {
    bool operator()(SpanImpl* a, SpanImpl* b) const
    {
        return a->filename_ == b->filename_ && a->beginLine_ == b->beginLine_ && a->endLine_ == b->endLine_ &&
               a->beginColumn_ == b->beginColumn_ && a->endColumn_ == b->endColumn_;
    }
};

static std::unordered_set<SpanImpl*, SpanImplHash, SpanImplEqual> registry;
static std::mutex mutex;

SpanImpl* SpanImpl::Create(const std::string& filename, int beginLine, int beginColumn, int endLine, int endColumn)
{
    auto span_ptr = new SpanImpl(filename, beginLine, beginColumn, endLine, endColumn);
    std::unique_lock lock(mutex);
    auto it = registry.find(span_ptr);
    if (it != registry.end()) {
        (*it)->refcnt++;
        lock.unlock();
        delete span_ptr;
        return *it;
    }
    registry.insert(span_ptr);
    return span_ptr;
}

void SpanImpl::Put()
{
    std::unique_lock lock(mutex);
    if (--refcnt == 0) {
        registry.erase(this);
        lock.unlock();
        delete this;
    }
}

Span::Span(const Span& other)
{
    impl_ = other.impl_;
    impl_->Get();
}

Span::Span()
{
    impl_ = Unknown().impl_;
    impl_->Get();
}

Span& Span::operator=(const Span& other)
{
    if (this != &other) {
        impl_->Put();
        impl_ = other.impl_;
        impl_->Get();
    }
    return *this;
}

Span::Span(Span&& other)
{
    impl_ = other.impl_;
    impl_->Get();
    other = Unknown();
}

Span& Span::operator=(Span&& other)
{
    if (this != &other) {
        impl_->Put();
        impl_ = other.impl_;
        impl_->Get();
        other = Unknown();
    }
    return *this;
}

Span::Span(const std::string& file, int beginLine, int beginColumn, int endLine, int endColumn)
{
    impl_ = SpanImpl::Create(file, beginLine, beginColumn, endLine, endColumn);
}

Span::~Span() { impl_->Put(); }

const std::string& Span::Filename() const { return impl_->filename_; }

int Span::BeginLine() const { return impl_->beginLine_; }

int Span::BeginColumn() const { return impl_->beginColumn_; }

int Span::EndLine() const { return impl_->endLine_; }

int Span::EndColumn() const { return impl_->endColumn_; }

std::string Span::ToString() const
{
    std::ostringstream oss;
    oss << impl_->filename_ << ":" << impl_->beginLine_ << ":" << impl_->beginColumn_;
    return oss.str();
}

Span& Span::Unknown()
{
    static Span span = Span("", -1, -1, -1, -1);
    return span;
}

void Span::SetCurrent(const Span& span) { Current() = span; }

Span& Span::Current()
{
    static Span span = Unknown();
    return span;
}

} // namespace ir
} // namespace pypto
