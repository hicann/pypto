/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_buffer.cpp
 * \brief
 */

#include "ir/serializer.h"

#include "securec.h"

namespace pto {
namespace serializer {

void MemoryIRBuffer::Write(const std::string &data) {
    buffer_.insert(buffer_.end(), data.begin(), data.end());
}

void MemoryIRBuffer::Write(const std::vector<uint8_t> &data) {
    buffer_.insert(buffer_.end(), data.begin(), data.end());
}

int64_t MemoryIRBuffer::Read(char *buf, int64_t size, bool tryRead) {
    int64_t rest = buffer_.size() - readPos_;
    int64_t readCount = std::min(size, rest);
    memcpy_s(buf, size, &buffer_[readPos_], readCount);
    if (!tryRead) {
        readPos_ += readCount;
    }
    return readCount;
}

int64_t MemoryIRBuffer::ReadSeek(int64_t offset, ReadSeekMode mode) {
    int64_t nextPos = 0;
    switch (mode) {
        case ReadSeekMode::Relative:
            nextPos += readPos_ + offset;
            break;
        case ReadSeekMode::Absolute:
            nextPos = offset;
            break;
        case ReadSeekMode::Position:
            return readPos_;
            break;
        default:
            return ErrorInvalidMode;
            break;
    }
    if (0 <= nextPos && nextPos <= (int64_t)buffer_.size()) {
        readPos_ = nextPos;
        return 0;
    } else {
        return ErrorInvalidOffset;
    }
}

} // namespace serializer
} // namespace pto
