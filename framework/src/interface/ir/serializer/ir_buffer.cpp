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

IRBuffer &IRBuffer::operator<<(const std::string &data) {
    Write(data);
    return *this;
}

IRBuffer &IRBuffer::operator<<(const std::vector<std::string> &dataList) {
    for (size_t k = 0; k < dataList.size(); k++) {
        Write(k == 0 ? "" : ", ");
        Write(dataList[k]);
    }
    return *this;
}

std::string IRBuffer::ReadSpace() {
    char ch;
    int readCount = Read(&ch, 1);
    std::string space;
    while (readCount == 1 && std::isspace(ch)) {
        space.push_back(ch);
        readCount = Read(&ch, 1);
    }
    if (readCount == 1) {
        // ch is not space
        ReadSeek(-1, ReadSeekMode::Relative);
    }
    return space;
}

std::string IRBuffer::ReadLine() {
    char ch;
    int readCount = Read(&ch, 1);
    std::string line;
    while (readCount == 1 && ch != '\n') {
        line.push_back(ch);
        readCount = Read(&ch, 1);
    }
    if (readCount == 1) {
        // ch must be '\n'
        line.push_back(ch);
    }
    return line;
}

#define READ_STACK_SIZE 4096
std::string IRBuffer::ReadUntil(const std::string &end) {
    char staticBuf[READ_STACK_SIZE + 1];
    std::string dynamicBuf;
    char *readBuf = nullptr;
    if (end.size() < READ_STACK_SIZE) {
        readBuf = staticBuf;
    } else {
        dynamicBuf.resize(end.size() + 1);
        readBuf = &dynamicBuf[0];
    }

    std::string text;
    int endSize = end.size();
    readBuf[endSize] = 0;
    int readCount = Read(readBuf, endSize, true);
    char ch;
    while (readCount == endSize && readBuf != end) {
        Read(&ch, 1);
        text.push_back(ch);
        readCount = Read(readBuf, end.size(), true);
    }
    if (readCount == endSize) {
        // readBuf == end
        ReadSeek(endSize, ReadSeekMode::Relative);
        text.insert(text.end(), readBuf, readBuf + readCount);
    } else {
        text.insert(text.end(), readBuf, readBuf + readCount);
    }
    return text;
}

std::string SourceReadIdentifier(IRBuffer &buf) {
    char ch = 0;
    int readCount = buf.Read(&ch, 1);
    std::string identifier;
    if (std::isalpha(ch) || ch == '_') {
        identifier.push_back(ch);
        readCount = buf.Read(&ch, 1);
        while (readCount == 1 && (std::isalnum(ch) || ch == '_')) {
            identifier.push_back(ch);
            readCount = buf.Read(&ch, 1);
        }
        if (readCount == 1) {
            // ch is not identifier
            buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Relative);
        }
    } else {
        buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Relative);
    }
    return identifier;
}

std::string SourceReadNumber(IRBuffer &buf) {
#define HEX_HEAD 2
    char head[HEX_HEAD] = {0};
    int readCount = buf.Read(head, HEX_HEAD, true);

    std::string number;
    if (readCount != 0 && std::isdigit(head[0])) {
        if (head[0] == '0' && (head[1] == 'x' || head[1] == 'X')) {
            // hex
            buf.ReadSeek(HEX_HEAD, IRBuffer::ReadSeekMode::Relative);
            number.insert(number.end(), &head[0], &head[HEX_HEAD]);

            char ch = 0;
            readCount = buf.Read(&ch, 1);
            while (readCount == 1 && std::isxdigit(ch)) {
                number.push_back(ch);
                readCount = buf.Read(&ch, 1);
            }
        } else {
            // dec
            buf.ReadSeek(1, IRBuffer::ReadSeekMode::Relative);
            number.insert(number.end(), &head[0], &head[1]);

            char ch = 0;
            readCount = buf.Read(&ch, 1);
            while (readCount == 1 && std::isdigit(ch)) {
                number.push_back(ch);
                readCount = buf.Read(&ch, 1);
            }
        }
        if (readCount == 1) {
            // ch is not number
            buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Relative);
        }
    }
    return number;
}


} // namespace serializer
} // namespace pto
