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
 * \file serializer_base.h
 * \brief
 */

#pragma once

#include "ir/function.h"
#include "ir/program.h"

namespace pto {
namespace serializer {

class IRBuffer {
public:
    virtual ~IRBuffer() = default;
    virtual void Write(const std::string &data) = 0;
    virtual void Write(const std::vector<uint8_t> &data) = 0;

    enum ErrorCode {
        OK = 0,
        ErrorBufferEnd = -1,
        ErrorInvalidMode = -2,
        ErrorInvalidOffset = -3,
    };
    virtual int64_t Read(char *buf, int64_t size, bool tryRead = false) = 0;

    /* Only affect read seek*/
    enum class ReadSeekMode {
        Relative,
        Absolute,
        Position,
    };
    virtual int64_t ReadSeek(int64_t offset, ReadSeekMode mode) = 0;

    IRBuffer &operator<<(const std::string &data);
    IRBuffer &operator<<(const std::vector<std::string> &data);

    std::string ReadSpace();
    std::string ReadLine();
    std::string ReadUntil(const std::string &end);
};

class MemoryIRBuffer : public IRBuffer {
public:
    virtual void Write(const std::string &data);
    virtual void Write(const std::vector<uint8_t> &data);
    virtual int64_t Read(char *buf, int64_t size, bool tryRead = false);
    virtual int64_t ReadSeek(int64_t offset, ReadSeekMode mode);

    std::string &GetRawBuffer() { return buffer_; }
private:
    std::string buffer_;
    int64_t readPos_{0};
};

class IRSerializer {
public:
    enum SerializerKind {
        /* assemble style */
        SOURCE_ASM,
        /* cplusplus style which is for codegen. Serializer is not responsible for whether the serialized code is further compiled
        * as AscendC or pure C++ code. */
        SOURCE_CPP,
    };
public:
    IRSerializer(SerializerKind kind) : kind_(kind) {}

    virtual ~IRSerializer() = default;
    virtual void Serialize(IRBuffer &buffer, const ProgramModulePtr &module) = 0;
    virtual ProgramModulePtr Deserialize(IRBuffer &buffer) = 0;
private:
    SerializerKind kind_;
};

struct SerializeUtils {
private:
    static inline void Append(std::vector<std::string> &holder, bool data) {
        holder.push_back(data ? "TRUE" : "FALSE");
    }
    static inline void Append(std::vector<std::string> &holder, int data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, long data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, long long data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, unsigned data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, unsigned long data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, unsigned long long data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, float data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, double data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, long double data) { holder.push_back(std::to_string(data)); }
    static inline void Append(std::vector<std::string> &holder, const std::string &data) {
        holder.push_back(data);
    }
    static inline void Append(std::vector<std::string> &holder, const char *data) {
        holder.push_back(data);
    }
    template<typename T>
    static inline void Append(std::vector<std::string> &holder, const std::vector<T> &data) {
        for (const auto &v : data) {
            Append(holder, v);
        }
    }
    template<typename Ty, typename ...TyArgs>
    static inline void ListAppend(std::vector<std::string> &holder, Ty &&arg, TyArgs... args) {
        Append(holder, arg);
        ListAppend(holder, args...);
    }

    static inline void ListAppend(std::vector<std::string> &holder) {
        (void)holder;
    }
public:
    template<typename ...TyArgs>
    static inline std::vector<std::string> List(TyArgs... args) {
        std::vector<std::string> holder;
        ListAppend(holder, args...);
        return holder;
    }
};

std::string SourceReadIdentifier(IRBuffer &buf);
std::string SourceReadNumber(IRBuffer &buf);

}
}
