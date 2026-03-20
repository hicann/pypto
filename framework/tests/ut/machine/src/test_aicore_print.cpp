/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aicore_print.cpp
 * \brief
 */
#include "gtest/gtest.h"

#include <string>
#include <vector>

// AiCore print helpers
#include "interface/machine/device/tilefwk/aicore_print.h"

using namespace npu::tile_fwk;

namespace {

// Simple mock logger that plugs into LogContext and records prints into a string.
struct MockLogger {
    LogContext ctx{};
    std::string buffer;

    static void PrintInt(LogContext *c, __gm__ const char ** /*fmt*/, int64_t val)
    {
        auto self = reinterpret_cast<MockLogger *>(c);
        self->buffer += std::to_string(val);
    }

    static void PrintFp32(LogContext *c, __gm__ const char ** /*fmt*/, float val)
    {
        auto self = reinterpret_cast<MockLogger *>(c);
        self->buffer += std::to_string(val);
    }

    static void PrintBf16(LogContext *c, __gm__ const char ** /*fmt*/, uint16_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger *>(c);
        self->buffer += std::to_string(DecodeBf16(rawBits));
    }

    static void PrintFp16(LogContext *c, __gm__ const char ** /*fmt*/, uint16_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger *>(c);
        self->buffer += std::to_string(DecodeF16(rawBits));
    }

    static void Print(LogContext *c, __gm__ const char *fmt)
    {
        auto self = reinterpret_cast<MockLogger *>(c);
        if (fmt != nullptr) {
            self->buffer += fmt;
        }
    }

    MockLogger()
    {
        ctx.PrintInt = &MockLogger::PrintInt;
        ctx.PrintFp32 = &MockLogger::PrintFp32;
        ctx.PrintBf16 = &MockLogger::PrintBf16;
        ctx.PrintFp16 = &MockLogger::PrintFp16;
        ctx.Print = &MockLogger::Print;
    }
};

} // namespace

TEST(AiCorePrintUTest, DecodeF16BasicValues)
{
    // 0.0
    EXPECT_FLOAT_EQ(0.0f, DecodeF16(0x0000u));
    // 1.0 -> 0x3C00
    EXPECT_NEAR(1.0f, DecodeF16(0x3C00u), 1e-6f);
    // -2.0 -> 0xC000
    EXPECT_NEAR(-2.0f, DecodeF16(0xC000u), 1e-6f);
}

TEST(AiCorePrintUTest, DecodeBf16BasicValues)
{
    // 0.0
    EXPECT_FLOAT_EQ(0.0f, DecodeBf16(0x0000u));
    // 1.0f -> 0x3F80 for BF16 (high 16 bits of 0x3F800000)
    EXPECT_NEAR(1.0f, DecodeBf16(0x3F80u), 1e-6f);
    // -1.0f -> 0xBF80
    EXPECT_NEAR(-1.0f, DecodeBf16(0xBF80u), 1e-6f);
}

TEST(AiCorePrintUTest, PrintFp16DecodedValue)
{
    MockLogger logger;

    // 1.5 in FP16: sign=0, exp=15+1, mant=0x200 -> bits 0x3E00
    uint16_t bits = 0x3E00u;
    float v = DecodeF16(bits);

    AiCoreLogF(&logger.ctx, "%f", v);

    // Expect that printed value is close to 1.5
    EXPECT_NE(std::string::npos, logger.buffer.find("1.5"))
        << "buffer: " << logger.buffer;
}

TEST(AiCorePrintUTest, PrintBf16DecodedValue)
{
    MockLogger logger;

    // 2.0f -> BF16 0x4000 (high 16 bits of 0x40000000)
    uint16_t bits = 0x4000u;
    float v = DecodeBf16(bits);

    AiCoreLogF(&logger.ctx, "%f", v);

    // Expect that printed value is close to 2.0
    EXPECT_NE(std::string::npos, logger.buffer.find("2"))
        << "buffer: " << logger.buffer;
}

TEST(AiCorePrintUTest, PrintIntAndFloat)
{
    MockLogger logger;
    const char *dummyFmt = "%d";
    __gm__ const char *fmtPtr = dummyFmt;
    __gm__ const char **fmt = &fmtPtr;

    // Directly call through LogContext to make sure function pointers work.
    logger.ctx.PrintInt(&logger.ctx, fmt, 42);
    logger.ctx.PrintFp32(&logger.ctx, fmt, 3.5f);

    // We only care that values are recorded in order.
    EXPECT_NE(std::string::npos, logger.buffer.find("42"));
    EXPECT_NE(std::string::npos, logger.buffer.find("3.5"));
}

TEST(AiCorePrintUTest, AiCorePrintGmTensorFloat)
{
    MockLogger logger;

    float data[3] = {1.0f, 2.0f, 3.5f};
    AiCorePrintGmTensor<float>(&logger.ctx, data, 3);

    // The header text and numbers are all appended into buffer.
    // Just check that each value appears at least once.
    std::string &buf = logger.buffer;
    EXPECT_NE(std::string::npos, buf.find("1"));
    EXPECT_NE(std::string::npos, buf.find("2"));
    EXPECT_NE(std::string::npos, buf.find("3.5"));
}

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)
TEST(AiCorePrintUTest, AiCorePrintShape2D)
{
    MockLogger logger;
    TileOp::Shape<int64_t, int64_t> shape = {3, 5};

    AiCorePrintShape(&logger.ctx, shape);

    std::string &buf = logger.buffer;
    // Expect formatted shape string: shape=[3,5]
    EXPECT_NE(std::string::npos, buf.find("shape=[3,5]"));
}
#endif


