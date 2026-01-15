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
 * \file test_ir_serializer.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "ir/utils_defop.h"
#include "ir/opcode.h"
#include "ir/serializer.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"

#include "../../../../common_case/ir/test_a_plus_b.h"

using namespace pto;
using namespace pto::serializer;

class IRSerializerTest : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IRSerializerTest, MemoryBuffer) {
    MemoryIRBuffer buf;
    buf.Write("abcd");
    std::vector<uint8_t> data = {'0', '1', '2', '3'};
    buf.Write(data);

    char ch = 0;
    EXPECT_EQ(buf.GetRawBuffer(), "abcd0123");
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(1, buf.Read(&ch, 1));
    EXPECT_EQ('a', ch);

    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(8, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(10, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(0, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(1, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(0, buf.Read(&ch, 1));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(1, buf.Read(&ch, 1));
    EXPECT_EQ('3', ch);
    EXPECT_EQ(0, buf.Read(&ch, 1));
}

TEST_F(IRSerializerTest, BufferRead) {
    {
        MemoryIRBuffer buf;
        buf.Write(" \t\t abcd\n0123");

        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ(" \t\t ", buf.ReadSpace());
        EXPECT_EQ("abcd\n", buf.ReadLine());
        EXPECT_EQ("0123", buf.ReadLine());
        EXPECT_EQ("", buf.ReadLine());
        EXPECT_EQ("", buf.ReadSpace());
    }
    {
        MemoryIRBuffer buf;
        buf.Write("a\r\n\t\t/*ccc\ncc*/12345");
        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ("a\r\n", buf.ReadLine());
        EXPECT_EQ("\t\t", buf.ReadSpace());
        EXPECT_EQ("/*ccc\ncc*/", buf.ReadUntil("*/"));
        EXPECT_EQ("12345", buf.ReadUntil("*/"));
    }
    {
        MemoryIRBuffer buf;
        buf.Write("ab123 123 0x123 0Xabc cde");
        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ("ab123", SourceReadIdentifier(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("123", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("0x123", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("0Xabc", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("cde", SourceReadIdentifier(buf));
    }
}

TEST_F(IRSerializerTest, SourceCppTokenization) {
    MemoryIRBuffer buf;
    buf.Write("r0(r1,r2){la0(la1,la2)}");

    SourceCppIRSerializer serializer;
    std::vector<SourceCppToken> tokenList;
    serializer.Tokenization(buf, tokenList);

    std::vector<std::string> tokenTextList;
    for (auto &token : tokenList) {
        tokenTextList.push_back(token.GetToken());
    }
    std::vector<std::string> resultList = {
        "r0", "(", "r1", ",", "r2", ")", "{", "la0", "(", "la1", ",", "la2", ")", "}"
    };
    EXPECT_EQ(resultList, tokenTextList);
}

TEST_F(IRSerializerTest, SourceCppASTNode) {
    auto root = std::make_shared<SourceCppASTNode>("r0", std::vector<std::string>({"r1", "r2"}));
    auto la = std::make_shared<SourceCppASTNode>("la0", std::vector<std::string>({"la1", "la2"}));
    auto lb = std::make_shared<SourceCppASTNode>("lb0", std::vector<std::string>({"lb1"}));
    auto lc = std::make_shared<SourceCppASTNode>("lc0", std::vector<std::string>());
    lb->push_back(lc);
    root->push_back(la);
    root->push_back(lb);
    MemoryIRBuffer buf;
    buf << "\n" << root;
    EXPECT_EQ(buf.GetRawBuffer(), R"(
r0(r1, r2) {
    la0(la1, la2)
    lb0(lb1) {
        lc0()
    }
}
)");
}

TEST_F(IRSerializerTest, Craft) {
    auto prog = CreateAdd();
    SourceCppIRSerializer serializer;

    MemoryIRBuffer buf;
    serializer.Serialize(buf, prog);

    std::vector<SourceCppToken> tokenList;
    serializer.Tokenization(buf, tokenList);
    EXPECT_EQ('#', tokenList[1].GetToken()[0]);
}
