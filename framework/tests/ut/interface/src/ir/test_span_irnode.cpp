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
 * \file test_span_irnode.cpp
 * \brief Unit tests for IR core classes (Span and IRNode)
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "ir/core.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

TEST(IRCoreTest, TestSpanBasic)
{
    // Test basic Span functionality
    Span span("test.py", 10, 5, 10, 15);

    ASSERT_EQ(span.filename_, "test.py");
    ASSERT_EQ(span.beginLine_, 10);
    ASSERT_EQ(span.beginColumn_, 5);
    ASSERT_EQ(span.endLine_, 10);
    ASSERT_EQ(span.endColumn_, 15);
}

TEST(IRCoreTest, TestSpanToString)
{
    // Test Span to_string() method
    Span span("test.py", 10, 5, 10, 15);
    std::string str = span.ToString();

    // Should contain meaningful location information
    ASSERT_TRUE(!str.empty());
    ASSERT_TRUE(str.find("test.py") != std::string::npos || str.find("10") != std::string::npos);
}

TEST(IRCoreTest, TestSpanMultiline)
{
    // Test Span spanning multiple lines
    Span span("file.py", 5, 10, 8, 20);

    ASSERT_EQ(span.beginLine_, 5);
    ASSERT_EQ(span.endLine_, 8);
    ASSERT_EQ(span.beginColumn_, 10);
    ASSERT_EQ(span.endColumn_, 20);
}

TEST(IRCoreTest, TestSpanSingleCharacter)
{
    // Test Span for a single character
    Span span("file.py", 5, 10, 5, 11);

    ASSERT_EQ(span.beginLine_, 5);
    ASSERT_EQ(span.endLine_, 5);
    ASSERT_EQ(span.beginColumn_, 10);
    ASSERT_EQ(span.endColumn_, 11);
}

TEST(IRCoreTest, TestIRNodeBasic)
{
    // Test basic IRNode functionality via ConstInt (IRNode is now abstract)
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    ASSERT_EQ(node->span_.filename_, span.filename_);
    ASSERT_EQ(node->span_.beginLine_, span.beginLine_);
}

TEST(IRCoreTest, TestIRNodeTypeName)
{
    // Test IRNode TypeName() method via ConstInt
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    std::string typeName = node->TypeName();
    ASSERT_EQ(typeName, "ConstInt");
}

TEST(IRCoreTest, TestMultipleIRNodes)
{
    // Test creating multiple IRNode instances via ConstInt
    Span span1("file1.py", 1, 0, 1, 10);
    Span span2("file2.py", 5, 5, 5, 15);
    Span span3("file3.py", 10, 0, 12, 0);

    auto node1 = std::make_shared<ConstInt>(1, DataType::INT32, span1);
    auto node2 = std::make_shared<ConstInt>(2, DataType::INT32, span2);
    auto node3 = std::make_shared<ConstInt>(3, DataType::INT32, span3);

    ASSERT_EQ(node1->span_.filename_, "file1.py");
    ASSERT_EQ(node2->span_.filename_, "file2.py");
    ASSERT_EQ(node3->span_.filename_, "file3.py");
}

TEST(IRCoreTest, TestSpanCopy)
{
    // Test copying Span objects
    Span original("test.py", 10, 5, 10, 15);
    Span copy = original;

    ASSERT_EQ(copy.filename_, original.filename_);
    ASSERT_EQ(copy.beginLine_, original.beginLine_);
    ASSERT_EQ(copy.beginColumn_, original.beginColumn_);
    ASSERT_EQ(copy.endLine_, original.endLine_);
    ASSERT_EQ(copy.endColumn_, original.endColumn_);
}

TEST(IRCoreTest, TestSpanComparison)
{
    // Test comparing Span objects
    Span span1("test.py", 10, 5, 10, 15);
    Span span2("test.py", 10, 5, 10, 15);
    Span span3("test.py", 20, 5, 20, 15);

    // Same location
    ASSERT_EQ(span1.filename_, span2.filename_);
    ASSERT_EQ(span1.beginLine_, span2.beginLine_);

    // Different location
    ASSERT_NE(span1.beginLine_, span3.beginLine_);
}

TEST(IRCoreTest, TestIRNodeSharedPtr)
{
    // Test IRNode with shared_ptr via ConstInt
    Span span("test.py", 1, 0, 1, 10);
    auto node = std::make_shared<ConstInt>(42, DataType::INT32, span);

    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->TypeName(), "ConstInt");
    ASSERT_EQ(node->span_.filename_, "test.py");
}

TEST(IRCoreTest, TestSpanZeroPosition)
{
    // Test Span with zero line and column
    Span span("test.py", 0, 0, 0, 1);

    ASSERT_EQ(span.beginLine_, 0);
    ASSERT_EQ(span.beginColumn_, 0);
}

TEST(IRCoreTest, TestSpanLargePosition)
{
    // Test Span with large line and column numbers
    Span span("test.py", 10000, 500, 10001, 600);

    ASSERT_EQ(span.beginLine_, 10000);
    ASSERT_EQ(span.beginColumn_, 500);
    ASSERT_EQ(span.endLine_, 10001);
    ASSERT_EQ(span.endColumn_, 600);
}

} // namespace ir
} // namespace pypto
