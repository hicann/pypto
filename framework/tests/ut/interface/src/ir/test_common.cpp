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
 * \file test_common.cpp
 * \brief Unit tests for core common definitions
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <string>

#include "core/common.h"

namespace pypto {

// ============================================================================
// Version Information Tests
// ============================================================================

TEST(CoreCommonTest, TestVersionMajor)
{
    // Test that PYPTO_VERSION_MAJOR is defined and has expected value
    ASSERT_EQ(PYPTO_VERSION_MAJOR, 0);
}

TEST(CoreCommonTest, TestVersionMinor)
{
    // Test that PYPTO_VERSION_MINOR is defined and has expected value
    ASSERT_EQ(PYPTO_VERSION_MINOR, 1);
}

TEST(CoreCommonTest, TestVersionPatch)
{
    // Test that PYPTO_VERSION_PATCH is defined and has expected value
    ASSERT_EQ(PYPTO_VERSION_PATCH, 0);
}

TEST(CoreCommonTest, TestVersionNumbers)
{
    // Test that version numbers are non-negative
    ASSERT_GE(PYPTO_VERSION_MAJOR, 0);
    ASSERT_GE(PYPTO_VERSION_MINOR, 0);
    ASSERT_GE(PYPTO_VERSION_PATCH, 0);
}

// ============================================================================
// IR Constants Tests
// ============================================================================

TEST(CoreCommonTest, TestDynamicDimValue)
{
    // Test that kDynamicDim has the expected value
    ASSERT_EQ(kDynamicDim, -1);
}

TEST(CoreCommonTest, TestDynamicDimType)
{
    // Test that kDynamicDim is of type int64_t
    static_assert(std::is_same<decltype(kDynamicDim), const int64_t>::value, "kDynamicDim should be int64_t");
    ASSERT_TRUE(true); // Compile-time check
}

TEST(CoreCommonTest, TestDynamicDimUsage)
{
    // Test using kDynamicDim in comparisons
    int64_t dim = kDynamicDim;
    ASSERT_EQ(dim, -1);
    ASSERT_LT(dim, 0);
    ASSERT_NE(dim, 0);
}

TEST(CoreCommonTest, TestDynamicDimInArray)
{
    // Test using kDynamicDim in array/vector
    int64_t dims[] = {10, 20, kDynamicDim, 30};
    ASSERT_EQ(dims[2], -1);
    ASSERT_EQ(dims[2], kDynamicDim);
}

// ============================================================================
// Nanobind Module Configuration Tests
// ============================================================================

TEST(CoreCommonTest, TestNanobindModuleDoc)
{
    // Test that PYPTO_NANOBIND_MODULE_DOC is defined
    std::string doc = PYPTO_NANOBIND_MODULE_DOC;
    ASSERT_FALSE(doc.empty());
    ASSERT_EQ(doc, "PyPTO core library");
}

TEST(CoreCommonTest, TestNanobindModuleDocLength)
{
    // Test that module doc has reasonable length
    std::string doc = PYPTO_NANOBIND_MODULE_DOC;
    ASSERT_GT(doc.length(), 0);
    ASSERT_LT(doc.length(), 1000); // Reasonable upper bound
}

// ============================================================================
// Compiler Hints and Attributes Tests
// ============================================================================

TEST(CoreCommonTest, TestAlwaysInlineMacro)
{
    // Test that PYPTO_ALWAYS_INLINE is defined
    // This is a compile-time test - if it compiles, the macro exists
    auto testFunc = []() PYPTO_ALWAYS_INLINE { return 42; };
    ASSERT_EQ(testFunc(), 42);
}

TEST(CoreCommonTest, TestUnusedMacro)
{
    // Test that PYPTO_UNUSED is defined
    // This is a compile-time test - if it compiles, the macro exists
    PYPTO_UNUSED int unusedVar = 100;
    ASSERT_TRUE(true); // If we get here, the macro worked
}

TEST(CoreCommonTest, TestStrConcatMacro)
{
// Test PYPTO_STR_CONCAT macro
#define TEST_PREFIX test_
#define TEST_SUFFIX value

    // This should concatenate to test_value
    int PYPTO_STR_CONCAT(TEST_PREFIX, TEST_SUFFIX) = 42;
    ASSERT_EQ(test_value, 42);

#undef TEST_PREFIX
#undef TEST_SUFFIX
}

TEST(CoreCommonTest, TestStrConcatWithNumbers)
{
// Test PYPTO_STR_CONCAT with numbers
#define VAR var_
    int PYPTO_STR_CONCAT(VAR, 1) = 10;
    int PYPTO_STR_CONCAT(VAR, 2) = 20;
    int PYPTO_STR_CONCAT(VAR, 3) = 30;

    ASSERT_EQ(var_1, 10);
    ASSERT_EQ(var_2, 20);
    ASSERT_EQ(var_3, 30);

#undef VAR
}

// ============================================================================
// Macro Expansion Tests
// ============================================================================

TEST(CoreCommonTest, TestStrConcatImplMacro)
{
    // Test PYPTO_STR_CONCAT_IMPL (internal implementation)
    // Note: PYPTO_STR_CONCAT_IMPL does not expand macros, it directly concatenates tokens
    int PYPTO_STR_CONCAT_IMPL(foo, bar) = 123;
    ASSERT_EQ(foobar, 123);
}

TEST(CoreCommonTest, TestNestedMacroExpansion)
{
// Test nested macro expansion with PYPTO_STR_CONCAT
#define OUTER outer_
#define INNER inner

    int PYPTO_STR_CONCAT(PYPTO_STR_CONCAT(OUTER, INNER), _value) = 456;
    ASSERT_EQ(outer_inner_value, 456);

#undef OUTER
#undef INNER
}

// ============================================================================
// Constant Value Range Tests
// ============================================================================

TEST(CoreCommonTest, TestDynamicDimRange)
{
    // Test that kDynamicDim is in valid range for int64_t
    ASSERT_GE(kDynamicDim, std::numeric_limits<int64_t>::min());
    ASSERT_LE(kDynamicDim, std::numeric_limits<int64_t>::max());
}

TEST(CoreCommonTest, TestDynamicDimDistinctFromZero)
{
    // Test that kDynamicDim is distinct from 0 (valid dimension)
    ASSERT_NE(kDynamicDim, 0);
}

TEST(CoreCommonTest, TestDynamicDimDistinctFromPositive)
{
    // Test that kDynamicDim is distinct from positive dimensions
    ASSERT_NE(kDynamicDim, 1);
    ASSERT_NE(kDynamicDim, 10);
    ASSERT_NE(kDynamicDim, 100);
    ASSERT_NE(kDynamicDim, 1000);
}

// ============================================================================
// Practical Usage Tests
// ============================================================================

TEST(CoreCommonTest, TestDynamicDimInConditional)
{
    // Test using kDynamicDim in conditional logic
    auto isDynamic = [](int64_t dim) { return dim == kDynamicDim; };

    ASSERT_TRUE(isDynamic(kDynamicDim));
    ASSERT_TRUE(isDynamic(-1));
    ASSERT_FALSE(isDynamic(0));
    ASSERT_FALSE(isDynamic(10));
}

TEST(CoreCommonTest, TestDynamicDimInSwitch)
{
    // Test using kDynamicDim in switch statement
    auto classifyDim = [](int64_t dim) -> std::string {
        if (dim == kDynamicDim) {
            return "dynamic";
        } else if (dim == 0) {
            return "zero";
        } else if (dim > 0) {
            return "positive";
        } else {
            return "negative";
        }
    };

    ASSERT_EQ(classifyDim(kDynamicDim), "dynamic");
    ASSERT_EQ(classifyDim(-1), "dynamic");
    ASSERT_EQ(classifyDim(0), "zero");
    ASSERT_EQ(classifyDim(10), "positive");
    ASSERT_EQ(classifyDim(-5), "negative");
}

TEST(CoreCommonTest, TestVersionComparison)
{
    // Test version comparison logic
    auto versionCode = PYPTO_VERSION_MAJOR * 10000 + PYPTO_VERSION_MINOR * 100 + PYPTO_VERSION_PATCH;

    ASSERT_EQ(versionCode, 100); // 0.1.0 = 100
}

TEST(CoreCommonTest, TestVersionString)
{
// Test constructing version string
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

    std::string version = std::string(TOSTRING(PYPTO_VERSION_MAJOR)) + "." +
                          std::string(TOSTRING(PYPTO_VERSION_MINOR)) + "." + std::string(TOSTRING(PYPTO_VERSION_PATCH));

    ASSERT_EQ(version, "0.1.0");

#undef STRINGIFY
#undef TOSTRING
}

// ============================================================================
// Attribute Application Tests
// ============================================================================

TEST(CoreCommonTest, TestAlwaysInlineFunction)
{
    // Test function with PYPTO_ALWAYS_INLINE attribute
    auto inlineFunc = []() PYPTO_ALWAYS_INLINE -> int { return 42; };

    int result = inlineFunc();
    ASSERT_EQ(result, 42);
}

TEST(CoreCommonTest, TestUnusedVariable)
{
    // Test that PYPTO_UNUSED suppresses warnings
    PYPTO_UNUSED int x = 10;
    PYPTO_UNUSED double y = 3.14;
    PYPTO_UNUSED bool z = true;

    // If we get here without warnings, the attribute works
    ASSERT_TRUE(true);
}

TEST(CoreCommonTest, TestUnusedParameter)
{
    // Test PYPTO_UNUSED with function parameters
    auto func = [](PYPTO_UNUSED int param) { return 100; };

    int result = func(42);
    ASSERT_EQ(result, 100);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST(CoreCommonTest, TestDynamicDimArithmetic)
{
    // Test arithmetic with kDynamicDim
    int64_t result1 = kDynamicDim + 1;
    int64_t result2 = kDynamicDim * 2;
    int64_t result3 = -kDynamicDim;

    ASSERT_EQ(result1, 0);
    ASSERT_EQ(result2, -2);
    ASSERT_EQ(result3, 1);
}

TEST(CoreCommonTest, TestDynamicDimComparison)
{
    // Test comparison operations with kDynamicDim
    ASSERT_TRUE(kDynamicDim < 0);
    ASSERT_TRUE(kDynamicDim <= -1);
    ASSERT_TRUE(kDynamicDim <= 0);
    ASSERT_FALSE(kDynamicDim > 0);
    ASSERT_FALSE(kDynamicDim >= 0);
}

TEST(CoreCommonTest, TestMultipleDynamicDims)
{
    // Test multiple dynamic dimensions
    int64_t dims[] = {kDynamicDim, kDynamicDim, kDynamicDim};

    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(dims[i], kDynamicDim);
    }
}

TEST(CoreCommonTest, TestMixedDimensions)
{
    // Test mixed static and dynamic dimensions
    int64_t dims[] = {10, kDynamicDim, 20, kDynamicDim, 30};

    ASSERT_EQ(dims[0], 10);
    ASSERT_EQ(dims[1], kDynamicDim);
    ASSERT_EQ(dims[2], 20);
    ASSERT_EQ(dims[3], kDynamicDim);
    ASSERT_EQ(dims[4], 30);
}

// ============================================================================
// Namespace Tests
// ============================================================================

TEST(CoreCommonTest, TestNamespaceScope)
{
    // Test that constants are in pypto namespace
    int64_t dim = pypto::kDynamicDim;
    ASSERT_EQ(dim, -1);
}

TEST(CoreCommonTest, TestConstantAccessibility)
{
    // Test that constants are accessible without namespace prefix (when using namespace)
    using namespace pypto;
    int64_t dim = kDynamicDim;
    ASSERT_EQ(dim, -1);
}

} // namespace pypto
