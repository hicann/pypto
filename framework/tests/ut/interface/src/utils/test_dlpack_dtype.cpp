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
 * \file test_dlpack_dtype.cpp
 * \brief Unit tests for DlpackDtypeToDataType.
 */

#include "gtest/gtest.h"
#include "interface/inner/dlpack_dtype.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestDlpackDtype : public testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TestDlpackDtype, IntTypes) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLInt, 8, 1, &out));
    EXPECT_EQ(out, DT_INT8);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLInt, 16, 1, &out));
    EXPECT_EQ(out, DT_INT16);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLInt, 32, 1, &out));
    EXPECT_EQ(out, DT_INT32);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLInt, 64, 1, &out));
    EXPECT_EQ(out, DT_INT64);
}

TEST_F(TestDlpackDtype, UintTypes) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLUInt, 8, 1, &out));
    EXPECT_EQ(out, DT_UINT8);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLUInt, 16, 1, &out));
    EXPECT_EQ(out, DT_UINT16);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLUInt, 32, 1, &out));
    EXPECT_EQ(out, DT_UINT32);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLUInt, 64, 1, &out));
    EXPECT_EQ(out, DT_UINT64);
}

TEST_F(TestDlpackDtype, FloatTypes) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLFloat, 16, 1, &out));
    EXPECT_EQ(out, DT_FP16);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLFloat, 32, 1, &out));
    EXPECT_EQ(out, DT_FP32);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLFloat, 64, 1, &out));
    EXPECT_EQ(out, DT_DOUBLE);
}

TEST_F(TestDlpackDtype, Bfloat16) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLBfloat, 16, 1, &out));
    EXPECT_EQ(out, DT_BF16);
}

TEST_F(TestDlpackDtype, Bool) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLBool, 8, 1, &out));
    EXPECT_EQ(out, DT_BOOL);
}

TEST_F(TestDlpackDtype, FP8) {
    DataType out;
    EXPECT_TRUE(DlpackDtypeToDataType(kDLFloat8_e5m2, 8, 1, &out));
    EXPECT_EQ(out, DT_FP8E5M2);
    EXPECT_TRUE(DlpackDtypeToDataType(kDLFloat8_e4m3, 8, 1, &out));
    EXPECT_EQ(out, DT_FP8E4M3);
}

TEST_F(TestDlpackDtype, UnsupportedLanes) {
    DataType out;
    EXPECT_FALSE(DlpackDtypeToDataType(kDLInt, 32, 4, &out));
}

TEST_F(TestDlpackDtype, UnsupportedCode) {
    DataType out;
    EXPECT_FALSE(DlpackDtypeToDataType(kDLOpaqueHandle, 8, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLComplex, 64, 1, &out));
}

TEST_F(TestDlpackDtype, UnsupportedBits) {
    DataType out;
    EXPECT_FALSE(DlpackDtypeToDataType(kDLInt, 4, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLInt, 128, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLFloat, 8, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLBfloat, 32, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLBool, 32, 1, &out));
    EXPECT_FALSE(DlpackDtypeToDataType(kDLFloat8_e5m2, 16, 1, &out));
}
