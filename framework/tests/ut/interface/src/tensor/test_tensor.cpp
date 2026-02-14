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
 * \file test_tensor.cpp
 * \brief
 */

#include <cstddef>
#include "gtest/gtest.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestTensor : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTensor, AssignWithData) {
    std::vector<int64_t> tshape = {100, 100};
    Tensor a(DT_FP32, tshape, "A");
    Tensor b(DT_FP32, tshape, "B");
    auto ptr1 = std::make_unique<uint8_t>(0);
    auto ptr2 = std::make_unique<uint8_t>(0);

    auto reset = [&a, &b]() {
        a.SetData(nullptr);
        b.SetData(nullptr);
    };

    {
        reset();
        b = a;
        EXPECT_EQ(b.GetData(), nullptr);
        EXPECT_EQ(a.GetData(), nullptr);
    }

    {
        reset();
        b.SetData(ptr1.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr1.get());
        EXPECT_EQ(a.GetData(), nullptr);
    }

    {
        reset();
        a.SetData(ptr1.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr1.get());
        EXPECT_EQ(a.GetData(), ptr1.get());
    }

    {
        reset();
        a.SetData(ptr2.get());
        b.SetData(ptr2.get());
        b = a;
        EXPECT_EQ(b.GetData(), ptr2.get());
        EXPECT_EQ(a.GetData(), ptr2.get());
    }
}

TEST_F(TestTensor, AssignWithData2) {
    std::vector<int64_t> tshape = {100, 100};
    Tensor b(DT_FP32, tshape, "B");
    auto ptr1 = std::make_unique<uint8_t>(0);
    auto ptr2 = std::make_unique<uint8_t>(0);

    auto reset = [&b]() {
        b.SetData(nullptr);
    };

    {
        Tensor a(DT_FP32, tshape, "A");
        reset();
        b = std::move(a);
        EXPECT_EQ(b.GetData(), nullptr);
    }

    {
        Tensor a(DT_FP32, tshape, "A");
        reset();
        b.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }

    {
        Tensor a(DT_FP32, tshape, "A");
        reset();
        a.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }

    {
        Tensor a(DT_FP32, tshape, "A");
        reset();
        a.SetData(ptr1.get());
        b.SetData(ptr1.get());
        b = std::move(a);
        EXPECT_EQ(b.GetData(), ptr1.get());
    }
}

TEST_F(TestTensor, GetShapeTest) {
    std::vector<int64_t> tshape = {16, 32, 16, 64};
    std::vector<int64_t> kshape = {};
    Tensor a(DT_FP32, tshape, "A");
    Tensor b(DT_FP32, kshape, "B");

    auto ashape = a.GetShape(1);
    EXPECT_EQ(ashape, 32);
    ashape = a.GetShape(-4);
    EXPECT_EQ(ashape, 16);

    auto validShape = a.GetValidShape();
    for (size_t i = 0; i < tshape.size(); i++) {
        EXPECT_EQ(validShape[i].Concrete(), tshape[i]);
    }
}

TEST_F(TestTensor, GetCachePolicyTest) {
    std::vector<int64_t> tshape = {4, 4};
    Tensor t1(DT_FP32, tshape, "T1");

    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));

    t1.SetCachePolicy(CachePolicy::PREFETCH, true);
    EXPECT_TRUE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));

    t1.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    EXPECT_TRUE(t1.GetCachePolicy(CachePolicy::PREFETCH));
    EXPECT_FALSE(t1.GetCachePolicy(CachePolicy::NONE_CACHEABLE));
}

TEST_F(TestTensor, RawTensorNegative) {
    RawTensor t(DT_INT8, {});
    t.SetRefCount(-2);
    t.AddRefCount(1);
    EXPECT_EQ(t.GetRefCount(), -1);
}