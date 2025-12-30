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
 * \file distributed_op_test_suite.h
 * \brief
 */

#ifndef DISTRIBUTED_OP_TEST_SUITE_H
#define DISTRIBUTED_OP_TEST_SUITE_H

namespace npu::tile_fwk {
namespace Distributed {

struct OpTestParam {
    char group[128]{0};
    int rankSize;
    int rankId;
};

void TestShmemMoeCombine(OpTestParam &testParam);
void TestAllGatherAttentionPostReducescatter(OpTestParam &testParam);
template<typename T>
void TestDynAllGather(OpTestParam &testParam);
template<typename T>
void TestShmemReduceScatter(OpTestParam &testParam);
template<typename T, bool useTwoShot>
void TestShmemAllReduce(OpTestParam &testParam);
void TestShmemMoeDispatch(OpTestParam &testParam);
template<typename T>
void TestShmemAllReduceAddAllReduce(OpTestParam &testParam);
} // namespace Distributed
} // namespace npu::tile_fwk

#endif // DISTRIBUTED_OP_TEST_SUITE_H