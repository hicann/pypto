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
 * \file test_aicore_runtime_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "execute/aicore_runtime_manager.h"
#include "securec.h"
#include "graph/buffer.h"
#include "graph/utils/attr_utils.h"

using namespace ge;
namespace npu::tile_fwk {
extern ge::graphStatus TileFwkHiddenInputsFunc(const ge::OpDescPtr &op_desc, std::vector<void *> &contexts);

class AicoreRuntimeManagerUnitTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(AicoreRuntimeManagerUnitTest, test_tile_fwk_hidden_input_for_ge) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  DevAscendProgram args;
  args.devArgs.nrAic = 0;
  args.devArgs.nrAiv = 0;
  args.devArgs.nrAicpu = 0;
  args.devArgs.nrValidAic = 0;
  args.devArgs.opaque = 0;
  args.devArgs.devQueueAddr = 0;
  args.devArgs.sharedBuffer = 0;
  args.devArgs.coreRegAddr = 0;
  args.devArgs.corePmuRegAddr = 0;
  args.devArgs.corePmuAddr = 0;
  args.devArgs.pmuEventAddr = 0;
  args.devArgs.taskType = 0;
  args.devArgs.machineConfig = 0;
  args.devArgs.taskId = 0;
  args.devArgs.taskData = 0;
  args.workspaceSize = 0;

  std::vector<uint8_t> op_binary_bin(sizeof(args), 0);
  memcpy_s(op_binary_bin.data(), sizeof(args), &args, sizeof(args));
  ge::Buffer op_binary_buffer =
      ge::Buffer::CopyFrom(reinterpret_cast<uint8_t*>(op_binary_bin.data()), op_binary_bin.size());
  ge::AttrUtils::SetBytes(op_desc_ptr, "_subkernel_op_binary", op_binary_buffer);
  ge::AttrUtils::SetInt(op_desc_ptr, "tvm_blockdim", 24);
  ge::AttrUtils::SetInt(op_desc_ptr, "_tile_fwk_op_config_key", 123);
  op_desc_ptr->SetWorkspaceBytes({100});
  std::vector<void *> contexts;
  TileFwkHiddenInputsFunc(op_desc_ptr, contexts);
  EXPECT_EQ(contexts.size(), 1);
  TileFwkHiddenInputsFunc(op_desc_ptr, contexts);
  EXPECT_EQ(contexts.size(), 2);
}

TEST_F(AicoreRuntimeManagerUnitTest, test_tile_fwk_hidden_input_for_aclnn) {
  OpDescPtr op_desc_ptr = make_shared<ge::OpDesc>("sigmod", "sigmod");
  DevAscendProgram args;
  args.devArgs.nrAic = 0;
  args.devArgs.nrAiv = 0;
  args.devArgs.nrAicpu = 0;
  args.devArgs.nrValidAic = 0;
  args.devArgs.opaque = 0;
  args.devArgs.devQueueAddr = 0;
  args.devArgs.sharedBuffer = 0;
  args.devArgs.coreRegAddr = 0;
  args.devArgs.corePmuRegAddr = 0;
  args.devArgs.corePmuAddr = 0;
  args.devArgs.pmuEventAddr = 0;
  args.devArgs.taskType = 0;
  args.devArgs.machineConfig = 0;
  args.devArgs.taskId = 0;
  args.devArgs.taskData = 0;
  args.workspaceSize = 0;

  std::vector<uint8_t> op_binary_bin(sizeof(args), 0);
  memcpy_s(op_binary_bin.data(), sizeof(args), &args, sizeof(args));
  int64_t *dev_args = AicoreRtManager::Instance().TileFwkHiddenInput(op_binary_bin, 234, 24, 100);
  EXPECT_NE(dev_args, nullptr);
}
}
