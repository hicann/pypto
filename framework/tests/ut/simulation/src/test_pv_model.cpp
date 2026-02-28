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
 * \file test_pv_model.cpp
 * \brief
 */

#include <fstream>
#include "gtest/gtest.h"
#include "cost_model/simulation_pv/PvModelImpl.h"
#include "cost_model/simulation/pv/PvModelFactory.h"

namespace CostModel {

TEST(PvModelTest, TestAddGlobalAttr) {
    std::string path("./local_function.cpp");
    std::fstream file(path, std::ios::out);

    if (!file.is_open()) {
        std::cerr << "open config file error: " << path << std::endl;
        return;
    }

    std::string code = R"!!!(
[aicore] void TENSOR_Matmul_T_root_3_0(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
}
[aicore] void TENSOR_Matmul_T_root_3_1(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
}
)!!!";

    file << code << std::endl;
    file.close();

    CostModel::PvModelCodegen::AddGlobalAttr(path);

    std::ifstream ifile(path);
    if (!file) {
        std::cerr << "open config file error: " << path << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << ifile.rdbuf();
    std::string actual(buffer.str());
    ifile.close();

    std::string expect = R"!!!(
extern "C" __global__ [aicore] void TENSOR_Matmul_T_root_3_0(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
}
extern "C" __global__ [aicore] void TENSOR_Matmul_T_root_3_1(__gm__ GMTensorInfo* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
}

)!!!";

    ASSERT_EQ(actual, expect);
}

TEST(PvModelTest, TestFactory) {
    auto pv = CostModel::PvModelFactory::Create();
    EXPECT_NE(pv, nullptr);
}

TEST(PvModelTest, TestDynFactory) {
    auto pv = CostModel::PvModelFactory::CreateDyn();
    EXPECT_NE(pv, nullptr);
}

TEST(PvModelTest, TestDynImpl) {
    auto pv = CostModel::PvModelFactory::CreateDyn();
    std::vector<uint8_t> test(16);
    auto addr = pv->CopyToDev(test.data(), test.size());
    EXPECT_NE(addr, nullptr);
    addr = pv->AllocWorkspaceDev(128);
    EXPECT_NE(addr, nullptr);
    std::vector<uint8_t> copy(16);
}

TEST(PvModelTest, TestDynCodegen) {
    std::string org = R"!!!(
#include "TileOpImpl.h"
[aicore] void TENSOR_PATH0_4_0(CoreFuncParam *param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo *oriAddrParam) {
}
)!!!";
    std::string srcFile("TENSOR_PATH0_4_0.cpp");
    std::ofstream ofs(srcFile);
    ofs << org;
    ofs.close();

    std::string dstFile("TENSOR_PATH0_4_0_pvmodel.cpp");
    npu::tile_fwk::CopyFile(srcFile, dstFile);
    PvModelCodegen::AddKernelEntry(dstFile);
    std::ifstream file(dstFile);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    std::string expect = R"!!!(#include "TileOpImpl.h"

extern "C" [aicore] void TENSOR_PATH0_4_0(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam);


extern "C" __global__ [aicore] void PvModelKernelEntry(__gm__ npu::tile_fwk::DynFuncData *funcData, __gm__ uint64_t *opAttrOffset) {
    CoreFuncParam param = {funcData, &funcData->opAttrs[opAttrOffset[0]], funcData->exprTbl};
    TENSOR_PATH0_4_0(&param, funcData->stackWorkSpaceAddr, (__gm__ int64_t *)funcData->startArgs->commContexts, (__gm__ GMTensorInfo*)NULL);
}


[aicore] void TENSOR_PATH0_4_0(CoreFuncParam *param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo *oriAddrParam) {
}
)!!!";
    EXPECT_EQ(expect, content);
}
}
