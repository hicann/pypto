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
 * \file test_distributed.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "test_common.h"
#include "distributed_op_test_suite.h"
#include "distributed_test_framework.h"

namespace npu::tile_fwk::Distributed {

struct OpMetaData {
    explicit OpMetaData(const nlohmann::json &testData)
        : testData_(testData) {}
    nlohmann::json testData_;
};

// 算子注册表
struct DisOpRegistry {
    std::unordered_map<std::string, std::function<void(OpTestParam&, const std::string&)>> registry;
    template <typename TFunc>
    void RegisterOp(const std::string& opName, TFunc func)
    {
        registry[opName] = [func](OpTestParam &testParam, const std::string &dtype)
        {
            if (dtype == "int32") func.template operator()<int32_t>(testParam);
            else if (dtype == "float16") func.template operator()<float16>(testParam);
            else if (dtype == "bfloat16") func.template operator()<bfloat16>(testParam);
            else if (dtype == "float32") func.template operator()<float>(testParam);
            else FAIL() << "Unsupported dtype: " << dtype;
        };
    }
    void Run(const std::string &opName, OpTestParam &testParam, const std::string &dtype)
    {
        if (!registry.count(opName)) {
            FAIL() << "Unsupported op: " << opName;
        }
        registry[opName](testParam, dtype);
    }
};


DisOpRegistry& GetRegistry()
{
    static DisOpRegistry registry;
    return registry;
}


// 各模板算子的Func
struct AllgatherFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestDynAllGather<T>(testParam);
    }
};

struct ReducescatterFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemReduceScatter<T>(testParam);
    }
};

struct AllreduceFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemAllReduce<T>(testParam);
    }
};

struct Allreduce_Add_AllreduceFunc {
    template <typename T>
    void operator()(OpTestParam &testParam) const
    {
        Distributed::TestShmemAllReduceAddAllReduce<T>(testParam);
    }
};


// 注册所有算子
void GegisterAllOps()
{
    auto& reg = GetRegistry();
    reg.RegisterOp("Allgather", AllgatherFunc{});  // 模板算子
    reg.RegisterOp("Reducescatter", ReducescatterFunc{});
    reg.RegisterOp("Allreduce", AllreduceFunc{});
    reg.RegisterOp("Allreduce_Add_Allreduce", Allreduce_Add_AllreduceFunc{});
    reg.registry["MoeCombine"] = [](OpTestParam &testParam, const std::string&) {
        Distributed::TestShmemMoeCombine(testParam);
    };
    reg.registry["Allgather_AttnPost_Reducescatter"] = [](OpTestParam &testParam, const std::string&) {
        Distributed::TestAllGatherAttentionPostReducescatter(testParam);
    };
    // 后续按照上面格式增加算子
}


template <typename T>
std::vector<T> GetOpMetaData(const std::string &op)
{
    auto caseFile = "../../../framework/tests/st/distributed/ops/test_case/" + op + "_st_test_cases.json";
    std::ifstream jsonFile(caseFile);
    if (!jsonFile.is_open()) {
        std::cerr << "Failed to open JSON file for op " << op << ". "
        << "Please check the path and ensure the file exists: " << caseFile << std::endl;
        return {};
    }
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    std::vector<T> testCaseList;
    for (auto &tc : jsonData.at("test_cases")) {
        testCaseList.emplace_back(tc);
    }
    if (testCaseList.empty()) {
        std::cerr << "No test cases found in json for op: " << op << ". "
        << "Please check the contents of: " << caseFile << std::endl;
    }
    return testCaseList;
}


class DistributedTest : public testing::TestWithParam<OpMetaData> {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase()
    {
        GegisterAllOps();
    }

    void SetUp() override
    {
        Distributed::TestFrameworkInit(testParam, hcomTestParam, physicalDeviceId);
        std::string folderPath = "output/output_" + getTimeStamp() + "_" + std::to_string(physicalDeviceId);
        setenv("TILE_FWK_OUTPUT_DIR", folderPath.c_str(), 0);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        DistributedTestDestroy();
        Distributed::TestFrameworkDestroy(timeout);
    }

    // 暴露超时设置接口
    void SetDestroyTimeout(int32_t destroyTimeout)
    {
        timeout = destroyTimeout;
    }

    // 通用测试入口
    void RunDistributedTestGeneric(const std::string& opName, const nlohmann::json& testData)
    {
        if (!testData.contains("input_tensors") || testData["input_tensors"].empty()) {
            FAIL() << "No input tensors in testData: " << testData.dump();
        }
        std::string dtype = testData["input_tensors"][0]["dtype"];
        GetRegistry().Run(opName, testParam, dtype);
    }

protected:
    void DistributedTestDestroy()
    {
        // 销毁集合通信域
        ASSERT(HcclCommDestroy(hcomTestParam.hcclComm) == 0);
        // 重置设备
        ASSERT(aclrtResetDevice(physicalDeviceId) == 0);
        // 设备去初始化
        ASSERT(aclFinalize() == 0);
    }

    Distributed::OpTestParam testParam;
    Distributed::HcomTestParam hcomTestParam;
    int32_t timeout = 10;
    int physicalDeviceId = 0;
};


INSTANTIATE_TEST_SUITE_P(TestAllgather, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allgather")));
TEST_P(DistributedTest, TestAllgather)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allgather", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestReducescatter, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Reducescatter")));
TEST_P(DistributedTest, TestReducescatter)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Reducescatter", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllreduce, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allreduce")));
TEST_P(DistributedTest, TestAllreduce)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allreduce", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestMoeCombine, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("MoeCombine")));
TEST_P(DistributedTest, TestMoeCombine)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("MoeCombine", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllreduce_Add_Allreduce, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allreduce_Add_Allreduce")));
TEST_P(DistributedTest, TestAllreduce_Add_Allreduce)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allreduce_Add_Allreduce", GetParam().testData_);
}

INSTANTIATE_TEST_SUITE_P(TestAllgather_AttnPost_Reducescatter, DistributedTest,
    ::testing::ValuesIn(GetOpMetaData<OpMetaData>("Allgather_AttnPost_Reducescatter")));
TEST_P(DistributedTest, TestAllgather_AttnPost_Reducescatter)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    RunDistributedTestGeneric("Allgather_AttnPost_Reducescatter", GetParam().testData_);
}

TEST_F(DistributedTest, shmem_allreduce_add_allreduce_bfloat16_256_102400_4)
{
    config::SetHostOption(ONLY_CODEGEN, true);
    Distributed::TestShmemAllReduceAddAllReduce<bfloat16>(testParam);
}
} // namespace npu::tile_fwk::Distributed
