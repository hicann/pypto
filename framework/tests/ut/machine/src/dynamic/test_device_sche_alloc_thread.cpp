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
 * \file test_device_sche_alloc_thread.cpp
 * \brief
 */

#include <atomic>
#include <gtest/gtest.h>

#include "machine/device/dynamic/device_sche_alloc_thread.h"
#include "machine/device/dynamic/device_sche_wait_ctrl.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
DeviceArgs MakeDevArgs(ArchInfo arch, uint32_t scheCpuNum, bool launchSchedSameCluster = false)
{
    DeviceArgs devArgs{};
    devArgs.archInfo = arch;
    devArgs.scheCpuNum = scheCpuNum;
    devArgs.launchSchedSameCluster = launchSchedSameCluster;
    devArgs.nrAicpu = scheCpuNum;
    return devArgs;
}

// 3 个 CPU 位于 die0 cluster（bit 4,5,6），用于 same-cluster 场景
constexpr uint64_t CPUMASK_3_IN_CLUSTER = (1ULL << 4) | (1ULL << 5) | (1ULL << 6);
// 2 个 CPU 位于 die0 cluster（bit 4,5）
constexpr uint64_t CPUMASK_2_IN_CLUSTER = (1ULL << 4) | (1ULL << 5);
} // namespace

struct ArbitrationTest : testing::Test {
    void SetUp() override {}
    void TearDown() override {}
};

// ---------------------------------------------------------------------------
// GetSameClusterCpuCnt
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, GetSameClusterCpuCnt_ZeroMask)
{
    std::atomic<uint64_t> cpumask{0};
    EXPECT_EQ(GetSameClusterCpuCnt(cpumask), 0);
}

TEST_F(ArbitrationTest, GetSameClusterCpuCnt_NoClusterBits)
{
    std::atomic<uint64_t> cpumask{0b111}; // bit 0,1,2 — 不在 cluster 范围
    EXPECT_EQ(GetSameClusterCpuCnt(cpumask), 0);
}

TEST_F(ArbitrationTest, GetSameClusterCpuCnt_ThreeInDie0Cluster)
{
    std::atomic<uint64_t> cpumask{CPUMASK_3_IN_CLUSTER};
    EXPECT_EQ(GetSameClusterCpuCnt(cpumask), 3);
}

TEST_F(ArbitrationTest, GetSameClusterCpuCnt_AllClusterBits)
{
    std::atomic<uint64_t> cpumask{CLUSTER_CPU_MASK}; // bit 4-7,12-15
    EXPECT_EQ(GetSameClusterCpuCnt(cpumask), 8);
}

TEST_F(ArbitrationTest, GetSameClusterCpuCnt_CrossDie)
{
    std::atomic<uint64_t> cpumask{(1ULL << 4) | (1ULL << 12) | (1ULL << 15)};
    EXPECT_EQ(GetSameClusterCpuCnt(cpumask), 3);
}

// ---------------------------------------------------------------------------
// IsSameCpuCluster
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, IsSameCpuCluster_Boundaries)
{
    EXPECT_FALSE(IsSameCpuCluster(3));
    EXPECT_TRUE(IsSameCpuCluster(4));
    EXPECT_TRUE(IsSameCpuCluster(7));
    EXPECT_FALSE(IsSameCpuCluster(8));
    EXPECT_FALSE(IsSameCpuCluster(11));
    EXPECT_TRUE(IsSameCpuCluster(12));
    EXPECT_TRUE(IsSameCpuCluster(15));
    EXPECT_FALSE(IsSameCpuCluster(16));
}

// ---------------------------------------------------------------------------
// CalculateArbitLevelFromArbitScheNum
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, CalculateArbitLevelFromArbitScheNum_AllBranches)
{
    constexpr int scheCpuNum = 3;

    // arbitScheNum == scheCpuNum: 取决于 same-cluster 计数
    std::atomic<uint64_t> cpumaskSameCluster{CPUMASK_3_IN_CLUSTER};
    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, 3, cpumaskSameCluster),
        ARBIT_A2A3_SAME_CLUSTER);

    std::atomic<uint64_t> cpumaskCross{CPUMASK_2_IN_CLUSTER};
    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, 3, cpumaskCross),
        ARBIT_A2A3_CROSS_CLUSTER);

    // arbitScheNum != scheCpuNum: 与 cpumask 无关
    std::atomic<uint64_t> cpumaskAny{0};
    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, DAV2201_DUAL_SCHE_NUM, cpumaskAny),
        ARBIT_A2A3_DUAL_SCHE);
    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, DAV2201_SINGLE_SCHE_NUM, cpumaskAny),
        ARBIT_A2A3_SINGLE_SCHE);

    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, 0, cpumaskAny), ARBIT_FAILED);
    EXPECT_EQ(CalculateArbitLevelFromArbitScheNum(scheCpuNum, 99, cpumaskAny), ARBIT_FAILED);
}

// ---------------------------------------------------------------------------
// GetScheNumByArbitrationLevel
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, GetScheNumByArbitrationLevel_AllBranches)
{
    constexpr int scheCpuNum = 3;

    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_A2A3_SAME_CLUSTER), 3);
    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_A2A3_CROSS_CLUSTER), 3);
    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_A2A3_DUAL_SCHE), DAV2201_DUAL_SCHE_NUM);
    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_A2A3_SINGLE_SCHE), DAV2201_SINGLE_SCHE_NUM);
    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_FAILED), 0);
    EXPECT_EQ(GetScheNumByArbitrationLevel(scheCpuNum, ARBIT_UNSET), 0);
}

// ---------------------------------------------------------------------------
// AllocThreadIdByArbitrationLevel
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, AllocThreadIdByArbitrationLevel_SameClusterThread)
{
    std::atomic<int> threadIdx{0};
    int curThreadIdx = -1;

    AllocThreadIdByArbitrationLevel(ARBIT_A2A3_SAME_CLUSTER, curThreadIdx, 1, threadIdx);
    EXPECT_EQ(curThreadIdx, 1);
    EXPECT_EQ(threadIdx.load(), 1);
}

TEST_F(ArbitrationTest, AllocThreadIdByArbitrationLevel_NonSameClusterThread)
{
    std::atomic<int> threadIdx{0};
    int curThreadIdx = -1;

    AllocThreadIdByArbitrationLevel(ARBIT_A2A3_SAME_CLUSTER, curThreadIdx, 0, threadIdx);
    EXPECT_EQ(curThreadIdx, -1);
    EXPECT_EQ(threadIdx.load(), 0);
}

TEST_F(ArbitrationTest, AllocThreadIdByArbitrationLevel_NonOptimalAlwaysIncrements)
{
    std::atomic<int> threadIdx{0};
    int curThreadIdx = -1;

    AllocThreadIdByArbitrationLevel(ARBIT_A2A3_CROSS_CLUSTER, curThreadIdx, 0, threadIdx);
    EXPECT_EQ(curThreadIdx, 1);

    AllocThreadIdByArbitrationLevel(ARBIT_A2A3_DUAL_SCHE, curThreadIdx, 0, threadIdx);
    EXPECT_EQ(curThreadIdx, 2);

    AllocThreadIdByArbitrationLevel(ARBIT_A2A3_SINGLE_SCHE, curThreadIdx, 1, threadIdx);
    EXPECT_EQ(curThreadIdx, 3);

    EXPECT_EQ(threadIdx.load(), 3);
}

// ---------------------------------------------------------------------------
// GetArbitTimeOutVal
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, GetArbitTimeOutVal_DispatchesByArch)
{
    // 非 __DEVICE__ 环境下 DEV_IF_DEBUG / DEV_IF_INFO 均为 if(true)，
    // arbitTimeout 总被覆盖为 *_20MIN，与 arch / isLastArbitration 无关
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_3510, true), TIMEOUT_A5_20MIN);
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_3510, false), TIMEOUT_A5_20MIN);
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_2201, true), TIMEOUT_A2A3_20MIN);
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_2201, false), TIMEOUT_A2A3_20MIN);
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_UNKNOWN, true), TIMEOUT_A2A3_20MIN);
    EXPECT_EQ(GetArbitTimeOutVal(ArchInfo::DAV_UNKNOWN, false), TIMEOUT_A2A3_20MIN);
}

// ---------------------------------------------------------------------------
// WaitForCpuMaskReadyForArbitration
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, WaitForCpuMaskReady_MaskAlreadyReady_ReturnsOk)
{
    std::atomic<uint64_t> cpumask{0b111};

    EXPECT_EQ(WaitForCpuMaskReadyForArbitration(ArchInfo::DAV_2201, 3, cpumask), DEVICE_MACHINE_OK);
    EXPECT_EQ(WaitForCpuMaskReadyForArbitration(ArchInfo::DAV_2201, 2, cpumask), DEVICE_MACHINE_OK);
    EXPECT_EQ(WaitForCpuMaskReadyForArbitration(ArchInfo::DAV_2201, 1, cpumask), DEVICE_MACHINE_OK);
}

// ---------------------------------------------------------------------------
// WaitForScheCpuInSameCluster
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, WaitForScheCpuInSameCluster_CntAlreadyReady_ReturnsOk)
{
    std::atomic<uint64_t> cpumask{CPUMASK_3_IN_CLUSTER};

    EXPECT_EQ(WaitForScheCpuInSameCluster(ArchInfo::DAV_2201, 3, cpumask), DEVICE_MACHINE_OK);
}

// ---------------------------------------------------------------------------
// ComputeArbitrationLevelDav2201
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, ComputeArbitrationLevelDav2201_CrossCluster)
{
    auto devArgs = MakeDevArgs(ArchInfo::DAV_2201, 3, false);
    std::atomic<uint64_t> cpumask{0b111}; // bit 0,1,2 — 非 cluster

    EXPECT_EQ(ComputeArbitrationLevelDav2201(&devArgs, cpumask), ARBIT_A2A3_CROSS_CLUSTER);
}

TEST_F(ArbitrationTest, ComputeArbitrationLevelDav2201_SameCluster_WhenAllInSameCluster)
{
    auto devArgs = MakeDevArgs(ArchInfo::DAV_2201, 3, false);
    std::atomic<uint64_t> cpumask{CPUMASK_3_IN_CLUSTER};

    EXPECT_EQ(ComputeArbitrationLevelDav2201(&devArgs, cpumask), ARBIT_A2A3_SAME_CLUSTER);
}

TEST_F(ArbitrationTest, ComputeArbitrationLevelDav2201_LaunchSchedSameCluster_SameCluster)
{
    auto devArgs = MakeDevArgs(ArchInfo::DAV_2201, 3, true);
    std::atomic<uint64_t> cpumask{CPUMASK_3_IN_CLUSTER};

    EXPECT_EQ(ComputeArbitrationLevelDav2201(&devArgs, cpumask), ARBIT_A2A3_SAME_CLUSTER);
}

// ---------------------------------------------------------------------------
// AllocThreadIdxForDav2201Impl
// 非 __DEVICE__ 环境下，函数被 #ifndef __DEVICE__ 短路：直接 ++threadIdx 并返回 OK，
// 不执行仲裁逻辑（arbitratedScheNum / globalArbitrationLevel 保持不变）
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, AllocThreadIdxForDav2201Impl_NonDeviceShortCircuits)
{
    auto devArgs = MakeDevArgs(ArchInfo::DAV_2201, 3, false);
    std::atomic<uint64_t> cpumask{0};
    std::atomic<int> globalArbitrationLevel{ARBIT_UNSET};
    std::atomic<int> threadIdx{0};
    int curThreadIdx = -1;
    int arbitratedScheNum = 0;

    EXPECT_EQ(AllocThreadIdxForDav2201Impl(&devArgs, 6, curThreadIdx, threadIdx, cpumask,
        arbitratedScheNum, globalArbitrationLevel), DEVICE_MACHINE_OK);
    // 短路分支：threadIdx 递增，仲裁状态不变
    EXPECT_EQ(curThreadIdx, 1);
    EXPECT_EQ(threadIdx.load(), 1);
    EXPECT_EQ(globalArbitrationLevel.load(), ARBIT_UNSET);
    EXPECT_EQ(arbitratedScheNum, 0);
}

// ---------------------------------------------------------------------------
// WaitForArbitrationLevel
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, WaitForArbitrationLevel_ReturnsPublishedLevel)
{
    std::atomic<int> globalArbitrationLevel{ARBIT_A2A3_SINGLE_SCHE};

    EXPECT_EQ(WaitForArbitrationLevel(ArchInfo::DAV_2201, globalArbitrationLevel), ARBIT_A2A3_SINGLE_SCHE);
}

// ---------------------------------------------------------------------------
// ApplyCtrlDecision
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, ApplyCtrlDecision_Failed_ReturnsError)
{
    int curThreadIdx = 1;
    int arbitratedScheNum = 3;
    EXPECT_EQ(ApplyCtrlDecision(nullptr, curThreadIdx, arbitratedScheNum, CTRL_WAIT_FAILED, ArchInfo::DAV_2201),
        DEVICE_MACHINE_ERROR);
    EXPECT_EQ(curThreadIdx, 1);
    EXPECT_EQ(arbitratedScheNum, 3);
}

TEST_F(ArbitrationTest, ApplyCtrlDecision_Ok_ReturnsOk)
{
    int curThreadIdx = 1;
    int arbitratedScheNum = 3;
    EXPECT_EQ(ApplyCtrlDecision(nullptr, curThreadIdx, arbitratedScheNum, CTRL_WAIT_OK, ArchInfo::DAV_2201),
        DEVICE_MACHINE_OK);
    EXPECT_EQ(curThreadIdx, 1);
    EXPECT_EQ(arbitratedScheNum, 3);
}

TEST_F(ArbitrationTest, ApplyCtrlDecision_Dropped_DropsLastThread)
{
    int curThreadIdx = 3; // == scheNum(3) → 被丢弃
    int arbitratedScheNum = 3;
    EXPECT_EQ(ApplyCtrlDecision(nullptr, curThreadIdx, arbitratedScheNum, CTRL_WAIT_DROPPED, ArchInfo::DAV_2201),
        DEVICE_MACHINE_OK);
    EXPECT_EQ(curThreadIdx, -1);
    EXPECT_EQ(arbitratedScheNum, 2);
}

// ---------------------------------------------------------------------------
// WaitCtrlDecision
// ---------------------------------------------------------------------------

TEST_F(ArbitrationTest, WaitCtrlDecision_ReturnsTerminalLevel_Ok)
{
    std::atomic<int> ctrlWaitLevel{CTRL_WAIT_OK};
    int outLevel = CTRL_WAIT_UNSET;

    EXPECT_EQ(WaitCtrlDecision(ArchInfo::DAV_2201, ctrlWaitLevel, outLevel), DEVICE_MACHINE_OK);
    EXPECT_EQ(outLevel, CTRL_WAIT_OK);
}

TEST_F(ArbitrationTest, WaitCtrlDecision_ReturnsTerminalLevel_Failed)
{
    std::atomic<int> ctrlWaitLevel{CTRL_WAIT_FAILED};
    int outLevel = CTRL_WAIT_UNSET;

    EXPECT_EQ(WaitCtrlDecision(ArchInfo::DAV_2201, ctrlWaitLevel, outLevel), DEVICE_MACHINE_OK);
    EXPECT_EQ(outLevel, CTRL_WAIT_FAILED);
}

TEST_F(ArbitrationTest, WaitCtrlDecision_ReturnsTerminalLevel_Dropped)
{
    std::atomic<int> ctrlWaitLevel{CTRL_WAIT_DROPPED};
    int outLevel = CTRL_WAIT_UNSET;

    EXPECT_EQ(WaitCtrlDecision(ArchInfo::DAV_2201, ctrlWaitLevel, outLevel), DEVICE_MACHINE_OK);
    EXPECT_EQ(outLevel, CTRL_WAIT_DROPPED);
}
