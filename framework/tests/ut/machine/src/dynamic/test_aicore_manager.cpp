/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>

#define private public
#define protected public
#include "machine/device/dynamic/aicore_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
struct MgrTestEnv {
    SchThreadStatus status;
    std::unique_ptr<AiCoreManager> mgr;
    std::unique_ptr<SchduleContext> ctxPtr;
    uint8_t* sharedBuf{nullptr};
    int64_t* regAddrsArr{nullptr};
    uint32_t* regMem{nullptr};
    static constexpr int AIC_NUM = 2;
    static constexpr int AIV_NUM = 0;

    MgrTestEnv() : ctxPtr(std::make_unique<SchduleContext>())
    {
        status.Init();
        mgr = std::make_unique<AiCoreManager>(status);
        mgr->aicNum_ = AIC_NUM;
        mgr->aivNum_ = AIV_NUM;
        mgr->aicStart_ = 0;
        mgr->aicEnd_ = AIC_NUM;
        mgr->aivStart_ = AIC_NUM;
        mgr->aivEnd_ = AIC_NUM;
        mgr->adjAicEnd_ = AIC_NUM;
        mgr->adjAivEnd_ = AIC_NUM;
        mgr->aicpuIdx_ = 0;
        mgr->schedIdx_ = 0;
        mgr->aicpuNum_ = 1;
        mgr->aicValidNum_ = AIC_NUM;
        mgr->archInfo_ = ArchInfo::DAV_2201;
        mgr->enableEslModel_ = false;
        mgr->hasAicpuTask_ = false;
        mgr->isMixPending_ = false;
        mgr->enableFairSch_ = false;
        mgr->enableL2CacheSch_ = false;
        mgr->validGetPgMask_ = false;
        mgr->releaseCoreByRegValFn_ = &AiCoreManager::ReleaseCoreByRegValByAsyncMode;
        mgr->pendingIds_.fill(AICORE_STATUS_INIT);
        mgr->runningIds_.fill(AICORE_STATUS_INIT);
        mgr->runningResolveIndexList_.fill(0);
        mgr->pendingResolveIndexList_.fill(0);
        mgr->pingPongFlag_.fill(0);
        mgr->context_ = ctxPtr.get();

        const size_t rawSize = SHARED_BUFFER_SIZE * (AIC_NUM + AIV_NUM);
        const size_t sharedSize = ((rawSize + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        sharedBuf = reinterpret_cast<uint8_t*>(aligned_alloc(PAGE_SIZE, sharedSize));
        memset(sharedBuf, 0, sharedSize);
        regMem = new uint32_t[256]();
        regAddrsArr = new int64_t[AIC_NUM + AIV_NUM]();
        for (int i = 0; i < AIC_NUM; i++) {
            regAddrsArr[i] = reinterpret_cast<int64_t>(&regMem[i * 64]);
        }

        DeviceArgs devArgs{};
        devArgs.sharedBuffer = reinterpret_cast<int64_t>(sharedBuf);
        devArgs.coreRegAddr = reinterpret_cast<int64_t>(regAddrsArr);
        devArgs.nrAic = AIC_NUM;
        devArgs.nrAiv = AIV_NUM;
        devArgs.archInfo = ArchInfo::DAV_2201;
        devArgs.enableEslModel = false;
        mgr->aicoreHal_.Init(&devArgs, &mgr->aicoreProf_);
        for (int i = 0; i < AIC_NUM; i++) {
            mgr->aicoreHal_.GetPhyIdByBlockId(i) = i;
        }

        auto& ctx = *ctxPtr;
        ctx.coreRunReadyCnt_[0] = 0;
        ctx.coreRunReadyCnt_[1] = 0;
        ctx.corePendReadyCnt_[0] = 0;
        ctx.corePendReadyCnt_[1] = 0;
        ctx.waitTaskCnt[0] = 0;
        ctx.waitTaskCnt[1] = 0;
        for (uint32_t i = 0; i < MAX_AICORE_NUM; i++) {
            ctx.coreIdxPosition_[i] = INVALID_COREIDX_POSITION;
            ctx.wrapCoreAvail_[i] = true;
        }
    }

    ~MgrTestEnv()
    {
        if (sharedBuf)
            free(sharedBuf);
        delete[] regMem;
        delete[] regAddrsArr;
    }

    SchduleContext& ctx() { return *ctxPtr; }
};
} // namespace

TEST(AicoreManagerTest, AddAndRemoveRunReadyCoreIdx)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);

    env.mgr->AddReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().runReadyCoreIdx_[type][0], 0u);

    env.mgr->AddReadyCoreIdx(1, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 2u);

    env.mgr->RemoveRunReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().coreIdxPosition_[0], INVALID_COREIDX_POSITION);
}

TEST(AicoreManagerTest, RemoveRunReadyCoreIdx_InvalidPos)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreIdxPosition_[0] = INVALID_COREIDX_POSITION;
    env.mgr->RemoveRunReadyCoreIdx(0, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 0u);
}

TEST(AicoreManagerTest, RemoveReadyCoreIdxTail)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);

    env.mgr->AddReadyCoreIdx(0, type);
    env.mgr->AddReadyCoreIdx(1, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 2u);

    env.mgr->RemoveReadyCoreIdxTail(1, type);
    EXPECT_EQ(env.ctx().coreRunReadyCnt_[type], 1u);
    EXPECT_EQ(env.ctx().coreIdxPosition_[1], INVALID_COREIDX_POSITION);
}

TEST(AicoreManagerTest, RemoveReadyCoreIdxTail_InvalidPos)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.ctx().coreIdxPosition_[0] = INVALID_COREIDX_POSITION;
    env.mgr->RemoveReadyCoreIdxTail(0, type);
}

TEST(AicoreManagerTest, AicpuIsBusyAndIdle)
{
    MgrTestEnv env;
    env.mgr->AicpuIsBusy(CoreType::AIC);
    EXPECT_FALSE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());

    env.mgr->AicpuIsIdle(CoreType::AIC);
    EXPECT_TRUE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, AicpuIsBusyAlreadyBusy)
{
    MgrTestEnv env;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(false);
    env.mgr->AicpuIsBusy(CoreType::AIC);
    EXPECT_FALSE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, AicpuIsIdleAlreadyIdle)
{
    MgrTestEnv env;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].store(true);
    env.mgr->AicpuIsIdle(CoreType::AIC);
    EXPECT_TRUE(env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][0].load());
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_SingleCpu)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 1;
    env.mgr->schedIdx_ = 0;
    EXPECT_FALSE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_OtherIdle)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][1].store(true);
    EXPECT_TRUE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, IsExistOtherAicpuIdle_OtherBusy)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 2;
    env.mgr->schedIdx_ = 0;
    env.status.isAicpuIdle[static_cast<int>(CoreType::AIC)][1].store(false);
    EXPECT_FALSE(env.mgr->IsExistOtherAicpuIdle(CoreType::AIC));
}

TEST(AicoreManagerTest, UpdateAicoreEnd_Normal)
{
    int end = 0;
    AiCoreManager::UpdateAicoreEnd(4, 0, 2, 1, 0, end);
    EXPECT_EQ(end, 2);

    AiCoreManager::UpdateAicoreEnd(5, 0, 2, 1, 0, end);
    EXPECT_EQ(end, 3);

    AiCoreManager::UpdateAicoreEnd(5, 1, 2, 1, 0, end);
    EXPECT_EQ(end, 2);
}

TEST(AicoreManagerTest, UpdateAicoreEnd_ZeroPart)
{
    int end = 99;
    AiCoreManager::UpdateAicoreEnd(4, 0, 0, 1, 5, end);
    EXPECT_EQ(end, 5);
}

TEST(AicoreManagerTest, GetPhyIdByBlockId)
{
    MgrTestEnv env;
    env.mgr->aicoreHal_.GetPhyIdByBlockId(0) = 5;
    EXPECT_EQ(env.mgr->GetPhyIdByBlockId(0), 5);
}

TEST(AicoreManagerTest, SyncAicpuTaskFinish_NoAicpuTask)
{
    MgrTestEnv env;
    env.mgr->hasAicpuTask_ = false;
    SchDeviceTaskContext devTaskCtx{};
    EXPECT_EQ(env.mgr->SyncAicpuTaskFinish(&devTaskCtx), DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, IsNeedProcAicpuTask)
{
    MgrTestEnv env;
    env.mgr->hasAicpuTask_ = false;
    EXPECT_FALSE(env.mgr->IsNeedProcAicpuTask());
    env.mgr->hasAicpuTask_ = true;
    EXPECT_TRUE(env.mgr->IsNeedProcAicpuTask());
}

TEST(AicoreManagerTest, GetAllAiCoreNum)
{
    MgrTestEnv env;
    EXPECT_EQ(env.mgr->GetAllAiCoreNum(), MgrTestEnv::AIC_NUM + MgrTestEnv::AIV_NUM);
}

TEST(AicoreManagerTest, AicoreType)
{
    MgrTestEnv env;
    EXPECT_EQ(env.mgr->AicoreType(0), CoreType::AIC);
    EXPECT_EQ(env.mgr->AicoreType(1), CoreType::AIC);
    env.mgr->aivStart_ = 2;
    env.mgr->aivEnd_ = 4;
    env.mgr->aicEnd_ = 2;
    EXPECT_EQ(env.mgr->AicoreType(2), CoreType::AIV);
}

TEST(AicoreManagerTest, SetDotStatus)
{
    MgrTestEnv env;
    env.mgr->SetDotStatus(42);
    EXPECT_EQ(env.mgr->dotStatus_, 42);
}

TEST(AicoreManagerTest, SetSchedSyncMode)
{
    MgrTestEnv env;
    env.mgr->SetSchedSyncMode(1);
    EXPECT_EQ(env.mgr->releaseCoreByRegValFn_, &AiCoreManager::ReleaseCoreByRegValBySyncMode);
    env.mgr->SetSchedSyncMode(0);
    EXPECT_EQ(env.mgr->releaseCoreByRegValFn_, &AiCoreManager::ReleaseCoreByRegValByAsyncMode);
}

TEST(AicoreManagerTest, CheckAndResetReg_InvalidPgMask)
{
    MgrTestEnv env;
    env.mgr->validGetPgMask_ = false;
    EXPECT_TRUE(env.mgr->CheckAndResetReg());
}

TEST(AicoreManagerTest, ForEachManageAicore)
{
    MgrTestEnv env;
    int count = 0;
    env.mgr->ForEachManageAicore([&count](int) { count++; });
    EXPECT_EQ(count, MgrTestEnv::AIC_NUM);
}

TEST(AicoreManagerTest, ForEachManageAicoreReverse)
{
    MgrTestEnv env;
    std::vector<int> order;
    env.mgr->ForEachManageAicoreReverse([&order](int i) { order.push_back(i); });
    EXPECT_EQ(order.size(), static_cast<size_t>(MgrTestEnv::AIC_NUM));
    if (order.size() >= 2) {
        EXPECT_GT(order[0], order[1]);
    }
}

TEST(AicoreManagerTest, ForEachManageAicoreWithRet_Success)
{
    MgrTestEnv env;
    int ret = env.mgr->ForEachManageAicoreWithRet([](int) -> int { return DEVICE_MACHINE_OK; });
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, ForEachManageAicoreWithRet_Error)
{
    MgrTestEnv env;
    int ret = env.mgr->ForEachManageAicoreWithRet([](int) -> int { return DEVICE_MACHINE_ERROR; });
    EXPECT_NE(ret, DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, NeedsHwStopOnLastDevTask_NonDevice)
{
    MgrTestEnv env;
    EXPECT_FALSE(env.mgr->NeedsHwStopOnLastDevTask());
}

TEST(AicoreManagerTest, RuntimeCopyOutResolveCounterDecode)
{
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0x00050000), 0u);
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0x00050003), 3u);
    EXPECT_EQ(AiCoreManager::RuntimeCopyOutResolveCounterDecode(0xFFFF), 0xFFFFu);
}

TEST(AicoreManagerTest, RecordResolveTask_Normal)
{
    MgrTestEnv env;
    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    env.mgr->RecordResolveTask(ctx, finishCnt, 0, 0x1234, 0);
    EXPECT_EQ(finishCnt, 1u);
    EXPECT_EQ(ctx[0].finishIds, 0x1234u);
    EXPECT_EQ(ctx[0].finishCoreIdx, 0);
}

TEST(AicoreManagerTest, RecordResolveTask_Overflow)
{
    MgrTestEnv env;
    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = MAX_RESOLVE_TASK_NUM;
    env.mgr->RecordResolveTask(ctx, finishCnt, 0, 0x1234, 0);
    EXPECT_EQ(finishCnt, MAX_RESOLVE_TASK_NUM);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValBySyncMode_NoMatch)
{
    MgrTestEnv env;
    int coreIdx = 0;
    env.mgr->pendingIds_[coreIdx] = 0x999;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;

    uint32_t resloveParallelIdx = 0;
    int32_t ret = env.mgr->ResolveWhenSyncMode(CoreType::AIC, 0x100, TASK_FIN_STATE, coreIdx, resloveParallelIdx);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], 0x999u);
}

TEST(AicoreManagerTest, DumpDfxWhenCoreNotStop)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(1);
    env.mgr->DumpDfxWhenCoreNotStop(&devTaskCtx);
}

TEST(AicoreManagerTest, DumpDfxWhenCoreNotStop_NotFinished)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(0);
    env.mgr->pendingIds_[0] = 0x100;
    env.mgr->runningIds_[0] = 0x200;
    env.mgr->DumpDfxWhenCoreNotStop(&devTaskCtx);
}

TEST(AicoreManagerTest, DumpLastWord)
{
    MgrTestEnv env;
    env.mgr->pendingIds_[0] = 0x100;
    env.mgr->runningIds_[0] = 0x200;
    env.mgr->DumpLastWord(0);
}

TEST(AicoreManagerTest, DumpLastWord_InitState)
{
    MgrTestEnv env;
    env.mgr->pendingIds_[0] = AICORE_TASK_INIT;
    env.mgr->runningIds_[0] = AICORE_TASK_INIT;
    env.mgr->DumpLastWord(0);
}

TEST(AicoreManagerTest, MarkCoreStoped)
{
    MgrTestEnv env;
    SchDeviceTaskContext devTaskCtx{};
    devTaskCtx.coreTaskFinished.fill(0);
    devTaskCtx.coreFinishedNum = 0;
    env.mgr->MarkCoreStoped(&devTaskCtx, 0);
    EXPECT_EQ(devTaskCtx.coreTaskFinished[0], 1);
    EXPECT_EQ(devTaskCtx.coreFinishedNum, 1u);
}

TEST(AicoreManagerTest, GetRunReadyCoreNum)
{
    MgrTestEnv env;
    env.ctx().coreRunReadyCnt_[static_cast<int>(CoreType::AIC)] = 3;
    EXPECT_EQ(env.mgr->GetRunReadyCoreNum(CoreType::AIC), 3u);
}

TEST(AicoreManagerTest, GetReadyCoreNum_NoTail)
{
    MgrTestEnv env;
    env.ctx().corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = 2;
    EXPECT_EQ(env.mgr->GetReadyCoreNum(CoreType::AIC, false), 2u);
}

TEST(AicoreManagerTest, CheckIsTailBatch_SingleCpu)
{
    MgrTestEnv env;
    env.mgr->aicpuNum_ = 1;
    SchDeviceTaskContext devTaskCtx{};
    uint64_t remaining = 0;
    EXPECT_FALSE(env.mgr->CheckIsTailBatch(&devTaskCtx, CoreType::AIC, remaining));
}

TEST(AicoreManagerTest, SetSchduleContext)
{
    MgrTestEnv env;
    auto newCtx = std::make_unique<SchduleContext>();
    env.mgr->SetSchduleContext(newCtx.get());
    EXPECT_EQ(env.mgr->context_, newCtx.get());
}

TEST(AicoreManagerTest, HandShakeCorrectReadyCore)
{
    MgrTestEnv env;
    int type = static_cast<int>(CoreType::AIC);
    env.mgr->adjAicEnd_ = 1;
    env.mgr->aicEnd_ = 2;
    env.mgr->AddReadyCoreIdx(0, type);
    env.mgr->AddReadyCoreIdx(1, type);
    env.ctx().corePendReadyCnt_[type] = 2;

    env.mgr->HandShakeCorrectReadyCore(CoreType::AIC);
    EXPECT_EQ(env.ctx().corePendReadyCnt_[type], 1);
}

TEST(AicoreManagerTest, CalcAdjAicoreEnd_NonDeviceNonEsl)
{
    MgrTestEnv env;
    env.mgr->enableEslModel_ = false;
    env.mgr->CalcAdjAicoreEnd(nullptr, true);
    EXPECT_EQ(env.mgr->adjAicEnd_, env.mgr->aicEnd_);
    EXPECT_EQ(env.mgr->adjAivEnd_, env.mgr->aivEnd_);
}

TEST(AicoreManagerTest, PostRun_Success)
{
    MgrTestEnv env;
    env.mgr->PostRun(DEVICE_MACHINE_OK);
}

TEST(AicoreManagerTest, BatchStopAllManagedCores_NonDeviceNonSim)
{
    MgrTestEnv env;
    env.mgr->BatchStopAllManagedCores();
}

TEST(AicoreManagerTest, ResetRegAll)
{
    MgrTestEnv env;
    env.mgr->ResetRegAll();
}

TEST(AicoreManagerTest, NormalStop)
{
    MgrTestEnv env;
    env.mgr->NormalStop();
}

TEST(AicoreManagerTest, AbnormalStop)
{
    MgrTestEnv env;
    env.mgr->AbnormalStop();
}

TEST(AicoreManagerTest, SendAllCoreStop_NonDeviceNonSim)
{
    MgrTestEnv env;
    env.mgr->SendAllCoreStop();
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingFinished)
{
    MgrTestEnv env;
    int coreIdx = 0;
    int type = static_cast<int>(CoreType::AIC);
    uint32_t taskId = 0x100;
    env.mgr->pendingIds_[coreIdx] = taskId;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;
    env.ctx().corePendReadyCnt_[type] = 0;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(taskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);
    uint32_t aicpuCallCode = 0;

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, aicpuCallCode, taskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_RunningFinished)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t runningTaskId = 0x200;
    env.mgr->pendingIds_[coreIdx] = AICORE_TASK_INIT;
    env.mgr->runningIds_[coreIdx] = runningTaskId;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(runningTaskId) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, runningTaskId, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_PendingAck)
{
    MgrTestEnv env;
    int coreIdx = 0;
    uint32_t taskId = 0x300;
    env.mgr->pendingIds_[coreIdx] = taskId;
    env.mgr->runningIds_[coreIdx] = AICORE_TASK_INIT;
    env.ctx().wrapCoreAvail_[coreIdx] = true;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(taskId) | (static_cast<uint64_t>(TASK_ACK_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, taskId, TASK_ACK_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
    EXPECT_EQ(env.mgr->runningIds_[coreIdx], taskId);
    EXPECT_EQ(env.mgr->pendingIds_[coreIdx], AICORE_TASK_INIT);
}

TEST(AicoreManagerTest, ReleaseCoreByRegValByAsyncMode_InconsistentState)
{
    MgrTestEnv env;
    int coreIdx = 0;
    env.mgr->pendingIds_[coreIdx] = 0x100;
    env.mgr->runningIds_[coreIdx] = 0x200;

    ResolveTaskContext ctx[MAX_RESOLVE_TASK_NUM];
    uint32_t finishCnt = 0;
    uint32_t resloveParallelIdx = 0;
    uint64_t finTaskRegVal = static_cast<uint64_t>(0x999) | (static_cast<uint64_t>(TASK_FIN_STATE) << 31);

    int32_t ret = env.mgr->ReleaseCoreByRegValByAsyncMode(CoreType::AIC, coreIdx, ctx, finishCnt, resloveParallelIdx,
                                                          finTaskRegVal, 0, 0x999, TASK_FIN_STATE);
    EXPECT_EQ(ret, DEVICE_MACHINE_OK);
}
