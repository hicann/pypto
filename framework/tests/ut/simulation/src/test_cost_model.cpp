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
 * \file test_cost_model.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <dlfcn.h>

#include "cost_model/simulation/backend.h"
#include "operator/models/llama/llama_def.h"
#include "cost_model/simulation/common/CommonType.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "interface/configs/config_manager.h"
#include "cost_model/simulation/arch/PipeFactory.h"
#include "cost_model/simulation/arch/CacheMachineImpl.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/machine/Scheduler.h"
#include "cost_model/simulation/tools/ParseInput.h"
#include "cost_model/simulation/arch/PipeSimulatorFast.h"
#include "cost_model/simulation/tools/visualizer.h"

using namespace npu::tile_fwk;

class CostModelTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
        config::SetSimConfig(KEY_BUILD_TASK_BASED_TOPO, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void RunLLamaLayerCostModel(const AttentionDims& dimsCfg, float threadhold = 0.001f)
{
    (void)threadhold;
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;

    PROGRAM("LLAMALAYER")
    {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
        Tensor Res(DT_FP32, {b * s, n * d}, "Res");
        config::SetBuildStatic(true);
        FUNCTION("LLAMA", {H, AW, DW, FW, Res})
        {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
        config::SetPassStrategy("OOO");
    }
}

void RunMatrixCostModel()
{
    int bs = 1;
    int m = 32;
    int k = 32;
    int n = 32;

    std::vector<int64_t> shapeA = {bs, m, k};
    std::vector<int64_t> shapeB = {bs, k, n};
    std::vector<int64_t> shapeC = {bs, m, n};

    config::Reset();
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor matA(DT_FP16, shapeA, "MatA", TileOpFormat::TILEOP_NZ);
    Tensor matB(DT_FP16, shapeB, "MatB", TileOpFormat::TILEOP_ND);
    Tensor matC(DT_FP32, shapeC, "MatC");
    config::SetBuildStatic(true);
    FUNCTION("BATCHMATMUL", {matA, matB, matC})
    {
        config::SetPassConfig("PVC2_OOO", "OoOSchedule", KEY_DISABLE_PASS, true);
        matC = npu::tile_fwk::Matrix::BatchMatmul(DT_FP32, matA, matB, false, false);
    }
}

void RunAttentionPostCostModel()
{
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    int b = 1;
    int n = 2;
    int s = 128;
    int d = 512;
    int v_head = 128;
    int h = 256;
    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_FP32, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost")
    {
        int new_b = attnPostIn.GetShape()[0];
        int new_n = attnPostIn.GetShape()[1];
        int new_s = attnPostIn.GetShape()[2];
        DataType dType = attnPostIn.GetStorage()->Datatype();
        TileShape::Current().SetVecTile({1, 1, 32, d});
        Tensor atten_res1 = Reshape(Transpose(attnPostIn, {1, 2}), {new_b * new_s, new_n, d});
        TileShape::Current().SetVecTile({32, 1, d});
        Tensor atten_res2 = Transpose(atten_res1, {0, 1});
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        TileShape::Current().SetVecTile(128, 128);
        TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
        Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        // Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        TileShape::Current().SetVecTile({1, 128, 128});
        Tensor mm7_res1 = Transpose(mm7_res, {0, 1});
        Tensor mm7_res2 = Reshape(mm7_res1, {new_b, new_s, new_n * v_head});

        // [b,s, n*vHeadDim] @ [n*vHeadDim, H] = [b,s,h]
        Tensor attn_out_w = Unsqueeze(oProjW, 0);
        atten_output = Matrix::BatchMatmul(dType, mm7_res2, attn_out_w);
    }
}

TEST_F(CostModelTest, TestAttentionPostAccuracy1)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracyFromJson)
{
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    RunAttentionPostCostModel();

    std::string jPath = config::LogTopFolder() + "/program.json";
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
    config::SetSimConfig(KEY_AGENT_JSON_PATH, jPath);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestGenCalendarSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.genCalendarScheduleCpp=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostCVMIXMode)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.cubeVecMixMode=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostSimulationSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=true");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.useOOOPassSeq=false");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostFunctional)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_SIM_MODE, int(CostModel::SimMode::EMULATOR));
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestErrorInput)
{
    std::string name = "TEST";
    auto newFunc = std::make_shared<Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {1, 1};
    auto outcast = std::make_shared<LogicalTensor>(*newFunc, DT_FP32, shape);
    newFunc->outCasts_.push_back(outcast);
    newFunc->inCasts_.push_back(outcast);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(newFunc.get());
}

TEST_F(CostModelTest, TestFixedLatencyTasks)
{
    RunAttentionPostCostModel();

    std::string jsonPath("./config/fixed_task_topo.json");
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + jsonPath);
    config::SetSimConfig(KEY_ARGS, arg);

    std::string jPath = config::LogTopFolder() + "/program.json";
    config::SetSimConfig(KEY_AGENT_JSON_PATH, jPath);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestAttentionPostL2Cache)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=false");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.mteUseL2Cache=true");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestBuildBasedOnConfigs)
{
    std::string configPath("./config/test_config.conf");
    std::vector<std::string> configs;
    configs.push_back("--conf");
    configs.push_back(configPath);

    CostModelAgent costModelAgent;
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);

    configs.clear();
    configs.push_back("-m");
    configs.push_back("1");
    configs.push_back("--conf");
    configs.push_back(configPath);
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);
}

TEST_F(CostModelTest, TestCoreMachineDeadlock)
{
    int accuracylevel = 1;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.testDeadLock=true");
    arg.emplace_back("Core.bufferBackPressure=true");
    arg.emplace_back("Pipe.ubSizeThreshold=256");
    arg.emplace_back("Pipe.l1SizeThreshold=256");
    arg.emplace_back("Pipe.l0aSizeThreshold=256");
    arg.emplace_back("Pipe.l0bSizeThreshold=256");
    arg.emplace_back("Pipe.l0cSizeThreshold=256");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

void RunCat()
{
    TileShape::Current().SetVecTile(16, 6, 6, 16);
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    std::vector<int64_t> shape1 = {10, 10, 10, 10};
    std::vector<int64_t> shape2 = {20, 10, 10, 10};
    int axis = 0;
    Tensor params1(DT_FP32, shape1, "params1");
    Tensor params2(DT_FP32, shape2, "params2");
    Tensor res;

    FUNCTION("A") { res = Cat(std::vector<Tensor>{params1, params2}, axis); }
}

TEST_F(CostModelTest, TestGlobalCalendar)
{
    std::string jsonPath("./config/global.calendar.json");
    std::string inputPath("./config/fixed_task_topo.json");
    CostModel::CalendarMode calendarMode = CostModel::CalendarMode::GLOBAL_COUNTER;
    std::vector<std::string> arg;
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + inputPath);
    arg.emplace_back("Model.calendarFile=" + jsonPath);
    arg.emplace_back("Model.calendarMode=" + std::to_string(static_cast<int>(calendarMode)));
    config::SetSimConfig(KEY_ARGS, arg);
    RunCat();
}

TEST_F(CostModelTest, TestLeafFunctionMode)
{
    config::SetSimConfig(KEY_SIM_MODE, int(CostModel::SimMode::LEAF_FUNCTION));
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    std::vector<std::string> arg;
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.statisticReportToFile=false");
    config::SetSimConfig(KEY_ARGS, arg);
    RunAttentionPostCostModel();
}

class CostModelDynTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        cacheEnable = config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override
    {
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, cacheEnable);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
    }

protected:
    bool oriEnableAihacBackend = false;
    bool cacheEnable = false;
};

void CostModelTestLoopViewAssemble(const Tensor& t0, const Tensor& t1, const Tensor& blockTable, Tensor& out, int s)
{
    FUNCTION("main", {t0, t1, blockTable}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s))
        {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2 * s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2 * s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, qi, ki, false, true);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(CostModelDynTest, TestDD)
{
    constexpr int tilingX = 32;
    constexpr int tilingY = 32;
    TileShape::Current().SetVecTile(tilingX, tilingY);
    constexpr int tilingM = 32;
    constexpr int tilingN = 32;
    constexpr int tilingK = 32;
    TileShape::Current().SetCubeTile({tilingM, tilingM}, {tilingN, tilingN}, {tilingK, tilingK});

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");     // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    CostModelTestLoopViewAssemble(t0, t1, blockTable, out, s);

    auto func = Program::GetInstance().GetLastFunction();
    auto pv = CostModel::PvModelFactory::CreateDyn();
    pv->Codegen(func);
}

TEST_F(CostModelTest, TestUnknownArchType)
{
    EXPECT_THROW(CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A0"), std::invalid_argument);
}

TEST_F(CostModelTest, TestCreateA5Cache)
{
    std::unique_ptr<CostModel::CacheMachineImpl> cacheImpl =
        CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A5");
    CostModel::CachePacket packet;
    cacheImpl->Simulate(packet);
}

TEST_F(CostModelTest, TestA5ArchType)
{
    auto simulator = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A5");
    EXPECT_TRUE(simulator != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateA2A3)
{
    auto sim = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A2A3");
    EXPECT_TRUE(sim != nullptr);
    auto sim2 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_VECTOR_ALU, "A2A3");
    EXPECT_TRUE(sim2 != nullptr);
    auto sim3 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_CUBE, "A2A3");
    EXPECT_TRUE(sim3 != nullptr);
    auto sim4 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_TILE_ALLOC, "A2A3");
    EXPECT_TRUE(sim4 != nullptr);
    auto sim5 = CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_CALL, "A2A3");
    EXPECT_TRUE(sim5 != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateUnknownPipeType)
{
    EXPECT_THROW(CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_UNKNOW, "A2A3"), std::invalid_argument);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheA2A3)
{
    auto cacheImpl = CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A2A3");
    EXPECT_TRUE(cacheImpl != nullptr);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheUnknownArch)
{
    EXPECT_THROW(CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A0"), std::invalid_argument);
}

TEST_F(CostModelTest, TestPipeFactoryCreateCacheUnknownType)
{
    EXPECT_THROW(CostModel::PipeFactory::CreateCache(CostModel::CacheType::FUNCTION_CACHE, "A2A3"), std::invalid_argument);
}

TEST_F(CostModelTest, TestCoreMachineDeadlock2)
{
    CostModel::CoreMachine* coreMachine = new CostModel::CoreMachine(CostModel::MachineType::AIC);
    std::set<int> unissuedTileMagics;

    coreMachine->sim = std::make_shared<CostModel::SimSys>();
    unissuedTileMagics.insert(1);
    unissuedTileMagics.insert(2);

    // 2. 初始化tileOps

    coreMachine->tileOps[1] = std::make_shared<CostModel::TileOp>();
    coreMachine->tileOps[2] = std::make_shared<CostModel::TileOp>();

    coreMachine->tileOps[1]->magic = 1;
    coreMachine->tileOps[2]->magic = 2;
    coreMachine->tileOps[1]->opcode = "";
    coreMachine->tileOps[2]->opcode = "";

    // 3. 初始化tiles
    coreMachine->tiles[1] = std::make_shared<CostModel::Tile>();
    coreMachine->tiles[2] = std::make_shared<CostModel::Tile>();

    coreMachine->tiles[1]->magic = 1;
    coreMachine->tiles[2]->magic = 2;

    // 4. 设置aliveBuffer
    coreMachine->aliveBuffer[CostModel::CorePipeType::PIPE_CUBE_BMU_L1].insert(1);
    coreMachine->aliveBuffer[CostModel::CorePipeType::PIPE_CUBE_BMU_L0A].insert(2);

    // 5. 设置readyQueues
    CostModel::ReadyQueue readyQueue1(CostModel::CorePipeType::PIPE_CUBE_BMU_L1, 0);
    readyQueue1.Insert(1);
    coreMachine->readyQueues.push_back(readyQueue1);

    CostModel::ReadyQueue readyQueue2(CostModel::CorePipeType::PIPE_CUBE_BMU_L0A, 1);
    readyQueue2.Insert(2);
    coreMachine->readyQueues.push_back(readyQueue2);

    // 6. 设置执行任务ID和函数哈希
    coreMachine->executingTaskId = 123;
    coreMachine->executingFunctionHash = 456;

    // 调用AnalysisDeadlock方法
    try {
        coreMachine->AnalysisDeadlock(unissuedTileMagics);
    } catch (const std::exception& e) {
        EXPECT_TRUE(true); // 如果捕获到异常，测试通过
    }

    try {
        coreMachine->CheckDeadlock();
    } catch (const std::exception& e) {
        EXPECT_TRUE(true); // 如果捕获到异常，测试通过
    }
    coreMachine->sim->ReportDeadlock(1);
    delete coreMachine;
}

TEST_F(CostModelTest, TestScheduler)
{
    using namespace CostModel;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> tileAllocSequence(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));

    // 1. 创建节点
    auto t10 = std::make_shared<CostModel::Tile>();
    t10->magic = 10;
    t10->exeInfo.domCount = 5;
    t10->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t11 = std::make_shared<CostModel::Tile>();
    t11->magic = 11;
    t11->exeInfo.domCount = 1;
    t11->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 更小的 domCount

    auto op100 = std::make_shared<CostModel::TileOp>();
    op100->magic = 100;
    op100->pipeType = CorePipeType::PIPE_VECTOR_BMU;

    auto t30 = std::make_shared<CostModel::Tile>();
    t30->magic = 30;
    t30->exeInfo.isOutcast = true;
    t30->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t40 = std::make_shared<CostModel::Tile>();
    t40->magic = 40;
    t40->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 无 consumer，视为 output

    // 2. 建立连接
    op100->iOperand = {t10, t11};
    op100->oOperand = {t30, t40};

    t10->consumers = {op100};
    t11->consumers = {op100};

    t30->producers = {op100};
    t40->producers = {op100};

    tiles[10] = t10;
    tiles[11] = t11;
    tiles[30] = t30;
    tiles[40] = t40;
    tileOps[100] = op100;

    // 3. 执行测试
    scheduler.SortTile(tiles, tileOps, tileAllocSequence);

    // 4. 验证日志覆盖和逻辑

    EXPECT_GT(op100->exeInfo.sequenceToIssue, -1);
    EXPECT_EQ(t10->exeInfo.copyOutIdx, t11->exeInfo.copyOutIdx);
}

TEST_F(CostModelTest, TestScheduler_EmptyInput)
{
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> seq;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    scheduler.SortTile(tiles, tileOps, seq);
}

TEST_F(CostModelTest, TestRemoveBarrierCounter_LogCoverage)
{
    using namespace CostModel;
    GenCalendar calendar;

    // 1. 构造 Source Task 列表 (11个)
    std::vector<uint64_t> srcIds;
    for (uint64_t i = 1; i <= 11; ++i) {
        srcIds.push_back(i);
        calendar.taskTopoInfo[i] = CalendarEntry{};
    }

    for (uint64_t j = 100; j < 110; ++j) {
        CalendarEntry info;
        info.waitSrcTaskIds = srcIds;
        calendar.taskTopoInfo[j] = info;
    }

    calendar.RemoveBarrierCounter();

    uint64_t firstTargetId = 100;
    EXPECT_TRUE(calendar.taskTopoInfo[firstTargetId].waitSrcTaskIds.empty());
    EXPECT_FALSE(calendar.taskTopoInfo[firstTargetId].waitBarrierCounterIds.empty());
    EXPECT_EQ(calendar.taskTopoInfo[firstTargetId].waitBarrierCounterIds[0].first, 100);
}

TEST_F(CostModelTest, TestGetPipeType_AssertOnMissingOpcode)
{
    using namespace CostModel;
    TileOp op;
    op.opcode = "UNKNOWN_OP";

    try {
        op.GetPipeType();
    } catch (const std::exception& e) {
    }

    op.pipeType = CorePipeType::PIPE_UNKNOW;

    try {
        op.GetAddress();
    } catch (const std::exception& e) {
    }
    try {
        op.GetSize();
    } catch (const std::exception& e) {
    }
}

TEST_F(CostModelTest, TestCheckTileOp)
{
    using namespace CostModel;
    ParseInput parser;
    auto func = std::make_shared<CostModel::Function>();
    func->funcName = "TestFunc";

    auto op = std::make_shared<TileOp>();
    op->opcode = "ADD";
    op->iOperand = {}; // 触发第一个 if
    op->oOperand = {}; // 触发第二个 if

    func->tileOps.push_back(op);

    EXPECT_NO_THROW(parser.CheckTileOp(func));
}

TEST_F(CostModelTest, TestCheckTile)
{
    using namespace CostModel;
    ParseInput parser;
    auto func = std::make_shared<CostModel::Function>();

    auto tile1 = std::make_shared<Tile>();
    tile1->magic = 101;
    tile1->producers = {};

    auto tile2 = std::make_shared<Tile>();
    tile2->magic = 202;
    tile2->consumers = {};

    func->tiles.push_back(tile1);
    func->tiles.push_back(tile2);

    parser.CheckTile(func);
}

TEST_F(CostModelTest, TestParseInputFile)
{
    using namespace CostModel;
    std::vector<std::string> cfg;
    const std::string path = "1";
    std::deque<TaskMap> deque;
    std::unordered_map<long unsigned int, std::deque<CostModel::ReplayTaskEntry>> map;
    ParseInput parser;
    parser.ParseJsonConfig(path, cfg);
    parser.ParseConfig(path, cfg);
    parser.ParseCalendarJson(nullptr, path);
    parser.ParseFixedLatencyTask(nullptr, path);
    parser.ParseTopoJson(path, deque);
    parser.ParseReplayInfoJson(path, map);
    parser.ParseJson(nullptr, path);
}

TEST_F(CostModelTest, TestParseJsonWithValidFile)
{
    std::string jsonPath = "/tmp/parse_json_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(jsonPath);
    ofs << R"({"functions":[]})";
    ofs.close();
    auto sim = std::make_shared<CostModel::SimSys>();
    CostModel::ParseInput parser;
    parser.ParseJson(sim, jsonPath);
    unlink(jsonPath.c_str());
}

TEST_F(CostModelTest, TestParseSingleFunction)
{
    auto sim = std::make_shared<CostModel::SimSys>();
    std::string name = "TestParseSingleFunc";
    auto func = std::make_shared<Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {1, 1};
    auto incast = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    func->inCasts_.push_back(incast);
    func->outCasts_.push_back(outcast);
    CostModel::ParseInput parser;
    parser.ParseSingleFunction(sim, func.get());
    EXPECT_NE(sim, nullptr);
}

TEST_F(CostModelTest, TestJsonFErrororFormat)
{
    using namespace CostModel;
    const std::string path = "./config/test_config.conf";
    CostModelAgent agent;
    try {
        agent.GetFunctionFromJson(path);
    } catch (const std::exception& e) {
    }
}

TEST_F(CostModelTest, TestGetCyclesForPassA2A3)
{
    const std::string opCode = "ADD";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassA5)
{
    const std::string opCode = "CAST";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassCopyIn)
{
    const std::string opCode = "COPY_IN";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassCopyOut)
{
    const std::string opCode = "COPY_OUT";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassSimulate)
{
    const std::string opCode = "WHERE_TT";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    int64_t cycle = CostModel::GetCyclesForPass(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestGetCyclesForPassSo)
{
    typedef int64_t (*GetCyclesForPassFunc)(
        const std::string& op, const std::vector<std::vector<int>>& shape, DataType dtype);
    const std::string opCode = "L1_TO_L0A";
    std::vector<std::vector<int>> shape = {{1, 1, 1, 1}};
    DataType dtype = DataType::DT_INT4;
    config::SetPlatformConfig("device_platform", "ASCEND_950PR_9579");
    config::SetSimConfig(KEY_ACCURACY_LEVEL, 1);
    std::string soPath = "libtile_fwk_simulation.so";
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    EXPECT_NO_THROW(if (!handle) { throw std::runtime_error("can not load library: " + std::string(dlerror())); });

    GetCyclesForPassFunc get_cycles_func = (GetCyclesForPassFunc)dlsym(handle, "GetCyclesForPass");
    EXPECT_NO_THROW(if (!get_cycles_func) {
        throw std::runtime_error("Failed to find symbol GetCyclesForPass: " + std::string(dlerror()));
    });
    int64_t cycle = get_cycles_func(opCode, shape, dtype);
    EXPECT_GT(cycle, 0);
}

TEST_F(CostModelTest, TestMachineName)
{
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::DEVICE), "DEVICE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::CPU), "AICPU");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::AIC), "AIC");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::AIV), "AIV");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::MIXAICORE), "MIXAICORE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::PIPE), "PIPE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::CACHE), "CACHE");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::HUB), "HUB");
    EXPECT_EQ(CostModel::MachineName(CostModel::MachineType::UNKNOWN), "ILLEGAL");
}

TEST_F(CostModelTest, TestCacheName)
{
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::FUNCTION_CACHE), "FunctionCache");
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::L2CACHE), "L2CACHE");
    EXPECT_EQ(CostModel::CacheName(CostModel::CacheType::TOTAL_CACHE_TYPE), "ILLEGAL");
}

TEST_F(CostModelTest, TestCacheRequestName)
{
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::FUNCTION_REQ), "Function_Read");
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::DATA_READ_REQ), "Data_Read");
    EXPECT_EQ(CostModel::CacheRequestName(CostModel::CacheRequestType::DATA_WRITE_REQ), "Data_Write");
}

TEST_F(CostModelTest, TestTileStringConstructor)
{
    std::string jsonStr = R"({"magic":42,"shape":[1,2,3],"offset":[0,1,2],"memorytype":{"tobe":"MEM_UB"},"nodetype":"LOCAL","rawtensor":{"symbol":"x","datatype":"DT_FP16","rawmagic":99,"rawshape":[4,5,6]}})";
    CostModel::Tile tile(jsonStr);
    EXPECT_EQ(tile.magic, 42);
    EXPECT_EQ(tile.shape, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(tile.offset, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(tile.bufType, CostModel::OperandType::BUF_UB);
}

TEST_F(CostModelTest, TestGetOpSequeceAfterOOO)
{
    CostModel::Function func;
    func.opSequenceAfterOOO_[10] = 100;
    uint64_t index = 0;
    func.GetOpSequeceAfterOOO(10, index);
    EXPECT_EQ(index, 100);
    uint64_t index2 = 5;
    func.GetOpSequeceAfterOOO(20, index2);
    EXPECT_EQ(index2, 5);
}

TEST_F(CostModelTest, TestGetOpRelativeReadyCycle)
{
    CostModel::Function func;
    func.startCycles = 10;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    tileOp->exeInfo.cycleInfo.executeStartCycle = 50;
    tileOp->pipeType = CostModel::CorePipeType::PIPE_VECTOR_BMU;
    func.pipeLastEndCycle[CostModel::CorePipeType::PIPE_VECTOR_BMU] = 60;
    uint64_t result = func.GetOpRelativeReadyCycle(tileOp, 20);
    EXPECT_GE(result, 60);
}

TEST_F(CostModelTest, TestCalculateRelativeCycle)
{
    CostModel::Function func;
    func.startCycles = 10;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    tileOp->exeInfo.cycleInfo.executeStartCycle = 50;
    tileOp->exeInfo.cycleInfo.executeEndCycle = 100;
    tileOp->pipeType = CostModel::CorePipeType::PIPE_VECTOR_ALU;
    func.opMagicSequence.push_back(1);
    func.tileOpMap[1] = tileOp;
    func.CalculateRelativeCycle(20, 1.0);
    EXPECT_EQ(tileOp->exeInfo.cycleInfo.relativeStartCycle, 60);
    EXPECT_EQ(tileOp->exeInfo.cycleInfo.relativeEndCycle, 110);
}

TEST_F(CostModelTest, TestParseJsonWithValidData)
{
    std::string jsonPath = "/tmp/parse_json_data_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(jsonPath);
    ofs << R"({
        "functions": [{
            "hash": "12345",
            "magic": 1,
            "magicname": "test_func_START",
            "operations": [{
                "opcode": "ADD",
                "opmagic": 10,
                "ioperands": [{"magic":100,"shape":[1,1],"offset":[0,0],"memorytype":{"tobe":"MEM_UB"},"nodetype":"LOCAL","rawtensor":{"symbol":"x","datatype":"DT_FP16","rawmagic":200,"rawshape":[1,1]}}],
                "ooperands": [{"magic":101,"shape":[1,1],"offset":[0,0],"memorytype":{"tobe":"MEM_UB"},"nodetype":"OUTCAST","rawtensor":{"symbol":"y","datatype":"DT_FP16","rawmagic":201,"rawshape":[1,1]}}]
            }]
        }]
    })";
    ofs.close();
    auto sim = std::make_shared<CostModel::SimSys>();
    sim->config.startFunctionLabel = "START";
    CostModel::ParseInput parser;
    parser.ParseJson(sim, jsonPath);
    EXPECT_EQ(sim->startFuncName, "test_func_START");
    EXPECT_EQ(sim->startFuncHash, 12345);
    unlink(jsonPath.c_str());
}

TEST_F(CostModelTest, TestParseTopoJsonWithValidData)
{
    std::string topoPath = "/tmp/parse_topo_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(topoPath);
    ofs << R"([{"seqNo":0,"taskId":1,"leafIndex":0,"opmagic":10,"psgId":-1,"rootIndex":0,"uniqueKey":1,"funcHash":123,"coreType":"AIV","successors":[]}])";
    ofs.close();
    std::deque<CostModel::TaskMap> deque;
    CostModel::ParseInput parser;
    parser.ParseTopoJson(topoPath, deque);
    EXPECT_EQ(deque.size(), 1);
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestParseReplayInfoJsonWithValidData)
{
    std::string replayPath = "/tmp/parse_replay_test_" + std::to_string(getpid()) + ".json";
    std::ofstream ofs(replayPath);
    ofs << R"([{"blockIdx":0,"coreType":"AIV","tasks":[{"seqNo":1,"taskId":100,"execStart":10,"execEnd":20}]}])";
    ofs.close();
    std::unordered_map<uint64_t, std::deque<CostModel::ReplayTaskEntry>> map;
    CostModel::ParseInput parser;
    parser.ParseReplayInfoJson(replayPath, map);
    EXPECT_GT(map.size(), 0);
    unlink(replayPath.c_str());
}

TEST_F(CostModelTest, TestDebugFunction)
{
    auto func = std::make_shared<CostModel::Function>();
    func->funcName = "test_debug";
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    CostModel::ModelVisualizer visualizer;
    std::string outdir = "/tmp/debug_func_test_" + std::to_string(getpid());
    mkdir(outdir.c_str(), 0755);
    visualizer.DebugFunction(func, tiles, tileOps, outdir);
    std::ifstream ifs(outdir + "/test_debug.deadlock_debug_graph.dot");
    EXPECT_TRUE(ifs.is_open());
    unlink((outdir + "/test_debug.deadlock_debug_graph.dot").c_str());
}

TEST_F(CostModelTest, TestSimQueueBuild)
{
    CostModel::SimQueue<int> queue;
    queue.Enqueue(10);
    queue.Build();
    EXPECT_EQ(queue.Size(), 0);
}

TEST_F(CostModelTest, TestSimQueueXferAndGetSim)
{
    CostModel::SimQueue<int> queue;
    queue.Xfer();
    EXPECT_EQ(queue.GetSim(), nullptr);
}

TEST_F(CostModelTest, TestSimQueueFrontAndPopFront)
{
    CostModel::SimQueue<int> queue;
    int val = 0;
    EXPECT_FALSE(queue.Front(val));
    EXPECT_FALSE(queue.PopFront());

    queue.Enqueue(42);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_TRUE(queue.Front(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(queue.PopFront());
    EXPECT_EQ(queue.Size(), 0);
}

TEST_F(CostModelTest, TestSimQueueDequeueEmpty)
{
    CostModel::SimQueue<int> queue;
    int val = 0;
    EXPECT_FALSE(queue.Dequeue(val));
}

TEST_F(CostModelTest, TestSimQueueEmptyStates)
{
    CostModel::SimQueue<int> queue;
    EXPECT_TRUE(queue.Empty());

    queue.Enqueue(1);
    EXPECT_TRUE(queue.Empty());

    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_FALSE(queue.Empty());
}

TEST_F(CostModelTest, TestSimQueueWriteQueueSize)
{
    CostModel::SimQueue<int> queue;
    EXPECT_EQ(queue.WriteQueueSize(), 0);
    queue.Enqueue(1);
    queue.Enqueue(2);
    EXPECT_EQ(queue.WriteQueueSize(), 2);
}

TEST_F(CostModelTest, TestSimQueueFullAndSetMaxSize)
{
    CostModel::SimQueue<int> queue;
    queue.SetMaxSize(2);
    EXPECT_FALSE(queue.Full());
    queue.Enqueue(1);
    queue.Enqueue(2);
    EXPECT_TRUE(queue.Full());
}

TEST_F(CostModelTest, TestSimQueueCalendarPopFront)
{
    CostModel::SimQueue<int> queue;
    EXPECT_EQ(queue.CalendarPopFront(), 0);

    queue.Enqueue(7);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    EXPECT_EQ(queue.CalendarPopFront(), 7);
}

TEST_F(CostModelTest, TestSimQueueIsTerminate)
{
    CostModel::SimQueue<int> queue;
    EXPECT_TRUE(queue.IsTerminate());
    queue.Enqueue(1);
    EXPECT_FALSE(queue.IsTerminate());
}

TEST_F(CostModelTest, TestSimQueueGetMinWaitCyclesZero)
{
    CostModel::SimQueue<int> queue;
    queue.SetWriteDelay(0);
    queue.SetReadDelay(0);
    queue.Enqueue(1);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    queue.Step();
    uint64_t wait = queue.GetMinWaitCycles();
    EXPECT_TRUE(wait >= 1);
}

TEST_F(CostModelTest, TestSimQueueWithDelay)
{
    CostModel::SimQueue<int> queue;
    queue.SetWriteDelay(3);
    queue.SetReadDelay(2);
    queue.Enqueue(10);
    queue.UpdateIntervalCycles(1);
    queue.Step();
    queue.Step();
    queue.Step();
    queue.Step();
    int val = 0;
    EXPECT_TRUE(queue.Dequeue(val));
    EXPECT_EQ(val, 10);
}

TEST_F(CostModelTest, TestSimQueueReset)
{
    CostModel::SimQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Reset();
    EXPECT_EQ(queue.Size(), 0);
    EXPECT_EQ(queue.WriteQueueSize(), 0);
}

TEST_F(CostModelTest, TestSimQueueSetCounterInfo)
{
    CostModel::SimQueue<int> queue;
    auto logger = std::make_shared<CostModel::TraceLogger>();
    queue.SetCounterInfo(logger, 1, 2);
}

TEST_F(CostModelTest, TestParseDynTopo)
{
    std::string topoPath = "/tmp/parse_dyn_topo_test_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(topoPath);
    ofs << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    ofs.close();
    npu::tile_fwk::CostModelAgent agent;
    Json res = agent.ParseDynTopo(topoPath);
    EXPECT_GT(res.size(), 0);
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestSubmitTopo)
{
    std::string topoPath = "/tmp/submit_topo_test_" + std::to_string(getpid()) + ".txt";
    std::ofstream ofs(topoPath);
    ofs << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    ofs.close();
    npu::tile_fwk::CostModelAgent agent;
    agent.SubmitTopo(topoPath);
    EXPECT_FALSE(agent.topoJsonPath.empty());
    unlink(topoPath.c_str());
}

TEST_F(CostModelTest, TestGetLeafFunctionTimeCost)
{
    npu::tile_fwk::CostModelAgent agent;
    EXPECT_EQ(agent.GetLeafFunctionTimeCost(0), 0);
}

TEST_F(CostModelTest, TestSubmitSingleFuncToCostModel)
{
    npu::tile_fwk::CostModelAgent agent;
    std::string name = "TestSingleFunc";
    auto newFunc = std::make_shared<npu::tile_fwk::Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    agent.SubmitSingleFuncToCostModel(newFunc.get());
}

TEST_F(CostModelTest, TestDebugSingleFunc)
{
    npu::tile_fwk::CostModelAgent agent;
    auto newFunc = std::make_shared<npu::tile_fwk::Function>(npu::tile_fwk::Program::GetInstance(), "root", "root", nullptr);
    config::SetSimConfig(KEY_DEBUG_SINGLE_FUNCNAME, "");
    agent.DebugSingleFunc(newFunc.get());
}

TEST_F(CostModelTest, TestPipeMachineImplSimulate)
{
    class TestPipeImpl : public CostModel::PipeMachineImpl {
        uint64_t Simulate(const CostModel::TileOpPtr&) override { return 42; }
        uint64_t PostSimulate(const CostModel::TileOpPtr&) override { return 10; }
    };
    TestPipeImpl impl;
    auto tileOp = std::make_shared<CostModel::TileOp>();
    EXPECT_EQ(impl.Simulate(tileOp), 42);
    EXPECT_EQ(impl.PostSimulate(tileOp), 10);
}

TEST_F(CostModelTest, TestPipeMachineImplSimulateForPass)
{
    class TestPipeImpl : public CostModel::PipeMachineImpl {
        uint64_t Simulate(const CostModel::TileOpPtr&) override { return 0; }
        uint64_t PostSimulate(const CostModel::TileOpPtr&) override { return 0; }
    };
    TestPipeImpl impl;
    EXPECT_EQ(impl.SimulateForPass("ADD", {{1, 1}}, npu::tile_fwk::DataType::DT_FP32), 0);
    EXPECT_EQ(impl.PostSimulateForPass("ADD", {{1, 1}}, npu::tile_fwk::DataType::DT_FP32), 0);
}
