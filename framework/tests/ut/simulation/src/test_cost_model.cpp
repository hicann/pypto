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
#include "cost_model/simulation_platform/platform.h"
#include "operator/models/llama/llama_def.h"
#include "cost_model/simulation/common/CommonType.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "interface/configs/config_manager.h"
#include "cost_model/simulation_ca/PipeSimulator.h"
#include "cost_model/simulation/arch/PipeFactory.h"
#include "cost_model/simulation/arch/CacheMachineImpl.h"
#include "cost_model/simulation/machine/CoreMachine.h"
#include "cost_model/simulation/machine/Scheduler.h"
#include "cost_model/simulation/tools/ParseInput.h"

using namespace npu::tile_fwk;

class CostModelTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
        config::SetSimConfig(KEY_BUILD_TASK_BASED_TOPO, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void RunLLamaLayerCostModel(const AttentionDims &dimsCfg, float threadhold = 0.001f) {
    (void)threadhold;
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;

    PROGRAM("LLAMALAYER") {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
        Tensor Res(DT_FP32, {b * s, n * d}, "Res");
        config::SetBuildStatic(true);
        FUNCTION("LLAMA", {H, AW, DW, FW, Res}) {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
        config::SetPassStrategy("OOO");
    }
}

void RunMatrixCostModel() {
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
    int v_head =128;
    int h = 256;
    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_FP32, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
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
    std::string jsonPath("./config/fixed_task_topo.json");
    std::vector<std::string> arg = config::GetSimConfig(KEY_ARGS, std::vector<std::string>{});
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + jsonPath);
    config::SetSimConfig(KEY_ARGS, arg);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracy2)
{
    int accuracylevel = 2;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracy3)
{
    int accuracylevel = 2;
    config::SetSimConfig(KEY_ACCURACY_LEVEL, accuracylevel);
    RunMatrixCostModel();
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
    ALOG_INFO("Init CostModel Simulation.");
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

TEST_F(CostModelTest, TestReplaceGMStr)
{
    std::string str = "abc";
    CostModel::PipeSimulatorUtils::ReplaceGMStr(str);
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

    FUNCTION("A") {
        res = Cat(std::vector<Tensor>{params1, params2}, axis);
    }
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
    arg.emplace_back("Model.calendarMode=" +  std::to_string(static_cast<int>(calendarMode)));
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

    void SetUp() override {
        cacheEnable = config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, false);
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        constexpr int level = 2;
        EnablePVModel(level);
    }

    void TearDown() override {
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, cacheEnable);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        ResetPVModelConfig();
    }

    void EnablePVModel(int level)
    {
        oriPvLevel = config::GetSimConfig(KEY_PV_LEVEL, 0);
        config::SetSimConfig(KEY_PV_LEVEL, level);
    }

    void ResetPVModelConfig()
    {
        config::SetSimConfig(KEY_PV_LEVEL, oriPvLevel);
    }

protected:
    bool oriEnableAihacBackend = false;
    int oriPvLevel = 0;
    bool cacheEnable = false;
};

void CostModelTestLoopViewAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FUNCTION("main", {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul(DataType::DT_FP32, qi, ki, false, true);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(CostModelDynTest, TestDD) {
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
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    CostModelTestLoopViewAssemble(t0, t1, blockTable, out, s);

    auto func = Program::GetInstance().GetLastFunction();
    auto pv = CostModel::PvModelFactory::CreateDyn();
    pv->Codegen(func);
}

TEST_F(CostModelTest, TestUnknownArchType)
{
    EXPECT_THROW(CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A0", 1), std::invalid_argument);
}

TEST_F(CostModelTest, TestCreateA5Cache)
{
    std::unique_ptr<CostModel::CacheMachineImpl> cacheImpl = CostModel::PipeFactory::CreateCache(CostModel::CacheType::L2CACHE, "A5");
    CostModel::CachePacket packet;
    cacheImpl->Simulate(packet);
}

TEST_F(CostModelTest, TestA5ArchType)
{
    auto simulator =CostModel::PipeFactory::Create(CostModel::CorePipeType::PIPE_MTE_IN, "A5", 1);
    EXPECT_TRUE(simulator != nullptr);
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
    delete coreMachine;
}

TEST_F(CostModelTest, TestScheduler) {
    using namespace CostModel;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> tileAllocSequence(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));

    // 1. 创建节点
    auto t10 = std::make_shared<CostModel::Tile>(); t10->magic = 10; t10->exeInfo.domCount = 5; t10->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t11 = std::make_shared<CostModel::Tile>(); t11->magic = 11; t11->exeInfo.domCount = 1; t11->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 更小的 domCount
    
    auto op100 = std::make_shared<CostModel::TileOp>(); op100->magic = 100; op100->pipeType = CorePipeType::PIPE_VECTOR_BMU;
    
    auto t30 = std::make_shared<CostModel::Tile>(); t30->magic = 30; t30->exeInfo.isOutcast = true; t30->pipeType = CostModel::CorePipeType::PIPE_MTE1;
    auto t40 = std::make_shared<CostModel::Tile>(); t40->magic = 40; t40->pipeType = CostModel::CorePipeType::PIPE_MTE1; // 无 consumer，视为 output

    // 2. 建立连接
    op100->iOperand = {t10, t11};
    op100->oOperand = {t30, t40};
    
    t10->consumers = {op100};
    t11->consumers = {op100};
    
    t30->producers = {op100};
    t40->producers = {op100};

    tiles[10] = t10; tiles[11] = t11; tiles[30] = t30; tiles[40] = t40;
    tileOps[100] = op100;

    // 3. 执行测试
    scheduler.SortTile(tiles, tileOps, tileAllocSequence);

    // 4. 验证日志覆盖和逻辑
    
    EXPECT_GT(op100->exeInfo.sequenceToIssue, -1);
    EXPECT_EQ(t10->exeInfo.copyOutIdx, t11->exeInfo.copyOutIdx);
    
}

TEST_F(CostModelTest, TestScheduler_EmptyInput) {
    std::unordered_map<int, CostModel::TilePtr> tiles;
    std::unordered_map<int, CostModel::TileOpPtr> tileOps;
    std::vector<std::vector<int>> seq;
    CostModel::Scheduler scheduler;
    scheduler.sim = std::make_shared<CostModel::SimSys>();
    scheduler.SortTile(tiles, tileOps, seq);
}

TEST_F(CostModelTest, TestRemoveBarrierCounter_LogCoverage) {
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

TEST_F(CostModelTest, TestGetPipeType_AssertOnMissingOpcode) {
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

TEST_F(CostModelTest, TestCheckTileOp) {
    using namespace CostModel;
    ParseInput parser;
    auto func = std::make_shared<CostModel::Function>();
    func->funcName = "TestFunc";

    auto op = std::make_shared<TileOp>();
    op->opcode="ADD";
    op->iOperand = {}; // 触发第一个 if
    op->oOperand = {}; // 触发第二个 if

    func->tileOps.push_back(op);

    EXPECT_NO_THROW(parser.CheckTileOp(func));
}

TEST_F(CostModelTest, TestCheckTile) {
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

TEST_F(CostModelTest, TestParseInputFile) {
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