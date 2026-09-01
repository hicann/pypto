/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "ir/span.h"
#include "ir/type.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "passes/tensor_graph_pass/merge_dangling_assemble_output.h"

namespace npu::tile_fwk {
namespace {

constexpr int64_t DIM = 16;

ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

class MergeDanglingAssembleOutputTest : public testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        function_ = std::make_shared<Function>(Program::GetInstance(), "merge_dangling_magic", "merge_dangling",
                                               nullptr);
        Program::GetInstance().SetCurrentFunction(function_.get());
    }

    void TearDown() override { Program::GetInstance().Reset(); }

    LogicalTensorPtr MakeTensor(const std::string& name)
    {
        return builder_.CreateTensorVar(*function_, DT_FP32, {DIM, DIM}, TileOpFormat::TILEOP_ND, name);
    }

    LogicalTensorPtr MakeVersion(const LogicalTensorPtr& tensor)
    {
        return builder_.CreateTensorVar(*function_, tensor->GetRawTensor(), tensor->GetOffset(), tensor->GetShape(),
                                        tensor->GetDynValidShape());
    }

    Operation& AddAssemble(const LogicalTensorPtr& source, const LogicalTensorPtr& output)
    {
        return function_->AddRawOperation(Opcode::OP_ASSEMBLE, {source}, {output});
    }

    Operation& AddConsumer(const LogicalTensorPtr& input)
    {
        auto rhs = MakeTensor("rhs");
        auto output = MakeTensor("consumer_output");
        return function_->AddRawOperation(Opcode::OP_ADD, {input, rhs}, {output});
    }

    void AddTokenDependency(Operation& producer, Operation& consumer)
    {
        if (producer.result_token_.empty()) {
            producer.result_token_ = {std::make_shared<ir::Var>("token_" + std::to_string(producer.GetOpMagic()),
                                                                ir::GetTokenType(), ir::Span::Unknown())};
        }
        auto token = producer.result_token_.front();
        consumer.tokens_.push_back(token);
        function_->GetVarDependency().AddProducer(token, ToStmtPtr(producer));
        function_->GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    }

    Status RunPass()
    {
        MergeDanglingAssembleOutput pass;
        return pass.Run(*function_, "MergeDanglingAssembleOutputTest", "MergeDanglingAssembleOutput");
    }

    IRBuilder builder_;
    std::shared_ptr<Function> function_;
};

/*
 * Scenario 3: Dangling T1 merges into consumed T2.
 * Before: A1->T1, A1->token->A2->T2->C2  (T1 has no consumer)
 * After:  A1/A2->T2->C2                   (T1 is merged into T2)
 */
TEST_F(MergeDanglingAssembleOutputTest, MergeIntoFirstObservableSuccessor)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(assemble1, assemble2);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
}

/*
 * Scenario 2: A1->T1->C1->token->A2->T2->C2.
 * Both assemble outputs are consumed, so the graph remains unchanged.
 */
TEST_F(MergeDanglingAssembleOutputTest, PreserveConsumedIntermediateVersion)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& consumer1 = AddConsumer(version1); // WAR: consumer1->assemble2 token ordering
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(consumer1, assemble2);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version1);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_FALSE(version1->GetConsumers().empty());
    EXPECT_FALSE(version2->GetConsumers().empty());
}

/*
 * Scenario 4: A1->T1->C1->token->A2->T2->C2->token->A3->T3->C3.
 * All three assemble outputs are consumed, so the graph remains unchanged.
 */
TEST_F(MergeDanglingAssembleOutputTest, Scenario4PreserveAllConsumedVersions)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& consumer1 = AddConsumer(version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(consumer1, assemble2);
    auto& consumer2 = AddConsumer(version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(consumer2, assemble3);
    AddConsumer(version3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version1);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_FALSE(version1->GetConsumers().empty());
    EXPECT_FALSE(version2->GetConsumers().empty());
    EXPECT_FALSE(version3->GetConsumers().empty());
}

TEST_F(MergeDanglingAssembleOutputTest, MergeAllUnconsumedVersionsIntoLastVersion)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(assemble1, assemble2);
    AddTokenDependency(assemble2, assemble3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version3);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version3);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_TRUE(version2->GetProducers().empty());
}

TEST_F(MergeDanglingAssembleOutputTest, MergeVersionsWithDifferentDynamicValidShapes)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    version1->UpdateDynValidShape({SymbolicScalar(DIM / 2), SymbolicScalar(DIM)});
    version2->UpdateDynValidShape({SymbolicScalar(DIM), SymbolicScalar(DIM)});

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(assemble1, assemble2);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(version2->GetDynValidShape(), (std::vector<SymbolicScalar>{SymbolicScalar(DIM), SymbolicScalar(DIM)}));
}

/*
 * Ordered outcasts do not represent an intermediate snapshot read.
 * Before: A1->T1(outcast), A1->token->A2->T2(outcast).
 * After:  A1/A2->T2, and both outcast slots refer to T2.
 */
TEST_F(MergeDanglingAssembleOutputTest, MergeOrderedOutcastVersions)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    function_->outCasts_.push_back(version1);
    function_->outCasts_.push_back(version2);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(assemble1, assemble2);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    ASSERT_EQ(function_->GetOutcast().size(), 2U);
    EXPECT_EQ(function_->GetOutcast()[0], version2);
    EXPECT_EQ(function_->GetOutcast()[1], version2);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
}

/*
 * Non-overlapping writes with no token still merge by version lineage.
 * Before: A1->T1 (no token), A2->T2->C2. T1 is dangling.
 * After:  A1/A2->T2 (T1 merged into T2). Version order alone decides the merge; tokens are irrelevant.
 */
TEST_F(MergeDanglingAssembleOutputTest, MergeByVersionOrderWithoutToken)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
}

/*
 * Scenario 5: Token chain ending at the only consumed version.
 * Before: A1->token->A2->token->A3, only T3 consumed.
 * After:  All A1/A2/A3 output T3 (T1 and T2 merged into T3).
 */
TEST_F(MergeDanglingAssembleOutputTest, MergeTokenChainEndingAtObservableVersion)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(assemble1, assemble2);
    AddTokenDependency(assemble2, assemble3);
    AddConsumer(version3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version3);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version3);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_TRUE(version2->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version2->GetMagic()), nullptr);
}

/*
 * Scenario 6: Merge only the unconsumed middle version.
 * Before: A1->T1->C1->token->A2->T2; A2->token->A3->T3->C3.
 * After:  A1 outputs T1 (consumed); A2/A3 output T3 (T2 merged into T3).
 */
TEST_F(MergeDanglingAssembleOutputTest, Scenario6MergeOnlyUnconsumedMiddleVersion)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& consumer1 = AddConsumer(version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(consumer1, assemble2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(assemble2, assemble3);
    AddConsumer(version3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version1);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version3);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_FALSE(version1->GetConsumers().empty());
    EXPECT_FALSE(version3->GetConsumers().empty());
    EXPECT_TRUE(version2->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version2->GetMagic()), nullptr);
}

/*
 * Scenario 7: Merge dangling first version into consumed second version.
 * Before: A1->T1 and A1->token->A2->T2->C2->token->A3->T3->C3.
 * After:  A1/A2 output T2 (T1 merged into T2); A3 outputs T3 unchanged.
 */
TEST_F(MergeDanglingAssembleOutputTest, Scenario7MergeFirstVersionIntoConsumedSecondVersion)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddTokenDependency(assemble1, assemble2);
    auto& consumer2 = AddConsumer(version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(consumer2, assemble3);
    AddConsumer(version3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_FALSE(version2->GetConsumers().empty());
    EXPECT_FALSE(version3->GetConsumers().empty());
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
}

/*
 * Version lineage ignores token forks: a dangling first version merges into its nearest live successor
 * regardless of how many token edges leave it.
 * Before: A1->T1 (token forks to A2 and A3); A2->T2->C2; A3->T3->C3. T1 is dangling.
 * After:  A1/A2->T2 (T1 merged into nearest live T2); A3->T3 unchanged.
 */
TEST_F(MergeDanglingAssembleOutputTest, MergeByVersionOrderIgnoringTokenFork)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddTokenDependency(assemble1, assemble2);
    AddTokenDependency(assemble1, assemble3);
    AddConsumer(version2);
    AddConsumer(version3);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    EXPECT_TRUE(version1->GetProducers().empty());
    EXPECT_EQ(function_->GetTensorMap().GetTensorByMagic(version1->GetMagic()), nullptr);
}

/*
 * Post-condition: no dangling residual. After the pass every assemble output either has an Operation
 * consumer or is the canonical sink (no compatible successor); no compatible dangling version survives.
 */
TEST_F(MergeDanglingAssembleOutputTest, PostConditionNoDanglingResidual)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);
    auto version3 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& assemble3 = AddAssemble(source, version3);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);

    // v1 dangling -> nearest live v2; v3 dangling with no live successor -> canonical sink (itself).
    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble3.GetOutputOperand(0), version3);
    // No surviving version is a compatible dangling one: v2 has a consumer, v3 is the last version.
    EXPECT_FALSE(version2->GetConsumers().empty());
    EXPECT_TRUE(version3->GetConsumers().empty());
}

/*
 * Idempotency: running the pass a second time must not change the graph.
 */
TEST_F(MergeDanglingAssembleOutputTest, IdempotentOnSecondRun)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    AddConsumer(version2);

    ASSERT_EQ(RunPass(), SUCCESS);
    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);

    // Second run is a no-op: the only remaining assemble outputs are the live version2.
    ASSERT_EQ(RunPass(), SUCCESS);
    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
}

/*
 * After merge, the producer's result token is consumed only by target's data consumer, so pruning that edge
 * leaves the token with no consumer. The producer must drop the now-orphan result token.
 */
TEST_F(MergeDanglingAssembleOutputTest, ClearProducerTokenWhenOrphanedAfterMerge)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& consumer2 = AddConsumer(version2);
    AddTokenDependency(assemble1, consumer2);

    ASSERT_FALSE(assemble1.result_token_.empty());

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(consumer2.tokens_.empty());
    EXPECT_TRUE(assemble1.result_token_.empty());
}

/*
 * The producer's result token is also consumed by an op that does not read the merge target. Pruning only
 * removes the redundant edge on target's consumer; the token still has a consumer, so it must be retained.
 */
TEST_F(MergeDanglingAssembleOutputTest, KeepProducerTokenWhenOtherConsumerRemains)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& consumer2 = AddConsumer(version2);
    AddTokenDependency(assemble1, consumer2);
    auto& unrelated = AddConsumer(MakeTensor("unrelated"));
    AddTokenDependency(assemble1, unrelated);

    ASSERT_EQ(RunPass(), SUCCESS);

    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(consumer2.tokens_.empty());
    EXPECT_FALSE(assemble1.result_token_.empty());
    ASSERT_EQ(unrelated.tokens_.size(), 1U);
    EXPECT_EQ(unrelated.tokens_[0].get(), assemble1.result_token_.front().get());
}

/*
 * After a merge that prunes a redundant token edge, the surviving operation token fields and the
 * Function-level VarDependency must agree, and no dependency entry may reference a detached
 * operation. The pass rebuilds VarDependency via TokenUtils::RebuildTokenDependencies, so every
 * producer/consumer statement must be a currently-owned operation.
 */
TEST_F(MergeDanglingAssembleOutputTest, TokenDependenciesAgreeWithFieldsAfterMerge)
{
    auto source = MakeTensor("source");
    auto base = MakeTensor("base");
    auto version1 = MakeVersion(base);
    auto version2 = MakeVersion(base);

    auto& assemble1 = AddAssemble(source, version1);
    auto& assemble2 = AddAssemble(source, version2);
    auto& consumer2 = AddConsumer(version2);
    AddTokenDependency(assemble1, consumer2);
    auto& unrelated = AddConsumer(MakeTensor("unrelated"));
    AddTokenDependency(assemble1, unrelated);

    ASSERT_FALSE(assemble1.result_token_.empty());
    auto survivingToken = assemble1.result_token_.front();

    ASSERT_EQ(RunPass(), SUCCESS);

    // Field-level expectations: the redundant edge to consumer2 is pruned; the unrelated consumer
    // keeps the producer's result token alive.
    EXPECT_EQ(assemble1.GetOutputOperand(0), version2);
    EXPECT_EQ(assemble2.GetOutputOperand(0), version2);
    EXPECT_TRUE(consumer2.tokens_.empty());
    EXPECT_FALSE(assemble1.result_token_.empty());
    ASSERT_EQ(unrelated.tokens_.size(), 1U);
    EXPECT_EQ(unrelated.tokens_[0].get(), survivingToken.get());

    // VarDependency must agree with the surviving fields.
    const auto& dep = function_->GetVarDependency();
    const auto& producers = dep.GetProducers(survivingToken);
    EXPECT_EQ(producers.size(), 1U);
    EXPECT_EQ(producers.count(ToStmtPtr(assemble1)), 1U);

    const auto& consumers = dep.GetConsumers(survivingToken);
    EXPECT_EQ(consumers.size(), 1U);
    EXPECT_EQ(consumers.count(ToStmtPtr(unrelated)), 1U);
    EXPECT_EQ(consumers.count(ToStmtPtr(consumer2)), 0U);

    // No dependency entry references a detached/stale operation: every producer and consumer
    // statement must be a currently-owned operation.
    std::unordered_set<ir::StmtPtr> currentStmts;
    for (auto& op : function_->Operations(false)) {
        currentStmts.insert(ToStmtPtr(op));
    }
    for (const auto& entry : dep.GetAllDependencies()) {
        for (const auto& stmt : entry.second.producers) {
            EXPECT_EQ(currentStmts.count(stmt), 1U);
        }
        for (const auto& stmt : entry.second.consumers) {
            EXPECT_EQ(currentStmts.count(stmt), 1U);
        }
    }
}
} // namespace
} // namespace npu::tile_fwk
