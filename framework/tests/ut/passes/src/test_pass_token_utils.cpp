/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/inner/tilefwk.h"
#include "ir/span.h"
#include "ir/type.h"
#include "passes/pass_utils/pass_token_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

class PassTokenUtilsTest : public testing::Test {
public:
    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

static ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

static Function* BuildFourOpTokenFunction(const std::string& name)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION(name)
    {
        Tensor a = Exp(input);
        Tensor b = Sqrt(a);
        Tensor c = Add(b, Element(DT_FP32, 1.0));
        output = Exp(c);
    }
    return Program::GetInstance().GetFunctionByRawName("TENSOR_" + name);
}

TEST_F(PassTokenUtilsTest, MoveTokenDependencyBeforeRemoveOp)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("PassUtilsTokenMoveBeforeRemove")
    {
        Tensor a = Exp(input);
        Tensor b = Sqrt(a);
        Tensor c = Add(b, Element(DT_FP32, 1.0));
        Tensor d = Exp(input);
        output = Add(c, d);
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PassUtilsTokenMoveBeforeRemove");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 5);

    auto tokenToRemoved = MakeToken("tokenToRemoved");
    auto tokenFromRemoved = MakeToken("tokenFromRemoved");
    auto& dep = func->GetVarDependency();
    dep.AddProducer(tokenToRemoved, std::static_pointer_cast<const ir::Stmt>(ops[3]));
    dep.AddConsumer(tokenToRemoved, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddProducer(tokenFromRemoved, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddConsumer(tokenFromRemoved, std::static_pointer_cast<const ir::Stmt>(ops[3]));
    ops[3]->result_token_ = {tokenToRemoved};
    ops[1]->tokens_.push_back(tokenToRemoved);
    ops[1]->result_token_ = {tokenFromRemoved};
    ops[3]->tokens_.push_back(tokenFromRemoved);

    PassTokenUtils::MoveTokenDependencyBeforeRemoveOp(*func, *ops[1]);

    EXPECT_TRUE(ops[1]->tokens_.empty());
    EXPECT_TRUE(ops[1]->result_token_.empty());
    EXPECT_TRUE(dep.HasConsumer(tokenToRemoved, std::static_pointer_cast<const ir::Stmt>(ops[2])));
    EXPECT_FALSE(dep.HasConsumer(tokenToRemoved, std::static_pointer_cast<const ir::Stmt>(ops[1])));
    ASSERT_FALSE(ops[0]->result_token_.empty());
    EXPECT_NE(ops[0]->result_token_.front(), tokenFromRemoved);
    EXPECT_FALSE(dep.HasDependency(tokenFromRemoved));
    EXPECT_TRUE(dep.HasProducer(ops[0]->result_token_.front(), std::static_pointer_cast<const ir::Stmt>(ops[0])));
    EXPECT_TRUE(dep.HasConsumer(ops[0]->result_token_.front(), std::static_pointer_cast<const ir::Stmt>(ops[3])));
    EXPECT_NE(std::find(ops[3]->tokens_.begin(), ops[3]->tokens_.end(), ops[0]->result_token_.front()),
              ops[3]->tokens_.end());
    EXPECT_EQ(std::find(ops[3]->tokens_.begin(), ops[3]->tokens_.end(), tokenFromRemoved), ops[3]->tokens_.end());
}

TEST_F(PassTokenUtilsTest, CopyTokenDependency)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("PassUtilsTokenCopy")
    {
        Tensor a = Exp(input);
        Tensor b = Sqrt(a);
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PassUtilsTokenCopy");
    ASSERT_NE(func, nullptr);

    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 3);

    auto inputToken = MakeToken("inputToken");
    auto outputToken = MakeToken("outputToken");
    auto& dep = func->GetVarDependency();
    dep.AddProducer(inputToken, std::static_pointer_cast<const ir::Stmt>(ops[0]));
    dep.AddConsumer(inputToken, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddProducer(outputToken, std::static_pointer_cast<const ir::Stmt>(ops[1]));
    dep.AddConsumer(outputToken, std::static_pointer_cast<const ir::Stmt>(ops[2]));
    ops[0]->result_token_ = {inputToken};
    ops[1]->tokens_.push_back(inputToken);
    ops[1]->result_token_ = {outputToken};
    ops[2]->tokens_.push_back(outputToken);

    auto& copiedOp = ops[1]->CloneOperation(*func, ops[1]->GetIOperands(), ops[1]->GetOOperands());
    PassTokenUtils::CopyTokenDependency(*func, *ops[1], copiedOp);

    EXPECT_EQ(copiedOp.tokens_.size(), 1);
    EXPECT_EQ(copiedOp.tokens_[0], inputToken);
    ASSERT_FALSE(copiedOp.result_token_.empty());
    EXPECT_NE(copiedOp.result_token_.front(), outputToken);
    EXPECT_TRUE(dep.HasConsumer(inputToken, std::static_pointer_cast<const ir::Stmt>(copiedOp.shared_from_this())));
    EXPECT_TRUE(dep.HasProducer(copiedOp.result_token_.front(),
                                std::static_pointer_cast<const ir::Stmt>(copiedOp.shared_from_this())));
    EXPECT_TRUE(dep.HasConsumer(copiedOp.result_token_.front(), std::static_pointer_cast<const ir::Stmt>(ops[2])));
    EXPECT_NE(std::find(ops[2]->tokens_.begin(), ops[2]->tokens_.end(), copiedOp.result_token_.front()),
              ops[2]->tokens_.end());
}

TEST_F(PassTokenUtilsTest, MoveResultTokensToProducers)
{
    Function* func = BuildFourOpTokenFunction("PassUtilsMoveResultToken");
    ASSERT_NE(func, nullptr);
    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 4);
    auto oldToken = MakeToken("oldResultToken");
    auto& dep = func->GetVarDependency();
    dep.AddProducer(oldToken, ToStmtPtr(*ops[1]));
    dep.AddConsumer(oldToken, ToStmtPtr(*ops[2]));
    dep.AddConsumer(oldToken, ToStmtPtr(*ops[3]));
    ops[1]->result_token_ = {oldToken};
    ops[2]->tokens_.push_back(oldToken);
    ops[3]->tokens_.push_back(oldToken);

    PassTokenUtils::MoveResultTokensToProducers(*func, {ops[1].get()}, {ops[0].get()}, {ToStmtPtr(*ops[3])});

    EXPECT_TRUE(ops[1]->result_token_.empty());
    EXPECT_FALSE(dep.HasDependency(oldToken));
    ASSERT_FALSE(ops[0]->result_token_.empty());
    EXPECT_TRUE(dep.HasProducer(ops[0]->result_token_.front(), ToStmtPtr(*ops[0])));
    EXPECT_TRUE(dep.HasConsumer(ops[0]->result_token_.front(), ToStmtPtr(*ops[2])));
    EXPECT_FALSE(dep.HasConsumer(ops[0]->result_token_.front(), ToStmtPtr(*ops[3])));
    EXPECT_NE(std::find(ops[2]->tokens_.begin(), ops[2]->tokens_.end(), ops[0]->result_token_.front()),
              ops[2]->tokens_.end());
    EXPECT_EQ(std::find(ops[3]->tokens_.begin(), ops[3]->tokens_.end(), oldToken), ops[3]->tokens_.end());
}

TEST_F(PassTokenUtilsTest, CleanupDeletedTokenDependency)
{
    Function* func = BuildFourOpTokenFunction("PassUtilsCleanupToken");
    ASSERT_NE(func, nullptr);
    auto& ops = func->operations_;
    ASSERT_GE(ops.size(), 4);
    auto externalToken = MakeToken("externalToken");
    auto deletedOnlyToken = MakeToken("deletedOnlyToken");
    auto& dep = func->GetVarDependency();
    dep.AddProducer(externalToken, ToStmtPtr(*ops[0]));
    dep.AddConsumer(externalToken, ToStmtPtr(*ops[1]));
    dep.AddProducer(deletedOnlyToken, ToStmtPtr(*ops[1]));
    dep.AddConsumer(deletedOnlyToken, ToStmtPtr(*ops[2]));
    ops[0]->result_token_ = {externalToken};
    ops[1]->tokens_.push_back(externalToken);
    ops[1]->result_token_ = {deletedOnlyToken};
    ops[2]->tokens_.push_back(deletedOnlyToken);

    PassTokenUtils::CleanupDeletedTokenDependency(*func, {ops[1].get(), ops[2].get()});

    EXPECT_TRUE(dep.HasDependency(externalToken));
    EXPECT_TRUE(dep.HasProducer(externalToken, ToStmtPtr(*ops[0])));
    EXPECT_FALSE(dep.HasConsumer(externalToken, ToStmtPtr(*ops[1])));
    EXPECT_FALSE(dep.HasDependency(deletedOnlyToken));
    EXPECT_TRUE(ops[1]->tokens_.empty());
    EXPECT_TRUE(ops[1]->result_token_.empty());
    EXPECT_TRUE(ops[2]->tokens_.empty());
}
