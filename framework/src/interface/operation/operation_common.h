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
 * \file operation_common.h
 * \brief
 */

#ifndef INTERFACE_MAIN_OPERATION_COMMON_H
#define INTERFACE_MAIN_OPERATION_COMMON_H

#include "interface/utils/common.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {
#define CALL(n, ...) Tensor##n(__VA_ARGS__)
#define RETURN_CALL(n, ...) return Tensor##n(__VA_ARGS__)

constexpr int32_t NUM_VALUE_0 = 0;
constexpr int32_t NUM_VALUE_1 = 1;
constexpr int32_t NUM_VALUE_2 = 2;
constexpr int32_t NUM_VALUE_3 = 3;
constexpr int32_t NUM_VALUE_4 = 4;
constexpr int32_t NUM_VALUE_5 = 5;
constexpr int32_t NUM_VALUE_8 = 8;
constexpr int32_t NUM_VALUE_10 = 10;
constexpr int32_t NUM_VALUE_16 = 16;
constexpr int32_t NUM_VALUE_31 = 31;
constexpr int32_t NUM_VALUE_32 = 32;
constexpr int32_t NUM_VALUE_64 = 64;
constexpr double NUM_VALUE_0_5 = 0.5;
constexpr double NUM_VALUE_EPS = 1e-9;

struct TileInfo {
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
    std::vector<SymbolicScalar> validShape;

    TileInfo(size_t shapeSize, size_t offsetSize) : shape(shapeSize), offset(offsetSize), validShape(shapeSize) {}

    TileInfo(std::vector<int64_t> aShape, std::vector<int64_t> aOffset, std::vector<SymbolicScalar> aValidShape = {})
        : shape(std::move(aShape)), offset(std::move(aOffset)), validShape(aValidShape) {}
};

struct Input {
    const Tensor tensor;
    TileInfo tileInfo;
};

void CheckTensorShape(const LogicalTensorPtr &tensor, const std::string &op);
std::vector<int> GetBroadCastShape(LogicalTensorPtr &operand1, LogicalTensorPtr &operand2);
std::vector<int> GetBroadcastAxes(const Shape &shape1, const Shape &shape2);
void CheckAxisRange(const Tensor &tensor, int &axis);

using TiledFuncType = std::function<void(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op)>;
class TiledFuncRegistry {
private:
    TiledFuncRegistry() = default;
    ~TiledFuncRegistry() = default;

public:
    static TiledFuncRegistry &GetInstance() {
        static TiledFuncRegistry instance;
        return instance;
    }

    void RegisterTiledFunc(const Opcode opcode, TiledFuncType func) { tiledFuncs_[opcode] = func; }

    TiledFuncType GetTiledFunc(const Opcode opcode) {
        auto it = tiledFuncs_.find(opcode);
        if (it == tiledFuncs_.end()) {
            return nullptr;
        }
        return tiledFuncs_[opcode];
    }

private:
    std::unordered_map<Opcode, TiledFuncType> tiledFuncs_;
};

#define REGISTER_OPERATION_TILED_FUNC(OpCoreStr, OpType, FuncName)                \
    class OpCoreStr##TiledRegister {                                              \
    public:                                                                       \
        OpCoreStr##TiledRegister() {                                              \
            TiledFuncRegistry::GetInstance().RegisterTiledFunc(OpType, FuncName); \
        }                                                                         \
    };                                                                            \
    static OpCoreStr##TiledRegister OpCoreStr##_tiled_register

class OpSyncQueue {
public:
    OpSyncQueue() {}
    OpSyncQueue(PipeType pipeId, PipeType trigPipeId, CoreType coreType, CoreType tirgCoreType, int evid)
        : pipeId_(pipeId), trigPipeId_(trigPipeId), coreType_(coreType), trigCoreType_(tirgCoreType), eventId_(evid) {}

    OpSyncQueue(int bufid, const std::vector<int> &offset, CoreType coreType, CoreType tirgCoreType)
        : coreType_(coreType), trigCoreType_(tirgCoreType), gMBufId(bufid), offset_(offset) {}

    PipeType pipeId_{PIPE_S};
    PipeType trigPipeId_{PIPE_S};
    CoreType coreType_{CoreType::AIV};
    CoreType trigCoreType_{CoreType::AIV};
    int eventId_{0};
    int gMBufId{0};
    std::vector<int> offset_;

    Json ToJson() const {
        Json j;
        j["pipe_id"] = pipeId_;
        j["trig_pipe"] = trigPipeId_;
        j["core_type"] = static_cast<int>(coreType_);
        j["tri_core_type"] = static_cast<int>(trigCoreType_);
        j["event_id"] = eventId_;
        j["gm_buf_id"] = gMBufId;
        j["offset"] = offset_;
        return j;
    }

    void FromJson(const Json &j) {
        pipeId_ = static_cast<PipeType>(j["pipe_id"].get<int>());
        trigPipeId_ = static_cast<PipeType>(j["trig_pipe"].get<int>());
        coreType_ = static_cast<CoreType>(j["core_type"].get<int>());
        trigCoreType_ = static_cast<CoreType>(j["tri_core_type"].get<int>());
        eventId_ = j["event_id"].get<int>();
        gMBufId = j["gm_buf_id"].get<int>();
        offset_ = j["offset"].get<std::vector<int>>();
    }

    std::string Dump() const {
        std::ostringstream oss;
        oss << GetPipeTypeDict().Find(pipeId_) << "," << GetPipeTypeDict().Find(trigPipeId_) << "," << eventId_;
        return oss.str();
    }
};
} // namespace npu::tile_fwk

#endif // INTERFACE_MAIN_OPERATION_COMMON_H
