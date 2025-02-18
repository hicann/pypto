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
 * \file infer_param_index.h
 * \brief
 */

#ifndef INFER_PARAM_INDEX_PASS_H_
#define INFER_PARAM_INDEX_PASS_H_
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/topo_program.h"

namespace npu {
namespace tile_fwk {
class InferParamIndex : public Pass {
public:
    InferParamIndex() : Pass("InferParamIndex") {}
    ~InferParamIndex() override {}
    Status RunOnFunction(Function &function) override;

private:
    std::string DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable);
    Status ResetOutputDynValidShape(const Operation &op);
    Status ResetViewDynValidShape(const Operation &op);
    Status ResetAssembleDynValidShape(const Operation &op);
    Status ResetDynValidShape(Function& function);
    Status UpdateValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified);
    Status SetSubValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified);
    Status UpdateParamIndex(Function &function);
    Status InferShape(Function& function);
};
}
}
#endif