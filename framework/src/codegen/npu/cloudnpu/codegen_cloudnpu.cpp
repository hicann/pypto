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
 * \file codegen_cloudnpu.cpp
 * \brief
 */
#include "codegen_cloudnpu.h"

#include "codegen_op_cloudnpu.h"
#include "codegen/utils/parallel_execute.h"

namespace npu::tile_fwk {

void CodeGenCloudNPU::GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss) const
{
    OperationsViewer operationList = subFunc.Operations(false);
    if (operationList.IsEmpty()) {
        CODEGEN_LOGW(
            "operationList from PASS is empty, func magic name: %s, func hash: %s", subFunc.GetMagicName().c_str(),
            subFunc.GetFunctionHash().c_str());
    }

    CODEGEN_LOGI(
        "TopFunc Type is %s\nFunction to codegen:\n %s\n", topFunc.GetFunctionTypeStr().c_str(),
        topFunc.Dump().c_str());

    std::shared_ptr<SymbolManager> symbolMgr = std::make_shared<SymbolManager>();
    std::shared_ptr<ForBlockManager> forBlkMgr = std::make_shared<ForBlockManager>(symbolMgr);
    FloatSpecValMgr floatSpecValMgr;
    std::string allocSourceRegion;
    allocSourceRegion.reserve(CODE_RESERVED_SIZE);
    std::string tileOpSourceRegion;
    tileOpSourceRegion.reserve(CODE_RESERVED_SIZE);
    auto locToOffsetMap = GenRealizeIdMap(subFunc.GetParameter());
    for (const auto& op : operationList) {
        CODEGEN_LOGI(
            "======================== Op CodeGenNPU Start ========================\nGen OP IS: %s", op.Dump().c_str());
        if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
            CODEGEN_LOGI("ignore this op\n------------------------ Op CodeGenNPU Finish -----------------------");
            continue;
        }

        std::string allocSourceCode = GenAllocForLocalBuffer(op, symbolMgr);
        floatSpecValMgr.UpdateByOp(op);

        CodeGenOpCloudNPU cop(
            {symbolMgr, topFunc, subFunc, op, locToOffsetMap, ctx.isMainBlock, ctx.isDynamicAligned, forBlkMgr});
        std::string tileOpSourceCode = cop.GenOpCode();
        ASSERT(GenCodeErr::GEN_OP_CODE_FAILED, tileOpSourceCode.find("CG_ERROR") == tileOpSourceCode.npos)
            << "Generate code of op failed, op is " << op.Dump();

        allocSourceRegion.append(allocSourceCode);
        tileOpSourceRegion.append(tileOpSourceCode);

        if (!allocSourceCode.empty()) {
            CODEGEN_LOGI(": extra alloc generated(moved up to alloc region): %s", allocSourceCode.c_str());
        }
        CODEGEN_LOGI("------------------------ Op CodeGenNPU Finish -----------------------");
    }
    floatSpecValMgr.PrintFloatSpecVal(oss);
    oss << allocSourceRegion << GenDynParamForExpr(subFunc) << symbolMgr->GenUsingList()
        << symbolMgr->GenTileTensorDefList() << tileOpSourceRegion;
}

} // namespace npu::tile_fwk
