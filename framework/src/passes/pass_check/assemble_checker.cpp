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
 * \file assemble_checker.cpp
 * \brief
 */

#include <utility>
#include "assemble_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "AssembleChecker"

namespace npu {
namespace tile_fwk {
// assemble存在dynOffset和输入存在dynValidShape场景暂不判断。
Status CheckDynSkip(const LogicalTensorPtr &outputTensor, bool &needSkip) {
    for (const auto &producerOp : outputTensor->GetProducers()){
        if (producerOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            needSkip = true;
            return SUCCESS;
        }
        auto assembleOpAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (assembleOpAttr) {
            if (assembleOpAttr->GetToDynOffset().size() != 0) {
                needSkip = true;
                return SUCCESS;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s[%d] has no valid assembleOpAttribute; Please check.",
                producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
            return FAILED;
        }
        auto input = producerOp->iOperand.front();
        if (input->GetDynValidShape().size() != 0) {
            needSkip = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}
/*
    在input->assemble->output的场景中，通过校验input之间是否每个轴都存在重叠来判断，input间是否存在覆盖output中同一数据块的情况。
    这种重叠可能由于两块数据到达时间不同，导致覆盖顺序不确定进而导致不确定的行为
*/
Status AssembleChecker::CheckAssembleOverlap(Function &function) {
    auto needSkip = [](const Shape &vec) -> bool {
        return std::any_of(vec.begin(), vec.end(), [](int64_t val) { return val == -1; });
    };
    for (const auto &tMap : function.GetTensorMap().tensorMap_) {
        for (const auto &outputTensor : tMap.second) {
            if (outputTensor->GetProducers().size() == 0){
                continue;
            }
            bool dynSkip = false;
            if (CheckDynSkip(outputTensor, dynSkip) == FAILED) {
                return FAILED;
            } 

            if (dynSkip) {
                continue;
            }
            coveredAreas_.clear();
            for (const auto &assembleOp : outputTensor->GetProducers()) {
                if (assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
                    continue;
                }
                auto assembleOffset = dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get())->GetToOffset();
                auto inputTensor = assembleOp->GetIOperands().front();
                auto inputShape = inputTensor->GetShape();
                if (needSkip(inputShape) || needSkip(assembleOffset)){
                    continue;
                }
                std::vector<std::pair<int64_t, int64_t>> curInputArea;
                if(assembleOffset.size() != inputShape.size()) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Dimension of assemble op[%d]'s toOffset(%s) "
                        "varies from its input[%d]'s shape(%s); Please check the function graph.",
                        assembleOp->GetOpMagic(), CommonUtils::ContainerToStr(assembleOffset).c_str(),
                        inputTensor->GetMagic(), CommonUtils::ContainerToStr(inputShape).c_str());
                    return FAILED;
                }
                for (size_t i = 0; i < inputShape.size(); i++){
                    curInputArea.emplace_back(assembleOffset[i], assembleOffset[i] + inputShape[i] - 1);
                }

                // 判断是否有重叠
                if (OverlapCurInput(curInputArea)) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Overlap input2: shape:%s offset:%s; Please check the function graph; Please check Tensor[%d] and its input.",
                        CommonUtils::ContainerToStr(inputShape).c_str(), CommonUtils::ContainerToStr(assembleOffset).c_str(), outputTensor->GetMagic());
                    return FAILED;
                }

                // 将覆盖区域添加到记录
                coveredAreas_.emplace_back(std::move(curInputArea));
            }
        }
    }
    return SUCCESS;
}

bool AssembleChecker::OverlapCurInput(const std::vector<std::pair<int64_t, int64_t>> &curInputArea) {
    for (const auto& recordedArea : coveredAreas_) {
        bool overlap = std::equal(curInputArea.begin(), curInputArea.end(),
            recordedArea.begin(),
            [](const auto& a, const auto& b) {
                return a.second >= b.first && a.first <= b.second;
            });
        if (overlap) {
            // 计算重叠的input的shape和offset
            Shape recordedShape;
            Shape recordedOffset;
            for (const auto& recordedDim : recordedArea){
                recordedOffset.emplace_back(recordedDim.first);
                recordedShape.emplace_back(recordedDim.second + 1 - recordedDim.first);
            }
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor produced by assemble has overlap inputs. Overlap input1: shape:%s offset:%s.",
                CommonUtils::ContainerToStr(recordedShape).c_str(), CommonUtils::ContainerToStr(recordedOffset).c_str());
            return true;
        }
    }
    return false;
}
} // namespace tile_fwk
} // namespace npu