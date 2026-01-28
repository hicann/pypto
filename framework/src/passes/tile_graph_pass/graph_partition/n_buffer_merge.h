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
 * \file n_buffer_merge.h
 * \brief
 */

#ifndef PASS_N_BUFFER_MERGE_H_
#define PASS_N_BUFFER_MERGE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
namespace npu::tile_fwk {

constexpr int64_t VEC_NBUFFER_SETTING_DEFAULT_MERGE_NUM_KEY = -1; // manualMerge模式配置默认合并粒度的key值，n个子图合并为一个

class NBufferMerge : public Pass {
public:
    NBufferMerge() : Pass("NBufferMerge") {}
    ~NBufferMerge() override = default;
private:
    Status RunOnFunction(Function &function) override;
    Status NBufferMergeProcess(Function &func);
    Status Init(Function &func);
    void InitParam(OperationsViewer &opOriList);
    void GetOpHash(std::vector<uint64_t> &hashList, const std::string op, size_t idx);
    void GetOpHashReverse(std::vector<uint64_t> &hashList, const std::string op, int idx);
    void GetColorHash(const OperationsViewer &opOriList,
                      std::vector<uint64_t> &hashColor,
                      std::map<uint64_t, std::vector<int>> &hashMap);
    Status CheckAndFixColorOrder(OperationsViewer &opOriList,
                               int &color1, std::vector<int> &colorCycles1,
                               std::vector<std::vector<int>> &colorNode1);
    std::map<uint64_t, size_t> GetIsoColorMergeNum(const std::map<uint64_t, std::vector<int>> &hashMap) const;
    std::vector<std::vector<int>> SortColorWithInput(std::vector<int> &colorValues) const;
    Status MergeProcess(const OperationsViewer &opOriList,
                        std::map<uint64_t, std::vector<int>> &hashMap,
                        std::map<uint64_t, size_t> &hashMergeNum,
                        std::vector<uint64_t> &hashColor);
    Status ColorTopo(int &color1,
                     std::vector<std::vector<int>> &inputColor,
                     std::vector<std::vector<int>> &outputColor,
                     OperationsViewer &opOriList);
    void MergePingPong(std::vector<std::vector<int>> &sortedColors,
                       const OperationsViewer &opOriList,
                       std::vector<uint64_t> &hashColor,
                       size_t &numDBmerge);
    std::map<uint64_t, size_t> SetNumDB(std::map<uint64_t, std::vector<int>> &hashMap);
    Status CheckVecNBufferSettingForManualMerge();
private:
    int color_{0};
    std::vector<std::vector<int>> inGraph_;
    std::vector<std::vector<int>> outGraph_;
    std::vector<std::vector<int>> inColor_;
    std::vector<std::vector<int>> outColor_;
    std::vector<std::vector<int>> colorNode_;
    std::vector<int> colorCycles_;
    int vecNBuffermode;
    int mgVecParallelLb;
    std::map<int64_t, int64_t> vecNBufferSetting;
    std::unordered_map<uint64_t, int> hashOrder;
    enum ModeType {
        noMerge = 0,
        autoMerge = 1,
        manualMerge = 2
    };
};
}  // namespace npu::tile_fwk
#endif  // PASS_N_BUFFER_MERGE_H_