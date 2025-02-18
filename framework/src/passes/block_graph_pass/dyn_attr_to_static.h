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
 * \file dyn_attr_to_static.h
 * \brief
 */

#ifndef PASS_DYNATTR_TO_STATIC_H_
#define PASS_DYNATTR_TO_STATIC_H_

#include <vector>
#include <unordered_map>
#include <regex>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/utils/log.h"
#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif

#define MODULE_NAME "DynAttrToStatic"

namespace npu {
namespace tile_fwk {

enum class CoaType {
    PARAM_OFFSET,
    PARAM_VALID_SHAPE,
    PARAM,
    INVALID
};

static const std::string COA_PREFIX = "RUNTIME_COA_GET_PARAM";
static const std::string MAYBE_CONST_POSTFIX = "MAYBE_CONST";

static const SymbolicScalar MAYBE_CONST_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE_MAYBE_CONST");
static const SymbolicScalar MAYBE_CONST_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM_MAYBE_CONST");

Status SToIWrapper(const std::string str, int& result);


constexpr int OFFSET_INDEX_ORDER = 0;
constexpr int SHAPE_INDEX_ORDER = 1;
constexpr int RAWSHAPE_INDEX_ORDER = 2;
constexpr int VALID_SHAPE_INDEX_ORDER = 3;
constexpr int INPUT_PARAM_POS_ONE = 1;
constexpr int INPUT_PARAM_POS_TWO = 2;
constexpr int INPUT_PARAM_POS_THREE = 3;

struct ValWithIdxs {
    SymbolicScalar val;                    // 存储的值
    std::vector<size_t> idxs; // 该值对应的索引列表
};

/**
 * @brief 校验vector形参调用时，是否存在索引组满足「每次调用内组内索引值相同」
 */
class VectorParamConsistencyChecker {
public:
    /**
     * @brief 注册一次函数调用的vector实参
     * @param args 本次调用的vector实参
     * @return 注册是否成功（长度非法时失败）
     */
    bool RegisterCall(const std::vector<SymbolicScalar>& args) {
        if (args.empty()) return false; // 空vector无意义
        
        // 首次调用：记录vector长度，后续调用需保持长度一致
        if (m_callCount == 0) {
            m_vecLen = args.size();
        } else if (args.size() != m_vecLen) {
            m_isValid = false; // 长度不一致，直接标记无效
            return false;
        }

        // 步骤1：生成本次调用的「值-索引列表」（仅用operator==，线性遍历）
        std::vector<ValWithIdxs> currValIdxs;
        for (size_t idx = 0; idx < args.size(); ++idx) {
            // 查找当前值是否已存在于currValIdxs中（仅用==匹配）
            auto it = FindValInList(currValIdxs, args[idx]);
            if (it != currValIdxs.end()) {
                it->idxs.push_back(idx); // 已存在，追加索引
            } else {
                currValIdxs.push_back({args[idx], {idx}});
            }
        }

        // 步骤2：更新候选索引组（首次调用初始化，后续调用筛选）
        if (m_callCount == 0) {
            // 首次调用：所有非空索引列表都作为候选组（去重+排序）
            for (auto& entry : currValIdxs) {
                if (!entry.idxs.empty()) {
                    // 索引组排序（保证相同索引组合的一致性，避免重复）
                    std::sort(entry.idxs.begin(), entry.idxs.end());
                    m_candidateGroups.push_back(entry.idxs);
                }
            }
            // 对候选组去重（避免首次调用就有重复的索引组）
            DeduplicateGroups(m_candidateGroups);
        } else {
            // 非首次调用：筛选候选组（仅保留在本次调用中值相同的索引组）
            std::vector<std::vector<size_t>> newCandidates;
            for (const auto& candidate : m_candidateGroups) {
                // 检查候选组是否在本次调用的某值索引列表中（子集判断）
                if (IsCandidateValidInCurrCall(candidate, currValIdxs)) {
                    newCandidates.push_back(candidate);
                }
            }
            // 去重后替换候选组
            DeduplicateGroups(newCandidates);
            m_candidateGroups.swap(newCandidates);
        }

        m_callCount++;
        return true;
    }

    /**
     * @brief 获取首个满足条件的索引组（便捷接口，兼容旧逻辑）
     * @return 第一个满足条件的索引组（空则无）
     */
    std::vector<size_t> GetConsistentIndexGroup() const {
        if (!m_isValid || m_candidateGroups.empty()) {
            return {};
        }
        return m_candidateGroups.front();
    }

    /**
     * @brief 获取所有满足条件的索引组（核心接口，返回全部有效组）
     * @return 所有符合条件的索引组（空则无）
     */
    std::vector<std::vector<size_t>> GetAllConsistentIndexGroups() const {
        if (!m_isValid || m_candidateGroups.empty()) {
            return {};
        }
        // 返回完整的候选组列表（已去重、有序）
        return m_candidateGroups;
    }

    std::string PrintIndexGroups(const std::vector<std::vector<size_t>>& groups) const {
        std::stringstream ss;
        ss << "ALL Consistent Index Group:  {";
        if (groups.empty()) {
            ss << "}";
        }
        for (size_t i = 0; i < groups.size(); ++i) {
            ss << "Consistent Index Group: " << (i + 1) << "{";
            for (size_t idx : groups[i]) {
                ss << idx << ", ";
            }
            ss << "}" << std::endl;
        }
        ss << "}";
        return ss.str();
    }

    /**
     * @brief 重置校验器（清空所有调用记录）
     */
    void Reset()  {
        m_callCount = 0;
        m_vecLen = 0;
        m_isValid = true;
        m_candidateGroups.clear();
    }

private:
    /**
     * @brief 查找值是否在ValWithIdxs列表中
     * @param list 「值-索引列表」
     * @param val 要查找的值
     * @return 找到则返回迭代器，否则返回end()
     */
    typename std::vector<ValWithIdxs>::iterator FindValInList(std::vector<ValWithIdxs>& list, const SymbolicScalar& val) {
        for (auto it = list.begin(); it != list.end(); ++it) {
            if (it->val.Dump() == val.Dump()) { // 仅依赖operator==
                return it;
            }
        }
        return list.end();
    }
    /**
     * @brief 检查候选索引组是否在本次调用中有效（组内索引值相同）
     * @param candidate 候选索引组
     * @param currValIdxs 本次调用的「值-索引列表」
     * @return 是否有效
     */
    bool IsCandidateValidInCurrCall(
        const std::vector<size_t>& candidate, const std::vector<ValWithIdxs>& currValIdxs) const {
        // 步骤1：获取候选组第一个索引在本次调用中对应的值
        if (candidate.empty()) return false;
        size_t firstIdx = candidate[0];
        const SymbolicScalar* targetVal = nullptr;
        // 查找第一个索引对应的value（线性遍历）
        for (const auto& entry : currValIdxs) {
            if (std::find(entry.idxs.begin(), entry.idxs.end(), firstIdx) != entry.idxs.end()) {
                targetVal = &entry.val;
                break;
            }
        }
        if (!targetVal) return false; // 索引不存在（理论上不会发生）

        // 步骤2：检查候选组所有索引是否都对应该值（仅用==）
        for (size_t idx : candidate) {
            bool isIdxMatch = false;
            for (const auto& entry : currValIdxs) {
                if (entry.val.Dump() == targetVal->Dump()) { // 仅依赖operator==
                    // 检查当前索引是否在该值的索引列表中
                    if (std::find(entry.idxs.begin(), entry.idxs.end(), idx) != entry.idxs.end()) {
                        isIdxMatch = true;
                        break;
                    }
                }
            }
            if (!isIdxMatch) {
                return false; // 有索引不匹配，候选组无效
            }
        }
        return true;
    }

    /**
     * @brief 对索引组列表去重（避免重复的索引组合）
     * @param groups 待去重的索引组列表
     */
    void DeduplicateGroups(std::vector<std::vector<size_t>>& groups) {
        if (groups.empty()) return;
        // 步骤1：先对每个索引组内部排序（保证 {3,1} 和 {1,3} 视为同一组）
        for (auto& group : groups) {
            std::sort(group.begin(), group.end());
        }
        // 步骤2：对索引组列表排序，便于去重
        std::sort(groups.begin(), groups.end(), 
            [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                if (a.size() != b.size()) return a.size() < b.size();
                for (size_t i = 0; i < a.size(); ++i) {
                    if (a[i] != b[i]) return a[i] < b[i];
                }
                return false;
            }
        );
        // 步骤3：去重
        auto last = std::unique(groups.begin(), groups.end());
        groups.erase(last, groups.end());
    }

    size_t m_callCount = 0; // 调用次数
    size_t m_vecLen = 0;    // vector固定长度
    bool m_isValid = true;  // 是否有效（长度一致）
    // 候选索引组：用普通vector存储，无需排序/哈希（已去重）
    std::vector<std::vector<size_t>> m_candidateGroups;
};

class DynAttrToStatic : public Pass {
public:
    DynAttrToStatic() : Pass("DynAttrToStatic") {}
    ~DynAttrToStatic() override = default;
private:
    std::unordered_map<Function*, std::vector<Operation*>> leaf2Caller;

    Status RunOnFunction(Function &function) override;
    std::vector<std::reference_wrapper<SymbolicScalar>> GetOpDynamicAttributeList(Operation &op);
    Status GetCallee(const Operation &callop, Function *&callFunc);
    void RefSpecifiedValue(std::vector<SymbolicScalar> &oriList,
        std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const;
    void FilterSpecifiedValue(std::vector<OpImmediate> &oriList,
        std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const;
    Status BuildLeafToCaller(Function *func);
    Status BuildNewCoa(
        std::reference_wrapper<SymbolicScalar>& dynScalar,
        std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim);
    Status TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList);
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_DYNATTR_TO_STATIC_H_
