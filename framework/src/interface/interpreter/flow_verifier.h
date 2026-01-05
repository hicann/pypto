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
 * \file flow_verifier.h
 * \brief
 */

#pragma once

#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"
#include "interface/interpreter/function.h"

namespace npu::tile_fwk {

constexpr int THREAD_THOUSAND = 1000;

class FlowVerifier {
public:
    static FlowVerifier &GetInstance();

    void VerifyTensorGraph(Function *entry, const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
        const std::shared_ptr<TensorSlotManager> &slotManager);
    void VerifyPass(Function *func, int passIndex, const std::string &passIdentifier);

    struct CompareElement {
        bool isError;
        size_t index;
        float goldenValue;
        float outputValue;
        float absDiff;
        float relDiff;

        CompareElement() = default;
        CompareElement(const CompareElement &) = default;
        CompareElement &operator=(const CompareElement &) = default;
        CompareElement(bool isError_, size_t index_, float goldenValue_, float outputValue_, float absDiff_, float relDiff_)
            : isError(isError_), index(index_), goldenValue(goldenValue_), outputValue(outputValue_), absDiff(absDiff_), relDiff(relDiff_) {}

        std::string Dump() const {
            std::ostringstream oss;
            oss << "index:" << index
                << " golden:" << goldenValue
                << " output:" << outputValue
                << " absDiff:" << absDiff
                << " relDiff:" << relDiff;
            return oss.str();
        }
    };
    struct CompareResult : std::vector<CompareElement> {
        CompareResult() {}
        CompareResult(int size, float eps, size_t errorCountThreshold = 0, size_t zeroCountThreshold = 0) : size_(size), eps_(eps) {
            errorCountThreshold_ = errorCountThreshold != 0 ? errorCountThreshold : static_cast<int>(size * eps);
            zeroCountThreshold_ = zeroCountThreshold != 0 ? zeroCountThreshold : THREAD_THOUSAND;
        }

        template<typename...TyArgs>
        void AppendError(TyArgs && ...args) {
            errorCount_++;
            this->emplace_back(args...);
        }
        template<typename...TyArgs>
        void AppendZero(TyArgs && ...args) {
            zeroCount_++;
            this->emplace_back(args...);
        }

        bool Check() const {
            return errorCount_ <= errorCountThreshold_ && zeroCount_ <= zeroCountThreshold_;
        }

        std::vector<std::string> Dump(int indent = 2, size_t maxPrint = 50) const {
            float maxAbsDiff = 0;
            float maxRelDiff = 0;
            CompareElement maxAbsElement;
            CompareElement maxRelElement;
            std::ostringstream oss;
            std::string space(indent, ' ');
            std::string infoError = "\n  " + space + "Error eps=" + std::to_string(eps_);
            std::string infoZero = "\n  " + space + "Zero";
            size_t count_ = 0;
            for (auto &element : *this) {
                auto [isError, index, goldenValue, outputValue, absDiff, relDiff] = element;
                (void)index;
                (void)goldenValue;
                (void)outputValue;
                if (absDiff > maxAbsDiff) {
                    maxAbsDiff = absDiff;
                    maxAbsElement = element;
                }
                if (relDiff > maxRelDiff) {
                    maxRelDiff = relDiff;
                    maxRelElement = element;
                }
                maxAbsDiff = std::max(maxAbsDiff, absDiff);
                maxRelDiff = std::max(maxRelDiff, relDiff);

                std::string info = "";
                if (isError) {
                    info = infoError.c_str();
                } else {
                    info = infoZero.c_str();
                }
                if (count_ <= maxPrint) {
                    oss << space << info << " " << element.Dump() << "";
                }
                count_++;
            }
            oss << "\n" << space << "All size:" << size_
                << " maxAbsDiff:" << maxAbsDiff
                << " maxRelDiff:" << maxRelDiff
                << " errorCount:" << errorCount_
                << " errorRatio:" << errorCount_ * 1.0 / size_
                << " zeroCount:" << zeroCount_
                << " zeroRatio:" << zeroCount_ * 1.0 / size_ << "\n";
            if (errorCount_ + zeroCount_ > 0) {
                oss << space << "maxAbs-> " << maxAbsElement.Dump() << "\n"
                    << space << "maxRel-> " << maxRelElement.Dump() << "\n";
            }
            if (!Check()) {ALOG_EVENT(oss.str());}
            return {std::to_string(maxAbsDiff), std::to_string(maxRelDiff),std::to_string(errorCount_),
                    std::to_string(errorCount_ * 1.0 / size_)};
        }

        float GetEps() const { return eps_; }
    private:
        size_t size_{0};
        float eps_{0};
        size_t errorCountThreshold_{0};
        size_t zeroCountThreshold_{0};
        size_t errorCount_ = 0;
        size_t zeroCount_ = 0;
    };

    template <typename DataType>
    static void CompareData(CompareResult &compareResult, size_t count, int64_t offset, const DataType *goldenValueList,
                            const DataType *outputValueList) {
        for (size_t index = 0; index < count; index++) {
            auto goldenValue = static_cast<float>(goldenValueList[index]);
            auto outputValue = static_cast<float>(outputValueList[index]);
            auto absDiff = std::abs(goldenValue - outputValue);
            auto relDiff = 0.0f;
            if (std::abs(goldenValue) < 1e-6) {
                relDiff = std::abs(absDiff) < 1e-6 ? 0 : 1;
            } else {
                relDiff = std::abs(absDiff / goldenValue);
            }
            if (std::abs(outputValue) <= 1e-6 && std::abs(goldenValue) > 1e-6) {
                compareResult.AppendZero(false, offset + index, goldenValue, outputValue, absDiff, relDiff);
            }
            if (absDiff > compareResult.GetEps() && relDiff > compareResult.GetEps()) {
                compareResult.AppendError(true, offset + index, goldenValue, outputValue, absDiff, relDiff);
            }
        }
    }

    template <typename DataType>
    static void CompareDataRecursive(CompareResult &compareResult, size_t axis, int64_t goldenOffset,
        int64_t outputOffset, const std::shared_ptr<LogicalTensorData> &goldenDataView,
        const std::shared_ptr<LogicalTensorData> &outputDataView) {
        auto &validShape = goldenDataView->GetValidShape();
        if (axis == validShape.size() - 1) {
            CompareData<DataType>(compareResult, validShape[axis], outputOffset, &goldenDataView->Get<DataType>(goldenOffset),
                &outputDataView->Get<DataType>(outputOffset));
        } else {
            for (int i = 0; i < validShape[axis]; i++) {
                int nGoldenOffset = goldenOffset + goldenDataView->GetStride(axis) * i;
                int nOutputOffset = outputOffset + outputDataView->GetStride(axis) * i;
                CompareDataRecursive<DataType>(
                    compareResult, axis + 1, nGoldenOffset, nOutputOffset, goldenDataView, outputDataView);
            }
        }
    }

    template <typename DataType>
    static CompareResult CompareData(const std::shared_ptr<LogicalTensorData> &goldenDataView,
        const std::shared_ptr<LogicalTensorData> &outputDataView, float eps, int errorCountThreshold = 0,
        int zeroCountThreshold = 0) {
        auto &validShape = goldenDataView->GetValidShape();
        auto size = std::accumulate(validShape.begin(), validShape.end(), 1, std::multiplies<>());
        CompareResult compareResult(size, eps, errorCountThreshold, zeroCountThreshold);
        CompareDataRecursive<DataType>(compareResult, 0, 0, 0, goldenDataView, outputDataView);
        return compareResult;
    }

    static CompareResult VerifyResult(
            const std::shared_ptr<LogicalTensorData> &goldenDataView,
            const std::shared_ptr<LogicalTensorData> &outputDataView, float eps);
    static bool VerifyResult(const std::string &key,
        const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList, float eps);
    bool VerifyResult(const std::string &key, const std::string tensorNameList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &tensorDataViewList, float eps);

private:
    void UpdateInterpreterCache();
    void Initialize(
            Function *entry,
            const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList,
            const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList,
            const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
            const std::shared_ptr<TensorSlotManager> &slotManager);

    template <typename type>
    static bool CompareDataAndCheck(const std::shared_ptr<RawTensorData> &goldenData,
        const std::shared_ptr<RawTensorData> &cmpData, float eps) {
        if (!CompareData<type>(goldenData->GetSize(), &goldenData->Get<type>(0), &cmpData->Get<type>(0), eps).Check()) {
            return false;
        }
        return true;
    }

private:
    Function *entry_;
    bool checkResult{true};
    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList_;
    std::vector<std::shared_ptr<LogicalTensorData>> outputDataViewList_;
    std::vector<std::shared_ptr<LogicalTensorData>> goldenDataViewList_;
    std::shared_ptr<FunctionInterpreter> functionInterpreter_;

    std::shared_ptr<FunctionControlFlowExecution> controlFlowExecution_;
    std::unordered_map<Function *, std::vector<std::shared_ptr<FunctionCaptureExecution>>> lastCaptureExecution_;
};

}
