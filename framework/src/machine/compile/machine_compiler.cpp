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
 * \file machine_compiler.cpp
 * \brief
 */

#include "machine/compile/machine_compiler.h"
#include <functional>
#include <unistd.h>
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/machine/host/host_machine.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
namespace {
constexpr int DUMP_INCAST = 2;
constexpr int DUMP_OUTCAST = 3;
std::shared_ptr<RawTensor> GetRawTensorByTensorMagic(const Function *compiledFunction, const int tensorMagic) {
    auto rawTensor = compiledFunction->GetTensorMap().GetRawTensorByRawMagic(tensorMagic);
    MACHINE_LOGD("magic is %d", tensorMagic);
    if (!rawTensor) {
        MACHINE_LOGW("Raw tensor is null for magic: %d", tensorMagic);
    }
    MACHINE_ASSERT(rawTensor);
    return rawTensor;
}

uint64_t CalcTensorOffset(const std::vector<int64_t> &offset, const std::vector<int64_t> &shape) {
    if (offset.size() != shape.size()) {
        MACHINE_LOGE("Offset size (%zu) does not match shape size (%zu)", offset.size(), shape.size());
    }
    MACHINE_ASSERT(offset.size() == shape.size());
    uint64_t offSetSize = 0;
    auto strideShapeFunc = [&shape](size_t i) -> auto
    {
        uint64_t stride = 1;
        for (size_t j = i; j < shape.size(); j++) {
            stride *= shape[j];
        }
        return stride;
    };
    for (size_t i = 0; i < shape.size(); i++) {
        offSetSize += offset[i] * strideShapeFunc(i + 1);
    }
    return offSetSize;
}

template<typename TensorParam>
void ProcWorkSpaceOffset(const TensorParam &tensorParam, const std::shared_ptr<RawTensor> &rawTensor,
                         InvokeParaOffset &paraOffset,
                         std::map<int, std::pair<uint64_t, uint64_t>> &rawTensorOffsetMap, uint64_t &totalSize) {
    const LogicalTensorPtr& tensor = tensorParam.tensor;
    const std::vector<int64_t>& rawShape = tensorParam.rawShape;
    const std::vector<int64_t>& offset = tensorParam.offset;

    int storageId = tensor->storage_ == nullptr ? rawTensor->GetRawMagic() : tensor->storage_->id_;
    uint64_t alignSize = 0;
    if (tensor->storage_ != nullptr) {
        alignSize = tensor->storage_->length_;
    } else {
        //  performance standpoint
        alignSize = (CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType()) + 511) / 512 * 512;
        MACHINE_LOGI("tensor %d raw %d storage is null, actual raw magic %d", tensor->magic, tensorParam.ddrId,
                    rawTensor->actualRawmagic);
    }

    uint64_t rawTensorOffset = 0;
    auto iter = rawTensorOffsetMap.find(storageId);
    if (iter != rawTensorOffsetMap.end()) {
        /* use raw tensor workspace offset + tensor view offset */
        rawTensorOffset = iter->second.first;
        MACHINE_LOGD("get old raw offset rawmagic %d storage id %d workspaceoffset %lu", tensorParam.ddrId,
            storageId, rawTensorOffset);
    } else {
        /* insert new offset */
        rawTensorOffset = totalSize;
        totalSize += alignSize;
        rawTensorOffsetMap[storageId] = std::make_pair(rawTensorOffset, alignSize);
        MACHINE_LOGD("insert new raw offset rawMagic:%d, storage id %d, rawOffset %lu, alignsize %lu", tensorParam.ddrId,
            storageId, rawTensorOffset, alignSize);
    }

    uint64_t offSetSize = CalcTensorOffset(offset, rawShape) * BytesOf(rawTensor->GetDataType());
    paraOffset.offset = rawTensorOffset + offSetSize;
    paraOffset.rawTensorOffset = rawTensorOffset;
}

template<typename TensorParam>
void PrintParaOffsetInfo(const TensorParam &tensorParam, const InvokeParaOffset &paraOffset) {
    MACHINE_LOGD("Tensor param: magic: %d, shape[%s], rawshape[%s], offset[%s], dtype[%d].",
                 tensorParam.ddrId, IntVecToStr(tensorParam.shape).c_str(), IntVecToStr(tensorParam.rawShape).c_str(),
                 IntVecToStr(tensorParam.offset).c_str(), static_cast<int>(tensorParam.dType));
    MACHINE_LOGD("Tensor offset: magic[%d], symbol[%s], offset[%lu], rawOffset[%lu], datatype[%zu], isTensorPara[%d].",
                 paraOffset.rawMagic, paraOffset.rawSymbol.c_str(), paraOffset.offset, paraOffset.rawTensorOffset,
                 static_cast<size_t>(paraOffset.datatype), paraOffset.isTensorParam);
}

template<typename TensorParam>
void ProcInOutCastParamOffset(const Function *compiledFunction, const TensorParam &tensorParam, const int castIdx,
    InvokeParaOffset &paraOffset, std::map<int, std::pair<uint64_t, uint64_t>> &rawTensorOffsetMap,
    uint64_t &totalSize) {
    paraOffset.funcitonMagic = compiledFunction->GetFuncMagic();
    auto rawTensor = GetRawTensorByTensorMagic(compiledFunction, tensorParam.ddrId);
    paraOffset.rawTensorAddr = nullptr;
    paraOffset.LogRawTensorInfo(rawTensor);
    paraOffset.isTensorParam = false;
    paraOffset.ioIndex = castIdx;
    ProcWorkSpaceOffset(tensorParam, rawTensor, paraOffset, rawTensorOffsetMap, totalSize);
    paraOffset.tensorShape = tensorParam.shape;
    paraOffset.rawTensorShape = tensorParam.rawShape;
    paraOffset.opMagic = tensorParam.opMagic;
    PrintParaOffsetInfo(tensorParam, paraOffset);
}

void ProcTensorParamOffset(const Function *compiledFunction, Function* function,
    const SubfuncInvokeInfoTy::TensorParamPackTy &tensorParam,
    InvokeParaOffset &paraOffset, std::map<int, std::pair<uint64_t, uint64_t>> &rawTensorOffsetMap,
    uint64_t &totalSize) {
    paraOffset.funcitonMagic = compiledFunction->GetFuncMagic();
    auto rawTensor = GetRawTensorByTensorMagic(compiledFunction, tensorParam.ddrId);
    /* begin function explicit 模式，等待后面run接口调用时候根据传入的op args确定rawtensor 地址 */
    paraOffset.rawTensorAddr = nullptr;
    paraOffset.opOriginArgsSeq = function->GetParamIndex(rawTensor);
    paraOffset.LogRawTensorInfo(rawTensor);
    paraOffset.isTensorParam = true;
    if (paraOffset.rawTensorAddr == nullptr && paraOffset.opOriginArgsSeq == INVALID_IN_OUT_INDEX) {
        /* tensor para 理论上不该存在此场景,
            * 等待前端graph&schedule解决此场景，此处兼容如果映射不到原始args上则申请workspace空间 */
        MACHINE_LOGD("Tensor param raw tensor addr is null, wait device agent alloc workspace");
        ProcWorkSpaceOffset(tensorParam, rawTensor, paraOffset, rawTensorOffsetMap, totalSize);
    } else {
        paraOffset.offset = CalcTensorOffset(tensorParam.offset, tensorParam.rawShape) * BytesOf(tensorParam.dType);
        paraOffset.paramType = tensorParam.isOutputToGM ? 0 : 1;
        paraOffset.tensorShape = tensorParam.shape;
        paraOffset.rawTensorShape = tensorParam.rawShape;
        paraOffset.opMagic = tensorParam.opMagic;
    }
    PrintParaOffsetInfo(tensorParam, paraOffset);
}
}

void CalcFunctionInvokeWorkespace(Function* cacheFunction, Function* function, MachineCompileInfo& compileInfo) {
    if (!function) {
        MACHINE_LOGW("Function pointer is null!");
        return;
    }
    MACHINE_LOGI("Begin calc invoke entry workespace!");
    Function *compiledFunction = cacheFunction != nullptr ? cacheFunction : function;

    /* rawtensor magic -> {workspace offset , shape size} */
    std::map<int, std::pair<uint64_t, uint64_t>> rawTensorOffsetMap;
    uint64_t totalSize = 0;
    for (uint64_t i = 0; i < compiledFunction->rootFunc_->Operations().size(); ++i) {
        const SubfuncInvokeInfoTy &subfuncInvoke = compiledFunction->rootFunc_->GetSubFuncInvokeInfo(i);
        auto &curSubFuncParaOffset = compileInfo.invokeParaOffset[i];
        MACHINE_LOGD("proc sub func: id[%lu] tensor param cnt[%zu]:", i, subfuncInvoke.GetTensorParamList().size());
        for (const SubfuncInvokeInfoTy::TensorParamPackTy &elm : subfuncInvoke.GetTensorParamList()) {
            InvokeParaOffset paraOffset;
            ProcTensorParamOffset(compiledFunction, function, elm, paraOffset, rawTensorOffsetMap, totalSize);
            curSubFuncParaOffset.push_back(paraOffset);
        }

        MACHINE_LOGD("Incast tensor param size: %zu", subfuncInvoke.GetIncastTensorParamList().size());
        int incastIndx = 0;
        for (const SubfuncInvokeInfoTy::IncastParamPackTy &elm : subfuncInvoke.GetIncastTensorParamList()) {
            InvokeParaOffset paraOffset;
            paraOffset.paramType = DUMP_INCAST;
            ProcInOutCastParamOffset(compiledFunction, elm, incastIndx, paraOffset, rawTensorOffsetMap, totalSize);
            curSubFuncParaOffset.push_back(paraOffset);
            incastIndx++;
        }

        MACHINE_LOGD("Outcast tensor param size: %zu", subfuncInvoke.GetOutcastTensorParamList().size());
        int outcastIndx = 0;
        for (const SubfuncInvokeInfoTy::OutcastParamPackTy &elm : subfuncInvoke.GetOutcastTensorParamList()) {
            InvokeParaOffset paraOffset;
            paraOffset.paramType = DUMP_OUTCAST;
            ProcInOutCastParamOffset(compiledFunction, elm, outcastIndx, paraOffset, rawTensorOffsetMap, totalSize);
            curSubFuncParaOffset.push_back(paraOffset);
            outcastIndx++;
        }
    }
    compileInfo.invokeParaWorkSpaceSize = totalSize;
    compileInfo.coreFunctionCnt = compiledFunction->rootFunc_->Operations().size();
    compileInfo.programFunctionCnt = compiledFunction->rootFunc_->programs_.size();
    compileInfo.Print();
}
} // namespace npu::tile_fwk
