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

#include "machine/host/machine_compiler.h"
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

constexpr int DUMP_INCAST = 2;
constexpr int DUMP_OUTCAST = 3;

namespace npu::tile_fwk {
void CalcFunctionInvokeWorkespace(Function* cacheFunction, Function* function,
                                  MachineCompileInfo& compileInfo)
{
    if (!function) {
        ALOG_WARN("Function  pointer is null!");
    }
    MACHINE_ASSERT(function);
    ALOG_INFO("Begin calc invoke entry workespace!");
    uint64_t totalSize = 0;
    Function *compiledFunction = cacheFunction ? cacheFunction : function;
    ASSERT(compiledFunction->rootFunc_ != nullptr) << "compiledFunction.rootFunc_ is nullptr , FuncMagic is %d \n"<< compiledFunction->GetFuncMagic();

    /* rawtensor magic -> {workspace offset , shape size} */
    std::map<int, std::pair<uint64_t, uint64_t>> rawTensorOffsetMap;

    auto getRawTensorByTensorMagic = [&compiledFunction](int tensorMagic) -> auto
    {
        auto rawTensor = compiledFunction->GetTensorMap().GetRawTensorByRawMagic(tensorMagic);
        ALOG_DEBUG_F("magic is %d", tensorMagic);
        if (!rawTensor) {
            ALOG_WARN("Raw tensor is null for magic: ");
        }
        MACHINE_ASSERT(rawTensor);
        return rawTensor;
    };

    auto calcOffsetFunc = [](const std::vector<int64_t> &offset, const std::vector<int64_t> &shape) -> uint64_t {
        if (offset.size() != shape.size()) {
            ALOG_ERROR_F("Offset size (%zu) does not match shape size (%zu)", offset.size(), shape.size());
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
    };

    auto workSpaceOffsetProcFunc = [&getRawTensorByTensorMagic, &rawTensorOffsetMap, &totalSize, &calcOffsetFunc,
                                    &compiledFunction](const LogicalTensorPtr& tensor, int rawMagic,
                                                       const std::vector<int64_t>& rawShape, const std::vector<int64_t>& offset,
                                                       std::list<InvokeParaOffset>& curSubFuncParaOffset,
                                                       bool isTensorPara) {
        InvokeParaOffset paraOffset;
        uint64_t rawTensorOffset = 0;
        auto& storage = tensor->storage_;
        auto rawTensor = getRawTensorByTensorMagic(rawMagic);
        int storageId = rawTensor->GetRawMagic();
        uint64_t alignSize = 0;
        if (storage != nullptr) {
            storageId = storage->id_;
            alignSize = storage->length_;
        } else {
            alignSize = (CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType()) + 511) / 512 *
                                 512; //  performance standpoint
            ALOG_INFO_F("tensor %d raw %d storage is null, actual raw magic %d", tensor->magic, rawMagic,
                rawTensor->actualRawmagic);
        }

        auto iter = rawTensorOffsetMap.find(storageId);
        if (iter != rawTensorOffsetMap.end()) {
            /* use raw tensor workspace offset + tensor view offset */
            rawTensorOffset = iter->second.first;
            ALOG_DEBUG_F("get old raw offset rawmagic %d storage id %d workspaceoffset %lu", rawMagic,
                storageId, rawTensorOffset);
        } else {
            /* insert new offset */
            rawTensorOffset = totalSize;
            totalSize += alignSize;
            rawTensorOffsetMap[storageId] = std::make_pair(rawTensorOffset, alignSize);
            ALOG_DEBUG_F("insert new raw offset rawMagic:%d, storage id %d, rawOffset %lu, alignsize %lu", rawMagic,
                storageId, rawTensorOffset, alignSize);
        }

        uint64_t offSetSize = calcOffsetFunc(offset, rawShape) * BytesOf(rawTensor->GetDataType());

        paraOffset.isTensorParam = isTensorPara;
        paraOffset.offset = rawTensorOffset + offSetSize;
        paraOffset.rawTensorOffset = rawTensorOffset;
        paraOffset.LogRawTensorInfo(rawTensor);
        paraOffset.rawTensorAddr = nullptr; // null express use workspace addr later
        paraOffset.funcitonMagic = compiledFunction->GetFuncMagic();
        ALOG_DEBUG_F("workspace tensor para offset, magic %d, offset [%lu, %lu], rawshape [%lu, %lu], datatype %zu, "
                   "rawOffset %lu, offsetSize %lu, isTensorPara %d",
            rawMagic, offset[0], offset[1], rawShape[0], rawShape[1], static_cast<size_t>(rawTensor->GetDataType()), rawTensorOffset,
            offSetSize, isTensorPara);
        ALOG_DEBUG_F(
            "workspace tensor para offset, magic %d offset [%d, %d], rawshape [%d, %d], datatype %d, rawOffset %lu, "
            "offsetSize %lu, isTensorPara %d\n",
            rawMagic, offset[0], offset[1], rawShape[0], rawShape[1], static_cast<int>(rawTensor->GetDataType()), rawTensorOffset,
            offSetSize, isTensorPara);
        curSubFuncParaOffset.push_back(paraOffset);
        return;
    };

    for (uint64_t i = 0; i < compiledFunction->rootFunc_->Operations().size(); ++i) {
        const SubfuncInvokeInfoTy &subfuncInvoke = compiledFunction->rootFunc_->GetSubFuncInvokeInfo(i);
        auto &curSubFuncParaOffset = compileInfo.invokeParaOffset[i];
        ALOG_DEBUG_F("proc sub func invoke info :  id = %lu", i);
        ALOG_DEBUG_F("tensor param cnt %zu:", subfuncInvoke.GetTensorParamList().size());
        for (const auto &elm : subfuncInvoke.GetTensorParamList()) {
            ALOG_DEBUG_F("ele Shape %s, RawShape %s, Offset %s", IntVecToStr(elm.shape).c_str(),
                IntVecToStr(elm.rawShape).c_str(), IntVecToStr(elm.offset).c_str());
            InvokeParaOffset paraOffset;
            auto rawTensor = getRawTensorByTensorMagic(elm.ddrId);
            paraOffset.offset = calcOffsetFunc(elm.offset, elm.rawShape) * BytesOf(elm.dType);
            paraOffset.paramType = elm.isOutputToGM ? 0 : 1;
            paraOffset.tensorShape = elm.shape;
            paraOffset.rawTensorShape = elm.rawShape;
            paraOffset.funcitonMagic = compiledFunction->GetFuncMagic();
            paraOffset.opMagic = elm.opMagic;
            /* begin function explicit 模式，等待后面run接口调用时候根据传入的op args确定rawtensor 地址 */
            paraOffset.rawTensorAddr = nullptr;
            paraOffset.opOriginArgsSeq = function->GetParamIndex(rawTensor);
            paraOffset.isTensorParam = true;
            paraOffset.LogRawTensorInfo(rawTensor);
            ALOG_DEBUG_F("tensor para offset, magic: %d, offset [%d, %d], rawshape:[%d, %d], rawAddr: %lx, offsetsize "
                       "%zu, orgArgsSeq %zu, symbol %s",
                elm.ddrId, elm.offset[0], elm.offset[1], elm.rawShape[0], elm.rawShape[1],
                reinterpret_cast<uint64_t>(paraOffset.rawTensorAddr), paraOffset.offset, paraOffset.opOriginArgsSeq,
                paraOffset.rawSymbol.c_str());
            // std::cout << "tensor para offset, magic:" << elm.DDRId << " offset:" << elm.Offset[0]
            //     << " " << elm.Offset[1] << " rawshape:" << elm.RawShape[0] << " " << elm.RawShape[1]
            //     << " rawAddr:" << reinterpret_cast<uint64_t>(paraOffset.rawTensorAddr)
            //     << " offsetsize: " << paraOffset.offset << " orgArgsSeq: " << paraOffset.opOriginArgsSeq
            //     << " symbol: " << paraOffset.rawSymbol << std::endl;
            if (paraOffset.rawTensorAddr == nullptr && paraOffset.opOriginArgsSeq == INVALID_IN_OUT_INDEX) {
                /* tensor para 理论上不该存在此场景,
                 * 等待前端graph&schedule解决此场景，此处兼容如果映射不到原始args上则申请workspace空间 */
                workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset,
                    curSubFuncParaOffset, true);
                ALOG_DEBUG_F("tensor param raw tensor addr is null , wait device agent alloc workspace,rawmagic= %d, "
                           "rawsymbol = %s, offset = %lu, datatype = %zu",
                    paraOffset.rawMagic, paraOffset.rawSymbol.c_str(), paraOffset.offset, static_cast<size_t>(paraOffset.datatype));

                ALOG_INFO << __FUNCTION__
                          << " tensor param raw tensor addr is null , wait device agent alloc workspace,"
                          << "rawmagic=" << paraOffset.rawMagic << " rawsymbol = " << paraOffset.rawSymbol
                          << " offset = " << paraOffset.offset << " datatype = " << elm.dType;
            } else {
                curSubFuncParaOffset.push_back(paraOffset);
            }
        }
        ALOG_DEBUG_F("incast cnt: %zu", subfuncInvoke.GetIncastTensorParamList().size());
        int incastIndx = 0;
        for (auto &elm : subfuncInvoke.GetIncastTensorParamList()) {
            workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset, curSubFuncParaOffset, false);
            curSubFuncParaOffset.back().ioIndex = incastIndx;
            curSubFuncParaOffset.back().paramType = DUMP_INCAST;
            curSubFuncParaOffset.back().tensorShape = elm.shape;
            curSubFuncParaOffset.back().rawTensorShape = elm.rawShape;
            curSubFuncParaOffset.back().opMagic = elm.opMagic;
            incastIndx++;
        }

        ALOG_DEBUG_F("outcast cnt: %zu", subfuncInvoke.GetOutcastTensorParamList().size());
        int outcastIndx = 0;
        for (auto &elm : subfuncInvoke.GetOutcastTensorParamList()) {
            workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset, curSubFuncParaOffset, false);
            curSubFuncParaOffset.back().ioIndex = outcastIndx;
            curSubFuncParaOffset.back().paramType = DUMP_OUTCAST;
            curSubFuncParaOffset.back().tensorShape = elm.shape;
            curSubFuncParaOffset.back().rawTensorShape = elm.rawShape;
            curSubFuncParaOffset.back().opMagic = elm.opMagic;
            outcastIndx++;
        }
    }
    compileInfo.invokeParaWorkSpaceSize = totalSize;
    compileInfo.coreFunctionCnt = compiledFunction->rootFunc_->Operations().size();
    compileInfo.programFunctionCnt = compiledFunction->rootFunc_->programs_.size();
    compileInfo.Print();
}
} // namespace npu::tile_fwk
