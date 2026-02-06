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
 * \file aicore_dump.h
 * \brief
 */

#ifndef AICORE_DUMP_H
#define AICORE_DUMP_H
#include <string>
#include <sstream>
#include "securec.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {
constexpr uint64_t DEV_DUMP_DATA_SIZE = 2 * 1024 * 1024;
using IDE_SESSION = void *;
enum IdeErrorT {};
extern "C" {
__attribute__((weak)) IDE_SESSION IdeDumpStart(const char *privInfo);
__attribute__((weak)) IdeErrorT IdeDumpData(IDE_SESSION session, const struct IdeDumpChunk *dumpChunk);
__attribute__((weak)) IdeErrorT IdeDumpEnd(IDE_SESSION session);
};

struct IdeDumpChunk {
    char  *fileName;           /**< absolute path */
    unsigned char *dataBuf;   /**< Buffer of input data */
    unsigned int bufLen;      /**< Buffer Length of input data */
    unsigned int isLastChunk; /**< isLastChunk data   0:Not last; 1：Is last */
    long long offset;         /**< The offset of file writing, -1 mean is written in append form */
    bool flag;                /**< flag */
};

struct DumpTensorInfo {
    uint32_t headSize;
    uint32_t funcId;
    uint32_t taskId;
    uint32_t callopMagic;
    int32_t coreId;
    int32_t dataType;  // INT8...
    int32_t rawMagic;
    int32_t dims;
    int64_t exeStart;
    int64_t exeEnd;
    uint64_t rootHash;
    uint64_t funcHash;
    uint64_t timeStamp;
    uint64_t shape[DEV_SHAPE_DIM_MAX];
    uint64_t offset[DEV_SHAPE_DIM_MAX];
    uint64_t rawShape[DEV_SHAPE_DIM_MAX];
    uint64_t tensorAddr{0};
};

struct DumpTensorData {
    uint64_t datasize{4};
    uint64_t data;
    std::uint8_t dataByte;
    uint64_t dataOffset{0};

    void TraverseAllAhapeIndexCombinations(const uint64_t shape[], const uint64_t stride[], const uint64_t offset[],
                                           uint32_t idx, uint32_t dims, uint64_t tensorAddr) {
        if (idx != dims - 1) {
            for (uint64_t k = 0; k < shape[idx]; k++) {
                auto newAddr = tensorAddr + offset[idx] * dataByte * stride[idx] + k * stride[idx] * dataByte;
                TraverseAllAhapeIndexCombinations(shape, stride, offset, idx + 1, dims, newAddr);
            }
        } else {
            auto ret = memcpy_s(reinterpret_cast<uint8_t *>(data) + dataOffset, shape[idx] * dataByte,
                        reinterpret_cast<const uint8_t *>(tensorAddr) + offset[idx] * dataByte, shape[idx] * dataByte);
            if (ret != 0) {
                DEV_ERROR("memcpy_s failed, ret is %d.", ret);
            }
            dataOffset = dataOffset + shape[idx] * dataByte;
        }
    }

    DumpTensorData(DumpTensorInfo info, uint64_t dataAddr) {
        dataByte = BytesOf(static_cast<DataType>(info.dataType));
        datasize = dataByte;
        for (int32_t i = 0; i < info.dims; i++) {
            datasize *= info.shape[i];
        }
        if (datasize > DEV_DUMP_DATA_SIZE) {
            return;
        }

        uint64_t stride[DEV_SHAPE_DIM_MAX];
        stride[info.dims - 1] = 1;

        for (int32_t k = info.dims - 2; k >= 0; k--) {
            stride[k] = stride[k + 1] * info.rawShape[k + 1];
        }

        data = dataAddr;
        TraverseAllAhapeIndexCombinations(info.shape, stride, info.offset, 0, info.dims, info.tensorAddr);
    }

    int GetDumpSize() const {
        DEV_DEBUG("Tensorinfo size is %zu, Tensor data size %lu.", sizeof(DumpTensorInfo), datasize);
        return datasize;
    }

};

class AicoreDump {
public:
    AicoreDump(){};
    ~AicoreDump(){};
    uint64_t dataSize_{0};
    void Init(DevStartArgs *startArgs, int schedIdx) {
        auto devProg = startArgs->devProg;
        auto deviceArgs = &devProg->devArgs;
        SetHostPid(deviceArgs->hostPid);
        if (enableDump_) {
            deviceId_ = deviceArgs->deviceId;
            uint64_t baseAddr = startArgs->contextWorkspaceAddr;
            baseAddr += devProg->memBudget.aicoreSpilled + devProg->memBudget.tensor.Total() + devProg->memBudget.debug.dumpTensor;

            dataAddr = baseAddr + schedIdx * DEV_DUMP_DATA_SIZE;
            DEV_DEBUG("dataAddr is %lu.", dataAddr);
        }
    }

    void DoDump(DeviceTask* devTask, std::string iOinfo, int32_t taskId, int32_t coreId, int64_t execStart = 0, int64_t execEnd = 0) {
        if (IsEnableDump()) {
            DumpInit(taskId, coreId, execStart, execEnd);
            DoDump(devTask, iOinfo);
        }
    }

    void DumpInit(int32_t taskId, int32_t coreId, int64_t execStart = 0, int64_t execEnd = 0) {
        taskId_ = taskId;
        coreId_ = coreId;
        execStart_ = execStart;
        execEnd_ = execEnd;
        timeStamp_ = GetTimeMonotonic();
    }

    void SetHostPid(uint32_t hostPid) {
        hostPid_ = hostPid;
        DEV_DEBUG("hostPid is %u.", hostPid_);
        enableDump_ = (hostPid_ != 0);
    }
    bool IsEnableDump() const { return enableDump_; }

    inline bool DumpData(const IDE_SESSION &ideSession, std::string &fileName, unsigned char *dataBuf,
        uint64_t dataSize, bool &isLast) const {
        IdeDumpChunk ideDumpChunk = {
            .fileName = const_cast<char *>(fileName.c_str()),
            .dataBuf = dataBuf,
            .bufLen = static_cast<unsigned int>(dataSize),
            .isLastChunk = isLast ? 1U : 0,
            .offset = -1,
            .flag = 0,
        };

        DEV_DEBUG("Start to ideDump tensor data.");
        const int ideState = IdeDumpData(ideSession, &ideDumpChunk);
        DEV_DEBUG("Finish ideDump with ideState is %d.", ideState);
        return ideState == 0;
    }

    void Dump(const IDE_SESSION &ideSession, DumpTensorInfo dumpTensorInfo, std::string &fileName,
        bool isLast) {
        DumpTensorData dumpTensorData(dumpTensorInfo, dataAddr);
        dataSize_ = dumpTensorData.GetDumpSize();
        if (dataSize_ > DEV_DUMP_DATA_SIZE) {
            DEV_WARN("Tensor data size %lu is larger than dump size %lu.", dataSize_, DEV_DUMP_DATA_SIZE);
            return;
        }
        bool ret = DumpData(ideSession, fileName, reinterpret_cast<uint8_t *>(&dumpTensorInfo),
            dumpTensorInfo.headSize, isLast);
        if (!ret) {
            DEV_ERROR("Dump Tensor info not successful.");
            return;
        }
        ret = DumpData(ideSession, fileName, reinterpret_cast<uint8_t *>(dumpTensorData.data), dataSize_, isLast);
        if (!ret) {
            DEV_ERROR("Dump Tensor data not successful.");
            return;
        }
    }

    void GetTensorShapeInfo(npu::tile_fwk::TensorInfo *info, std::string &shapeInfo) {
        std::ostringstream oss;
        for (uint32_t i = 0; i < info->dims; i++) {
            oss << "_" << std::to_string(info->shape[i]);
        }
        shapeInfo = oss.str();
    }

    uint64_t GetRawTensorAddr(DevAscendRawTensor* rawTensor, DevAscendFunctionDuppedData* dupData) const {

        uint64_t addr = 0ULL;
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            AddressDescriptor incast = dupData->GetIncastAddress(rawTensor->ioIndex);
            addr = incast.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            AddressDescriptor outcast = dupData->GetOutcastAddress(rawTensor->ioIndex);
            addr = outcast.addr;
        } else {
            uintdevptr_t runtimeWorkspace = dupData->GetRuntimeWorkspace();
            addr = runtimeWorkspace + rawTensor->addrOffset;
        }
        return addr;
    }

    DumpTensorInfo GetDumpTensorInfo(DynDeviceTask* dyntask, std::string iOinfo, int32_t tensorIdx) {
        auto opIdx = TaskID(taskId_);
        auto func = dyntask->dynFuncDataCacheList[FuncID(taskId_)].devFunc;
        auto dupData = dyntask->dynFuncDataCacheList[FuncID(taskId_)].duppedData;
        DumpTensorInfo dumpTensorInfo;

        auto setDumpTensorInfo = [&](DevAscendRawTensor* rawTensor, int32_t idx, bool isIOperand) {
            uint32_t dimSize = rawTensor->GetDim();
            dumpTensorInfo.headSize = sizeof(DumpTensorInfo);
            dumpTensorInfo.funcId = FuncID(taskId_);
            dumpTensorInfo.callopMagic = func->GetOperationDebugOpmagic(opIdx);
            dumpTensorInfo.taskId = taskId_;
            dumpTensorInfo.rawMagic = rawTensor->rawMagic;
            dumpTensorInfo.coreId = coreId_;
            dumpTensorInfo.dataType = static_cast<uint32_t>(rawTensor->dataType);
            dumpTensorInfo.dims = dimSize;
            dumpTensorInfo.exeStart = execStart_;
            dumpTensorInfo.exeEnd = execEnd_;
            dumpTensorInfo.rootHash = func->rootHash;
            dumpTensorInfo.funcHash = dyntask->cceBinary[func->GetOperationAttrCalleeIndex(opIdx)].funcHash;
            GetTensorOffsetAndShape<false>(func, dumpTensorInfo.offset, dumpTensorInfo.shape,
                    &(dupData->GetExpression(0)), dimSize, opIdx, idx, isIOperand);
            dumpTensorInfo.tensorAddr = GetRawTensorAddr(rawTensor, dupData);
            for (uint32_t i = 0; i < dimSize; i++) {
                dumpTensorInfo.rawShape[i] = rawTensor->shape.At(i, dupData->GetExpressionAddr());
            }
            dumpTensorInfo.timeStamp = timeStamp_;
        };
        if (iOinfo == "input") {
            uint64_t rawIdx = func->GetOperationIOperand(opIdx, tensorIdx)->rawIndex;
            auto *rawTensor = func->GetRawTensor(rawIdx);
            setDumpTensorInfo(rawTensor, tensorIdx, true);
        } else {
            uint64_t rawIdx = func->GetOperationOOperand(opIdx, tensorIdx)->rawIndex;
            auto *rawTensor = func->GetRawTensor(rawIdx);
            setDumpTensorInfo(rawTensor, tensorIdx, false);
        }
        return dumpTensorInfo;
    }

    void DoDump(DeviceTask* devTask, std::string iOinfo) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(devTask);
        auto func = dyntask->dynFuncDataCacheList[FuncID(taskId_)].devFunc;
        auto opIdx = TaskID(taskId_);
        int32_t tensorNum = (iOinfo == "input") ? func->GetOperationIOperandSize(opIdx) :
            func->GetOperationOOperandSize(opIdx);
        if (!IdeDumpStart || !IdeDumpData || !IdeDumpEnd) {
            DEV_ERROR("IdeDumpStart, IdeDumpData, IdeDumpEnd function not found.");
            return;
        }

        std::string dumpPath = "output/dump_tensor/device_" + std::to_string(deviceId_) + "/";
        // ip: port only matches parameter rules with code, without communication funciton
        const std::string privateInfo =
            "127.0.0.1:22118;" + std::to_string(deviceId_) + ";" + std::to_string(hostPid_);
        const IDE_SESSION ideSession = IdeDumpStart(privateInfo.c_str()); // 建立通道过程 device
        DEV_DEBUG("Current pid is %d, deviceId is %u, privateInfo is %s.", (int)hostPid_, deviceId_, privateInfo.c_str());

        if (ideSession == nullptr) {
            DEV_WARN("Created ideSession failed.");
            return;
        }
        auto seqNo = dyntask->GetDynFuncDataList()->seqNo;
        for (int i = 0; i < tensorNum; i++) {
            auto info = GetDumpTensorInfo(dyntask, iOinfo, i);
            bool isLast = (i == tensorNum - 1) ? true : false;
            std::string tensorInfos = std::to_string(taskId_) + "_" + std::to_string(seqNo) + "_"+ std::to_string(info.callopMagic) + "_" +
                                      std::to_string(info.rootHash) + "_" + std::to_string(info.funcHash) + "_" +
                                      std::to_string(info.rawMagic) + "_" + std::to_string(timeStamp_) + "_" +
                                      DataType2CCEStr(static_cast<DataType>(info.dataType)) + "_" + iOinfo + std::to_string(i) + ".tdump";
            std::string fileName = dumpPath + tensorInfos;
            Dump(ideSession, info, fileName, isLast);
        }
        DEV_DEBUG("Now to close the tensor dump.");
        int m = IdeDumpEnd(ideSession);
        if (m != 0) {
            DEV_WARN("Close ideSession failed with state %d.", m);
        }
    }

private:
    int32_t taskId_{0};
    int32_t coreId_{0};
    int64_t execStart_{0};
    int64_t execEnd_{0};
    uint32_t deviceId_{0};
    uint32_t hostPid_{0};
    uint64_t timeStamp_{0};
    uint64_t dataAddr;
    bool enableDump_{false};
};
} // namespace npu::tile_fwk
#endif
