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
 * \file aicore_dump.h
 * \brief
 */

#ifndef AICORE_DUMP_H
#define AICORE_DUMP_H
#include <string>
#include <mutex>
#include <sstream>
#include "machine/utils/machine_ws_intf.h"
#include "securec.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk {
static std::mutex dumpLock;
constexpr int LOCAL_INCAST = 1;
constexpr int LOCAL_OUTCAST = 0;
constexpr int DUMP_INCAST = 2;
constexpr int DUMP_OUTCAST = 3;
using IDE_SESSION = void *;
enum IdeErrorT {};
extern "C" {
__attribute__((weak)) IDE_SESSION IdeDumpStart(const char *privInfo);
__attribute__((weak)) IdeErrorT IdeDumpData(IDE_SESSION session, const struct IdeDumpChunk *dumpChunk);
__attribute__((weak)) IdeErrorT IdeDumpEnd(IDE_SESSION session);
__attribute__((weak)) IdeErrorT drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t *local_dev_id);
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
    int functionMagic;
    uint32_t subgraphId;
    uint32_t taskId;
    uint32_t tensorMagic;
    int32_t coreId;
    int32_t dataType;  // INT8...
    int32_t paramType; // 0:func incast; 1:func outcas; 2:mid incast; 3:mid outcast
    int32_t dumpType;  // 00: no dump; 01:func dunp; 10:subgraph dump; 11:func dunp & subgraph dump
    uint32_t idx;
    uint32_t dims;
    int64_t exeStart;
    int64_t exeEnd;
    int shape[MAX_DIMS];
    uint32_t stride[MAX_DIMS];
};

struct DumpTensorData {
    int32_t datasize{4};
    DumpTensorInfo dumpTensorInfo;
    void *data;
    std::uint8_t dataByte;

    void SetDumpTensorInfo(npu::tile_fwk::TensorInfo *info, const int32_t subgraphId, const int32_t &taskId,
        const int32_t &coreId, int64_t execStart, int64_t execEnd) {
        dumpTensorInfo.headSize = sizeof(DumpTensorInfo);
        dumpTensorInfo.functionMagic = info->functionMagic;
        dumpTensorInfo.subgraphId = subgraphId;
        dumpTensorInfo.taskId = taskId;
        dumpTensorInfo.tensorMagic = info->opMagic;
        dumpTensorInfo.coreId = coreId;
        dumpTensorInfo.dataType = info->dataType;
        dumpTensorInfo.paramType = info->paramType;
        dumpTensorInfo.idx = info->idx;
        dumpTensorInfo.dims = info->dims;
        dumpTensorInfo.exeStart = execStart;
        dumpTensorInfo.exeEnd = execEnd;
    }

    int64_t CalculateOffset(const int &idx, const int32_t shapeStride[], const uint32_t dims) const {
        int offset = 1;
        for (uint32_t i = idx; i <= dims; i++) {
            offset *= shapeStride[i];
        }
        return offset;
    }

    bool IsLastShapeIndexCombination(const int32_t shape[], const uint32_t dims, const int32_t indices[]) {
        for (size_t i = 0; i < dims; ++i) {
            if (indices[i] != shape[i] - 1) {
                return false;
            }
        }
        return true;
    }

    bool IsNotLastShapeIndexComBination(const int32_t shape[], const uint32_t dims, const int32_t indices[]) {
        for (size_t i = 0; i < dims; ++i) {
            if (indices[i] < shape[i] - 1) {
                return true;
            }
        }
        return false;
    }

    void GetNextShapeIndexCombination(const int32_t shape[], const uint32_t dims, int32_t indices[]) {
        for (size_t i = 0; i < dims; ++i) {
            indices[i]++;
            if (indices[i] < shape[i]) {
                return;
            }
            indices[i] = 0;
        }
    }

    void CopyTensorData(const int32_t shape[], const int32_t stride[], uint32_t dims, uint64_t tensorAddr,
                        const int32_t indices[]) {
        int64_t dataTotalOffset = 0;
        int64_t dataAddrTotalOffset = 0;
        std::ostringstream oss;
        for (uint32_t i = 0; i < dims; i++) {
            oss << "_" << std::to_string(indices[i]);
        }
        for (uint32_t i = 1; i <= dims; ++i) {
            dataTotalOffset += indices[i - 1] * CalculateOffset(i, shape, dims);
            dataAddrTotalOffset += indices[i - 1] * CalculateOffset(i, stride, dims);
        }
        memcpy_s(reinterpret_cast<void *>(reinterpret_cast<char *>(data) + dataTotalOffset * dataByte),
            shape[dims] * dataByte, reinterpret_cast<const void *>(tensorAddr + dataAddrTotalOffset * dataByte),
            shape[dims] * dataByte);
    }

    void TraverseAllAhapeIndexCombinations(const int32_t shape[], const int32_t stride[],
                                           uint32_t dims, uint64_t tensorAddr) {
        int32_t indices[MAX_DIMS] = {0};
        // special case for [1, 1, x] tensor
        uint64_t tensorNum = datasize / dataByte;
        if ((tensorNum / shape[dims]) == 1) {
            memcpy_s(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(data)), shape[dims] * dataByte,
                reinterpret_cast<const void *>(tensorAddr), shape[dims] * dataByte);
            DEV_DEBUG("Shape is [1, 1, x], tensor data is copied.");
            return;
        }
        // normal case
        while (IsNotLastShapeIndexComBination(shape, dims, indices)) {
            CopyTensorData(shape, stride, dims, tensorAddr, indices);
            GetNextShapeIndexCombination(shape, dims, indices);
            if (IsLastShapeIndexCombination(shape, dims, indices)) {
                CopyTensorData(shape, stride, dims, tensorAddr, indices);
            }
        }
    }

    DumpTensorData(npu::tile_fwk::TensorInfo *info, uint64_t tensorAddr) {
        std::lock_guard<std::mutex> lock(dumpLock);
        if (info == nullptr) {
            DEV_WARN("Tensor info is null.");
            return;
        }
        dataByte = info->dataByte;
        datasize = info->dataByte;
        for (uint32_t i = 0; i < info->dims; i++) {
            DEV_INFO(
                "Tensor infor is not null shape[%u]: %d, stride[%u] : %d.", i, info->shape[i], i, info->stride[i]);
            datasize *= info->shape[i];
            dumpTensorInfo.shape[i] = info->shape[i];
            dumpTensorInfo.stride[i] = info->stride[i];
        }

        data = malloc(datasize);
        DEV_DEBUG("Start to MemCpy the tensor to host , pragramtype is %d.", info->paramType);
        TraverseAllAhapeIndexCombinations(info->shape, info->stride, info->dims - 1, tensorAddr);
    }

    int GetDumpSize() const {
        DEV_DEBUG("Tensorinfo size is %zu, Tensor data size %d.", sizeof(dumpTensorInfo), datasize);
        return datasize;
    }

    ~DumpTensorData() { free(data); }
};

class AicoreDump {
public:
    AicoreDump(){};
    ~AicoreDump(){};
    uint64_t dataSize_{0};
    void DumpInit(int32_t subgraphId, int32_t taskId, int32_t coreId, int64_t execStart = 0, int64_t execEnd = 0) {
        subGraphId_ = subgraphId;
        taskId_ = taskId;
        coreId_ = coreId;
        execStart_ = execStart;
        execEnd_ = execEnd;
    }

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

    void Dump(const IDE_SESSION &ideSession, npu::tile_fwk::TensorInfo *info, uint64_t tensorAddr, std::string &fileName,
        bool isLast) {
        DumpTensorData dumpTensorData(info, tensorAddr);
        dumpTensorData.SetDumpTensorInfo(info, subGraphId_, taskId_, coreId_, execStart_, execEnd_);
        dataSize_ = dumpTensorData.GetDumpSize();
        bool ret = DumpData(ideSession, fileName, reinterpret_cast<uint8_t *>(&dumpTensorData.dumpTensorInfo),
            dumpTensorData.dumpTensorInfo.headSize, isLast);
        if (!ret) {
            DEV_WARN("Dump Tensor info not successful.");
            return;
        }
        ret = DumpData(ideSession, fileName, reinterpret_cast<uint8_t *>(dumpTensorData.data), dataSize_, isLast);
        if (!ret) {
            DEV_WARN("Dump Tensor data not successful.");
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

    void DoDump(int64_t tensorInfo, int64_t tensorNum, int64_t tensorAddr, std::string iOinfo) {
        npu::tile_fwk::TensorInfo *info = reinterpret_cast<npu::tile_fwk::TensorInfo *>(tensorInfo);
        if (info == nullptr || info->hostpid == 0) {
            DEV_DEBUG("Current Datadump is not abailable, please check the state of AST_DATADUMP_PATH.");
            return;
        }
        uint32_t deviceid = 0;
        drvGetLocalDevIDByHostDevID(info->deviceId, &deviceid);
        DEV_DEBUG("Current host deviceid is %u, devicedeviceId is %u.", info->deviceId, deviceid);
        std::string dumpPath = "/data/local/tmp/device_" + std::to_string(info->deviceId) + "/";
        // ip: port only matches parameter rules with code, without communication funciton
        const std::string privateInfo =
            "127.0.0.1:22118;" + std::to_string(deviceid) + ";" + std::to_string(info->hostpid);
        const IDE_SESSION ideSession = IdeDumpStart(privateInfo.c_str()); // 建立通道过程 device
        DEV_DEBUG("Current pid is %d, privateInfo is %s.", (int)info->hostpid, privateInfo.c_str());

        if (ideSession == nullptr) {
            DEV_WARN("Created ideSession failed.");
            return;
        }
        for (int i = 0; i < tensorNum; i++) {
            info = reinterpret_cast<npu::tile_fwk::TensorInfo *>(tensorInfo + i * sizeof(npu::tile_fwk::TensorInfo));
            uint64_t tmpTensorAddr = tensorAddr + i * sizeof(uint64_t);
            DEV_DEBUG("Current info: magic is %u, datatype is %d, paramType is %d, idx is %u.", info->rawMagic,
                info->dataType, info->paramType, info->idx);
            bool isLast = (i == tensorNum - 1) ? true : false;
            std::string shapeInfo = "";
            GetTensorShapeInfo(info, shapeInfo);
            std::string tensorInfos = std::to_string(info->functionMagic) + "_" + std::to_string(subGraphId_) + "/" +
                                       std::to_string(taskId_) + "/" + iOinfo + "_" + std::to_string(info->rawMagic) +
                                       "_" + std::to_string(info->opMagic) + shapeInfo + ".bin";
            std::string fileName = dumpPath + tensorInfos;
            if (iOinfo == "input" && (info->paramType == LOCAL_INCAST || info->paramType == DUMP_INCAST)) {
                DEV_DEBUG("Input File_name is %s, with paramType is %d.", fileName.c_str(), info->paramType);
                Dump(ideSession, info, tmpTensorAddr, fileName, isLast);
            }
            if (iOinfo == "output" && (info->paramType == LOCAL_OUTCAST || info->paramType == DUMP_OUTCAST)) {
                DEV_DEBUG("Output File_name is %s, with paramType is %d.", fileName.c_str(), info->paramType);
                Dump(ideSession, info, tmpTensorAddr, fileName, isLast);
            }
        }
        DEV_DEBUG("Now to close the tensor dump.");
        int m = IdeDumpEnd(ideSession);
        if (m != 0) {
            DEV_WARN("Close ideSession failed with state %d.", m);
        }
    }

private:
    int32_t subGraphId_{0};
    int32_t taskId_{0};
    int32_t coreId_{0};
    int64_t execStart_{0};
    int64_t execEnd_{0};
};
} // namespace npu::tile_fwk
#endif
