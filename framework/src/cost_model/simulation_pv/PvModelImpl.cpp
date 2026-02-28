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
 * \file PvModelImpl.cpp
 * \brief
 */

#include <iostream>
#include <string>

#include "PvModelConfig.h"
#include "codegen/codegen.h"
#include "PvModelImpl.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "tilefwk/tilefwk_log.h"

using namespace npu::tile_fwk;

namespace CostModel {
const uint32_t PV_REG_PC = 0;
const uint32_t PV_REG_PARA_BASE = 4;
const uint32_t PV_REG_BLOCK_DIM = 9;
const uint32_t PV_REG_TASK_CFG = 163;
const uint32_t PV_STEP_PIPE_ID = 2;
const int VECTOR_SIZE = 20;
const uint64_t HBM_SATRT_ADDR = 0xffff8000;

void PvModelBinHelper::DumpBin(std::vector<uint8_t> &bytes, uint64_t size, std::string path)
{
    std::ofstream outFile(path, std::ios::binary);
    if (!outFile.is_open()) {
        return;
    }

    outFile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (bytes.size() < size) {
        std::vector<uint8_t> zeros(size-bytes.size(), 0);
        outFile.write(reinterpret_cast<const char*>(zeros.data()), zeros.size());
    }

    outFile.close();
    return;
}

void PvModelBinHelper::ReadBin(std::string path, std::vector<uint8_t> &bytes)
{
    std::ifstream inFile(path, std::ios::binary);

    if (!inFile.is_open()) {
        SIMULATION_LOGE("open bin file error: %s", path.c_str());
        return;
    }

    for (size_t i = 0; i < bytes.size(); i++) {
        char ch = 0;
        if (!inFile.eof()) {
            inFile.read(&ch, 1);
        }
        bytes[i] = ch;
    }

    inFile.close();
    return;
}

uint64_t PvModelBinHelper::GetBinSize(std::string path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        SIMULATION_LOGE("open file error: %s", path.c_str());
        return 0;
    }

    uint64_t fileSize = file.tellg();
    file.close();
    return fileSize;
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::Submit(npu::tile_fwk::Function *func, PvData *data, int level, std::string dir) {
    dir_ = dir;
    data_ = data;
    level_ = level;
    func_ = func;
    if (level_ > 0) {
        allocator_ = std::make_unique<PvMemAllocator>();
        Prepare(func);
        CodeGen(func);
        BinGen(func);
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::CalcInvokeWorkespace(npu::tile_fwk::Function *function, PvModelInvoke &invoke) {
    uint64_t totalSize = 0;
    npu::tile_fwk::Function *compiledFunction = function;

    /* rawtensor magic -> {workspace offset , shape size} */
    std::map<int, std::pair<uint64_t, uint64_t>> rawTensorOffsetMap;

    auto getRawTensorByTensorMagic = [&compiledFunction](int tensorMagic) -> auto {
        auto rawTensor = compiledFunction->GetTensorMap().GetRawTensorByRawMagic(tensorMagic);
        return rawTensor;
    };

    auto calcOffsetFunc = [](const std::vector<int64_t> &offset, const std::vector<int64_t> &shape) -> uint64_t {
        uint64_t offSetSize = 0;
        auto strideShapeFunc = [&shape](size_t i) -> auto {
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
                                       &compiledFunction](const npu::tile_fwk::LogicalTensorPtr &tensor, int rawMagic,
                                       const std::vector<int64_t> &rawShape, const std::vector<int64_t> &offset,
                                       std::list<InvokeParaOffset> &curSubFuncParaOffset, bool isTensorPara) {
        InvokeParaOffset paraOffset;
        uint64_t rawTensorOffset = 0;
        auto &storage = tensor->storage_;
        auto rawTensor = getRawTensorByTensorMagic(rawMagic);
        int storageId = rawTensor->GetRawMagic();
        uint64_t alignSize = 0;
        if (storage != nullptr) {
            storageId = storage->id_;
            alignSize = storage->length_;
        } else {
            alignSize = (CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType()) + 511) / 512 *
                        512; //  performance standpoint
        }

        auto iter = rawTensorOffsetMap.find(storageId);
        if (iter != rawTensorOffsetMap.end()) {
            /* use raw tensor workspace offset + tensor view offset */
            rawTensorOffset = iter->second.first;
        } else {
            /* insert new offset */
            rawTensorOffset = totalSize;
            totalSize += alignSize;
            rawTensorOffsetMap[storageId] = std::make_pair(rawTensorOffset, alignSize);
        }

        uint64_t offSetSize = calcOffsetFunc(offset, rawShape) * BytesOf(rawTensor->GetDataType());

        paraOffset.isTensorParam = isTensorPara;
        paraOffset.offset = rawTensorOffset + offSetSize;
        paraOffset.rawTensorOffset = rawTensorOffset;
        paraOffset.LogRawTensor(rawTensor);
        paraOffset.rawTensorAddr = nullptr; // null express use workspace addr later
        paraOffset.funcitonMagic = compiledFunction->GetFuncMagic();
        curSubFuncParaOffset.push_back(paraOffset);
        return;
    };

    for (uint64_t i = 0; i < compiledFunction->Operations().size(); ++i) {
        const npu::tile_fwk::SubfuncInvokeInfoTy &subfuncInvoke = compiledFunction->GetSubFuncInvokeInfo(i);
        auto &curSubFuncParaOffset = invoke.invokeParaOffset[i];
        for (const auto &elm : subfuncInvoke.GetTensorParamList()) {
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
            paraOffset.LogRawTensor(rawTensor);
            if (paraOffset.rawTensorAddr == nullptr && paraOffset.opOriginArgsSeq == INVALID_ARG_INDEX) {
                /* tensor para 理论上不该存在此场景,
                 * 等待前端graph&schedule解决此场景，此处兼容如果映射不到原始args上则申请workspace空间 */
                workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset, curSubFuncParaOffset, true);
            } else {
                curSubFuncParaOffset.push_back(paraOffset);
            }
        }
        int incastIndx = 0;
        for (auto &elm : subfuncInvoke.GetIncastTensorParamList()) {
            workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset, curSubFuncParaOffset, false);
            curSubFuncParaOffset.back().ioIndex = incastIndx;
            curSubFuncParaOffset.back().tensorShape = elm.shape;
            curSubFuncParaOffset.back().rawTensorShape = elm.rawShape;
            curSubFuncParaOffset.back().opMagic = elm.opMagic;
            incastIndx++;
        }

        int outcastIndx = 0;
        for (auto &elm : subfuncInvoke.GetOutcastTensorParamList()) {
            workSpaceOffsetProcFunc(elm.tensor, elm.ddrId, elm.rawShape, elm.offset, curSubFuncParaOffset, false);
            curSubFuncParaOffset.back().ioIndex = outcastIndx;
            curSubFuncParaOffset.back().tensorShape = elm.shape;
            curSubFuncParaOffset.back().rawTensorShape = elm.rawShape;
            curSubFuncParaOffset.back().opMagic = elm.opMagic;
            outcastIndx++;
        }
    }
    invoke.invokeParaWorkSpaceSize = totalSize;
    invoke.coreFunctionCnt = compiledFunction->Operations().size();
    invoke.programFunctionCnt = compiledFunction->programs_.size();
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::Prepare(npu::tile_fwk::Function *func) {
    task_.stackSize = func->GetStackWorkespaceSize();
    if (task_.stackSize == 0) {
        task_.stackSize = 1;
    }
    task_.stack.resize(task_.stackSize, 0);
    task_.stackAddr = allocator_->AllocWorkspace(task_.stack.size());

    CalcInvokeWorkespace(func, task_.invoke);
    task_.workspaceSize = task_.invoke.invokeParaWorkSpaceSize;

    if (task_.workspaceSize == 0) {
        task_.workspaceSize = 1;
    }
    task_.workspace.resize(task_.workspaceSize, 0);
    task_.workspaceAddr = allocator_->AllocWorkspace(task_.workspace.size());

    constexpr int hcclContextSize = 1024;
    task_.hcclContextSize = hcclContextSize;
    task_.hcclContextAddr = allocator_->AllocArg(task_.hcclContextSize);

    funcDir_ = dir_ + "/" + func->GetRawName() + "_" + std::to_string(func->GetFuncMagic());
    if (npu::tile_fwk::IsPathExist(funcDir_)) {
        npu::tile_fwk::DeleteDir(funcDir_);
    }
    npu::tile_fwk::CreateDir(funcDir_);

    task_.oriArgs = func->GetOpOriginArgsInfo();
    for (auto &arg : task_.oriArgs) {
        task_.args.emplace_back(std::vector<uint8_t>(data_->Get((void *)arg.addr)));
        if (task_.args.back().size() < arg.size) {
            task_.args.back().resize(arg.size, 0);
        }
        auto pvAddr = allocator_->AllocArg(task_.args.back().size());
        task_.oriArgsMap[arg.addr] = pvAddr;
        task_.oriArgsAddr.emplace_back(pvAddr);
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::CodeGen(npu::tile_fwk::Function *func) {
    func->rootFunc_ = func;
    npu::tile_fwk::CodeGenCtx ctxRoot("", dir_);
    npu::tile_fwk::CodeGen g(ctxRoot);
    if (level_ > 1) {
        g.GenCode(*func,{});
    }

    // add global function
    for (auto &subFuncPair : func->programs_) {
        auto leafFuncAttr = subFuncPair.second->GetLeafFuncAttribute();
        auto binPath = leafFuncAttr == nullptr ? "" : leafFuncAttr->binPath;
        auto srcPath = binPath.substr(0, binPath.length()-1) + "cpp";
        PvModelCodegen::AddGlobalAttr(srcPath);
        npu::tile_fwk::CodeGenCtx ctx;
        npu::tile_fwk::CodeGenCloudNPU cga(ctx);
        auto coreType = leafFuncAttr == nullptr ? npu::tile_fwk::CoreType::INVALID : leafFuncAttr->coreType;
        bool isCube = coreType == npu::tile_fwk::CoreType::AIC;
        npu::tile_fwk::CompileInfo compileInfo(
            *func, ctx, subFuncPair, isCube, subFuncPair.second->IsUnderDynamicFunction());
        compileInfo.SetCCEAbsPath(srcPath);
        compileInfo.SetBinAbsPath(binPath);
        cga.CompileCCE(compileInfo, "");
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::BinGen(npu::tile_fwk::Function *func) {
    for (auto &subFuncPair : func->programs_) {
        auto leafFuncAttr = subFuncPair.second->GetLeafFuncAttribute();
        auto binPath = leafFuncAttr == nullptr ? "" : leafFuncAttr->binPath;
        task_.objPath[subFuncPair.first] = binPath;
        task_.binPath[subFuncPair.first] =
            binPath.length() > 1 ? binPath.substr(0, binPath.length() - 1) + "bin" : std::string("null.bin");
        task_.binType[subFuncPair.first] =
            leafFuncAttr == nullptr ? npu::tile_fwk::CoreType::INVALID : leafFuncAttr->coreType;

        if (level_ > 0) {
            char cmd[2048];
            (void)snprintf_s(cmd, sizeof(cmd), sizeof(cmd) - 1, "llvm-objcopy -O binary -j .text %s %s",
                task_.objPath[subFuncPair.first].c_str(), task_.binPath[subFuncPair.first].c_str());

            int ret = std::system(cmd);
            if (ret != 0) {
                SIMULATION_LOGE("cmd error: %s", cmd);
            }

            auto size = PvModelBinHelper::GetBinSize(task_.binPath[subFuncPair.first]);
            task_.binAddr[subFuncPair.first] = allocator_->AllocCode(size);
        }
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::PrepareInvoke(
    int esgId, std::vector<uint64_t> &invokeOffsetVec, std::vector<uint64_t> &invokeOffsetOriVec) {
    uint64_t paraWorkSpaceAddr = task_.workspaceAddr;
    std::list<InvokeParaOffset> &invokeParaOffsetList = task_.invoke.invokeParaOffset[esgId];
    for (auto &elm : invokeParaOffsetList) {
        uint64_t value;
        uint64_t oriValue;
        if (elm.isTensorParam) {
            if (elm.opOriginArgsSeq != INVALID_ARG_INDEX) {
                value = task_.oriArgsAddr[elm.opOriginArgsSeq] + elm.offset;
                invokeOffsetVec.push_back(value);
                oriValue = task_.oriArgsAddr[elm.opOriginArgsSeq];
                invokeOffsetOriVec.push_back(oriValue);
            } else {
                value = reinterpret_cast<uint64_t>(paraWorkSpaceAddr) + elm.offset;
                invokeOffsetVec.push_back(value);
                oriValue = reinterpret_cast<uint64_t>(paraWorkSpaceAddr);
                invokeOffsetOriVec.push_back(oriValue);
            }
        } else {
            /* raw_tensor_addr_ 为空代表是incast outcast，插入新申请的workspace地址偏移 */
            value = reinterpret_cast<uint64_t>(paraWorkSpaceAddr) + elm.offset;
            invokeOffsetVec.push_back(value);
            oriValue = reinterpret_cast<uint64_t>(paraWorkSpaceAddr);
            invokeOffsetOriVec.push_back(oriValue);
        }
    }
    return;
}

static std::string FileName(const std::string& path) {
    const char* separator = "/";

    size_t pos = path.find_last_of(separator);
    if (pos == std::string::npos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::SetUp(int esgId, int psgId, std::string esgDir) {
    SystemConfig sconfig;
    sconfig.Dump(esgDir+"/spec.toml");

    CaseConfig cconfig;
    cconfig.SetTitle(std::string("esg") + std::to_string(esgId));

    // program
    std::string binName = FileName(task_.binPath[psgId]);
    cconfig.SetBin(task_.binAddr[psgId], "../../"+binName);

    std::vector<uint64_t> invokeOffsetVec;
    std::vector<uint64_t> invokeOffsetOriVec;
    PrepareInvoke(esgId, invokeOffsetVec, invokeOffsetOriVec);

    // param
    std::vector<uint8_t> invokeOffsetByte;
    invokeOffsetByte.reserve(invokeOffsetVec.size() * sizeof(uint64_t));
    const uint8_t* src = reinterpret_cast<const uint8_t*>(invokeOffsetVec.data());
    std::copy(src, src + invokeOffsetVec.size() * sizeof(uint64_t), std::back_inserter(invokeOffsetByte));
    std::string paramPath = esgDir + "/param.bin";
    PvModelBinHelper::DumpBin(invokeOffsetByte, invokeOffsetByte.size(), paramPath);
    auto addr = allocator_->AllocArg(invokeOffsetByte.size());
    cconfig.AddInputArg(addr, invokeOffsetByte.size(), "param.bin");

    // stack
    std::string stackPath = esgDir + "/stack.bin";
    PvModelBinHelper::DumpBin(task_.stack, task_.stackSize, stackPath);
    cconfig.AddInputArg(task_.stackAddr, task_.stackSize, "stack.bin");

    // hccl context
    std::string hcclContextPath = esgDir + "/hcclContext.bin";
    std::vector<uint8_t> hccl(task_.hcclContextSize, 0);
    PvModelBinHelper::DumpBin(hccl, task_.hcclContextSize, hcclContextPath);
    cconfig.AddInputArg(task_.hcclContextAddr, task_.hcclContextSize, "hcclContext.bin");

    // oriAddrParam
    std::vector<uint8_t> invokeOffsetOriByte;
    invokeOffsetOriByte.reserve(invokeOffsetOriVec.size() * sizeof(uint64_t));
    src = reinterpret_cast<const uint8_t*>(invokeOffsetOriVec.data());
    std::copy(src, src + invokeOffsetOriVec.size() * sizeof(uint64_t), std::back_inserter(invokeOffsetOriByte));
    std::string oriAddrParamPath = esgDir + "/oriAddrParam.bin";
    PvModelBinHelper::DumpBin(invokeOffsetOriByte, invokeOffsetOriByte.size(), oriAddrParamPath);
    addr = allocator_->AllocArg(invokeOffsetOriByte.size());
    cconfig.AddInputArg(addr, invokeOffsetOriByte.size(), "oriAddrParam.bin");

    // workspace
    PvModelBinHelper::DumpBin(task_.workspace, task_.workspaceSize, esgDir+"/workspace.bin");
    cconfig.AddInputArg(task_.workspaceAddr, task_.workspaceSize, "workspace.bin");

    // args
    for (size_t i = 0; i < task_.oriArgs.size(); i++) {
        std::string argPath = esgDir + "/" + std::to_string(i) + ".bin";
        PvModelBinHelper::DumpBin(task_.args[i], task_.oriArgs[i].size, argPath);
        cconfig.AddInputArg(task_.oriArgsAddr[i], task_.args[i].size(), std::to_string(i) + ".bin");
    }

    // stack out
    cconfig.AddOutputArg(task_.stackAddr, task_.stackSize, "stack_out.bin");

    // workspace out
    cconfig.AddOutputArg(task_.workspaceAddr, task_.workspaceSize, "workspace_out.bin");

    // out args
    for (size_t i = 0; i < task_.oriArgs.size(); i++) {
        cconfig.AddOutputArg(task_.oriArgsAddr[i], task_.args[i].size(), std::to_string(i) + "_out.bin");
    }
    cconfig.Dump(esgDir+"/config.toml");
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::Run(int esgId, int psgId) {
    if (level_ > 0) {
        SIMULATION_LOGI("[PVMODEL]Run esgId: %d,psgId: %d", esgId, psgId);
        std::string esgDir = funcDir_ + "/esg" + std::to_string(esgId);
        (void)npu::tile_fwk::CreateDir(esgDir);
        SetUp(esgId, psgId, esgDir);
        RunModel(esgDir);
        TearDown(esgDir);
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::RunModel(std::string esgDir)
{
    char cmd[2048];
    (void)snprintf_s(cmd, sizeof(cmd), sizeof(cmd) - 1,
    "cd %s/ && ../../../../../../../../PvModel%s --gtest_filter=test_st_case.test_st_pv --spec=spec.toml", esgDir.c_str(), arch_.c_str());
    SIMULATION_LOGI("[PVMODEL] %s", cmd);

    int result = std::system(cmd);
    if (result != 0) {
        SIMULATION_LOGE("cmd error: %s", cmd);
    }
}

template <typename SystemConfig, typename CaseConfig>
void PvModelImpl<SystemConfig, CaseConfig>::TearDown(std::string esgDir)
{
    PvModelBinHelper::ReadBin(esgDir+"/stack_out.bin", task_.stack);
    PvModelBinHelper::ReadBin(esgDir+"/workspace_out.bin", task_.workspace);

    // out args
    for (size_t i = 0; i < task_.oriArgs.size(); i++) {
        PvModelBinHelper::ReadBin(esgDir + "/" + std::to_string(i) + "_out.bin", task_.args[i]);
    }
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::Run(DynFuncData *funcdata, int coreId, int funcId, int taskId)
{
    SIMULATION_LOGI("[AICORE] core  %d, func %d, task %d", coreId, funcId, taskId);
    auto data = &funcdata[funcId];
    auto opAttrs = &data->opAttrs[data->opAtrrOffsets[taskId]];
    auto psgId = opAttrs[0];
    auto cce = &cceBin[psgId];
    std::string dir(dir_ + "/leaf_" + std::to_string(funcId) + "_" + std::to_string(taskId));
    std::string coreType[] = {"AIV", "AIC", "MIX", "AICPU", "HUB", "GMATOMIC", "INVALID"};
    dir += "_" + coreType[static_cast<int>(cce->coreType)];
    (void)CreateDir(dir);

    if (cce->coreType != CoreType::AIV && cce->coreType != CoreType::AIC && cce->coreType != CoreType::MIX) {
        return;
    }

    DynFuncData dupData;
    memset_s(&dupData, sizeof(dupData), 0, sizeof(dupData));
    SetUp(cce, data, static_cast<uint64_t>(data->opAtrrOffsets[taskId]), dir, &dupData);
    RunModel(dir);
    TearDown(dir, &dupData);
}

template <typename SystemConfig, typename CaseConfig>
uint64_t DynPvModelImpl<SystemConfig, CaseConfig>::LookupWorkspace(uint64_t addr)
{
    if (addr >= workspace_.hostPtr && addr <= workspace_.hostPtr + workspace_.size) {
        return addr - workspace_.hostPtr + workspace_.devPtr;
    }
    return 0;
}

template <typename SystemConfig, typename CaseConfig>
uint64_t DynPvModelImpl<SystemConfig, CaseConfig>::LookupData(uint64_t addr)
{
    for (auto &d : data_) {
        if (addr >= d.hostPtr && addr <= d.hostPtr + d.size) {
            return addr - d.hostPtr + d.devPtr;
        }
    }
    return 0;
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::BuildFuncData(DynFuncData *funcdata, DynFuncData *dupData, uint64_t *refAddr, uint64_t *refSize, std::vector<uint8_t> *ref_data)
{
    uint64_t opAttrSize = funcdata->opAttrSize * sizeof(uint64_t);
    uint64_t exprSize = funcdata->exprNum * sizeof(uint64_t);
    uint64_t rawDescSize = funcdata->rawTensorDescSize * sizeof(DevRawTensorDesc);
    uint64_t rawTensorSize = funcdata->rawTensorAddrSize * sizeof(uint64_t);
    *refSize = opAttrSize + exprSize + rawTensorSize + rawDescSize;

    std::vector<uint8_t> ref(*refSize, 0);
    uint64_t offset = 0;
    auto p = reinterpret_cast<uint8_t *>(funcdata->opAttrs);
    std::copy(p, p + opAttrSize, ref.begin() + offset);
    offset += opAttrSize;

    p = reinterpret_cast<uint8_t *>(funcdata->exprTbl);
    std::copy(p, p + exprSize, ref.begin() + offset);
    offset += exprSize;

    p = reinterpret_cast<uint8_t *>(funcdata->rawTensorDesc);
    std::copy(p, p + rawDescSize, ref.begin() + offset);
    offset += rawDescSize;

    constexpr uint32_t RAW_TENSOR_OFFSET_SIZE = 63;
    std::vector<uint64_t> tensorAddr(funcdata->rawTensorAddrSize);
    for (size_t i = 0; i < funcdata->rawTensorAddrSize; i++) {
        auto addr = reinterpret_cast<uint64_t>(funcdata->rawTensorAddr[i]) & ((1UL << RAW_TENSOR_OFFSET_SIZE) - 1);
        tensorAddr[i] = LookupData(addr);
        if (!tensorAddr[i]) {
            throw std::runtime_error(std::string("bad incast tensor addr: ") + std::to_string(addr));
        }
    }
    auto err = memcpy_s(ref.data() + offset, rawTensorSize, tensorAddr.data(), rawTensorSize);
    ASSERT(err == 0) << "[SIMULATION]: tensorAddr copy failed. error=" << err;
    *ref_data = ref;

    auto addr = allocator_->AllocArg(*refSize);
    *refAddr = addr;
    dupData->opAttrs = reinterpret_cast<uint64_t *>(addr);
    addr += opAttrSize;
    dupData->exprTbl = reinterpret_cast<uint64_t *>(addr);
    addr += exprSize;
    dupData->rawTensorDesc = reinterpret_cast<DevRawTensorDesc *>(addr);
    addr += rawDescSize;
    dupData->rawTensorAddr = reinterpret_cast<uint64_t *>(addr);
    dupData->opAttrSize = funcdata->opAttrSize;
    dupData->rawTensorAddrSize = funcdata->rawTensorAddrSize;
    dupData->rawTensorDescSize = funcdata->rawTensorDescSize;
    dupData->exprNum = funcdata->exprNum;
    BuildFuncDataWorkSpace(funcdata, dupData);
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::BuildFuncDataWorkSpace(DynFuncData *funcdata, DynFuncData *dupData)
{
    if (funcdata->workspaceAddr) {
        constexpr int workspaceSize = 10 * 1024 * 1024;
        dupData->workspaceAddr = allocator_->AllocWorkspace(workspaceSize);
        if (!dupData->workspaceAddr) {
            throw std::runtime_error(std::string("bad workspace addr: ") + std::to_string(funcdata->workspaceAddr));
        }
    } else {
        dupData->workspaceAddr = 0;
    }

    if (funcdata->stackWorkSpaceSize) {
        dupData->stackWorkSpaceAddr = LookupWorkspace(funcdata->stackWorkSpaceAddr);
        if (!dupData->stackWorkSpaceAddr) {
            throw std::runtime_error(std::string("bad stack addr: ") + std::to_string(funcdata->stackWorkSpaceAddr));
        }
    } else {
        dupData->stackWorkSpaceAddr = 0;
    }
    dupData->workspaceAddr = workspace_.devPtr;
    dupData->stackWorkSpaceSize = funcdata->stackWorkSpaceSize;
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::SetUp(PvModelCceBin *cce, DynFuncData *funcdata, uint64_t opAttrOffset, std::string dir, DynFuncData *dupData)
{
    SystemConfig sconfig;
    sconfig.Dump(dir + "/spec.toml");
    CaseConfig config;
    caseConfig = &config;

    caseConfig->SetTitle(std::string("func_") + std::to_string(cce->funcHash));

    // program
    auto binName = FileName(cce->binPath);
    (void)CopyFile(cce->binPath, dir + "/" + binName);
    auto srcName = FileName(cce->srcPath);
    (void)CopyFile(cce->srcPath, dir + "/" + srcName);
    auto binSize = PvModelBinHelper::GetBinSize(cce->binPath);
    auto binAddr = allocator_->AllocCode(binSize);
    caseConfig->SetBin(binAddr, "./" + binName);

    // aic/aic flag
    uint64_t aicFlag = 0;
    if (binName.find("aiv") != std::string::npos) {
        aicFlag = static_cast<uint64_t>(1);
    }
    caseConfig->SetCoreType(aicFlag);
    pv_set_toml((dir + "/spec.toml").c_str());
    pv_init_(0, 0, 1, (dir + std::string("/../pvlog/")).c_str(), 0);
    pv_launch_sub_core_(binAddr, (dir + "/" + binName).c_str(), caseConfig->GetCoreType(), 0);

    uint64_t hbm_para_start_addr = HBM_SATRT_ADDR;
    uint8_t value_1_ = 1;
    uint8_t *value_1_ptr = &value_1_;
    pv_reg_write_((uint32_t)1, PV_REG_PC, (uint8_t *)&binAddr, caseConfig->GetCoreType(), 0);
    pv_reg_write_((uint32_t)1, PV_REG_PARA_BASE, (uint8_t *)&hbm_para_start_addr, caseConfig->GetCoreType(), 0);
    pv_reg_write_((uint32_t)1, PV_REG_BLOCK_DIM, value_1_ptr, caseConfig->GetCoreType(), 0);
    pv_reg_write_((uint32_t)1, PV_REG_TASK_CFG, value_1_ptr, caseConfig->GetCoreType(), 0);
    LoadPvConfig(funcdata, opAttrOffset, dir, dupData, hbm_para_start_addr);
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::LoadPvConfig(DynFuncData *funcdata, uint64_t opAttrOffset, std::string dir, DynFuncData *dupData, uint64_t hbm_para_start_addr) {
    std::vector<uint64_t> para_arg(VECTOR_SIZE, 0);
    uint32_t para_offset = 0;

    // funcdata
    uint64_t refAddr;
    uint64_t refSize;
    std::vector<uint8_t> ref_data;
    BuildFuncData(funcdata,  dupData, &refAddr, &refSize, &ref_data);
    std::vector<uint8_t> dup(reinterpret_cast<uint8_t *>(dupData), reinterpret_cast<uint8_t *>(dupData) + sizeof(DynFuncData));
    auto addr = allocator_->AllocArg(dup.size());
    PvModelBinHelper::DumpBin(dup, dup.size(), dir + "/funcdata.bin");
    caseConfig->AddInputArg(addr, dup.size(), "funcdata.bin");
    pv_mem_write_(0, addr, dup.size(), dup.data(), caseConfig->GetCoreType(), 0);
    para_arg[para_offset] = addr;
    para_offset++;

    // attr offset
    std::vector<uint8_t> offset(sizeof(uint64_t), 0);
    memcpy_s(offset.data(), sizeof(uint64_t), &opAttrOffset, sizeof(uint64_t));
    addr = allocator_->AllocArg(offset.size());
    PvModelBinHelper::DumpBin(offset, offset.size(), dir + "/offset.bin");
    caseConfig->AddInputArg(addr, offset.size(), "offset.bin");
    pv_mem_write_(0, addr, offset.size(), offset.data(), caseConfig->GetCoreType(), 0);
    para_arg[para_offset] = addr;
    para_offset++;

    // ref
    PvModelBinHelper::DumpBin(ref_data, ref_data.size(), dir + "/ref.bin");
    caseConfig->AddInputArg(refAddr, ref_data.size(), "ref.bin");
    pv_mem_write_(0, refAddr, refSize, ref_data.data(), caseConfig->GetCoreType(), 0);
    para_arg[para_offset] = refAddr;
    para_offset++;

    // input tensor
    for (size_t i = 0; i < data_.size(); i++) {
        std::string name = std::string("tensor_") + std::to_string(i) + ".bin";
        std::vector<uint8_t> tensorData(reinterpret_cast<uint8_t *>(data_[i].hostPtr), reinterpret_cast<uint8_t *>(data_[i].hostPtr) + data_[i].size);
        PvModelBinHelper::DumpBin(tensorData, data_[i].size, dir + "/" + name);
        caseConfig->AddInputArg(data_[i].devPtr, data_[i].size, name);
        pv_mem_write_(0, data_[i].devPtr, data_[i].size, tensorData.data(), caseConfig->GetCoreType(), 0);
        para_arg[para_offset] = data_[i].devPtr;
        para_offset++;
    }

    // input workspace
    if (workspace_.size) {
        std::vector<uint8_t> workspaceData(reinterpret_cast<uint8_t *>(workspace_.hostPtr), reinterpret_cast<uint8_t *>(workspace_.hostPtr) + workspace_.size);
        PvModelBinHelper::DumpBin(workspaceData, workspaceData.size(), dir+"/workspace.bin");
        caseConfig->AddInputArg(workspace_.devPtr, workspace_.size, "workspace.bin");
        pv_mem_write_(0, workspace_.devPtr, workspace_.size, workspaceData.data(), caseConfig->GetCoreType(), 0);
        para_arg[para_offset] = dupData->stackWorkSpaceAddr;
        para_offset++;
    }

    // input stack workspace
    if (funcdata->stackWorkSpaceSize) {
        std::vector<uint8_t> stackData(reinterpret_cast<uint8_t *>(funcdata->stackWorkSpaceAddr), reinterpret_cast<uint8_t *>(funcdata->stackWorkSpaceAddr) + funcdata->stackWorkSpaceSize);
        PvModelBinHelper::DumpBin(stackData, stackData.size(), dir + "/stack.bin");
        caseConfig->AddInputArg(dupData->stackWorkSpaceAddr, dupData->stackWorkSpaceSize, "stack.bin");
        pv_mem_write_(0, dupData->stackWorkSpaceAddr, dupData->stackWorkSpaceSize, stackData.data(), caseConfig->GetCoreType(), 0);
        para_arg[para_offset] = dupData->stackWorkSpaceAddr;
        para_offset++;
    }

    // output tensor
    for (size_t i = 0; i < data_.size(); i++) {
        std::string name = std::string("tensor_") + std::to_string(i) + "_out.bin";
        caseConfig->AddOutputArg(data_[i].devPtr, data_[i].size, name);
        para_arg[para_offset] = data_[i].devPtr;
        para_offset++;
    }

    // output stack
    if (funcdata->stackWorkSpaceSize) {
        caseConfig->AddOutputArg(dupData->stackWorkSpaceAddr, dupData->stackWorkSpaceSize, "stack_out.bin");
        para_arg[para_offset] = dupData->stackWorkSpaceAddr;
        para_offset++;
    }

    if (workspace_.size) {
        caseConfig->AddOutputArg(workspace_.devPtr, workspace_.size, "workspace_out.bin");
        para_arg[para_offset] = workspace_.devPtr;
        para_offset++;
    }

    caseConfig->Dump(dir + "/config.toml");

    pv_mem_write_(uint32_t(0), hbm_para_start_addr, para_arg.size() * sizeof(uint64_t), (uint8_t *)(&para_arg[0]),
        caseConfig->GetCoreType(), 0);
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::RunModel(std::string dir) {
    step_status_t step_status;
    SIMULATION_LOGI("RunModel here");
    do {
        step_status = (step_status_t)pv_step_(PV_STEP_PIPE_ID, caseConfig->GetCoreType(), 0, 0);
    } while (step_status != step_status_t::END && step_status != step_status_t::TIME_OUT);

    for (size_t i = 0; i < data_.size(); i++) {
        std::string name = std::string("tensor_") + std::to_string(i) + "_out.bin";
        uint64_t addr = data_[i].devPtr;
        uint64_t size = data_[i].size;
        readHbmData(dir, name, addr, size);
    }

    readHbmData(dir, "workspace_out.bin", workspace_.devPtr, workspace_.size);
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::readHbmData(
    std::string dir, std::string name, uint64_t addr, uint64_t size) {
    const uint64_t MAX_READ_SIZE = 2048;
    std::vector<uint8_t> model_data(MAX_READ_SIZE);
    std::ofstream outFile(dir + "/" + name, std::ios::binary);
    if (!outFile.is_open()) {
        return;
    }
    for (uint64_t i = 0; i < size; i += MAX_READ_SIZE) {
        const uint32_t read_size = std::min(MAX_READ_SIZE, (size - i));
        pv_mem_read_(0, addr + i, read_size, model_data.data(), caseConfig->GetCoreType(), 0);
        outFile.write(reinterpret_cast<const char *>(model_data.data()), read_size);
    }
    outFile.close();
}

template <typename SystemConfig, typename CaseConfig>
void DynPvModelImpl<SystemConfig, CaseConfig>::TearDown(std::string dir, DynFuncData *fundata)
{
    // update tensor
    for (size_t i = 0; i < data_.size(); i++) {
        std::string name = std::string("tensor_") + std::to_string(i) + "_out.bin";
        std::vector<uint8_t> d(reinterpret_cast<uint8_t *>(data_[i].hostPtr), reinterpret_cast<uint8_t *>(data_[i].hostPtr) + data_[i].size);
        PvModelBinHelper::ReadBin(dir + "/" + name, d);
        memcpy_s(reinterpret_cast<void *>(data_[i].hostPtr), data_[i].size, d.data(), data_[i].size);
    }

    // update stack
    if (fundata->stackWorkSpaceSize) {
        std::vector<uint8_t> s(reinterpret_cast<uint8_t *>(fundata->stackWorkSpaceAddr), reinterpret_cast<uint8_t *>(fundata->stackWorkSpaceAddr) + fundata->stackWorkSpaceSize);
        PvModelBinHelper::ReadBin(dir + "/stack_out.bin", s);
        memcpy_s(reinterpret_cast<void*>(fundata->stackWorkSpaceAddr), fundata->stackWorkSpaceSize, s.data(), fundata->stackWorkSpaceSize);
    }

    if (workspace_.size) {
        std::vector<uint8_t> s(reinterpret_cast<uint8_t *>(workspace_.hostPtr),
        reinterpret_cast<uint8_t *>(workspace_.hostPtr) + workspace_.size);
        PvModelBinHelper::ReadBin(dir + "/workspace_out.bin", s);
        memcpy_s(reinterpret_cast<void*>(workspace_.hostPtr), workspace_.size, s.data(), workspace_.size);
    }
}

template class PvModelImpl<PvModelSystemConfig, PvModelCaseConfig>;

extern "C" std::shared_ptr<PvModel> CreatePvModelImplA2A3() {
    return std::make_shared<PvModelImpl<PvModelSystemConfig, PvModelCaseConfig>>("A2A3");
}

template class DynPvModelImpl<PvModelSystemConfig, PvModelCaseConfig>;

extern "C" std::shared_ptr<DynPvModel> CreateDynPvModelImpl() {
    return std::make_shared<DynPvModelImpl<PvModelSystemConfig, PvModelCaseConfig>>();
}
} // namespace CostModel
