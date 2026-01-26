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
 * \file PvModelImpl.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <list>
#include <regex>
#include <dlfcn.h>
#include "interface/utils/file_utils.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation_pv/PvMemAllocator.h"
#include "cost_model/simulation/base/ModelLogger.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "tilefwk/core_func_data.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/platform.h"

constexpr int INVALID_ARG_INDEX = 0xFFFFFFFF;

namespace CostModel {

inline int64_t CalcShapeSizeFunc(const std::vector<int64_t> &shape) {
    int64_t size = 1;
    for (auto &i : shape) {
        size *= i;
    }
    return size;
}

struct InvokeParaOffset {
    uint8_t *rawTensorAddr{nullptr}; // 原始input output tensor基地址, 如果是子图间workspace incast outcast 则为null
    uint64_t offset{0};
    uint64_t rawTensorOffset{0};
    bool isTensorParam{false};
    uint64_t rawShapeSize{0};
    int rawMagic{0};
    std::string rawSymbol{""};
    int opOriginArgsSeq{INVALID_ARG_INDEX}; // map origin args seq no
    int funcitonMagic{-1};
    int8_t ioIndex{-1};
    int8_t paramType{-1};
    std::vector<int64_t> tensorShape;
    int opMagic{0};
    npu::tile_fwk::DataType datatype{npu::tile_fwk::DataType::DT_INT32};
    std::vector<int64_t> rawTensorShape;
    void LogRawTensor(std::shared_ptr<npu::tile_fwk::RawTensor> rawTensor) {
        auto &rawShape = rawTensor->GetRawShape();
        rawShapeSize = CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType());
        rawMagic = rawTensor->GetRawMagic();
        rawSymbol = rawTensor->GetSymbol();
        datatype = rawTensor->GetDataType();
    }
};

struct PvModelInvoke {
    uint64_t programFunctionCnt; // 同构后的funciton 个数
    uint64_t coreFunctionCnt;
    uint64_t workSpaceStackSize{0}; // ooo 调度use stack workspace
    uint64_t invokeParaWorkSpaceSize{0};
    size_t invokeOffsetSize{0};
    std::vector<std::string> commGroups;
    std::map<uint64_t, std::list<InvokeParaOffset>> invokeParaOffset; // map esgid to all para list
    std::map<uint64_t, uint64_t> coreFunctionIdToProgramId;           // 对应graph 里 esgid map psgid
    std::vector<uint64_t> readyAicIdVec;
    std::vector<uint64_t> readyAivIdVec;
    std::vector<uint64_t> coreFuncBinOffset;
    std::vector<std::vector<uint64_t>> invokeArgsOffset; // esgid to all para offset
    std::vector<std::vector<int64_t>>
        invokeTensorsIdx; // esgid to all para tensorIdx: input0 input1 ... output0 output1, -1 means workspace
    std::vector<uint64_t> coreFunctionInvokeEntryOffset;
    std::vector<uint64_t> coreFunctionTensorInfoOffset;
    std::vector<uint64_t> coreTensorNum;
};

class PvModelTask {
public:
    std::vector<uint8_t> stack;
    uint64_t stackAddr;
    uint64_t stackSize;
    uint64_t hcclContextAddr;
    uint64_t hcclContextSize;
    std::vector<uint8_t> workspace;
    uint64_t workspaceAddr;
    uint64_t workspaceSize;
    PvModelInvoke invoke;
    std::map<uint64_t, uint64_t> binAddr;
    std::map<uint64_t, std::string> objPath;
    std::map<uint64_t, std::string> binPath;
    std::map<uint64_t, npu::tile_fwk::CoreType> binType;
    std::vector<npu::tile_fwk::OriArgInfo> oriArgs;
    std::vector<std::vector<uint8_t>> args;
    std::vector<uint64_t> oriArgsAddr;
    std::map<uint64_t, uint64_t> oriArgsMap;
    std::map<int, uint64_t> stubOutRawTensorAddr;
};

class PvModelBinHelper {
public:
    static void ReadBin(std::string path, std::vector<uint8_t> &bytes);
    static uint64_t GetBinSize(std::string path);
    static void DumpBin(std::vector<uint8_t> &bytes, uint64_t size, std::string path);
};

template <typename SystemConfig, typename CaseConfig>
class PvModelImpl : public PvModel {
private:
    std::string arch_;
    npu::tile_fwk::Function *func_;
    std::string dir_;
    std::string funcDir_;
    PvData *data_;
    PvModelTask task_;
    int level_;
    std::unique_ptr<PvMemAllocator> allocator_;

public:
    PvModelImpl(std::string arch) : arch_(arch) {}
    void Submit(npu::tile_fwk::Function *func, PvData *data, int level, std::string dir);
    void Run(int esgId, int psgId);

private:
    void Prepare(npu::tile_fwk::Function *func);
    void CodeGen(npu::tile_fwk::Function *func);
    void BinGen(npu::tile_fwk::Function *func);
    void CalcInvokeWorkespace(npu::tile_fwk::Function *function, PvModelInvoke &invoke);
    void PrepareInvoke(int esgId, std::vector<uint64_t> &invokeOffsetVec, std::vector<uint64_t> &invokeOffsetOriVec);
    void SetUp(int esgId, int psgId, std::string esgDir);
    void RunModel(std::string esgDir);
    void TearDown(std::string esgDir);
};

class PvModelCodegen {
public:
    static void AddGlobalAttr(std::string srcPath) {
        const std::string searchStr = "[aicore]";
        const std::string replaceStr = "extern \"C\" __global__ [aicore]";

        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        content = ReplaceAll(content, searchStr, replaceStr);

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        outFile << content;
        outFile.close();
    }

    static void AddKernelEntry(std::string srcPath) {
        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        std::string line;
        std::string include_lines;
        std::string other_lines;
        SeparateHeadersAndContent(include_lines, content, other_lines);

        outFile << include_lines;
        auto name = ExtractFunctionName(content);

        std::string decName = R"!!!(
extern "C" [aicore] void {KernelName}(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam);

)!!!";
        std::string entry = R"!!!(
extern "C" __global__ [aicore] void PvModelKernelEntry(__gm__ npu::tile_fwk::DynFuncData *funcData, __gm__ uint64_t *opAttrOffset) {
    CoreFuncParam param = {funcData, &funcData->opAttrs[opAttrOffset[0]], funcData->exprTbl};
    {KernelName}(&param, funcData->stackWorkSpaceAddr, (__gm__ int64_t *)funcData->hcclContext, (__gm__ GMTensorInfo*)NULL);
}

)!!!";
        decName = ReplaceAll(decName, "{KernelName}", name);
        entry = ReplaceAll(entry, "{KernelName}", name);
        outFile << decName;
        outFile << entry;
        outFile << other_lines;
        outFile.close();
    }

private:
    static void SeparateHeadersAndContent(std::string &headers, const std::string &content, std::string &otherContent) {
        std::istringstream stream(content);
        std::string line;

        while (std::getline(stream, line)) {
            if (line.find("#include") == 0) {
                headers += line + "\n";
            } else {
                otherContent += line + "\n";
            }
        }
    }

    static std::string ReplaceAll(std::string str, const std::string &from, const std::string &to) {
        size_t startPos = 0;
        while ((startPos = str.find(from, startPos)) != std::string::npos) {
            str.replace(startPos, from.length(), to);
            startPos += to.length();
        }
        return str;
    }

    static std::string ExtractFunctionName(const std::string &code) {
        std::string functionName;
        std::regex functionPattern(R"(\b\w+\s+(\w+)\s*\([^)]*\))");
        std::smatch match;

        std::string::const_iterator searchStart(code.cbegin());
        std::regex_search(searchStart, code.cend(), match, functionPattern);
        if (match.size() > 1) {
            functionName = match[1].str();
        }

        return functionName;
    }
};

// Dynamic
template <typename SystemConfig, typename CaseConfig>
class DynPvModelImpl : public DynPvModel {
private:
    npu::tile_fwk::Function *func_;
    std::string dir_;
    std::unique_ptr<PvMemAllocator> allocator_;
    struct DataMap {
        uint64_t hostPtr;
        uint64_t devPtr;
        uint64_t size;
    };
    std::vector<DataMap> data_;
    DataMap workspace_;
    std::vector<std::vector<uint8_t>> storage_;

    struct PvModelCceBin {
        uint32_t psgId;
        uint64_t funcHash;
        npu::tile_fwk::CoreType coreType;
        std::string srcPath;
        std::string binPath;
        PvModelCceBin(uint32_t p, uint64_t h, npu::tile_fwk::CoreType t, std::string s = "", std::string b = "") : psgId(p), funcHash(h), coreType(t), srcPath(s), binPath(b) {
        }
    };
    std::vector<PvModelCceBin> cceBin;

public:
    using PvInitFunc = void (*)(
        int pv_mode, int hj_switch, int pv_wrap, const char *out_dir, uint32_t core_id);
    using PvLaunchSubCoreFunc = void (*)(uint64_t pc, const char *bin_file, uint32_t sub_core_id, uint32_t core_id);
    using PvStepFunc = uint32_t (*)(uint32_t pipe_id, uint32_t sub_core_id, uint32_t core_id, uint32_t warp_id);
    using PvMemWriteFunc = void (*)(
        uint32_t mem_type, uint64_t addr, uint64_t size, uint8_t *buf, uint32_t sub_core_id, uint32_t core_id);
    using PvMemReadFunc = void (*)(
        uint32_t mem_type, uint64_t addr, uint64_t size, uint8_t *buf, uint32_t sub_core_id, uint32_t core_id);
    using PvRegWriteFunc = void (*)(
        uint32_t reg_type, uint32_t reg_id, uint8_t *buf, uint32_t sub_core_id, uint32_t core_id);
    using PvSetTomalFunc = void(*)(const char *toml_name);
    CaseConfig *caseConfig;

    explicit DynPvModelImpl() {
        allocator_ = std::make_unique<PvMemAllocator>();
        dir_ = npu::tile_fwk::config::LogTopFolder() + "/PvModelOutput";
        if (npu::tile_fwk::IsPathExist(dir_)) {
            npu::tile_fwk::DeleteDir(dir_);
        }
        npu::tile_fwk::CreateDir(dir_);
    }

    void InitPv() {
        auto socVersion = npu::tile_fwk::Platform::Instance().GetSoc().GetSocVersion();
        std::string soPath = std::string(std::getenv("ASCEND_HOME_PATH")) + "/toolkit/tools/simulator/" + npu::tile_fwk::SocVersionToString(socVersion) + "/lib/libpem_davinci.so";
        void *handle = dlopen((soPath.c_str()), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error("can not load library: " + soPath);
        }
        // Load function symbols
        this->pv_init_ = (PvInitFunc)load_symbol(handle, "pv_init");
        this->pv_launch_sub_core_ = (PvLaunchSubCoreFunc)load_symbol(handle, "pv_launch_sub_core");
        this->pv_step_ = (PvStepFunc)load_symbol(handle, "pv_step");
        this->pv_mem_write_ = (PvMemWriteFunc)load_symbol(handle, "pv_mem_write");
        this->pv_mem_read_ = (PvMemReadFunc)load_symbol(handle, "pv_mem_read");
        this->pv_reg_write_ = (PvRegWriteFunc)load_symbol(handle, "pv_reg_write");
        this->pv_set_toml = (PvSetTomalFunc)load_symbol(handle, "set_toml");
    }

    void* load_symbol(void* handle, std::string symbol) {
        void* func = dlsym(handle, symbol.c_str());
        if (!func) {
            dlclose(handle);
            throw std::runtime_error("Cannot load symbol: " + symbol);
        }
        return func;
    }

    uint64_t *GetDataHostPtr(int index) { return reinterpret_cast<uint64_t *>(data_[index].hostPtr); }

    int GetOutIndex(int index, int out_size) { return data_.size() - out_size + index; }

    void Codegen(npu::tile_fwk::Function *func) {
        auto attr = func->GetDyndevAttribute();
        std::map<std::uint64_t, npu::tile_fwk::Function *> leafDict;
        for (size_t i = 0; i < attr->funcGroup.devRootList.size(); i++) {
            npu::tile_fwk::Function *devRoot = attr->funcGroup.devRootList[i];
            for (auto &[hash, leaf] : devRoot->programs_) {
                (void) hash;
                if (!leafDict.count(leaf->GetFunctionHash().GetHash())) {
                    leafDict[leaf->GetFunctionHash().GetHash()] = leaf;
                }
            }
        }

        cceBin.emplace_back(PvModelCceBin(0, 0, npu::tile_fwk::CoreType::HUB));
        int Len2 = 2;
        int Len3 = 3;
        for (auto &[name, leaf] : leafDict) {
            (void) name;
            if (leaf->IsDummyFunction()) {
                cceBin.emplace_back(PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(), npu::tile_fwk::CoreType::HUB));
            } else {
                auto leafFuncAttr = leaf->GetLeafFuncAttribute();
                auto binPath = leafFuncAttr == nullptr ? "" : leafFuncAttr->binPath;
                auto orgSrcPath = binPath.substr(0, binPath.length() - 1) + "cpp";
                auto srcPath = binPath.substr(0, binPath.length() - Len2) + "_pvmodel.cpp";
                npu::tile_fwk::CopyFile(orgSrcPath, srcPath);
                PvModelCodegen::AddKernelEntry(srcPath);

                auto objPath = srcPath.substr(0, srcPath.length() - Len3) + "o";
                npu::tile_fwk::CodeGenCtx ctx;
                npu::tile_fwk::CodeGenCloudNPU cga(ctx);
                auto coreType = leafFuncAttr == nullptr ? npu::tile_fwk::CoreType::INVALID : leafFuncAttr->coreType;
                bool isCube = coreType == npu::tile_fwk::CoreType::AIC;
                npu::tile_fwk::CompileInfo compileInfo(
                    *func, ctx, {leaf->GetProgramId(), leaf}, isCube, leaf->IsUnderDynamicFunction());
                compileInfo.SetCCEAbsPath(srcPath);
                compileInfo.SetBinAbsPath(objPath);
                cga.CompileCCE(compileInfo, "");

                binPath = srcPath.substr(0, srcPath.length() - Len3) + "bin";
                constexpr int cmdLen = 2048;
                char cmd[cmdLen];
                (void)snprintf_s(cmd, sizeof(cmd), sizeof(cmd)-1, "llvm-objcopy -O binary -j .text %s %s", objPath.c_str(), binPath.c_str());

                int ret = std::system(cmd);
                if (ret != 0) {
                    MLOG_ERROR("cmd error: ", cmd);
                }

                cceBin.emplace_back(
                    PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(), coreType, srcPath, binPath));
            }
        }
    }

    uint8_t *CopyToDev(const uint8_t *data, uint64_t size) {
        std::vector<uint8_t> s(data, data + size);
        uint8_t *hostPtr = s.data();
        storage_.emplace_back(std::move(s));
        return hostPtr;
    }

    uint8_t *CopyTensorToDev(const uint8_t *data, uint64_t size) {
        std::vector<uint8_t> s(data, data + size);
        uint8_t *hostPtr = s.data();
        storage_.emplace_back(std::move(s));
        uint64_t devPtr = allocator_->AllocArg(size);
        DataMap m = {reinterpret_cast<uint64_t>(hostPtr), devPtr, size};
        data_.emplace_back(m);
        std::cout << "[PVMODEL]tensor map host: " << std::hex << m.hostPtr << ", dev: " << std::hex << m.devPtr << ", size: " << m.size << std::endl;
        return hostPtr;
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) { memcpy_s(data, size, devPtr, size); }

    uint8_t *AllocWorkspaceDev(size_t size) {
        std::vector<uint8_t> s(size, 0);
        uint8_t *hostPtr = s.data();
        storage_.emplace_back(std::move(s));
        uint64_t devPtr = allocator_->AllocWorkspace(size);
        DataMap m = {reinterpret_cast<uint64_t>(hostPtr), devPtr, size};
        workspace_ = m;
        std::cout << "[PVMODEL]workspace map host: " << std::hex << m.hostPtr << ", dev: " << std::hex << m.devPtr << ", size: " << m.size << std::endl;
        return hostPtr;
    }

    void Run(npu::tile_fwk::DynFuncData *funcdata, int coreId, int funcId, int taskId);

private:
    void LoadPvConfig(npu::tile_fwk::DynFuncData *funcdata, uint64_t opAttrOffset, std::string dir, npu::tile_fwk::DynFuncData *dupData, uint64_t hbm_para_start_addr);
    void SetUp(PvModelCceBin *cce, npu::tile_fwk::DynFuncData *funcdata, uint64_t opAttrOffset, std::string dir, npu::tile_fwk::DynFuncData *dupData);
    void RunModel(std::string dir);
    void readHbmData(std::string dir, std::string name, uint64_t addr, uint64_t size);
    void TearDown(std::string dir, npu::tile_fwk::DynFuncData *fundata);
    void BuildFuncData(npu::tile_fwk::DynFuncData *funcdata, npu::tile_fwk::DynFuncData *dupData, uint64_t *refAddr, uint64_t *refSize, std::vector<uint8_t> *ref_data);
    void BuildFuncDataWorkSpace(npu::tile_fwk::DynFuncData *funcdata, npu::tile_fwk::DynFuncData *dupData);
    uint64_t LookupWorkspace(uint64_t addr);
    uint64_t LookupData(uint64_t addr);

    PvInitFunc pv_init_;
    PvLaunchSubCoreFunc pv_launch_sub_core_;
    PvStepFunc pv_step_;
    PvMemWriteFunc pv_mem_write_;
    PvMemReadFunc pv_mem_read_;
    PvRegWriteFunc pv_reg_write_;
    PvSetTomalFunc pv_set_toml;
    enum class step_status_t { END = 0, NORMAL = 1, TIME_OUT = 2, CONTINUE = 3, UNDEF };
};
} // namespace CostModel
