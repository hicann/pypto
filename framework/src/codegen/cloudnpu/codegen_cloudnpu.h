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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_CLOUDNPU_H
#define CODEGEN_CLOUDNPU_H

#include <string>
#include <unordered_set>
#include <utility>

#include "tilefwk/platform.h"
#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
class CompileInfo {
public:
    CompileInfo(Function &topFunc, std::string cceDir, const std::pair<uint64_t, Function *> &subFuncPair, bool isCube,
        bool isUnderDyn)
        : userSpecCCEDir_(std::move(cceDir)),
          isCube_(isCube),
          isUnderDyn_(isUnderDyn),
          attr_(subFuncPair.second->GetLeafFuncAttribute()) {
        Init(topFunc, subFuncPair.first);
    };
    std::string GetCCEAbsPath() const { return cceAbsPath_; }
    void SetCCEAbsPath(const std::string &cceAbsPath) { cceAbsPath_ = cceAbsPath; }

    std::string GetBinAbsPath() const { return binAbsPath_; }
    void SetBinAbsPath(const std::string &binAbsPath) { binAbsPath_ = binAbsPath; }
    static bool IsNeedCompileCCE() {
        bool isNeedCompile = ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, true);
        return isNeedCompile;
    }

    void SetKernelName(const std::string &kernelName) { kernelName_ = kernelName; }
    std::string GetKernelName() const { return kernelName_; }
    void SetFuncDeclare(const std::string &funcDeclare) { funcDeclare_ = funcDeclare; }
    std::string GetFuncDeclare() const { return funcDeclare_; }
    bool IsCube() const { return isCube_; }
    bool isUnderDyn() const { return isUnderDyn_; }

private:
    void Init(Function &topFunc, uint64_t subProgramId) {
        std::string coreType = isCube_ ? "aic" : "aiv";
        std::ostringstream ss;
        std::ostringstream tailStr;

        if ((attr_ != nullptr) && (attr_->mixId != -1)) {
            tailStr << "mix" << attr_->mixId << "_" << coreType;
            if (!isCube_) {
                int aivId = static_cast<int>(attr_->aivCore);
                if (aivId == -1) {
                    tailStr << "x";
                } else {
                    tailStr << aivId;
                }
            }
        } else {
            tailStr << coreType;
        }
        ss << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subProgramId << "_" << tailStr.str();
        cceFileName_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << GetSuffix();
        cceAbsPath_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".o";
        binAbsPath_ = ss.str();
    }
    std::string GetSuffix() const {
        std::string suffix = ".cpp";
        return suffix;
    }

    std::string userSpecCCEDir_;
    bool isCube_{false};
    bool isUnderDyn_{false};
    std::string cceFileName_;
    std::string cceAbsPath_;
    std::string binAbsPath_;
    std::string kernelName_;
    std::string funcDeclare_;
    std::shared_ptr<LeafFuncAttribute> attr_{nullptr};
};

class CodeGenCloudNPU : public CodeGenCCE {
public:
    explicit CodeGenCloudNPU(const CodeGenCtx &cgCtx) : CodeGenCCE(cgCtx) {
        platform_ = Platform::Instance().GetSoc().GetNPUArch();
    };
    ~CodeGenCloudNPU() override = default;

    void GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    std::pair<int, std::string> CompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const;
    std::optional<std::string> GenExtraAlloc(
        const std::shared_ptr<SymbolManager> &sm, const std::shared_ptr<LogicalTensor> &tensor) const;
    std::string GenAllocForLocalBuffer(const Operation &op, const std::shared_ptr<SymbolManager> &sm) const;
    std::string GetCoreArch(const CompileInfo &compileInfo) const;

private:
    std::string GenFuncBodyBefore(
        const std::pair<uint64_t, Function *> &subFuncPair, Function &topFunc, CompileInfo &compileInfo) const;
    std::string GenInclude() const;
    static std::string GenCommentBeforeFuncHeader(Function &subFunc);
    std::string GenFuncHeader(uint64_t programId, Function &topFunc, CompileInfo &compileInfo) const;
    std::string GenFuncBody(Function &subFunc, Function &topFunc) const;
    static std::string GenFuncEnd();
    static std::string GenKernelName(Function &topFunc, uint64_t programId);
    std::string GenLimitValue(FloatSaturateStatus &fs) const;

    bool IsNeedDumpCCE(const std::string &inputFile) const;
    void DumpCCE(const std::string &name, const std::string &code) const;

    void DoCompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const;
    std::string BuildCompileOptions(const CompileInfo &compileInfo, const std::string &compileOptions) const;
    void BuildIncludes(std::ostringstream &oss) const;
    void BuildLLVMParams(std::ostringstream &oss) const;

    std::string GenAlloc(const std::shared_ptr<SymbolManager> &manager, BufferType bufferType,
        npu::tile_fwk::DataType dataType, const npu::tile_fwk::TileRange &range) const;

    std::string GetParamType(const Function &func, bool isUnderDynFunc) const;

    std::string GenDynParamForExpr(const npu::tile_fwk::Function &func) const;

    bool HandleForAICpuSubFunc(Function &subFunc);

    void UpdateSubFunc(std::pair<uint64_t, Function *> subFuncPair, const CompileInfo &compileInfo) const;

    NPUArch platform_;

    std::string GetIncludePathForCompileCCE() const;
    std::string GetPtoTileLibPathByEnv() const;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
