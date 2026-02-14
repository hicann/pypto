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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H
#define SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H

#include <cstdint>
#include <vector>

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_tensor_creator.h"

namespace npu::tile_fwk::dynamic {
using DeviceStream = unsigned long long;
DeviceStream DeviceGetAicpuStream();
DeviceStream DeviceGetAicoreStream();

class DeviceTensorData {
public:
    DeviceTensorData() = default;
    DeviceTensorData(DataType dtype, void *addr, const std::vector<int64_t> &shape,
        TileOpFormat format = TileOpFormat::TILEOP_ND)
        : dtype_(dtype), addr_(addr), shape_(shape), format_(format) {}
    DeviceTensorData(DataType dtype, uintptr_t addr, const std::vector<int64_t> &shape,
        TileOpFormat format = TileOpFormat::TILEOP_ND)
        : dtype_(dtype), addr_((void *)addr), shape_(shape), format_(format) {}

    void *GetAddr() const { return addr_; }

    const std::vector<int64_t> &GetShape() const { return shape_; }

    DataType GetDataType() const { return dtype_; }

    TileOpFormat Format() const { return format_; }

    int64_t GetDataSize() const {
        return std::accumulate(shape_.begin(), shape_.end(), BytesOf(dtype_), std::multiplies<>());
    }

private:
    DataType dtype_;
    void *addr_;
    std::vector<int64_t> shape_;
    TileOpFormat format_;
};

struct DeviceLauncherConfig {
    bool onBoard{true};
    int blockdim{0};
    int aicpuNum{5};
    int64_t dynWorkspaceSize{0};
    int64_t repeatNum{1};
    bool runModel{true};
    std::vector<uint64_t> hcclContext;
    bool controlFlowCache{false};
    bool cpuSeparate{false};
    uint64_t workspaceAddr{0};
    bool isCacheOriginShape{true}; // infer cache shape or origin shape

    DeviceLauncherConfig() = default;
    DeviceLauncherConfig(bool onboard, int tblockdim, int taicpunum)
        : onBoard(onboard), blockdim(tblockdim), aicpuNum(taicpunum) {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize, int64_t trepeatNum)
        : dynWorkspaceSize(tdynWorkspaceSize), repeatNum(trepeatNum) {}
    DeviceLauncherConfig(const std::vector<std::uint64_t> &addrs) : hcclContext(addrs) {}

    static DeviceLauncherConfig CreateConfigWithWorkspaceAddr(uint64_t workspaceAddr) {
        DeviceLauncherConfig config;
        config.workspaceAddr = workspaceAddr;
        return config;
    }
};

struct OperatorTensorPara {
    std::vector<DevTensorData> inputTensorParaList;
    std::vector<DevTensorData> outputTensorParaList;
    bool operator==(const OperatorTensorPara &other) const {
        if (inputTensorParaList.size() != other.inputTensorParaList.size()) {
            return false;
        }
        for (size_t i = 0; i < inputTensorParaList.size(); i++) {
            if (!inputTensorParaList[i].shape.Equal(other.inputTensorParaList[i].shape)) {
                return false;
            }
        }

        if (outputTensorParaList.size() != other.outputTensorParaList.size()) {
            return false;
        }
        for (size_t i = 0; i < outputTensorParaList.size(); i++) {
            if (!outputTensorParaList[i].shape.Equal(other.outputTensorParaList[i].shape)) {
                return false;
            }
        }
        return true;
    }
};

struct OperatorTensorParaHash {
    std::size_t operator()(const OperatorTensorPara& para) const {
        std::size_t hash = 0;
        hash_combine(hash, para.inputTensorParaList.size());
        for (const auto& tensor : para.inputTensorParaList) {
            hash_combine(hash, tensor.shape.dimSize);
            for (int i = 0; i < tensor.shape.dimSize; i++) {
                hash_combine(hash, tensor.shape.dim[i]);
            }
        }
        hash_combine(hash, para.outputTensorParaList.size());
        for (const auto& tensor : para.outputTensorParaList) {
            hash_combine(hash, tensor.shape.dimSize);
            for (int i = 0; i < tensor.shape.dimSize; i++) {
                hash_combine(hash, tensor.shape.dim[i]);
            }
        }
        return hash;
    }

private:
    template <class T>
    static void hash_combine(std::size_t& seed, const T& v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class CachedOperator {
public:
    static uint8_t **GetWorkspaceDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->workspaceDevAddr_;
    }
    static uint8_t **GetCfgDataDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->cfgDataDevAddr_;
    }
    static uint8_t **GetMetaDataDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->metaDataDevAddr_;
    }
    static void *GetBinHandleHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->binHandle_;
    }

    uint8_t* FindCtrlFlowCache(
            const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList) {
        auto it = devCtrlFlowCacheMap_.find(BuildOperatorTensorPara(inputList, outputList));
        if (it != devCtrlFlowCacheMap_.end()) {
            return it->second;
        }
        return nullptr;
    }

    void InsertCtrlFlowCache(const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList, uint8_t* cache) {
        devCtrlFlowCacheMap_[BuildOperatorTensorPara(inputList, outputList)] = cache;
    }

private:
    OperatorTensorPara BuildOperatorTensorPara(
            const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList) {
        OperatorTensorPara para; 
        for (const auto& input : inputList) {
            para.inputTensorParaList.emplace_back(DevAscendTensorDataCreator::Create(0, input.GetShape()));
        }

        for (const auto& output : outputList) {
            para.outputTensorParaList.emplace_back(DevAscendTensorDataCreator::Create(0, output.GetShape()));
        }
        return para;
    }

private:
    uint8_t *workspaceDevAddr_{nullptr};
    uint8_t *cfgDataDevAddr_{nullptr};
    uint8_t *metaDataDevAddr_{nullptr};
    void *binHandle_{nullptr};
    std::unordered_map<OperatorTensorPara, uint8_t*, OperatorTensorParaHash> devCtrlFlowCacheMap_;
};

struct Evaluator {
    const std::map<std::string, int64_t> &symbolDict;
    const std::vector<DeviceTensorData> &inputs;
    const std::vector<DeviceTensorData> &outputs;

    int Evaluate(SymbolicScalar &ss) { return Evaluate(ss.Raw()); }

private:
    int64_t GetinputShapeDim(int64_t argIdx, int64_t dim) {
        if (argIdx < (int64_t)inputs.size()) {
            return inputs[argIdx].GetShape()[dim];
        } else {
            return outputs[argIdx - inputs.size()].GetShape()[dim];
        }
    }

    int64_t GetViewValidShapeDim(int64_t validshape, int64_t viewoffset, int64_t viewshape) {
        validshape -= viewoffset;
        if (validshape > viewshape)
            validshape = viewshape;
        else if (validshape < 0)
            validshape = 0;
        return validshape;
    }

    int64_t EvaluateSymbolicCall(const std::string &name, std::vector<int64_t> &vals) {
        if (name == "RUNTIME_GetInputShapeDim") {
            return GetinputShapeDim(vals[0], vals[1]);
        } else if  (name == "RUNTIME_GetViewValidShapeDim") {
            return GetViewValidShapeDim(vals[0], vals[1], vals[0x2]);
        } else {
            ASSERT(false) << "unsupported call " << name;
            return 0;
        }
    }

    int Evaluate(RawSymbolicScalarPtr ss) {
        switch (ss->Kind()) {
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
                auto imm = std::static_pointer_cast<RawSymbolicImmediate>(ss);
                return imm->Immediate();
            }
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
                auto sym = std::static_pointer_cast<RawSymbolicSymbol>(ss);
                ASSERT(symbolDict.count(sym->Name())) << "symbol " << sym->Name() << " not found";
                return symbolDict.find(sym->Name())->second;
            }
            case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
                auto expr = std::static_pointer_cast<RawSymbolicExpression>(ss);
                auto iops = expr->OperandList();
                auto opcode = expr->Opcode();
                if (opcode == SymbolicOpcode::T_MOP_CALL) {
                    std::vector<int64_t> vals;
                    for (size_t i = 1; i < iops.size(); i++) {
                        vals.emplace_back(Evaluate(iops[i]));
                    }
                    auto name = std::static_pointer_cast<RawSymbolicSymbol>(iops[0])->Name();
                    return EvaluateSymbolicCall(name, vals);
                } else if (SymbolicOpcode::T_UOP_BEGIN <= opcode && opcode< SymbolicOpcode::T_UOP_END) {
                    return RawSymbolicExpression::GetSymbolicCalcUnary(opcode)(Evaluate(iops[0]));
                } else if (SymbolicOpcode::T_BOP_BEGIN <= opcode && opcode< SymbolicOpcode::T_BOP_END) {
                    return RawSymbolicExpression::GetSymbolicCalcBinary(opcode)(
                        Evaluate(iops[0]), Evaluate(iops[1]));
                } else {
                    ASSERT(false);
                    return 0;
                }
            }
            default: {
                ASSERT(false);
                return 0;
            }
        }
    }
};

class ExportedOperator : public CachedOperator {
public:
    void ResetFunction(Function *func) { func_ = Program::GetInstance().GetFunctionSharedPtr(func); }

    Function *GetFunction() const { return func_.get(); }

    int64_t AlignUp(int64_t x) const { return (x + 511) & (!511); } // 511 cacheline mask

    uint64_t GetWorkSpaceSize(const std::vector<DeviceTensorData> &inputs,
        const std::vector<DeviceTensorData> &outputs) const {
        auto dynAttr = func_->GetDyndevAttribute();
        std::vector<uint8_t> &devProgData = dynAttr->devProgBinary;
        auto *devProg = reinterpret_cast<DevAscendProgram *>(devProgData.data());
        Evaluator eval{dynAttr->inputSymbolDict, inputs, outputs};
        if (devProg == nullptr) {
            return 0;
        }
        devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem);
        return devProg->memBudget.Total();
    }

private:
    std::shared_ptr<Function> func_;
};

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(ExportedOperator *op,
    const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
    DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize, uint8_t* devCtrlCache = nullptr,
    const DeviceLauncherConfig &config = DeviceLauncherConfig());

int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream);

int DeviceRunOnce(Function *function, uint8_t* hostCtrlCache = nullptr, const DeviceLauncherConfig &config = DeviceLauncherConfig());

int HasInplaceArgs(Function *function);

void DeviceLauncherInit();

void DeviceLauncherFini();

ExportedOperator *ExportedOperatorBegin();

void ExportedOperatorEnd(ExportedOperator *op);

void CopyDevToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor);

void CopyHostToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor);

uint8_t* CopyHostToDev(uint8_t* data, uint64_t size);
void ChangeCaptureModeRelax();
void ChangeCaptureModeGlobal();

} // namespace npu::tile_fwk::dynamic

#endif // SRC_MACHINE_DEVICE_LAUNCHER_H
