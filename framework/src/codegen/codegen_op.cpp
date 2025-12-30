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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"

namespace npu::tile_fwk {
namespace {
bool IsCopyOpWithShapeOffsetAttr(Opcode opcode) {
    bool result = opcode == Opcode::OP_COPY_IN || opcode == Opcode::OP_COPY_OUT ||
                  opcode == Opcode::OP_TRANSPOSE_MOVEOUT || opcode == Opcode::OP_TRANSPOSE_MOVEIN ||
                  opcode == Opcode::OP_INDEX_OUTCAST || opcode == Opcode::OP_LOCAL_COPY_OUT ||
                  opcode == Opcode::OP_REMOTE_REDUCE || opcode == Opcode::OP_REMOTE_GATHER ||
                  opcode == Opcode::OP_FFN_SCHED || opcode == Opcode::OP_FFN_BATCHING ||
                  opcode == Opcode::OP_FFN_COMBINEINFO || opcode == Opcode::OP_FFN_VALIDCNT ||
                  opcode == Opcode::OP_COPY_TO_LOCAL_EXPERT || opcode == Opcode::OP_SHMEM_PUT ||
                  opcode == Opcode::OP_SHMEM_PUT_UB2GM || opcode == Opcode::OP_SHMEM_SIGNAL ||
                  opcode == Opcode::OP_SHMEM_GET || opcode == Opcode::OP_SHMEM_GET_GM2UB ||
                  opcode == Opcode::OP_SHMEM_REDUCE || opcode == Opcode::OP_SHMEM_SET ||
                  opcode == Opcode::OP_SHMEM_MOE_COMBINE_SEND || opcode == Opcode::OP_SHMEM_MOE_COMBINE_RECEIVE;
    return result;
}
} // namespace

template <typename T>
void CombineLastTwoAxis(std::vector<T> &shape, size_t shapeSize) {
    if (shape.size() < NUM2) {
        return;
    }
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = 1;
}

void CodeGenOp::CombineAxis(const Operation &oper, int operandIdx, bool isInput, size_t ioIdx) {
    size_t dim = rawShape[operandIdx].size();
    if (dim <= 1) {
        ALOG_WARN_F("raw shape dim is %d, return", dim);
        return;
    }

    ALOG_INFO_F("operandIdx %d, isInput: %d, ioIdx is %d ", operandIdx, isInput, ioIdx);

    std::vector<bool> needCombineIOIdx;
    if ((isInput && oper.GetAttr(OpAttributeKey::inputCombineAxis, needCombineIOIdx) && needCombineIOIdx[ioIdx]) ||
        (!isInput && oper.GetAttr(OpAttributeKey::outputCombineAxis, needCombineIOIdx) && needCombineIOIdx[ioIdx])) {
        ALOG_INFO_F("needCombineIOIdx is %s", IntVecToStr(needCombineIOIdx).c_str());
        CombineLastTwoAxis(shape[operandIdx], dim);
        CombineLastTwoAxis(rawShape[operandIdx], dim);
        CombineLastTwoAxis(originShape[operandIdx], dim);
        CombineLastTwoAxis(dynamicValidShape[operandIdx], dim);
    }
}

void CodeGenOp::UpdateShape(
    const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx, bool isInput, size_t ioIdx) {
    ALOG_INFO_F("op code %s, operandIdx: %d, raw shape is %s, originShape is %s, dynamicValidShape is %s",
        oper.GetOpcodeStr().c_str(), operandIdx, IntVecToStr(logicalTensor.tensor->rawshape).c_str(),
        IntVecToStr(logicalTensor.oriShape).c_str(), IntVecToStr(logicalTensor.GetDynValidShape()).c_str());

    rawShape[operandIdx] = logicalTensor.tensor->rawshape;
    // need adapt unaligned scene after
    originShape[operandIdx] = logicalTensor.oriShape;
    if (isDynamicFunction) {
        dynamicValidShape[operandIdx] = logicalTensor.GetDynValidShape();
    }

    ASSERT(logicalTensor.shape.size() <= MAX_DIM) << "only support max dim: " << MAX_DIM;

    Opcode opcode = oper.GetOpcode();
    bool useAttrForGM = IsCopyOpWithShapeOffsetAttr(opcode);
    // Local Tensor shape just use shape from LogicalTensor
    if (!useAttrForGM || logicalTensor.GetMemoryTypeOriginal() != MEM_DEVICE_DDR) {
        shape[operandIdx] = logicalTensor.shape;
        if (isDynamicFunction) { // NEXTNEXT: stack gm should also has dynShape_ later
            ASSERT(!logicalTensor.GetDynValidShape().empty())
                << "LogicalTensor::dynShape_ can not empty in Dynamic Unaligned Scene";
        }
    } else {
        // used for spilling GM scene
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
        ASSERT(attr != nullptr) << ": missing OpAttr in copy op: \n" << oper.Dump();
        shape[operandIdx] = attr->GetSpecifiedShape(1);
        ALOG_INFO_F("attrShape(from op CopyOpAttribute) = %s", IntVecToStr(shape[operandIdx]).c_str());
    }

    CombineAxis(oper, operandIdx, isInput, ioIdx);
}

void CodeGenOp::UpdateOffsetValueForGM(const std::vector<OpImmediate> &offsets, int operandIdx) {
    std::vector<SymbolicScalar> dynOffset(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i].IsSpecified()) {
            auto val = offsets[i].GetSpecifiedValue();
            dynOffset[i] = val;
        }
    }
    offsetGmSymbolic[operandIdx] = dynOffset;
    ALOG_INFO_F("UpdateOffsetValueForGM , offsetGmSymbolic is %s", IntVecToStr(dynOffset).c_str());
}

void CodeGenOp::UpdateOffsetForInput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx) {
    const std::set<Opcode> cubeMDLOpCode = {Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0_AT,
        Opcode::OP_L1_TO_L0_BT, Opcode::OP_L1_TO_BT, Opcode::OP_L1_TO_FIX_QUANT_PRE};
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    bool cubeMDLCondition = cubeMDLOpCode.count(opCode) && (attr != nullptr);
    bool useAttrShapeOffsetForInputGM = OpcodeManager::Inst().IsCopyIn(opCode);
    if (cubeMDLCondition || (useAttrShapeOffsetForInputGM && logicalTensor.GetMemoryTypeOriginal() == MEM_DEVICE_DDR)) {
        // only used for 1. L1 Copy; 2. spilling into gm scene(e.g., ooo spilling); 3. matmul Multi-Data Load scene.
        ALOG_INFO_F("start update offset for GM input");
        ASSERT(attr != nullptr) << ": missing OpAttr in copy in op: \n" << oper.Dump();
        UpdateOffsetValueForGM(attr->GetCopyInAttr().first, operandIdx);
        return;
    }

    offset[operandIdx] = logicalTensor.offset; // Local Tensor offset just use offset from LogicalTensor
    ALOG_INFO_F("UpdateOffsetForInput offset is %s", IntVecToStr(offset[operandIdx]).c_str());
}

void CodeGenOp::UpdateOffsetForOutput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx) {
    bool useAttrShapeOffsetForOutputGM = OpcodeManager::Inst().IsCopyOut(opCode);
    if (!useAttrShapeOffsetForOutputGM || logicalTensor.GetMemoryTypeOriginal() != MEM_DEVICE_DDR) {
        offset[operandIdx] = logicalTensor.offset; // Local Tensor offset just use offset from LogicalTensor
        ALOG_INFO_F("UpdateOffsetForOutput offset is %s", IntVecToStr(offset[operandIdx]).c_str());
        return;
    }

    // only used for 1. L1 Copy; 2. spilling into gm scene(e.g., ooo spilling)
    ALOG_INFO_F("start update offset for GM output");
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    ASSERT(attr != nullptr) << ": missing OpAttr in copy out op: \n" << oper.Dump();
    UpdateOffsetValueForGM(attr->GetCopyOutAttr().second, operandIdx);
}

void CodeGenOp::UpdateScalarValue(const npu::tile_fwk::Operation &ops) {
    if (ops.HasAttr(OpAttributeKey::scalar)) {
        extOperandVal = ops.GetElementAttribute(OpAttributeKey::scalar);
    }
    if (ops.HasAttr(OpAttributeKey::dynScalar)) {
        extSymbolicScalar = ops.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    }
    if (ops.HasAttr(OpAttributeKey::vectorScalar)) {
        extScalarVec = ops.GetVectorElementAttribute(OpAttributeKey::vectorScalar);
    }
}

void CodeGenOp::Init(const npu::tile_fwk::Operation &ops) {
    ASSERT(ops.iOperand.size() + ops.oOperand.size() <= MAX_OPERANDS)
        << "can not support ops.iOperand.size: " << ops.iOperand.size()
        << ", ops.oOperand.size: " << ops.oOperand.size();

    isDynamicFunction = functionType == FunctionType::DYNAMIC_LOOP_PATH;
    isSupportDynamicAligned = config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_ALIGNED);
    ALOG_INFO_F(
        "%s: init CodeGenOp from npu::tile_fwk::Operation, isDynamicFunction is %d, isSupportDynamicAligned is %d",
        __FUNCTION__, isDynamicFunction, isSupportDynamicAligned);

    UpdateTileOpInfo(ops);
    ASSERT(!tileOpName.empty()) << "empty tileOpName for ops: " << ops.Dump();

    // opcode would be refreshed by UpdateTileOpInfo
    isSupportLayout = ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false) &&
                      SUPPORT_TILETENSOR_OPS.find(opCode) != SUPPORT_TILETENSOR_OPS.end();

    opCodeStr = OpcodeManager::Inst().GetOpcodeStr(opCode);

    int operandIdx = 0;
    for (size_t i = 0; i < ops.oOperand.size(); ++i) {
        const auto &output = ops.oOperand[i];
        UpdateCodegenOpInfoByTensor(ops, false, output, operandIdx, i);
    }

    // if no output like WriteRemote OP, set operandIdx=1 for input
    if (operandIdx == 0) {
        operandIdx = 1;
    }

    for (size_t i = 0; i < ops.iOperand.size(); ++i) {
        const auto &input = ops.iOperand[i];
        UpdateCodegenOpInfoByTensor(ops, true, input, operandIdx, i);
    }

    operandCnt = ops.oOperand.size() + ops.iOperand.size();

    GetGmParamIdx(ops);
    syncQueue = ops.syncQueue_;
    UpdateScalarValue(ops);
    UpdateOpAttribute(ops);
}

void CodeGenOp::UpdateCodegenOpInfoByTensor(
    const Operation &ops, bool isInput, const std::shared_ptr<LogicalTensor> &tensor, int &operandIdx, size_t ioIdx) {
    operand[operandIdx] = tensor->GetMemoryTypeOriginal() == MEM_DEVICE_DDR ? tensor->tensor->GetRawMagic() :
                                                                              -tensor->tensor->GetRawMagic();
    operandWithMagic[operandIdx] = tensor->GetMagic();
    UpdateShape(ops, *tensor, operandIdx, isInput, ioIdx);
    if (isInput) {
        UpdateOffsetForInput(ops, *tensor, operandIdx);
    } else {
        UpdateOffsetForOutput(ops, *tensor, operandIdx);
    }
    operandDtype[operandIdx] = tensor->tensor->datatype;
    auto it = OPERAND_TYPE_TO_MEMORY_TYPE.find(tensor->GetMemoryTypeOriginal());
    ASSERT(it != OPERAND_TYPE_TO_MEMORY_TYPE.end())
        << "can not support memory type: " << static_cast<size_t>(tensor->GetMemoryTypeOriginal());
    operandType[operandIdx] = it->second;
    ++operandIdx;
}

void CodeGenOp::UpdateOpAttribute(const npu::tile_fwk::Operation &ops) {
    opAttrs = ops.GetAllAttr();
    isInputForceCombineAxis = ops.HasAttr(OpAttributeKey::inputCombineAxis);

    ConvertAttribute(ops);
}

std::string CodeGenOp::GenOpAttr(bool hasExistingParam) const {
    if (opAttrs.empty()) {
        return {};
    }

    std::vector<std::string> attrList;
    for (const auto &kv : opAttrs) {
        if (kv.first.substr(0, OP_ATTR_PREFIX.size()) != OP_ATTR_PREFIX) {
            continue;
        }
        if (kv.second.Type() == typeid(int64_t)) {
            attrList.push_back(std::to_string(npu::tile_fwk::AnyCast<int64_t>(kv.second)));
        } else if (kv.second.Type() == typeid(bool)) {
            attrList.push_back(std::to_string(npu::tile_fwk::AnyCast<bool>(kv.second)));
        } else if (kv.second.Type() == typeid(std::vector<int64_t>)) {
            auto vec = npu::tile_fwk::AnyCast<std::vector<int64_t>>(kv.second);
            for (auto v : vec) {
                attrList.push_back(std::to_string(v));
            }
        }
    }

    if (attrList.empty()) {
        return {};
    }

    std::string joined = JoinString(attrList, CONN_COMMA);
    return hasExistingParam ? CONN_COMMA + joined : joined;
}

void CodeGenOp::ConvertPoolAttribute(const Operation &operation) {
    auto opc = operation.GetOpcode();
    if (opc != Opcode::OP_MAX_POOL) {
        return;
    }

    std::vector<std::string> intAttrStrList{
        ConvOpAttributeKey::paddingLeft,
        ConvOpAttributeKey::paddingTop,
        ConvOpAttributeKey::paddingRight,
        ConvOpAttributeKey::paddingBottom,
        ConvOpAttributeKey::strideh,
        ConvOpAttributeKey::stridew,
        PoolOpAttributeKey::poolh,
        PoolOpAttributeKey::poolw,
    };
    for (size_t i = 0; i < intAttrStrList.size(); i++) {
        poolParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
    }
}

void CodeGenOp::ConvertAttribute(const Operation &operation) {
    ASSERT(operation.iOperand.size() + operation.oOperand.size() <= MAX_OPERANDS)
        << "can not support operation.iOperand.size: " << operation.iOperand.size()
        << ", operation.oOperand.size: " << operation.oOperand.size();
    if (opCode == Opcode::OP_CONV || opCode == Opcode::OP_CONV_ADD) {
        std::vector<std::string> intAttrStrList{
            ConvOpAttributeKey::cin,
            ConvOpAttributeKey::cout,
            ConvOpAttributeKey::paddingLeft,
            ConvOpAttributeKey::paddingTop,
            ConvOpAttributeKey::paddingRight,
            ConvOpAttributeKey::paddingBottom,
            ConvOpAttributeKey::strideh,
            ConvOpAttributeKey::stridew,
            ConvOpAttributeKey::hposX,
            ConvOpAttributeKey::hsteP,
            ConvOpAttributeKey::wposX,
            ConvOpAttributeKey::wstep,
            ConvOpAttributeKey::hoffsetY,
            ConvOpAttributeKey::woffsetY,
            ConvOpAttributeKey::reluType,
            ConvOpAttributeKey::reluAlpha,
            ConvOpAttributeKey::clearFlag,
            ConvOpAttributeKey::hasAccFlag,
            ConvOpAttributeKey::hasEltFlag,
            ConvOpAttributeKey::hasBiasFlag,
            ConvOpAttributeKey::eltBrcbFlag,
            ConvOpAttributeKey::eltMode,
        };
        // (Cin, Cout, PaddingLeft, PaddingTop, PaddingRight, PaddingBottom, Stride1, Stride2, HPosX, HStep, WPosX,
        // WStep, HOffsetY, WOffsetY, reluType, relu_alpha, clearFlag, has_acc_flag, has_elt_flag, has_bias_flag,
        // elt_brcb_flag, elt_mode, hasQuantPreVector, hasQuantPostVector, hasAntiqVector)
        for (size_t i = 0; i < intAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
        }
        std::vector<std::string> longAttrStrList{
            FixpOpAttributeKey::quantPreScalar,
            FixpOpAttributeKey::quantPostScalar,
            FixpOpAttributeKey::antiqScalar,
        };
        for (size_t i = 0; i < longAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(longAttrStrList[i]));
        }
    }
    if (opCode == Opcode::OP_L1_COPY_IN_FRACTAL_Z) {
        convParams.push_back(operation.GetIntAttribute(ConvOpAttributeKey::fmapC0));
    }

    if (opCode == Opcode::OP_L1_TO_FIX || opCode == Opcode::OP_L1_TO_FIX_RELU_PRE ||
        opCode == Opcode::OP_L1_TO_FIX_RELU_POST || opCode == Opcode::OP_L1_TO_FIX_QUANT_POST ||
        opCode == Opcode::OP_L1_TO_FIX_ELT_ANTIQ || opCode == Opcode::OP_L1_TO_FIX_MTE2_ANTIQ) {
        convParams.push_back(operation.GetIntAttribute(FixpOpAttributeKey::fbAddrSpace));
    }

    ConvertPoolAttribute(operation);
}

void CodeGenOp::UpdateTileOpInfo(const Operation &ops) {
    opCode = ops.GetOpcode();
    tileOpName = GetTileOpName(opCode);

    ALOG_INFO_F(
        "enter tileOpName is %s, opcode = %s", tileOpName.c_str(), OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());

    if (opCode == Opcode::OP_COPY_IN && !ops.oOperand.empty()) {
        npu::tile_fwk::MemoryType memtype = ops.oOperand[0]->GetMemoryTypeOriginal();
        if (memtype == npu::tile_fwk::MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyIn";
            opCode = Opcode::OP_UB_COPY_IN;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyIn";
            opCode = Opcode::OP_L1_COPY_IN;
        }
    } else if (opCode == Opcode::OP_COPY_OUT && !ops.iOperand.empty()) {
        npu::tile_fwk::MemoryType memtype = ops.iOperand[0]->GetMemoryTypeOriginal();
        if (memtype == npu::tile_fwk::MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyOut";
            opCode = Opcode::OP_UB_COPY_OUT;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyOut";
            opCode = Opcode::OP_L1_COPY_OUT;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L0C) {
            tileOpName = "TileOp::L0CCopyOut";
            opCode = Opcode::OP_L0C_COPY_OUT;
        }
    }

    if (!isDynamicFunction || DISTRIBUTED_OPS.count(opCode)) {
        return;
    }

    std::string dynPrefix = "Dyn";
    size_t nameSpaceLen = std::strlen("TileOp::");
    bool isNeedInsertDynPrefix =
        isDynamicFunction && SUPPORT_DYNAMIC_UNALIGNED_OPS.find(opCode) != SUPPORT_DYNAMIC_UNALIGNED_OPS.end();
    ALOG_INFO_F("isNeedInsertDynPrefix is %d, opcode = %s", isNeedInsertDynPrefix,
        OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
    if (isNeedInsertDynPrefix) {
        tileOpName.insert(nameSpaceLen, dynPrefix);
    }

    ALOG_INFO_F("after UpdateTileOpInfo: tileOpName = %s, opCode = %s", tileOpName.c_str(),
        OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
}

void CodeGenOp::GetGmParamIdx(const npu::tile_fwk::Operation &oper) {
    if (!isUnderDynamicFunction || oper.IsNeedStackGM()) {
        auto inParamLocSize = oper.inParamLocation_.size();
        auto outParamLocSize = oper.outParamLocation_.size();

        // Ops like UB_ALLOC have output operands, but does not have output
        // param locs, so here we should not assert 'outParamLocSize == outputTensors.size()' !
        ASSERT(inParamLocSize <= oper.iOperand.size()) << "size of Op.inParamLocation_ is larger than input operands";
        ASSERT(outParamLocSize <= oper.oOperand.size())
            << "size of Op.outParamLocation_ is larger than output operands";

        ALOG_INFO_F("%d: inParamLocation = %s", __FUNCTION__, IntVecToStr(oper.inParamLocation_).c_str());
        ALOG_INFO_F("%d: outParamLocation = %s", __FUNCTION__, IntVecToStr(oper.outParamLocation_).c_str());

        std::copy(oper.outParamLocation_.begin(), oper.outParamLocation_.end(), paramLocation);
        std::copy(oper.inParamLocation_.begin(), oper.inParamLocation_.end(), paramLocation + oper.oOperand.size());
        return;
    }

    if (OpcodeManager::Inst().IsSharedMemory(oper.GetOpcode())) {
        for (size_t i = 0; i < oper.GetOOperands().size(); ++i) {
            if (oper.GetOOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                paramLocation[i] = oper.GetOOpAttrOffset(i);
            }
        }
        size_t iOffset = oper.GetOOperands().size() == 0 ? 1 : oper.GetOOperands().size();
        for (size_t i = 0; i < oper.GetIOperands().size(); ++i) {
            if (oper.GetIOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                paramLocation[i + iOffset] = oper.GetIOpAttrOffset(i);
            }
        }
        return;
    }

    if (oper.GetOpcode() == Opcode::OP_LOAD) {
        paramLocation[0] = oper.GetIOpAttrOffset(0);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        return;
    }

    if (oper.GetOpcode() == Opcode::OP_GATHER_IN_L1 || oper.GetOpcode() == Opcode::OP_GATHER_IN_UB) {
        paramLocation[ID0] = oper.GetIOpAttrOffset(ID0);
        paramLocation[ID1] = oper.GetIOpAttrOffset(ID1);
        paramLocation[ID2] = oper.GetIOpAttrOffset(ID2);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        return;
    }

    if (OpcodeManager::Inst().IsCopyIn(oper.GetOpcode())) {
        const std::shared_ptr<OpAttribute> &attr = oper.GetOpAttribute();
        ASSERT(attr != nullptr) << "Copy In attr is null";
        std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
        paramLocation[1] = oper.GetIOpAttrOffset(0);
        ALOG_INFO_F("Gm Param Index of Copy In Op %s is %d", tileOpName.c_str(), paramLocation[1]);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        ALOG_INFO_F("%s GmTensorParamIdxInCallFunc: %d", __FUNCTION__, GmTensorParamIdxInCallFunc);
        return;
    }

    if (OpcodeManager::Inst().IsCopyOut(oper.GetOpcode())) {
        const std::shared_ptr<OpAttribute> &attr = oper.GetOpAttribute();
        ASSERT(attr != nullptr) << "Copy In attr is null";
        std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
        paramLocation[0] = oper.GetOOpAttrOffset(0);
        ALOG_INFO_F("Gm Param Index of Copy Out Op %s is %d", tileOpName.c_str(), paramLocation[0]);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        ALOG_INFO_F("%s GmTensorParamIdxInCallFunc: %d", __FUNCTION__, GmTensorParamIdxInCallFunc);
        return;
    }
}

std::string CodeGenOp::GenBarrier() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "pipe_barrier(%s);\n", pipeId1.c_str());
    if (ret < 0) {
        ALOG_INFO_F("genBarrier snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOp::GenSyncSetOp() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "set_flag(%s, %s, EVENT_ID%d);\n", pipeId1.c_str(),
        pipeId2.c_str(), syncQueue.eventId_);
    if (ret < 0) {
        ALOG_INFO_F("genSyncSetOp snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOp::GenSyncWaitOp() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "wait_flag(%s, %s, EVENT_ID%d);\n",
        pipeId1.c_str(), pipeId2.c_str(), syncQueue.eventId_);
    if (ret < 0) {
        ALOG_INFO_F("genSyncWaitOp snprintf_s failed %d", ret);
    }
    return buffer;
}

} // namespace npu::tile_fwk
