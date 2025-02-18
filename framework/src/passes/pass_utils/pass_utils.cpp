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
 * \file pass_utils.cpp
 * \brief
 */

#include "pass_utils.h"
#include <climits>
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

void FunctionUtils::RelinkOperationInput(Operation *op, const size_t inputIndex, const Operation *targetOp,
                                         const size_t outputIndex) {
    if (op == nullptr || targetOp == nullptr) {
        return;
    }
    if (inputIndex >= op->GetIOperands().size() || outputIndex >= targetOp->GetOOperands().size()) {
        return;
    }
    LogicalTensorPtr inputTenosr = op->GetIOperands().at(inputIndex);
    LogicalTensorPtr targetOutputTenosr = targetOp->GetOOperands().at(outputIndex);
    // update consumer of operation
    inputTenosr->RemoveConsumer(*op);
    targetOutputTenosr->AddConsumer(*op);
    // replace input tensor of op with the output tensor of target op
    op->ReplaceInputOperand(inputTenosr, targetOutputTenosr);
}

bool IsOverlapping(const LogicalTensor& a, const LogicalTensor& b) {
    for (size_t i = 0; i < a.shape.size(); ++i) {
        int aStart = a.offset[i];
        int aEnd = aStart + a.shape[i];
        int bStart = b.offset[i];
        int bEnd = bStart + b.shape[i];

        // 如果任意一维不重叠，则整体不重叠
        if (aEnd <= bStart || aStart >= bEnd) {
            return false;
        }
    }
    return true;
}

// 计算矩形的体积（面积、体积等）
int CalculateVolume(const LogicalTensor& tensor) {
    int volume = 1;
    for (int dim : tensor.shape) {
        volume *= dim;
    }
    return volume;
}

// 判断一组 LogicalTensor 是否可以拼接成一个大矩形
bool FunctionUtils::IsContinuous(const std::vector<std::shared_ptr<LogicalTensor>> &tensors) {
    if (tensors.empty()) {
        return false;
    }

    size_t numDims = tensors[0]->shape.size();

    // 计算整体边界
    std::vector<int64_t> minCoords(numDims, INT_MAX);
    std::vector<int64_t> maxCoords(numDims, INT_MIN);

    for (const auto& tensor : tensors) {
        for (size_t i = 0; i < numDims; ++i) {
            minCoords[i] = std::min(minCoords[i], tensor->offset[i]);
            maxCoords[i] = std::max(maxCoords[i], tensor->offset[i] + tensor->shape[i]);
        }
    }

    // 计算整体体积
    int totalVolume = 1;
    for (size_t i = 0; i < numDims; ++i) {
        totalVolume *= (maxCoords[i] - minCoords[i]);
    }

    // 计算所有矩形的总体积
    int sumVolume = 0;
    for (const auto& tensor : tensors) {
        sumVolume += CalculateVolume(*tensor);
    }

    // 如果总体积不等于整体体积，说明有缝隙
    if (sumVolume != totalVolume) {
        return false;
    }

    // 检查是否有重叠
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            if (IsOverlapping(*tensors[i], *tensors[j])) {
                return false;
            }
        }
    }

    return true;
}

namespace {
const size_t POS_TWO = 2;
const int SPACE_NUM_16 = 16;
} // namespace
void SubfuncInvokeInfoTy::ConstructActualInvokeParam(int esgId) {
    if (!isFinalized_) {
        ALOG_ERROR("Error: does not finalized before constructing InvokeParam");
        return;
    }

    int paramLoc = 0;
    for (auto &tensorArg : tensorArgs_) {
        ALOG_DEBUG_F("######## Construct TA for %d ########", esgId);
        tensorParamList_.emplace_back(paramLoc, tensorArg.realDDRId, tensorArg.offset, tensorArg.shape,
            tensorArg.rawShape, tensorArg.dType, tensorArg.isOutputToGM, tensorArg.tensor, tensorArg.opMagic,
            tensorArg.operandIdx);
        paramLoc++;
    }

    int iParamLoc = 0 | 0x10000000;
    for (auto &conn : connections_) {
        InCastInfoTy &inCastInfo = std::get<2>(conn);
        // Note: here incast tensors are not combined to a huge one, offset are
        // always zero. If we do incast tensor combine later, offset need update
        // accordinarly.
        incastTensorParamList_.emplace_back(
            IncastParamPackTy{iParamLoc, inCastInfo.realIncastDDRId, inCastInfo.offset, inCastInfo.shape,
                              inCastInfo.rawShape, inCastInfo.dType, inCastInfo.tensor, inCastInfo.opMagic,
                              inCastInfo.operandIdx});
        iParamLoc++;
    }

    int oParamLoc = 0 | 0x20000000;
    for (auto &outCast : outCasts_) {
        outcastTensorParamList_.emplace_back(oParamLoc, outCast.realOutCastDDRId, outCast.refCount, outCast.shape,
            outCast.rawShape, outCast.offset, outCast.dType, outCast.tensor, outCast.opMagic, outCast.operandIdx);

        oParamLoc++;
    }
}

void SubfuncInvokeInfoTy::PrintInvokeInfo(const std::string &extraInfo) const {
    std::stringstream ss;
    ss << extraInfo;
    ss << "(";
    ss << "Tensors[";
    for (auto &tensorParam : tensorParamList_) {
        tensorParam.Print(ss);
        ss << ", ";
    }
    ss << "] Incast[";
    for (auto &incastTensorParam : incastTensorParamList_) {
        incastTensorParam.Print(ss);
        ss << ", ";
    }
    ss << "] OCast[";
    for (auto &outcastTensorParam : outcastTensorParamList_) {
        outcastTensorParam.Print(ss);
        ss << ", ";
    }
    ss << "])\n";
    ALOG_DEBUG_F("%s", ss.str().c_str());
}

void SubfuncInvokeInfoTy::PrettyPrintInvokeInfo(const int subgraphId) const {
    std::stringstream ss;
    ss << "INVOKE[" << subgraphId << "]" << std::endl;
    ss << "|--CALL SUB_GRAPH[" << programSubgraphId_ << "]" << std::endl;
    for (auto &tensorParam : tensorParamList_) {
        ss << "|--TENSOR";
        tensorParam.Print(ss);
        ss << std::endl;
    }
    for (auto &incastTensorParam : incastTensorParamList_) {
        ss << "|--INCAST";
        incastTensorParam.Print(ss);
        ss << std::endl;
    }
    for (auto &outcastTensorParam : outcastTensorParamList_) {
        ss << "|--OUTCAST";
        outcastTensorParam.Print(ss);
        ss << std::endl;
    }
    ss << std::endl;
    ALOG_DEBUG_F("%s", ss.str().c_str());
}

void SubfuncInvokeInfoTy::DumpInvokeInfo(int64_t invokeParamMemOffset, int64_t *invokeParamPtr) const {
    std::vector<int64_t> invokeParam;
    invokeParam.emplace_back(static_cast<int64_t>(programSubgraphId_));
    invokeParam.emplace_back(static_cast<int64_t>(tensorParamList_.size()));
    invokeParam.emplace_back(static_cast<int64_t>(incastTensorParamList_.size()));
    invokeParam.emplace_back(static_cast<int64_t>(outcastTensorParamList_.size()));

    for (auto &tensorParam : tensorParamList_) {
        tensorParam.DumpTensor(invokeParam);
    }

    for (auto &incastTensorParam : incastTensorParamList_) {
        incastTensorParam.DumpIncastInfo(invokeParam);
    }

    for (auto &outcastTensorParam : outcastTensorParamList_) {
        outcastTensorParam.DumpOutcastInfo(invokeParam);
    }

    (void)memcpy_s(invokeParamPtr + invokeParamMemOffset / sizeof(int64_t), invokeParam.size() * sizeof(int64_t),
                   invokeParam.data(), invokeParam.size() * sizeof(int64_t));
}

std::tuple<int, int, int> SubfuncInvokeInfoTy::LookupInvokeArgs(const int paramLoc) const {
    switch (paramLoc >> ParamLocOffset) {
        case ParamLocTensor:
            for (auto &tensorParam : tensorParamList_) {
                if (tensorParam.paramLoc == paramLoc) {
                    return std::tuple<int, int, int>{tensorParam.ddrId, tensorParam.offset[0],
                                                     tensorParam.offset[1]};
                }
            }
            assert(0 && "not found param");
            return std::tuple<int, int, int>{0, 0, 0};

        case ParamLocIncast:
            for (auto &incastTensorParam : incastTensorParamList_) {
                if (incastTensorParam.paramLoc == paramLoc) {
                    return std::tuple<int, int, int>{incastTensorParam.ddrId, incastTensorParam.offset[0],
                                                     incastTensorParam.offset[1]};
                }
            }
            assert(0 && "not found param");
            return std::tuple<int, int, int>{0, 0, 0};

        case ParamLocOutcast:
            for (auto &outcastTensorParam : outcastTensorParamList_) {
                if (outcastTensorParam.paramLoc == paramLoc) {
                    return std::tuple<int, int, int>{outcastTensorParam.ddrId, outcastTensorParam.offset[0],
                                                     outcastTensorParam.offset[1]};
                }
            }
            assert(0 && "not found param");
            return std::tuple<int, int, int>{0, 0, 0};
        default: assert("Invalid parameter location"); return std::tuple<int, int, int>{0, 0, 0};
    }
}

void SubfuncInvokeInfoTy::DoFinishRecord() {
    isFinalized_ = true;
}

void SubfuncInvokeInfoTy::Print(const std::string &extInfo) const {
    std::stringstream ss;
    ss << "-- SubgraphInvokeInfo: " << extInfo << "\n";

    ss << "---- Tensors: \n";
    for (auto &tensorArg : tensorArgs_) {
        ss << "Op:" << tensorArg.operandIdx << ", $" << tensorArg.realDDRId;
        ss << "\n";
    }

    auto printIncast = [](std::ostream &osm, const ExeSubgraphEdgeTy &conn) {
        osm << "SrcESgId: " << std::get<0>(conn) << ", ";
        osm << "DstESgId: " << std::get<1>(conn) << ", ";
        const InCastInfoTy &icInfo = std::get<2>(conn);
        osm << "DstOprn: " << icInfo.operandIdx << ",";
        int ddrId = icInfo.realIncastDDRId;
        osm << "ddrId: " << (ddrId != -1 ? ("$" + std::to_string(ddrId)) : "NOT_CONNECTED") << "\n";
    };

    ss << "---- Incast: \n";
    for (auto &conn : connections_) {
        printIncast(ss, conn);
    }

    ss << "---- OutCast: \n";
    for (auto &outCast : outCasts_) {
        ss << "SrcESgId: " << outCast.srcESgId << ", ";
        ss << "RefCount: " << outCast.refCount << ", ";
        int ddrId = outCast.realOutCastDDRId;
        ss << "ddrId: " << (ddrId != -1 ? ("$" + std::to_string(ddrId)) : "NOT_CONNECTED");
        auto printLeadingSpace = [](std::ostream &osm, const int numSpace) {
            for (int i = 0; i < numSpace; i++) {
                osm << " ";
            }
        };
        ss << "[\n";
        for (auto &succIncast : outCast.successorIncastInfo) {
            printLeadingSpace(ss, SPACE_NUM_16);
            if (succIncast.successorIncast) {
                printIncast(ss, *succIncast.successorIncast);
            }
            ss << "\n";
        }
        ss << "]\n";
    }

    ss << "\n\n";
    ALOG_DEBUG_F("%s", ss.str().c_str());
}

Json SubfuncInvokeInfoTy::DumpJson() const {
    Json ret;
    // json中只需要记录magic即可，可以从对应的root graph的"tensors"中获取具体的tensor信息
    ret["incast_params"] = Json::array();
    auto &incastArray = ret["incast_params"];
    for (const auto &incast : incastTensorParamList_) {
        Json incastJson;
        incastJson["param_loc"] = incast.paramLoc;
        incastJson["tensor"] = incast.tensor->magic;
        incastJson["offset"] = incast.offset;
        incastJson["shape"] = incast.shape;
        incastJson["op_magic"] = incast.opMagic;
        incastJson["operandIdx"] = incast.operandIdx;
        incastArray.emplace_back(incastJson);
    }

    ret["outcast_params"] = Json::array();
    auto &outcastArray = ret["outcast_params"];
    for (const auto &outcast : outcastTensorParamList_) {
        Json outcastJson;
        outcastJson["param_loc"] = outcast.paramLoc;
        outcastJson["ref_count"] = outcast.refCount;
        outcastJson["tensor"] = outcast.tensor->magic;
        outcastJson["offset"] = outcast.offset;
        outcastJson["shape"] = outcast.shape;
        outcastJson["op_magic"] = outcast.opMagic;
        outcastJson["operandIdx"] = outcast.operandIdx;
        outcastArray.emplace_back(outcastJson);
    }

    ret["tensor_params"] = Json::array();
    auto &tensorArray = ret["tensor_params"];
    for (const auto &tensor : tensorParamList_) {
        Json tensorJson;
        tensorJson["param_loc"] = tensor.paramLoc;
        tensorJson["tensor"] = tensor.tensor->magic;
        tensorJson["offset"] = tensor.offset;
        tensorJson["shape"] = tensor.shape;
        tensorJson["op_magic"] = tensor.opMagic;
        tensorJson["is_output"] = tensor.isOutputToGM;
        tensorJson["operandIdx"] = tensor.operandIdx;
        tensorArray.emplace_back(tensorJson);
    }
    ret["program_id"] = programSubgraphId_;
    ret["graph_type"] = static_cast<int>(graphType_);
    return ret;
}

void SubfuncInvokeInfoTy::LoadIncastFromJson(const Json& incastJson, Function* belongTo) {
    int paramLoc = incastJson["param_loc"].get<int>();
    int opMagic = incastJson["op_magic"].get<int>();
    int operandIdx = incastJson["operandIdx"].get<int>();
    std::shared_ptr<LogicalTensor> tensorPtr = belongTo->GetTensorMap().GetTensorByMagic(
        incastJson["tensor"].get<int>());
    if (tensorPtr == nullptr) {
        ALOG_ERROR_F("Tile FWK for incast %d op %d is nullptr, function type %d name %s",
            incastJson["tensor"].get<int>(), opMagic, belongTo->GetFunctionType(),
            belongTo->GetMagicName().c_str());
        return;
    }
    incastTensorParamList_.emplace_back(IncastParamPackTy(paramLoc, tensorPtr->GetRawMagic(),
        incastJson["offset"].get<std::vector<int64_t>>(), incastJson["shape"].get<std::vector<int64_t>>(),
        tensorPtr->tensor->rawshape, tensorPtr->tensor->GetDataType(), tensorPtr, opMagic, operandIdx));
}

void SubfuncInvokeInfoTy::LoadOutcastFromJson(const Json& outcastJson, Function* belongTo) {
    int paramLoc = outcastJson["param_loc"].get<int>();
    int refCount = outcastJson["ref_count"].get<int>();
    int opMagic = outcastJson["op_magic"].get<int>();
    int operandIdx = outcastJson["operandIdx"].get<int>();
    std::shared_ptr<LogicalTensor> tensorPtr = belongTo->GetTensorMap().GetTensorByMagic(
        outcastJson["tensor"].get<int>());
    if (tensorPtr == nullptr) {
        ALOG_ERROR_F("Tile FWK for outcast %d op %d is nullptr function type %d name %s",
            outcastJson["tensor"].get<int>(), opMagic, belongTo->GetFunctionType(),
            belongTo->GetMagicName().c_str());
        return;
    }
    outcastTensorParamList_.emplace_back(OutcastParamPackTy(paramLoc, tensorPtr->GetRawMagic(), refCount,
        outcastJson["shape"].get<std::vector<int64_t>>(), tensorPtr->tensor->rawshape,
        outcastJson["offset"].get<std::vector<int64_t>>(), tensorPtr->tensor->GetDataType(), tensorPtr, opMagic,
        operandIdx));
}

void SubfuncInvokeInfoTy::LoadTensorFromJson(const Json& tensorJson, Function* belongTo) {
    int paramLoc = tensorJson["param_loc"].get<int>();
    int opMagic = tensorJson["op_magic"].get<int>();
    int operandIdx = tensorJson["operandIdx"].get<int>();
    std::shared_ptr<LogicalTensor> tensorPtr = belongTo->GetTensorMap().GetTensorByMagic(
        tensorJson["tensor"].get<int>());
    if (tensorPtr == nullptr) {
        ALOG_ERROR_F("Tile FWK for tensor %d op %d is nullptr, function type %d name %s",
            tensorJson["tensor"].get<int>(), opMagic, belongTo->GetFunctionType(),
            belongTo->GetMagicName().c_str());
        return;
    }
    bool isOutput = tensorJson["is_output"].get<bool>();
    tensorParamList_.emplace_back(TensorParamPackTy(paramLoc, tensorPtr->GetRawMagic(),
        tensorJson["offset"].get<std::vector<int64_t>>(), tensorJson["shape"].get<std::vector<int64_t>>(),
        tensorPtr->tensor->rawshape, tensorPtr->tensor->GetDataType(), isOutput, tensorPtr, opMagic, operandIdx));
}

void SubfuncInvokeInfoTy::LoadJson(const Json &invokeInfoJson, Function *belongTo) {
    for (const Json &incastJson : invokeInfoJson["incast_params"]) {
        LoadIncastFromJson(incastJson, belongTo);
    }

    for (const Json &outcastJson : invokeInfoJson["outcast_params"]) {
        LoadOutcastFromJson(outcastJson, belongTo);
    }

    for (const Json &tensorJson : invokeInfoJson["tensor_params"]) {
        LoadTensorFromJson(tensorJson, belongTo);
    }

    programSubgraphId_ = invokeInfoJson["program_id"].get<int>();
    graphType_ = static_cast<CoreType>(invokeInfoJson["graph_type"].get<int>());
}

Json SubfuncInvokeInfoTy::ToJson() const {
    Json j, jins, jouts, jtensors;

    for (const auto &conn : connections_) {
        Json jdata;
        auto &incast = std::get<2>(conn);
        jdata["operandIdx"] = incast.operandIdx;
        jdata["ddrId"] = incast.realIncastDDRId;
        jdata["shape"] = incast.shape;
        jdata["offset"] = incast.offset;
        jdata["dtype"] = incast.dType;
        jins.push_back(jdata);
    }

    for (const auto &outcast : outCasts_) {
        Json jdata;
        jdata["operandIdx"] = outcast.operandIdx;
        jdata["ddrId"] = outcast.realOutCastDDRId;
        jdata["shape"] = outcast.shape;
        jdata["offset"] = outcast.offset;
        jdata["dtype"] = outcast.dType;
        std::vector<int> esgIds;
        for (auto &succEsg : outcast.successorIncastInfo) {
            esgIds.push_back(succEsg.successorESgId);
        }
        jdata["succEsgIds"] = esgIds;
        jouts.push_back(jdata);
    }

    for (const auto &tensor : tensorArgs_) {
        Json jdata;
        jdata["operandIdx"] = tensor.operandIdx;
        jdata["ddrId"] = tensor.realDDRId;
        jdata["shape"] = tensor.shape;
        jdata["offset"] = tensor.offset;
        jdata["dtype"] = tensor.dType;
        jdata["isOutput"] = tensor.isOutputToGM;
        jtensors.push_back(jdata);
    }

    j["incasts"] = jins;
    j["outcasts"] = jouts;
    j["tensors"] = jtensors;
    return j;
}

bool SubfuncInvokeInfoTy::operator==(const SubfuncInvokeInfoTy &other) const
{
    auto &thisTensorList = tensorParamList_;
    auto &otherTensorList = other.GetTensorParamList();
    if (thisTensorList.size() != otherTensorList.size()) {
        return false;
    }
    for (size_t i = 0; i < thisTensorList.size(); i++) {
        if (thisTensorList[i] != otherTensorList[i]) {
            return false;
        }
    }

    auto &thisIncastList = incastTensorParamList_;
    auto &otherIncastList = other.GetIncastTensorParamList();
    if (thisIncastList.size() != otherIncastList.size()) {
        return false;
    }
    for (size_t i = 0; i < thisIncastList.size(); i++) {
        if (thisIncastList[i] != otherIncastList[i]) {
            return false;
        }
    }

    auto &thisOutcastList = outcastTensorParamList_;
    auto &otherOutcastList = other.GetOutcastTensorParamList();
    if (thisOutcastList.size() != otherOutcastList.size()) {
        return false;
    }
    for (size_t i = 0; i < thisOutcastList.size(); i++) {
        if (thisOutcastList[i] != otherOutcastList[i]) {
            return false;
        }
    }
    return true;
}

bool SubfuncInvokeInfoTy::operator!=(const SubfuncInvokeInfoTy &other) const
{
    if (*this == other) {
        return false;
    }
    return true;
}

void SubfuncInvokeInfoTy::TensorParamPackTy::Print(std::ostream &osm) const {
    osm << IntVecToStr(offset);
    osm << IntVecToStr(shape);
    osm << IntVecToStr(rawShape);
    osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
}

void SubfuncInvokeInfoTy::TensorParamPackTy::DumpTensor(std::vector<int64_t> &invokeParam) const {
    invokeParam.emplace_back(static_cast<int64_t>(ddrId));
}

bool SubfuncInvokeInfoTy::TensorParamPackTy::operator==(const TensorParamPackTy &other) const {
    if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
        rawShape != other.rawShape || dType != other.dType || isOutputToGM != other.isOutputToGM ||
        tensor->GetMagic() != other.tensor->GetMagic() ||
        tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
        return false;
    }
    return true;
}

bool SubfuncInvokeInfoTy::TensorParamPackTy::operator!=(const TensorParamPackTy &other) const {
    return !(*this == other);
}

void SubfuncInvokeInfoTy::IncastParamPackTy::Print(std::ostream &osm) const {
    osm << IntVecToStr(offset);
    osm << IntVecToStr(shape);
    osm << IntVecToStr(rawShape);
    osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
}

void SubfuncInvokeInfoTy::IncastParamPackTy::DumpIncastInfo(std::vector<int64_t> &invokeParam) const {
    invokeParam.emplace_back(static_cast<int64_t>(ddrId));
}

bool SubfuncInvokeInfoTy::IncastParamPackTy::operator==(const IncastParamPackTy &other) const {
    if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
        rawShape != other.rawShape || dType != other.dType || tensor->GetMagic() != other.tensor->GetMagic() ||
        tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
        return false;
    }
    return true;
}

bool SubfuncInvokeInfoTy::IncastParamPackTy::operator!=(const IncastParamPackTy &other) const {
    return !(*this == other);
}

void SubfuncInvokeInfoTy::OutcastParamPackTy::Print(std::ostream &osm) const {
    osm << "[RC:" << refCount << "]";
    osm << IntVecToStr(offset);
    osm << IntVecToStr(shape);
    osm << IntVecToStr(rawShape);
    osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
}

void SubfuncInvokeInfoTy::OutcastParamPackTy::DumpOutcastInfo(std::vector<int64_t> &invokeParam) const {
    invokeParam.emplace_back(static_cast<int64_t>(ddrId));
}

bool SubfuncInvokeInfoTy::OutcastParamPackTy::operator==(const OutcastParamPackTy &other) const {
    if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
        rawShape != other.rawShape || dType != other.dType || tensor->GetMagic() != other.tensor->GetMagic() ||
        tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
        return false;
    }
    return true;
}

bool SubfuncInvokeInfoTy::OutcastParamPackTy::operator!=(const OutcastParamPackTy &other) const {
    return !(*this == other);
}
Json SubfuncParam::ToJson() const {
    Json j, jins, jouts, jtensors;
    for (auto &incast : inCastArgs_) {
        Json jdata;
        jdata["operandIdx"] = incast.operandIdx;
        jdata["ddrId"] = incast.symDDRId;
        jdata["shape"] = incast.shape;
        jdata["offset"] = incast.offset;
        jdata["name"] = incast.symName;
        jdata["symbol"] = incast.symbol;
        jdata["loc"] = incast.paramLoc;
        jdata["data_type"] = static_cast<int>(incast.dataType);
        jins.push_back(jdata);
    }

    for (auto &outcast : outCastArgs_) {
        Json jdata;
        jdata["operandIdx"] = outcast.operandIdx;
        jdata["ddrId"] = outcast.symDDRId;
        jdata["shape"] = outcast.shape;
        jdata["offset"] = outcast.offset;
        jdata["name"] = outcast.symName;
        jdata["symbol"] = outcast.symbol;
        jdata["loc"] = outcast.paramLoc;
        jdata["data_type"] = static_cast<int>(outcast.dataType);
        jouts.push_back(jdata);
    }

    for (auto &tensor : tensorsArgs_) {
        Json jdata;
        jdata["operandIdx"] = tensor.operandIdx;
        jdata["ddrId"] = tensor.symDDRId;
        jdata["shape"] = tensor.shape;
        jdata["offset"] = tensor.symOffset;
        jdata["name"] = tensor.symName;
        jdata["symbol"] = tensor.symbol;
        jdata["loc"] = tensor.paramLoc;
        jdata["data_type"] = static_cast<int>(tensor.dataType);
        jtensors.push_back(jdata);
    }

    j["incasts"] = jins;
    j["outcasts"] = jouts;
    j["tensors"] = jtensors;
    return j;
}

void SubfuncParam::FromJson(const Json& params) {
    inCastArgs_.clear();
    tensorsArgs_.clear();
    outCastArgs_.clear();
    for (auto &ele : params["incasts"]) {
        AppendIncastParam(ele["operandIdx"].get<int>(), ele["ddrId"].get<int>(),
            ele["shape"].get<std::vector<int64_t>>(), ele["offset"].get<std::vector<int64_t>>(),
            ele["name"].get<std::string>(), ele["loc"].get<int>(), ele["symbol"].get<std::string>(),
            static_cast<DataType>(ele["data_type"].get<int>()));
    }

    for (auto &ele : params["outcasts"]) {
        AppendOutcastParam(ele["operandIdx"].get<int>(), ele["ddrId"].get<int>(), 0,
            ele["shape"].get<std::vector<int64_t>>(), ele["offset"].get<std::vector<int64_t>>(),
            ele["name"].get<std::string>(), ele["loc"].get<int>(), ele["symbol"].get<std::string>(),
            static_cast<DataType>(ele["data_type"].get<int>()));
    }

    for (auto &ele : params["tensors"]) {
        AppendTensorParam(ele["operandIdx"].get<int>(), ele["ddrId"].get<int>(),
            ele["shape"].get<std::vector<int64_t>>(), ele["offset"].get<std::vector<int64_t>>(),
            ele["name"].get<std::string>(), ele["loc"].get<int>(), ele["symbol"].get<std::string>(),
            static_cast<DataType>(ele["data_type"].get<int>()));
    }
}

void SubfuncParam::PrettyPrint(const int psgId, std::ostream &osm) const {
    osm << "PARAM_LIST[" << psgId << "]:\n";
    for (auto &tensor : tensorsArgs_) {
        osm << "|--";
        tensor.Print(osm);
    }

    for (auto &ins : inCastArgs_) {
        osm << "|--";
        ins.Print(osm);
    }

    for (auto &outs : outCastArgs_) {
        osm << "|--";
        outs.Print(osm);
    }
}

void SubfuncParam::InCastParamTy::Print(std::ostream &osm) const {
    osm << "INCAST";
    osm << IntVecToStr(offset);
    osm << IntVecToStr(shape);
    osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]\n";
}

bool SubfuncParam::InCastParamTy::CompareParam(const SubfuncInvokeInfoTy::IncastParamPackTy &esgParam) const {
    return (paramLoc == esgParam.paramLoc) && (shape == esgParam.shape) &&
           (dataType == esgParam.dType);
}

void SubfuncParam::OutCastParamTy::Print(std::ostream &osm) const {
    osm << "OUTCAST";
    osm << "[" << refCount << "]";
    osm << IntVecToStr(offset);
    osm << IntVecToStr(shape);
    osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]" << std::endl;
}

bool SubfuncParam::OutCastParamTy::CompareParam(const SubfuncInvokeInfoTy::OutcastParamPackTy &esgParam) const {
    return (paramLoc == esgParam.paramLoc) && (refCount == esgParam.refCount) &&
           (shape == esgParam.shape) && (dataType == esgParam.dType);
}

void SubfuncParam::TensorParamTy::Print(std::ostream &osm) const {
    osm << IntVecToStr(symOffset);
    osm << IntVecToStr(shape);
    osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]" << std::endl;
}

bool SubfuncParam::TensorParamTy::CompareParam(const SubfuncInvokeInfoTy::TensorParamPackTy &esgParam) const {
    return (paramLoc == esgParam.paramLoc) &&
        (shape == esgParam.shape) && (dataType == esgParam.dType);
}
namespace {
const int32_t MAGIC_NUM_TWO = 2;
}

void SubfuncTopologyInfoTy::AddEntry(const int esgId, const int readState, const setType &succ) {
    Entry entry;
    entry.esgId = esgId;
    entry.readyState = readState;
    entry.outGraph = succ;
    topology_.push_back(entry);
    if (readState == 0) {
        readyIds_.emplace_back(esgId);
    }
}

void SubfuncTopologyInfoTy::UpdateEntry(const uint32_t extType, const uint32_t extParamNum, const std::vector<int64_t> &extParams) {
    auto &entry = topology_.back();
    entry.extType = extType;
    entry.extParamNum = extParamNum;
    entry.extParams = extParams;
}

std::vector<int> SubfuncTopologyInfoTy::TopoSort() {
    std::vector<int> res;
    res.reserve(topology_.size());

    // make a copy
    std::vector<Entry> tmpTopo = topology_;
    std::vector<int> ready;
    for (auto &entry : tmpTopo) {
        if (entry.readyState == 0) {
            res.push_back(entry.esgId);
            ready.push_back(entry.esgId);
        }
    }

    while (!ready.empty()) {
        auto iter = ready.begin();
        int index = *iter;
        ready.erase(iter);
        for (auto succId : tmpTopo[index].outGraph) {
            tmpTopo[succId].readyState++;
            if (tmpTopo[succId].readyState == 0) {
                ready.push_back(succId);
                res.push_back(succId);
            }
        }
    }
    return res;
}

void SubfuncTopologyInfoTy::Print(std::ostream &osm) const {
    osm << "-- SrcESgId -- Ready? -- [OutGraphIds .......] \n";
    for (auto &entry : topology_) {
        char bufStr[32] = {'\0'};
        sprintf_s(bufStr, sizeof(bufStr), "-- %6d   %6d     [", entry.esgId, entry.readyState);
        osm << bufStr;

        auto iter = entry.outGraph.begin();
        for (int i = 0; i < maxM_; i++) {
            int og = -1;
            if (iter != entry.outGraph.end()) {
                og = *iter;
                iter++;
            }

            char buf[8] = {'\0'};
            sprintf_s(buf, sizeof(buf), "%4d, ", og);
            osm << buf;
        }
        osm << "]\n";
    }
}

void SubfuncTopologyInfoTy::DumpEachEntryInfo(int esgId, CoreType coreType, int64_t entryOffset,
    int64_t *entryParamPtr, int32_t *readyStatePtr) const { // dump each entry
    std::vector<int64_t> entryParam;
    entryParam.clear();

    // high 32 bit of the graphID is coretype
    entryParam.emplace_back((static_cast<uint64_t>(coreType) << 32) | static_cast<int64_t>(esgId));
    entryParam.emplace_back(static_cast<int64_t>(topology_[esgId].readyState));
    entryParam.emplace_back(static_cast<int64_t>(topology_[esgId].outGraph.size()));
    for (auto &num : topology_[esgId].outGraph) {
        entryParam.emplace_back(static_cast<int64_t>(num));
    }
    (void)memcpy_s(entryParamPtr + entryOffset / sizeof(int64_t), entryParam.size() * sizeof(int64_t),
                   entryParam.data(), entryParam.size() * sizeof(int64_t));
    *(readyStatePtr + static_cast<int32_t>(esgId) * MAGIC_NUM_TWO) = static_cast<int32_t>(topology_[esgId].readyState);
    *(readyStatePtr + static_cast<int32_t>(esgId) * MAGIC_NUM_TWO + 1) = static_cast<int32_t>(coreType);
}

bool SubfuncTopologyInfoTy::IsEsgReady(const int esgId) const {
    return topology_[esgId].readyState == 0;
}

std::vector<int> SubfuncTopologyInfoTy::GetSuccs(int esgId) const {
    std::vector<int> succs;
    for (auto &entry : topology_) {
        if (esgId == entry.esgId) {
            succs.insert(succs.end(), entry.outGraph.begin(), entry.outGraph.end());
            break;
        }
    }
    return succs;
}

Json SubfuncTopologyInfoTy::DumpJson() const
{
    Json ret;
    ret["entrys"] = Json::array();
    auto &entrys = ret["entrys"];
    for (auto &entry : topology_) {
        Json entryJson;
        entryJson["esg_id"] = entry.esgId;
        entryJson["ready_state"] = entry.readyState;
        Json outGraphJson = Json::array();
        for (auto out : entry.outGraph) {
            outGraphJson.emplace_back(out);
        }

        entryJson["out_graph"] = outGraphJson;
        entryJson["ext_params"] = entry.extParams;
        entryJson["ext_type"] = entry.extType;
        entryJson["ext_param_num"] = entry.extParamNum;
        entrys.emplace_back(entryJson);
    }
    return ret;
}

void SubfuncTopologyInfoTy::LoadJson(const Json &topoJson)
{
    topology_.clear();
    readyIds_.clear();
    for (auto &ele : topoJson["entrys"]) {
        setType outGraph;
        for (auto out : ele["out_graph"]) {
            outGraph.emplace(out);
        }
        AddEntry(ele["esg_id"], ele["ready_state"], outGraph);
        UpdateEntry(ele["ext_type"], ele["ext_param_num"], ele["ext_params"].get<std::vector<int64_t>>());
    }
}
} // namespace npu::tile_fwk
