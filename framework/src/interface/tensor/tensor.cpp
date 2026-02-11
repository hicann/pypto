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
 * \file tensor.cpp
 * \brief
 */

#include "tilefwk/tensor.h"
#include "logical_tensor.h"
#include "raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"
#include "tilefwk/error.h"
#include "interface/utils/id_gen.h"


using namespace npu::tile_fwk;

Tensor::Tensor() : storage_(nullptr), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    Program::GetInstance().InsertAliveTensor(this);
}

Tensor::~Tensor() {
    Program::GetInstance().GetTensorSlotManager()->TensorDestruct(*this);

    Program::GetInstance().EraseAliveTensor(this);
    if (storage_ == nullptr) {
        return;
    }
    ASSERT(storage_->tensor != nullptr);
    storage_->tensor->AddRefCount(-1);
}

Tensor::Tensor(std::shared_ptr<LogicalTensor> s) : storage_(std::move(s)), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    ASSERT(storage_ != nullptr && storage_->tensor != nullptr);
    Program::GetInstance().InsertAliveTensor(this);
    storage_->tensor->AddRefCount(1);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
}

static std::vector<SymbolicScalar> ToDynShape(const std::string &tname, const Shape &shape) {
    auto dynShape = SymbolicScalar::FromConcrete(shape);
    for (size_t dim = 0; dim < shape.size(); dim++) {
        CHECK(shape[dim] >= -1) << "Invalid shape " << shape[dim];
        if (shape[dim] == -1) {
            auto name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputShapeDim);
            auto handler = SymbolicScalar(AddRuntimePrefix(name));
            auto input = SymbolicScalar(AddArgPrefix(tname));
            dynShape[dim] = handler(input, dim);
        }
    }
    return dynShape;
}

template <typename T>
void CheckShapeValid(DataType &dataType, T &shape, TileOpFormat &format) {
    if (format == TileOpFormat::TILEOP_NZ && shape.back() != -1) {
        ASSERT(shape.back() * BytesOf(dataType) % ALIGN_SIZE_32 == 0)
            << "Current inner axis: " << shape.back() << ", when input "
            << "is NZ format, inner axis shape must be 32-byte aligned\n";
    }
}

Tensor::Tensor(DataType dataType, const Shape &shape, std::string name, TileOpFormat format)
    : index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    CheckShapeValid(dataType, shape, format);
    auto dynShape = ToDynShape(name, shape);
    storage_ = std::make_shared<LogicalTensor>(
        *Program::GetInstance().GetCurrentFunction(), dataType, shape, dynShape, format, name, NodeType::LOCAL);
    storage_->tensor->AddRefCount(1);
    Program::GetInstance().GetTensorSlotManager()->TensorConstruct(*this);

    Program::GetInstance().InsertAliveTensor(this);
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    Program::GetInstance().GetTensorSlotManager()->TensorSymbol(*this, name);
}

Tensor::Tensor(DataType dataType, std::vector<SymbolicScalar> shape, std::string name, TileOpFormat format)
    : Tensor(dataType, SymbolicScalar::Concrete(shape, -1), name, format) {
    CheckShapeValid(dataType, shape, format);
    auto rawTensor = storage_->GetRawTensor();
    for (size_t axis = 0; axis < shape.size(); axis++) {
        if (shape[axis].ConcreteValid() && shape[axis].Concrete() == -1) {
            shape[axis] = rawTensor->GetDynRawShape(axis);
        }
    }
    storage_->UpdateDynValidShape(shape);
    rawTensor->UpdateDynRawShape(shape);
}

void Tensor::SetData(BinDataPtr data) {
    data_ = data;
}

const LogicalTensor &Tensor::operator*() const {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage_;
}

LogicalTensor &Tensor::operator*() {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage_;
}

const std::shared_ptr<LogicalTensor> &Tensor::GetStorage(bool readSlot) const
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage_;
}

std::shared_ptr<LogicalTensor> &Tensor::GetStorage(bool readSlot)
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage_;
}

namespace npu {
namespace tile_fwk {
void AssignTensorData(Tensor &lhs, const Tensor &rhs) {
    if (lhs.GetData() != nullptr) {
        if (rhs.GetData() != nullptr) {
            CHECK(lhs.GetData() == rhs.GetData()) << "Prohibit self-assignment.";
        }
    } else {
        lhs.SetData(rhs.GetData());
    }
}
}
}

Tensor &Tensor::operator=(const Tensor &rhs) {
    if (this == &rhs) {
        return *this;
    }
    AssignTensorData(*this, rhs);
    if (storage_ != nullptr && storage_->tensor != nullptr) {
        rhs.GetStorage()->tensor->symbol = storage_->tensor->symbol;
    }
    if (storage_ != nullptr) {
        storage_->tensor->AddRefCount(-1);
    }
    storage_ = rhs.GetStorage();
    if (storage_ != nullptr) {
        storage_->tensor->AddRefCount(1);
    }
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&rhs) noexcept {
    if (this == &rhs) {
        return *this;
    }
    AssignTensorData(*this, rhs);
    rhs.SetData(nullptr);
    if (storage_ != nullptr && storage_->tensor != nullptr) {
        rhs.GetStorage()->tensor->symbol = storage_->tensor->symbol;
    }
    if (storage_ != nullptr) {
        storage_->tensor->AddRefCount(-1);
    }
    storage_ = std::move(rhs.GetStorage());
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
    return *this;
}

Tensor::Tensor(const Tensor &rhs) : storage_(rhs.GetStorage()), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    if (storage_ != nullptr) {
        storage_->tensor->AddRefCount(1);
    }
    SetData(rhs.GetData());
    Program::GetInstance().InsertAliveTensor(this);
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

Tensor::Tensor(Tensor &&rhs) : storage_(std::move(rhs.GetStorage())), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    Program::GetInstance().InsertAliveTensor(this);
    SetData(rhs.GetData());
    rhs.SetData(nullptr);
    if (storage_ != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

DataType Tensor::GetDataType() const {
    return storage_->Datatype();
}

const Shape &Tensor::GetShape() const {
    return storage_->shape;
}

uint64_t Tensor::Dim() const {
    if (storage_ != nullptr) {
        return storage_->shape.size();
    }
    else {
        return 0;
    }
}

bool Tensor::IsEmpty() const {
    return storage_ == nullptr;
}

int32_t Tensor::GetShape(int axis) const {
    const size_t dimCount = storage_->shape.size();
    ASSERT(dimCount > 0) << "Tensor has no dimensions!";
    if (axis < 0) {
        axis += static_cast<int>(dimCount);
    }
    ASSERT(axis >= 0 && static_cast<size_t>(axis) < dimCount)
        << "Axis index " << axis << " is out of range [0, " << (dimCount - 1) << "].";
    return storage_->shape[axis];
}

std::vector<SymbolicScalar> &Tensor::GetValidShape() const {
    return storage_->GetDynValidShape();
}

TileOpFormat Tensor::Format() const {
    return storage_->Format();
}

void Tensor::SetCachePolicy(CachePolicy policy, bool value) {
  if (storage_ != nullptr) {
    storage_->SetCachePolicy(policy, value);
  }
}

bool Tensor::GetCachePolicy(CachePolicy policy) const {
  if (storage_ != nullptr) {
    return storage_->GetCachePolicy(policy);
  }
  return false;
}

void Tensor::SetName(const std::string &name) const {
    if (storage_) {
        storage_->tensor->SetSymbol(name);
    }
}

std::string Tensor::GetName() const {
    return storage_ ? storage_->tensor->GetSymbol() : "";
}

SymbolicScalar npu::tile_fwk::GetInputShape(const Tensor &t, int n) {
    auto rawTensor = t.GetStorage(false)->GetRawTensor();
    return rawTensor->GetDynRawShape(n);
}

const std::vector<SymbolicScalar>& npu::tile_fwk::GetInputShape(const Tensor &t) {
    auto rawTensor = t.GetStorage(false)->GetRawTensor();
    return rawTensor->GetDynRawShape();
}

namespace npu::tile_fwk {

static
SymbolicScalar GetInputDataInt32Dim1(const Tensor &t, SymbolicScalar off0) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim1Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim1);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim1Name = AddRuntimePrefix(getInputDataInt32Dim1Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim1(getInputDataInt32Dim1Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim1(input, off0);
}

static
SymbolicScalar GetInputDataInt32Dim2(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim2Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim2);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim2Name = AddRuntimePrefix(getInputDataInt32Dim2Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim2(getInputDataInt32Dim2Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim2(input, off0, off1);
}

static
SymbolicScalar GetInputDataInt32Dim3(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1, SymbolicScalar off2) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim3Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim3);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim3Name = AddRuntimePrefix(getInputDataInt32Dim3Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim3(getInputDataInt32Dim3Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim3(input, off0, off1, off2);
}

static
SymbolicScalar GetInputDataInt32Dim4(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1,
    SymbolicScalar off2, SymbolicScalar off3) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim4Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim4);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim4Name = AddRuntimePrefix(getInputDataInt32Dim4Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim4(getInputDataInt32Dim4Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim4(input, off0, off1, off2, off3);
}

SymbolicScalar GetInputData(const Tensor &t, const std::vector<SymbolicScalar> &offset) {
    ASSERT(t.Dim() == offset.size());
    ASSERT(t.Dim() >0 && t.Dim() <= 0x4);
    if (t.Dim() == 0x1) {
        return GetInputDataInt32Dim1(t, offset[0]);
    }
    else if (t.Dim() == 0x2) {
        return GetInputDataInt32Dim2(t, offset[0], offset[1]);
    }
    else if (t.Dim() == 0x3){
        return GetInputDataInt32Dim3(t, offset[0], offset[1], offset[2]);
    }
    else {
        return GetInputDataInt32Dim4(t, offset[0], offset[1], offset[2], offset[3]);
    }
}

static
SymbolicScalar DoGetTensorDataInt32(SymbolHandlerId handlerId, const Tensor &t, const std::vector<SymbolicScalar> &offset) {
    ASSERT(t.GetShape().size() == offset.size()) << "Mismatch dimension: " << t.GetShape().size() << " vs " << offset.size() << "\n";
    Program::GetInstance().GetTensorSlotManager()->TensorRead(t);

    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    ASSERT(currDynFunc != nullptr) << "Not under dynamic function!\n";

    auto currDynAttr = currDynFunc->GetDyndevAttribute();
    int getTensorDataIndex = currDynAttr->getTensorDataCount++;

    auto assemble = std::make_shared<Tensor>(TensorExtract(t, offset));

    auto emuopAssemble = *assemble->GetStorage()->GetProducers().begin();
    auto emuopMark = *emuopAssemble->GetIOperands()[0]->GetProducers().begin();
    auto emuopView = *emuopMark->GetIOperands()[0]->GetProducers().begin();
    GetTensorDataSetIndex(emuopView, getTensorDataIndex);
    GetTensorDataSetIndex(emuopMark, getTensorDataIndex);
    GetTensorDataSetIndex(emuopAssemble, getTensorDataIndex);

    auto &desc = currDynAttr->getTensorDataDescDict[getTensorDataIndex];
    desc.assembleTensor = assemble;

    std::string getName = SymbolHandler::GetNameByHandlerId(handlerId);
    std::string getRuntimeName = AddRuntimePrefix(getName);
    SymbolicScalar getRuntimeHandler(getRuntimeName);
    std::vector<SymbolicScalar> argList = {getTensorDataIndex, -1, -1, -1};
    argList.insert(argList.end(), offset.begin(), offset.end());
    return getRuntimeHandler(argList);
}

static std::vector<std::reference_wrapper<const Tensor>>::iterator FindTensor (
    const Tensor &key, std::vector<std::reference_wrapper<const Tensor>> &vec) {
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (&key == &(it->get())) {
            return it;
        }
    }
    return vec.end();
}

constexpr int MAX_GET_TENSOR_DATA_DIM = 4;
SymbolicScalar GetTensorData(const Tensor &t, const std::vector<SymbolicScalar> &offset) {
    CHECK(t.GetDataType() == DT_INT32);
    auto funcPtr = Program::GetInstance().GetCurrentDynamicFunction();
    if (funcPtr) {
        auto inputTensorList = funcPtr->GetDyndevAttribute()->startArgsInputTensorList;
        if (FindTensor(t, inputTensorList) != inputTensorList.end()) {
            return GetInputData(t, offset);
        }
    }

    CHECK(offset.size() <= MAX_GET_TENSOR_DATA_DIM);
    SymbolHandlerId handlerId = static_cast<SymbolHandlerId>(static_cast<int>(SymbolHandlerId::GetTensorDataInt32Dim1) + offset.size() - 1) ;
    return DoGetTensorDataInt32(handlerId, t, offset);
}

static
void DoSetTensorDataInt32(const SymbolicScalar &v, const std::vector<SymbolicScalar> &off, Tensor &t) {
    CHECK(t.GetShape().size() == off.size()) << "Mismatch dimen:" << t.GetShape().size() << " vs " << off.size() << "\n";
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(t);

    auto currDynFunc = Program::GetInstance().GetCurrentDynamicFunction();
    ASSERT(currDynFunc != nullptr) << "Not under dynamic function!\n";

    Shape vShape = Shape(t.GetShape().size(), 1);
    auto tmp = Full(v, t.GetDataType(), vShape);
    TensorInsert(tmp, off, t);
}

void SetTensorData(const SymbolicScalar &v, const std::vector<SymbolicScalar> &off, Tensor &dst) {
    CHECK(dst.GetDataType() == DT_INT32);
    return DoSetTensorDataInt32(v, off, dst);
}

}

SymbolicScalar npu::tile_fwk::IsLoopBegin(const SymbolicScalar &symbol, const SymbolicScalar &begin) {
    std::string isLoopBeginName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin);
    isLoopBeginName = AddRuntimePrefix(isLoopBeginName);
    SymbolicScalar isLoopBegin(isLoopBeginName);
    auto result = isLoopBegin(symbol, begin);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}

SymbolicScalar npu::tile_fwk::IsLoopEnd(const SymbolicScalar &symbol, const SymbolicScalar &end) {
    std::string isLoopEndName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd);
    isLoopEndName = AddRuntimePrefix(isLoopEndName);
    SymbolicScalar isLoopEnd(isLoopEndName);
    auto result = isLoopEnd(symbol, end);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}
