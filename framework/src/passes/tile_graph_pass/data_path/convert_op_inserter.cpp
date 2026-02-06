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
 * \file convert_op_inserter.cpp
 * \brief
 */
#include "convert_op_inserter.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu{
namespace tile_fwk {

const std::unordered_set<DataType> kA2A3SupportedDtypes = {DT_INT4, DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16};
const std::unordered_set<DataType> kA5SupportedDtypes = {DT_INT4, DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_HF8, DT_FP8, DT_FP32};

const static std::unordered_map<NPUArch, std::unordered_set<DataType>> kArch2SupportedDtypes = {
    {NPUArch::DAV_1001, kA2A3SupportedDtypes},
    {NPUArch::DAV_2201, kA2A3SupportedDtypes},
    {NPUArch::DAV_3510, kA5SupportedDtypes},
    {NPUArch::DAV_UNKNOWN, kA2A3SupportedDtypes}
};

// 设置指定tensor的指定consumer op所需的mem tobe 类型
void ConvertInserter::UpdateTensorTobeMap(const LogicalTensorPtr &tensor, Operation &operation, MemoryType t) {
    // 传入op必须为tensor的consumer
    if(!tensor->HasConsumer(operation)){
        APASS_LOG_ERROR_F(Elements::Tensor, "Operation %d is not a consumer of tensor %d; "
            "Please make sure the operation is relative to the tensor to be mapped.",
            operation.GetOpMagic(), tensor->GetMagic());
        return;
    }
    if (tensorTobeMap.count(tensor) == 0) {
        // 首次插入
        std::map<Operation *, MemoryType> tobeMap;
        tobeMap.emplace(&operation, t);
        tensorTobeMap[tensor] = tobeMap;
        APASS_LOG_DEBUG_F(Elements::Tensor, "Tensor %d first set toBeMap(%s[%d]) as %s",
            tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
            BriefMemoryTypeToString(t).c_str());
        return;
    }
    if (tensorTobeMap[tensor].count(&operation) == 0) {
        // 已存在，且首次设置该consumer的tobe mem
        tensorTobeMap[tensor].emplace(&operation, t);
        APASS_LOG_DEBUG_F(Elements::Tensor, "First Set magic: %d, new: %s.",
            tensor->GetMagic(), BriefMemoryTypeToString(t).c_str());
        return;
    }
    if (tensorTobeMap[tensor][&operation] == MemoryType::MEM_UNKNOWN && t != MemoryType::MEM_UNKNOWN) {
        tensorTobeMap[tensor][&operation] = t;
        return;
    }
    if (tensorTobeMap[tensor][&operation] == t) {
        return;
    }
    tensorTobeMap[tensor][&operation] = t;
    APASS_LOG_DEBUG_F(Elements::Tensor, "Update tensor %d toBeMap(%s[%d]) from %s to %s,",
        tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
        BriefMemoryTypeToString(tensorTobeMap[tensor][&operation]).c_str(),
        BriefMemoryTypeToString(t).c_str());
}

// 将指定tensor的tobe map中的unknown项更新为指定的mem类型
void ConvertInserter::UpdateTensorTobeMapUnknown(LogicalTensorPtr &tensor, MemoryType t) {
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has not been inserted yet; "
            "Please make sure tensor in the tobe map.", tensor->GetMagic());
        return;
    }
    if (t == MemoryType::MEM_UNKNOWN) {
        return;
    }
    for (auto &item : tensorTobeMap[tensor]) {
        if (item.second == MemoryType::MEM_UNKNOWN) {
            item.second = t;
        }
    }
}


// 打印指定tensor的tobe map
void ConvertInserter::PrintTensorTobeMap(LogicalTensorPtr &tensor) const {
    APASS_LOG_INFO_F(Elements::Tensor, "PrintTensorTobeMap tensor %d.", tensor->GetMagic());
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has not been inserted yet; "
            "Please make sure tensor in the tobe map.", tensor->GetMagic());
        return;
    }
    APASS_LOG_INFO_F(Elements::Tensor, "Size: %d.", tensorTobeMap.at(tensor).size());
    for (const auto &item : tensorTobeMap.at(tensor)) {
        APASS_LOG_INFO_F(Elements::Tensor, "\t|--- TensorTobeMap: %s --> %s[%d].", BriefMemoryTypeToString(item.second).c_str(),
            item.first->GetOpcodeStr().c_str(), item.first->GetOpMagic());
    }
}

// 提取指定tensor的tobe map，默认格式，key为consumer op，val为对应的mem类型
std::map<Operation *, MemoryType> ConvertInserter::GetTobeDefault(LogicalTensorPtr &tensor) const {
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has not been inserted yet; "
            "Please make sure tensor in the tobe map.", tensor->GetMagic());
        return {};
    }
    return tensorTobeMap.at(tensor);
}

// 提取指定tensor的tobe map，新格式，key为Mem类型，val为需要改mem类型的op指针set
std::map<MemoryType, std::set<Operation *>> ConvertInserter::GetRequiredTobe(LogicalTensorPtr &tensor) const {
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has not been inserted yet; "
            "Please make sure tensor in the tobe map.", tensor->GetMagic());
        return {};
    }
    std::map<MemoryType, std::set<Operation *>> result;
    for (const auto &item : tensorTobeMap.at(tensor)) {
        result[item.second].insert(item.first);
    }
    return result;
}

// 提取指定tensor的指定consumer op所需的mem类型
MemoryType ConvertInserter::GetMemoryTypeFromTensorTobeMap(LogicalTensorPtr &tensor, Operation &operation) const {
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor, "Tensor %d has not been inserted yet; "
            "Please make sure tensor in the tobe map.", tensor->GetMagic());
        return MemoryType::MEM_UNKNOWN;
    }
    return tensorTobeMap.at(tensor).at(&operation);
}

// 提取指定tensor的所有consumer op和所需的mem类型
std::map<Operation *, MemoryType> ConvertInserter::GetMemoryTypeFromTensorTobeMap(LogicalTensorPtr &tensor) const {
    auto it = tensorTobeMap.find(tensor);
    if (it != tensorTobeMap.end()) {
        return it->second;
    }
    return {};
}


std::map<MemoryType, std::set<Operation *>> ConvertInserter::ReformMap(std::map<Operation *, MemoryType> &oriMap) const {
    std::map<MemoryType, std::set<Operation *>> result;
    for (const auto &item : oriMap) {
        result[item.second].insert(item.first);
    }
    return result;
}

// 过滤得到所有有conflict的Tensor信息
void ConvertInserter::FilterConflictTensor() {
    for (const auto &pairLocal : tensorTobeMap) {
        auto tensorLocal = pairLocal.first;
        std::map<Operation *, MemoryType> oriMap = pairLocal.second;
        std::map<MemoryType, std::set<Operation *>> tobeMap = ReformMap(oriMap);
        if (tobeMap.size() == 1) {
            MemoryType requiredMemoryType = tobeMap.begin()->first;
            if (tensorLocal->GetMemoryTypeOriginal() == requiredMemoryType) {
                continue;
            }
        }
        conflictMap[tensorLocal->magic] = tobeMap;
    }
    APASS_LOG_INFO_F(Elements::Tensor, "--- ConflictMap size: %d ---", conflictMap.size());
}

// 将 tensor tobe map初始化当前tensor的memory type original
void ConvertInserter::RefreshTensorTobeMap(Function &function) {
    for (const auto &ele : function.GetTensorMap().tensorMap_) {
        for (const auto &tensor : ele.second) {
            for (const auto &consumerOp : tensor->GetConsumers()) {
                UpdateTensorTobeMap(tensor, *consumerOp, tensor->GetMemoryTypeOriginal());
            }
        }
    }
}

// 判断path路径中是否包含DDR
bool ConvertInserter::CrossCore(const MemoryType from, const MemoryType to) const {
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);

    return std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
}

// graph重连
void ConvertInserter::UpdateConsumerAndReconnect(
    std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor, Operation *op) const {
    newTensor->AddConsumer(op);
    auto updateViewOffset = [](std::shared_ptr<LogicalTensor> oldTensor1,
                               std::shared_ptr<LogicalTensor> newTensor1, Operation *op1) {
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op1->GetOpAttribute().get());
        if (viewOpAttribute != nullptr) {
            auto oldOffset = viewOpAttribute->GetFromOffset();
            if (oldTensor1->offset != newTensor1->offset) {
                for (size_t i = 0; i < oldOffset.size(); i++) {
                    oldOffset[i] -= oldTensor1->offset[i];
                }
            }
            viewOpAttribute->SetFromOffset(oldOffset, viewOpAttribute->GetFromDynOffset());
        }
    };
    for (size_t i = 0; i < op->iOperand.size(); ++i) {
        if ((op->iOperand[i]->magic == oldTensor->magic) &&
            (op->iOperand[i]->tensor->rawmagic == oldTensor->tensor->rawmagic)) {
            if (op->GetOpcode() == Opcode::OP_VIEW) {
                updateViewOffset(oldTensor, newTensor, op);
            }
            op->ReplaceIOperand(i, newTensor);
        }
    }
}

// 遍历所有tensor，如果有Mem conflict，记录到converts中
Status ConvertInserter::RecordConflict(Function &function) {
    oldRawToNewRaw.clear();
    converts.clear();
    std::vector<int> visitedTensor;
    for (const auto &op : function.Operations()) {
        for (auto &iOperand : op.iOperand) {
            if(iOperand->GetProducers().size()>=1){
                continue;
            }
            iOperand->SetMemoryTypeToBe(iOperand->GetMemoryTypeOriginal());
        }
        for (auto &oOperand : op.oOperand) {
            oOperand->SetMemoryTypeToBe(oOperand->GetMemoryTypeOriginal());
            //step1:当tensor不在conflictMap中或者已经在visitedTensor中被处理过，可以跳过，否则需要处理内存冲突
            if (SkipOperand(oOperand,visitedTensor)) {
                continue;
            }
            //step2:解析冲突需求
            std::map<MemoryType, std::set<Operation *>> tobeMap = conflictMap.at(oOperand->magic);
            for (const auto &item : tobeMap) {
                MemoryType requiredMemoryType = item.first;
                std::set<Operation *> consumers = item.second;
                if (requiredMemoryType == oOperand->GetMemoryTypeOriginal()) {
                    continue;
                }
                
                //step3：处理特殊生产者消费者场景
                ProcessSpecialProducersOrConsumers(op, oOperand, consumers, requiredMemoryType);
                if (requiredMemoryType == oOperand->GetMemoryTypeOriginal()) {
                    continue;
                }

                //step4:构造转换路径
                std::vector<MemoryType> paths;
                Status status = ProcessConvertPath(op, oOperand, requiredMemoryType, paths);
                if (status != SUCCESS) {return status;}

                //step5：对每个消费者插入的Convert Op并更新图链接
                InsertConvertOpForEachConsumer(function, op, oOperand, consumers, paths);

                //step6：标记已处理
                visitedTensor.push_back(oOperand->magic);
            }
        }
    }
    return SUCCESS;
}

//为每个存在内存冲突的消费者插入convert op
void ConvertInserter::InsertConvertOpForEachConsumer(Function &function, const Operation &op, const std::shared_ptr<LogicalTensor> &oOperand,
 std::set<Operation *> &consumers, std::vector<MemoryType> &paths) {
    for (auto consumer : consumers) {
        auto output = RecordInsertConvertOp(oOperand,paths,function,op);
        if (consumer->BelongTo() == &function) {
            UpdateConsumerAndReconnect(oOperand, output, consumer);
        }
    }
}

//特殊场景处理：生成者均为Assemble或者消费者均为View/Assemble，且mem路径中经过DDR
void ConvertInserter::ProcessSpecialProducersOrConsumers(const Operation &op, const std::shared_ptr<LogicalTensor> &oOperand,
    std::set<Operation *> &consumers, MemoryType &requiredMemoryType) {
    //case1:当tensor的生产者都是assemble，并且tensor的mem路径需要经过DDR，则将tensor的ori刷成DDR
    APASS_LOG_DEBUG_F(Elements::Operation, "Operation %s[%d] has output %d ori and tobe conflict.",
        op.GetOpcodeStr().c_str(), op.GetOpMagic(), oOperand->magic);
    bool crossCore = CrossCore(oOperand->GetMemoryTypeOriginal(), requiredMemoryType);
    bool producedByAssemble = isAllProducerAssemble(oOperand);
    if (producedByAssemble && crossCore) {
        oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        oOperand->SetMemoryTypeToBe(oOperand->GetMemoryTypeOriginal());
    }
    //case2:当tensor的消费者都是view或者assemble，并且tensor的mem路径需要经过DDR时，将tensor的tobe刷成DDR
    bool canSetBoth = isAllConsumersValid(consumers);
    if (canSetBoth && crossCore) {
        requiredMemoryType = MEM_DEVICE_DDR;
    }
}

bool ConvertInserter::IsNotValidDataType(const std::shared_ptr<LogicalTensor> &firstCVOutput) const {
    // 1. 获取当前NPU架构
    const NPUArch currentArch = Platform::Instance().GetSoc().GetNPUArch();
    
    // 2. 查找当前架构对应的支持类型集合（容错：找不到则用DAV_UNKNOWN兜底）
    auto archIter = kArch2SupportedDtypes.find(currentArch);
    if (archIter == kArch2SupportedDtypes.end()) {
        archIter = kArch2SupportedDtypes.find(NPUArch::DAV_UNKNOWN);
    }
    const auto& supportedDtypes = archIter->second;

    // 3. 判断当前张量类型是否不在支持列表中（不在则返回true，表示无效）
    const DataType tensorDtype = firstCVOutput->Datatype();
    return supportedDtypes.find(tensorDtype) == supportedDtypes.end();
}

//构造转换路径
Status ConvertInserter::ProcessConvertPath(const Operation &op, const std::shared_ptr<LogicalTensor> &oOperand,
    MemoryType requiredMemoryType, std::vector<MemoryType> &paths) {
    auto currTensorMemOri = oOperand->GetMemoryTypeOriginal();
    if(currTensorMemOri == MemoryType::MEM_L0C && requiredMemoryType == MemoryType::MEM_L1) {
        //特殊处理L0C2L1：针对不支持的数据类型场景路径中插入DDR
        bool needDDRTrans = IsNotValidDataType(oOperand);
        if(needDDRTrans) {
            paths = {currTensorMemOri, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1};
        } else {
            paths = {currTensorMemOri, MemoryType::MEM_L1};
        }
    } else {
        //常规场景：查platform硬件配置获取path处理
        Status status = ConstructPath(oOperand->GetMemoryTypeOriginal(),requiredMemoryType,paths,oOperand,op);
        if (status != SUCCESS) {return status;}
    }
    return SUCCESS;
}

//检查from和to之间是否不存在数据通路
Status ConvertInserter::ConstructPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths,
    const std::shared_ptr<LogicalTensor> &oOperand,const Operation &op) const {
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (paths.empty()) {
        //path为空的两种场景:1、from和to内存类型一致；2、from和to不一致，且未找到数据通路。这里处理场景2，报错退出
        APASS_LOG_ERROR_F(Elements::Operation, "No memory path found from %s to %s for tensor %d in operation %s[%d]. %s",
            BriefMemoryTypeToString(from).c_str(),
            BriefMemoryTypeToString(to).c_str(),
            oOperand->magic,
            op.GetOpcodeStr().c_str(),
            op.GetOpMagic(),
            GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

//检查tensor是否需要跳过
bool ConvertInserter::SkipOperand(const std::shared_ptr<LogicalTensor> &oOperand, const std::vector<int> visitedTensor) const{
    return ((conflictMap.find(oOperand->magic) == conflictMap.end()) ||
            (std::find(visitedTensor.begin(), visitedTensor.end(), oOperand->magic) != visitedTensor.end()));
}

//检查tensor生产者是否都是assemble
bool ConvertInserter::isAllProducerAssemble(const std::shared_ptr<LogicalTensor> &oOperand) const{
    auto producers = oOperand->GetProducers();
    return std::all_of(producers.begin(), producers.end(),
            [](const Operation *producerOp) { return producerOp->GetOpcode() == Opcode::OP_ASSEMBLE; });
}

//检查tensor所有的消费者是否是view或者assemble
bool ConvertInserter::isAllConsumersValid(const std::set<Operation *> &consumers) const{
    for (const auto consumer : consumers){
        if (consumer->GetOpcode() != Opcode::OP_VIEW && consumer->GetOpcode() != Opcode::OP_ASSEMBLE){
            return false;
        }
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE &&
            (consumer->GetOOperands().front()->nodetype != NodeType::OUTCAST)){
            return false;
        }
    }
    return true;
}

//记录需要插入的convert op
std::shared_ptr<LogicalTensor> ConvertInserter::RecordInsertConvertOp(const std::shared_ptr<LogicalTensor> &oOperand,
    const std::vector<MemoryType> &paths,Function &function,const Operation &op){
    std::shared_ptr<LogicalTensor> input = oOperand;
    for (size_t i = 0; i < paths.size() - 1; ++i) {
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(
            input->Datatype(), input->GetShape(), input->Format());;
        input->SetMemoryTypeToBe(paths[i]); // 后续删除
        std::vector<int64_t> newoffset(input->offset.size(), 0);
        std::shared_ptr<LogicalTensor> output = std::make_shared<LogicalTensor>(function, newRawTensor, newoffset, input->shape);
        output->SetMemoryTypeBoth(paths[i + 1]); // 后续只用设置original
        converts.emplace_back(ConvertOpInfo{paths[i], paths[i + 1], input, output});
        APASS_LOG_DEBUG_F(Elements::Tensor, "%s[%d] --> tensor[%d](%s) --> Convert --> %s.", op.GetOpcodeStr().c_str(),
            op.GetOpMagic(), input->magic,
            BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
            BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
        APASS_LOG_DEBUG_F(Elements::Tensor, "--- %s --> %s ---",
            BriefMemoryTypeToString(paths[i]).c_str(),
            BriefMemoryTypeToString(paths[i + 1]).c_str());
        input = output;
    }
    return input;
}

//graph重连
void ConvertInserter::GraphReconnect(const std::shared_ptr<LogicalTensor> &oOperand, std::shared_ptr<LogicalTensor> output,
        const std::set<Operation *> &consumers,Function &function) const {
    for (const auto &consumer : consumers) {
        if (consumer->BelongTo() == &function) {
            UpdateConsumerAndReconnect(oOperand, output, consumer);
        }
    }
}

// 合法性校验
void ConvertInserter::CheckUnknown(Function &function) const {
    auto opList = function.Operations();
    ParallelTool::Instance().Parallel_for(0, opList.size(), 1, [&](int st, int et, int tid) {
        (void)tid;
        for (int opIdx = st; opIdx < et; opIdx++) {
            auto &op = opList[opIdx];
            std::unordered_set<MemoryType> supportedMemType = {MemoryType::MEM_UB, MemoryType::MEM_L1,
                MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_L2, MemoryType::MEM_L3,
                MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_HOST1, MemoryType::MEM_FAR1, MemoryType::MEM_FAR2};
            switch (op.GetCoreType()) {
                case CoreType::AIC:
                    supportedMemType = {MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                        MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_FIX,
                        MemoryType::MEM_FIX_QUANT_PRE, MemoryType::MEM_FIX_RELU_PRE, MemoryType::MEM_FIX_RELU_POST,
                        MemoryType::MEM_FIX_QUANT_POST, MemoryType::MEM_FIX_ELT_ANTIQ, MemoryType::MEM_FIX_MTE2_ANTIQ};
                    break;
                case CoreType::AIV: supportedMemType = {MemoryType::MEM_UB}; break;
                case CoreType::GMATOMIC: supportedMemType = {MemoryType::MEM_DEVICE_DDR}; break;
                default: break;
            }
            for (const auto &i : op.GetIOperands()) {
                if(supportedMemType.count(i->GetMemoryTypeToBe()) == 0){
                    APASS_LOG_DEBUG_F(Elements::Operation, "Op %s[%d] input[%d] has unsupported mem type %s.",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), i->magic, MemoryTypeToString(i->GetMemoryTypeToBe()).c_str());
                }
            }
            for (const auto &o : op.GetOOperands()) {
                if(supportedMemType.count(o->GetMemoryTypeToBe()) == 0){
                    APASS_LOG_DEBUG_F(Elements::Operation, "Op %s[%d] output[%d] has unsupported mem type %s.",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), o->magic,MemoryTypeToString(o->GetMemoryTypeToBe()).c_str());
                }
                if(o->GetMemoryTypeOriginal() != o->GetMemoryTypeToBe()){
                    APASS_LOG_DEBUG_F(Elements::Operation, "Op %s[%d] output[%d] has two mem type %s and %s.",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), o->magic,MemoryTypeToString(o->GetMemoryTypeToBe()).c_str(),
                        MemoryTypeToString(o->GetMemoryTypeToBe()).c_str());
                }
            }
        }
    });
}

void ConvertInserter::CreateMoveOpForConvert(Operation &op) {
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();

    if (from == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_VIEW); // 将convert根据View, 后续GenerateMoveOp Pass会转化为copyin
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(op.iOperand.front()->GetOffset(), to,
            op.iOperand.front()->GetDynOffset(),
            op.iOperand.front()->GetDynValidShape()));
        auto childOp = *op.oOperand.front()->GetConsumers().begin();
        op.UpdateSubgraphID(childOp->GetSubgraphID());
        return;
    }

    if (to == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_ASSEMBLE); //将convert根据Assemble, 后续GenerateMoveOp Pass会转化为copyout
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(from, op.oOperand.front()->GetOffset(),
            op.oOperand.front()->GetDynOffset(),
            op.iOperand.front()->GetDynValidShape()));
        auto parentOp = *op.oOperand.front()->GetProducers().begin();
        op.UpdateSubgraphID(parentOp->GetSubgraphID());
    }
}

// 根据已记录的converts插入OP_CONVERT
void ConvertInserter::InsertConvertOps(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "--- Need to insert %d convert operations ---", converts.size());
    for (const auto &c : converts) {
        GraphUtils::CopyDynStatus(c.output, c.input);
        auto &convertOp = function.AddRawOperation(Opcode::OP_CONVERT, {c.input}, {c.output});
        convertOp.SetOpAttribute(std::make_shared<ConvertOpAttribute>(c.from, c.to));
        CreateMoveOpForConvert(convertOp);
    }
}

// 对外总接口
Status ConvertInserter::DoInsertion(Function &function) {
    FilterConflictTensor();
    Status status = RecordConflict(function);
    if(status != SUCCESS) { return status; }
    InsertConvertOps(function);
    CheckUnknown(function);
    APASS_LOG_INFO_F(Elements::Function, "After Insert Convert, total op Num: %d.",
        function.Operations().size());
    return SUCCESS;
}
} //namespace tile_fwk
} // namespace npu
