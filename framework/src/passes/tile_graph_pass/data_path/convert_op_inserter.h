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
 * \file convert_op_inserter.h
 * \brief
 */

#ifndef PASS_CONVERT_OP_INSERTER_H_
#define PASS_CONVERT_OP_INSERTER_H_

#include <unordered_map>

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_utils/parallel_tool.h"

namespace npu{
    namespace tile_fwk {

struct ConvertOpInfo {
    MemoryType from;
    MemoryType to;
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> output;
};

class ConvertInserter {
public:
    ConvertInserter() = default;
    ~ConvertInserter() = default;

    std::vector<ConvertOpInfo> converts;
    std::unordered_map<int, std::shared_ptr<RawTensor>> oldRawToNewRaw;

    /*
        key: Tensor 指针
        value: consumer op的指针到该op所需内存类型的映射map
    */
    std::unordered_map<LogicalTensorPtr, std::map<Operation *, MemoryType>> tensorTobeMap;
    std::unordered_map<int, std::map<MemoryType, std::set<Operation *>>> conflictMap;

    // 设置指定tensor的指定consumer op所需的mem tobe 类型
    void UpdateTensorTobeMap(const LogicalTensorPtr &tensor, Operation &operation, MemoryType t);

    // 将指定tensor的tobe map中的unknown项更新为指定的mem类型
    void UpdateTensorTobeMapUnknown(LogicalTensorPtr &tensor, MemoryType t);

    // 打印指定tensor的tobe map
    void PrintTensorTobeMap(LogicalTensorPtr &tensor) const;

    // 提取指定tensor的tobe map，默认格式，key为consumer op，val为对应的mem类型
    std::map<Operation *, MemoryType> GetTobeDefault(LogicalTensorPtr &tensor) const;

    // 提取指定tensor的tobe map，新格式，key为Mem类型，val为需要改mem类型的op指针set
    std::map<MemoryType, std::set<Operation *>> GetRequiredTobe(LogicalTensorPtr &tensor) const;

    // 过滤得到所有有conflict的tensor信息
    void FilterConflictTensor();

    //tobe Map转换类型，以memory type为key
    std::map<MemoryType, std::set<Operation *>> ReformMap(std::map<Operation *, MemoryType> &oriMap) const;

    // 提取指定tensor的指定consumer op所需的mem类型
    MemoryType GetMemoryTypeFromTensorTobeMap(LogicalTensorPtr &tensor, Operation &operation) const;

    // 将 tensor tobe map初始化当前tensor的memory type original
    void RefreshTensorTobeMap(Function &function);

    // 遍历所有tensor，如果有Mem conflict，记录到converts中
    Status RecordConflict(Function &function);

    // 根据已记录的converts插入OP_CONVERT
    void InsertConvertOps(Function &function);

    // 将插入的OP_CONVERT转化为View和Assemble，后续GenerateMoveOp时会转化为copy类Op
    void CreateMoveOpForConvert(Operation &op);

    // 判断是否跨Memory层级
    bool CrossCore(const MemoryType from, const MemoryType to) const;

    // 更新消费者并重连graph
    void UpdateConsumerAndReconnect(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor, 
        Operation* op) const;

    // 合法性校验
    void CheckUnknown(Function &function) const;

    // 对外总接口
    Status DoInsertion(Function &function);

    //构建转换路径
    Status ConstructPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths,
        const std::shared_ptr<LogicalTensor> &oOperand,const Operation &op) const;

    // 检查tensor是否需要跳过
    bool SkipOperand(const std::shared_ptr<LogicalTensor> &oOperand, const std::vector<int> visitedTensor) const;
    
    //检查tensor生产者是否都是assemble
    bool isAllProducerAssemble(const std::shared_ptr<LogicalTensor> &oOperand) const;
    
    //检查tensor所有的消费者是否都有效
    bool isAllConsumersValid(const std::set<Operation *> &consumers) const;
    
    //记录需要插入的convert op
    std::shared_ptr<LogicalTensor> RecordInsertConvertOp(const std::shared_ptr<LogicalTensor> &oOperand, const std::vector<MemoryType> &paths,
        Function &function,const Operation &op);

    //graph重连
    void GraphReconnect(const std::shared_ptr<LogicalTensor> &oOperand, std::shared_ptr<LogicalTensor> output, 
        const std::set<Operation *> &consumers,Function &function) const;
    
    //cube级联场景
    bool IsNotValidDataType(const std::shared_ptr<LogicalTensor> &firstCVOutput) const;

    //特殊场景处理：生成者均为Assemble或者消费者均为View/Assemble，且mem路径中经过DDR
    void ProcessSpecialProducersOrConsumers(const Operation &op, const std::shared_ptr<LogicalTensor> &oOperand,
        std::set<Operation *> &consumers, MemoryType &requiredMemoryType);

    //构造转换路径
    Status ProcessConvertPath(const Operation &op, const std::shared_ptr<LogicalTensor> &oOperand,
        MemoryType requiredMemoryType, std::vector<MemoryType> &paths);
};
} 
}// namespace npu::tile_fwk
#endif // PASS_CONVERT_OP_INSERTER_H_