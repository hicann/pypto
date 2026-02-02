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
 * \file graph_utils.h
 * \brief
 */

#pragma once
#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H
#include <vector>
#include <queue>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "pass_common_defs.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
class GraphUtils {
public:
    /**
     * @brief Add an operation and set the DynValidShape of the output.
     *
     * @param function the target function for the operation to be added.
     * @param opCode type of the operation to be added (Besides Assemble, View, Convert, CopyIn, CopyOut, Reshape)
     * @param iOperands LogicalTensors, indicating the input of the op to be added
     * @param oOperands LogicalTensors, indicating the output of the op to be added
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     * @return the operation to be added
     */
    static Operation &AddDynOperation(Function &function, const Opcode opCode, LogicalTensors iOperands, const LogicalTensors &oOperands,
                                      const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Add a raw operation by AddRawOperation and set the DynValidShape of the output.
     *
     * @param function the target function for the operation to be added.
     * @param opCode type of the operation to be added (Besides Assemble, View, Convert, CopyIn, CopyOut, Reshape)
     * @param iOperands LogicalTensors, indicating the input of the op to be added
     * @param oOperands LogicalTensors, indicating the output of the op to be added
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     * @return the operation to be added
     */
    static Operation &AddDynRawOperation(Function &function, const Opcode opCode, LogicalTensors iOperands, const LogicalTensors &oOperands,
                                        const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Add a view operation.
     *        Set the DynValidShape of the output.
     *        Update the ViewOpAttribute of the view operation. The toDynValidShape value is set by the calculated value of SetDynShape.
     *
     * @param function the target function for the view operation.
     * @param view ViewOp, indicating the input, output, toType, fromOffSet
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     * @return the operation to be added
     */
    static Operation &AddViewOperation(Function &function, const ViewOp &view, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Add an assemble operation.
     *        Update the AssembleOpAttribute of the assemble operation. The fromDynValidShape value is set by the DynValidShape of input.
     *        Inherit the operation attribute and scopeId when given an origin assemble op.
     *        Set the DynValidShape of the output.
     *
     * @param function the target function for the assemble operation.
     * @param assemble AssembleOp, indicating the basic information of the added assemble operation.
     *                 The information includes the memoryType of assemble OpAttribute, assemble offset, input and output of assemble.
     *                 The information also indicates the origin assemble op (if exist) that the added operation
     *                 should inherit attribute and scope id from.
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape of each output.
     *                    The AssembleOpAttribute does not require dynamic attributes for output, so the SetDynShape is executed at last. 
     * @return the operation to be added
     */
    static Operation &AddAssembleOperation(Function &function, const AssembleOp &assemble, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Add a reshape operation.
     *        Set the DynValidShape of the output.
     *        Inherit the operation attribute and scope id when given a legal origin reshape operation pointer.
     *        Update the op_attr_validShape of the reshape operation by the DynValidShape of output.
     *
     * @param function the target function for the reshape operation.
     * @param iOperand LogicalTensorPtr, indicating the input of the op
     * @param oOperand LogicalTensorPtr, indicating the output of the op
     * @param originOp Pointer of operation, indicating an origin operation the added reshape operation should inherit attribute and scopeId from.
     *                 Skip inherit attribute and scopeId if the pointer is nullptr.
     * @param outDynShape the DynValidShape of the output and the value of op_attr_validShape. The default value is {}.
     *                    If outDynShape is empty, uses CallInferShapeFunc to calculate the DynValidShape.
     * @return the operation to be added
     */
    static Operation &AddReshapeOperation(Function &function, const LogicalTensorPtr iOperand, const LogicalTensorPtr &oOperand,
        const ReshapeOp &reshapeOp, const std::vector<SymbolicScalar> &outDynShape = {});
    /**
     * @brief Add a copyin operation.
     *        Update the CopyOpAttribute of the copyin operation by the CopyInOutOp object.
     *        Set the DynValidShape of the output.
     *        Update the SubgraphID.
     *
     * @param function the target function for the copyin operation.
     * @param copy CopyInOutOp, indicating the input, output, fromtype, Offset, shape, rawShape, fromDynValidShape
     * @param outDynShape the DynValidShape of the output and the value of op_attr_validShape. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape.
     *                    Since the CallInferShapeFunc interface requires the CopyOpAttribute, set the CopyOpAttribute before exectuing the SetDynShape.
     * @return the operation to be added
     */
    static Operation &AddCopyInOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Add a copyout operation.
     *        Update the CopyOpAttribute of the copyout operation by the CopyInOutOp object.
     *        Set the DynValidShape of the output.
     *        Update the SubgraphID.
     *
     * @param function the target function for the copyout operation.
     * @param copy CopyInOutOp, indicating the input, output, fromtype, Offset, shape, rawShape, fromDynValidShape
     * @param outDynShape the DynValidShape of the output and the value of op_attr_validShape. The default value is {}.
     *                    If outDynShape is empty, uses SetDynShape to calculate the DynValidShape.
     *                    Since the CallInferShapeFunc interface requires the CopyOpAttribute, set the CopyOpAttribute before exectuing the SetDynShape.
     * @return the operation to be added
     */
    static Operation &AddCopyOutOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Set the DynValidShape of dstTensor by the DynValidShape of srcTensor.
     *
     * @param function the target function, consisting the target op.
     * @param op the target view op.
     */
    static void CopyDynStatus(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor);
    /**
     * @brief Update FromDynOffset of a view op when the input or output is incast or outcast.
     *
     * @param function the target function, consisting the target op.
     * @param op the target view op.
     */
    static void UpdateViewAttr(Function &function, Operation &op);
    /**
     * @brief Set the DynValidShape of the output for the specified op.
     *
     * @param newOp the target operation having oOperands without DynValidShape.
     * @param outDynShape the DynValidShape of each output. The default value is {}.
     *                    If outDynShape is empty, set the DynValidShape of each output by CallInferShapeFunc.
     */
    static void SetDynShape(Operation *newOp, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    /**
     * @brief Set the CopyOpAttribute for a copyin op.
     *
     * @param op the target copyin operation.
     * @param copy CopyInOutOp, consisting of input, output, memorytype, shape and rawshape.
     */
    static void SetCopyInAttr(Operation &op, const CopyInOutOp &copy);
    /**
     * @brief Set the CopyOpAttribute for a copyout op.
     *
     * @param op the target copyout operation.
     * @param copy CopyInOutOp, consisting of input, output, memorytype, shape and rawshape.
     */
    static void SetCopyOutAttr(Operation &op, const CopyInOutOp &copy);
    /**
     * @brief Set the ViewOpAttribute for a view op.
     *
     * @param op the target view operation.
     * @param copy ViewOp, consisting of input, output, totype, fromOffset.
     */
    static void SetViewAttr(Function &function, Operation &op, const ViewOp &view);
    /**
     * @brief Set the AssembleOpAttribute for a assemble op.
     *
     * @param op the target assemble operation.
     * @param copy AssembleOp, consisting of input, output, fromtype, toOffset.
     */
    static void SetAssembleAttr(Operation &op, const AssembleOp &assemble);
    /**
     * @brief Determine it is a CV seperate or CV mix platform.
     */
    static bool IsCVMixPlatform();
};
}
}
#endif