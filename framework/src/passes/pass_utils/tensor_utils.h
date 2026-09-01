/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_utils.h
 * \brief utils for querying logical tensors sharing the same rawMagic.
 *
 * In the new graph representation, multiple assemble operations may write to different
 * logical tensors that share the same rawMagic (i.e. the same underlying RawTensor/address).
 * For example: tensor1->assemble1->tensor2, tensor3->assemble2->tensor4, where tensor2 and
 * tensor4 share the same rawMagic. These utils recover the "same address" relationship that
 * was previously expressed by sharing a single logical tensor.
 */

#ifndef PASS_TENSOR_UTILS_H_
#define PASS_TENSOR_UTILS_H_

#include <vector>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu {
namespace tile_fwk {
class TensorUtils {
public:
    /**
     * @brief Get ALL logical tensors that share the same rawMagic, INCLUDING the input tensor itself.
     *
     * @param function the target function to search in.
     * @param tensor the logical tensor used to locate the rawMagic bucket.
     * @return a vector of LogicalTensorPtrs sharing the same rawMagic (including self).
     */
    static std::vector<LogicalTensorPtr> GetSameRawMagicLogicalTensors(Function& function,
                                                                       const LogicalTensorPtr& tensor);

    /**
     * @brief Get the producers of ALL logical tensors sharing the same rawMagic,
     *        INCLUDING the input tensor itself.
     *
     * This returns every producer (deduplicated, non-deleted, belonging to the given
     * function) of the input tensor and its sibling logical tensors that share the
     * same rawMagic. Use this to recover the complete set of operations writing to
     * the same underlying address (RawTensor bucket).
     *
     * @param function the target function to search in.
     * @param tensor the logical tensor used to locate the rawMagic bucket.
     * @return a vector of producer Operations of the input tensor and its siblings.
     */
    static std::vector<Operation*> GetProducersOfSameRawMagicLogicalTensors(Function& function,
                                                                            const LogicalTensorPtr& tensor);
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_TENSOR_UTILS_H_
