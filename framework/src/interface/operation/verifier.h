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
 * \file verifier.h
 * \brief
 */

#include <ostream>

namespace npu::tile_fwk {
    class Function;
    class Operation;
    /**
     * \brief Entry for verifying operation
     *
     * \param func : the function that contains the operation.
     * \param op : the target operation.
     * \param oss : stream printer
     * \return bool : true means success. false means failure.
     */
    using VerifyOperationEntry = std::function<bool(const Function &func, const Operation &op, std::ostream &oss)>;
}
