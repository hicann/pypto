/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Minimal stub of the real Ascend ``graph/operator_reg.h`` for compiling the pypto export
// op-def codegen test only. The new op-def TU emits, inside ``namespace ge { ... }``:
//
//   REG_OP(MyOp)
//       .INPUT(in0, TensorType({DT_FLOAT16}))
//       .OUTPUT(out0, TensorType({DT_FLOAT16}))
//       .OP_END_FACTORY_REG(MyOp);
//
// The real header provides the REG_OP/INPUT/OUTPUT/OP_END_FACTORY_REG macros + TensorType; our
// existing ``register/register.h`` stub already mirrors exactly those (and pulls DataType from
// ``gert_ge_minimal.hpp``), so reuse it as the single source of truth.
#ifndef PYPTO_EXPORT_FIXTURE_GRAPH_OPERATOR_REG_H
#define PYPTO_EXPORT_FIXTURE_GRAPH_OPERATOR_REG_H

#include "register/register.h"

#endif // PYPTO_EXPORT_FIXTURE_GRAPH_OPERATOR_REG_H
