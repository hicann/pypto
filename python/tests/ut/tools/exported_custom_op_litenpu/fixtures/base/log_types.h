/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Minimal mock of CANN ``base/log_types.h`` for codegen compile tests.
//
// The real header (in the CANN install: ``$ASCEND/.../include/base/log_types.h``)
// defines the slog module-id enum. We only need the ids the generated executor uses
// via ``PTO_CUSTOM_LOGD`` -> ``dlog_debug(<module>, ...)``. Keep values in sync with the
// real header.
#ifndef PYPTO_MOCK_BASE_LOG_TYPES_H
#define PYPTO_MOCK_BASE_LOG_TYPES_H

enum {
    RUNTIME = 7,
    APP = 33,
    FE = 39,
    GE = 45,
    ASCENDCL = 48,
    TBE = 57,
    OP = 63,
};

#endif // PYPTO_MOCK_BASE_LOG_TYPES_H
