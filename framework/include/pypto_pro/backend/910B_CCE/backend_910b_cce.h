/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_BACKEND_910B_CCE_BACKEND_910B_CCE_H_
#define PYPTO_BACKEND_910B_CCE_BACKEND_910B_CCE_H_

#include <string>

#include "backend/common/backend.h"

namespace pypto {
namespace backend {

/**
 * \brief Backend implementation for 910B hardware with CCE code generation
 *
 * Provides CCE (C++ pto-isa) code generation for 910B architecture.
 * Uses shared 910B SoC configuration created by Create910BSoC().
 * Operators are registered via REGISTER_BACKEND_OP macro in separate
 * compilation units.
 */
class Backend910B_CCE : public Backend {
public:
    /**
     * \brief Get registration instance for static operator registration
     *
     * Returns a singleton instance used during static initialization
     * to register operators via REGISTER_BACKEND_OP macro.
     *
     * \return Reference to registration instance
     */
    static Backend910B_CCE& Instance();

    /**
     * \brief Get backend type name
     *
     * \return "910B_CCE"
     */
    [[nodiscard]] std::string GetTypeName() const override { return "910B_CCE"; }

private:
    /**
     * \brief Private constructor (singleton pattern)
     *
     * Constructor is private to enforce singleton pattern.
     * Use Instance() to get the singleton instance.
     */
    Backend910B_CCE();
};

} // namespace backend
} // namespace pypto

#endif // PYPTO_BACKEND_910B_CCE_BACKEND_910B_CCE_H_
