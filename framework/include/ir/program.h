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
 * \file program.h
 * \brief
 */

#pragma once

#include "ir/object.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pto {

class Function;

// Represents the top-level program.module container.
class ProgramModule : public Object {
public:
    explicit ProgramModule(std::string name);

    ObjectType GetObjectType() const override { return ObjectType::Program; }

    // Entrypoints.
    void SetProgramEntry(const std::shared_ptr<Function>& programEntry);
    const std::shared_ptr<Function> GetProgramEntry() const { return programEntry_; }

    // Functions (defined in func.h).
    void AddFunction(const std::shared_ptr<Function>& function);
    const std::vector<std::shared_ptr<Function>> GetFunctions() const { return functions_; }

    // Pretty-print to a textual PTO-IR-like form.
    void Print(std::ostream& os, int indent = 0) const;

private:
    std::shared_ptr<Function> programEntry_;
    std::vector<std::shared_ptr<Function>> functions_;
};

using ProgramModulePtr = std::shared_ptr<ProgramModule>;

// Helper for convenient streaming: std::cout << module;
std::ostream& operator<<(std::ostream& os, const ProgramModule& module);
using ProgramModulePtr = std::shared_ptr<ProgramModule>;
} // namespace pto


