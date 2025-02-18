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
 * \file SimulatorAdaptor.cpp
 * \brief
 */

#include <sstream>
#include <algorithm>
#include "SimulatorAdaptor.h"

namespace CostModel
{
    static std::vector<std::string> ParseTokens(std::string input)
    {
        std::vector<std::string> result;
        std::istringstream stream(input);
        std::string token;

        while (std::getline(stream, token, ' ')) { 
            if (!token.empty()) { 
                result.push_back(token);
            }
        }

        return result;
    }

    std::vector<std::string> SimulatorAdaptor::Rewrite(const std::vector<std::string> &program) const
    {
        std::vector<std::string> np;
        for (const auto &s : program) {
            std::string ln;
            size_t firstArg = 0;
            auto tokens = ParseTokens(s);
            size_t threshold1 = 6;
            size_t threshold2 = 4;
            size_t threshold3 = 2;
            if (tokens.size() >= threshold1 && !std::isdigit(tokens[1][0]) && tokens[1][0] != '-'
                && !std::isdigit(tokens[3][0]) && tokens[3][0] != '-' && !std::isdigit(tokens[5][0]) && tokens[5][0] != '-') {
                ln += tokens[0] + "<" + tokens[1] + "," + tokens[3] + "," + tokens[5] + ">(" + tokens[2] + ", " + tokens[4] + ", ";
                firstArg = threshold1;
            }
            else if (tokens.size() >= threshold2 && !std::isdigit(tokens[1][0]) && tokens[1][0] != '-'
                && !std::isdigit(tokens[3][0]) && tokens[3][0] != '-') {
                ln += tokens[0] + "<" + tokens[1] + "," + tokens[3] + ">(" + tokens[2] + ", ";
                firstArg = threshold2;
            }
            else if (tokens.size() >= threshold3 && !std::isdigit(tokens[1][0]) && tokens[1][0] != '-') {
                ln += tokens[0] + "<" + tokens[1] + ">(";
                firstArg = threshold3;
            }
            else {
                ln += tokens[0] + "(";
                firstArg = 1;
            }

            for (size_t i = firstArg; i < tokens.size(); i++) {
                ln += tokens[i];
                if (i < tokens.size() - 1) {
                    ln += ", ";
                }
            }
            ln += ")";
            
            np.push_back(ln);
        }
        return np;
    }
} // namespace CostModel

