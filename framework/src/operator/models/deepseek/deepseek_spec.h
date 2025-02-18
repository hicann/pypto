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
 * \file deepseek_spec.h
 * \brief
 */

#pragma once

#include <map>
#include <string>
#include <variant>

static std::map<std::string, std::variant<bool, int, float, std::string>> deepseekConfig1 = {
    {          "architectures", "DeepseekForCausalLM"},
    {         "attention_bias",                 false},
    {      "attentionDropout",                     0},
    {             "AutoConfig",      "g_deepseekConfig"},
    {              "AutoModel",       "DeepseekModel"},
    {   "AutoModelForCausalLM", "DeepseekForCausalLM"},
    {         "auxLossAlpha",                0.001f},
    {           "bosTokenId",                100000},
    {           "eosTokenId",                100001},
    {                "epSize",                     1},
    {  "firstKDenseReplace",                     3},
    {             "hiddenAct",                "silu"},
    {      "initializerRange",                 0.02f},
    {      "intermediateSize",                 18432},
    {                "lmHead",                 false},
    {"maxPositionEmbeddings",                  4096},
    {             "modelType",         "deepseek_v3"},
    {  "moeIntermediateSize",                  2048},
    {         "moeLayerFreq",                     1},
    {                "nGroup",                     8},
    {       "nRoutedExperts",                   256},
    {       "nSharedExperts",                     1},
    {         "normTopkProb",                  true},
    {    "numExpertsPerTok",                     8},
    {      "numHiddenLayers",                    61},
    {    "numKeyValueHeads",                   128},
    {         "pretrainingTp",                     1},
    {    "numAttentionHeads",                     2}, // 128 // 4
    {            "hiddenSize",                   256}, // 7168
    {            "qLoraRank",                   512}, // 1536
    {           "kvLoraRank",                   512},
    {       "qkNopeHeadDim",                   128},
    {       "qkRopeHeadDim",                    64},
    {                "rmHead",                 false},
    {           "rmsNormEps",                1e-06f},
    {           "ropeScaling",                     1},
    {             "ropeTheta",                 10000},
    {  "routedScalingFactor",                  2.5f},
    {           "scoringFunc",             "sigmoid"},
    {                "seqAux",                  true},
    {    "tieWordEmbeddings",                 false},
    {             "topkGroup",                     4},
    {            "topkMethod",            "noaux_tc"},
    {            "torchDtype",            "bfloat16"},
    {   "transformersVersion",              "4.33.1"},
    {              "useCache",                  true},
    {             "vHeadDim",                   128},
    {             "vocabSize",                129280},
    {             "fp8Format",                "e4m3"},
    {        "initFp8Params",                  true}
};

static std::map<std::string, std::variant<bool, int, float, std::string>> deepseekConfigMoE = {
    {          "architectures", "DeepseekForCausalLM"},
    {         "attention_bias",                 false},
    {      "attentionDropout",                     0},
    {             "AutoConfig",      "g_deepseekConfig"},
    {              "AutoModel",       "DeepseekModel"},
    {   "AutoModelForCausalLM", "DeepseekForCausalLM"},
    {         "auxLossAlpha",                0.001f},
    {           "bosTokenId",                100000},
    {           "eosTokenId",                100001},
    {                "epSize",                     1},
    {  "firstKDenseReplace",                     3},
    {             "hiddenAct",                "silu"},
    {      "initializerRange",                 0.02f},
    {      "intermediateSize",                  512 },  //18432
    {                "lmHead",                 false},
    {"maxPositionEmbeddings",                  4096},
    {             "modelType",         "deepseek_v3"},
    {  "moeIntermediateSize",                   512},  //2048
    {         "moeLayerFreq",                     1},
    {                "nGroup",                     8},
    {       "nRoutedExperts",                   256},
    {       "nSharedExperts",                     1},
    {         "normTopkProb",                  true},
    {    "numExpertsPerTok",                     8},
    {      "numHiddenLayers",                    61},
    {    "numKeyValueHeads",                   128},
    {         "pretrainingTp",                     1},
    {    "numAttentionHeads",                     2}, // 128 // 4
    {            "hiddenSize",                   256}, // 7168
    {            "qLoraRank",                   512}, // 1536
    {           "kvLoraRank",                   512},
    {       "qkNopeHeadDim",                   128},
    {       "qkRopeHeadDim",                    64},
    {                "rmHead",                 false},
    {           "rmsNormEps",                1e-06f},
    {           "ropeScaling",                     1},
    {             "ropeTheta",                 10000},
    {  "routedScalingFactor",                  2.5f},
    {           "scoringFunc",             "sigmoid"},
    {                "seqAux",                  true},
    {    "tieWordEmbeddings",                 false},
    {             "topkGroup",                     4},
    {            "topkMethod",            "noaux_tc"},
    {            "torchDtype",            "bfloat16"},
    {   "transformersVersion",              "4.33.1"},
    {              "useCache",                  true},
    {             "vHeadDim",                   128},
    {             "vocabSize",                129280},
    {             "fp8Format",                "e4m3"},
    {        "initFp8Params",                  true}
};
