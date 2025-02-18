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
 * \file deepseek_mla.h
 * \brief
 */

#pragma once
#ifndef DEEPSEEK_MLA_H
#define DEEPSEEK_MLA_H

#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

constexpr int SCATTER_UPADATE_DIM = -2;
constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_4 = 4;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_20 = 20;
constexpr int NUM_24 = 24;
constexpr int NUM_32 = 32;
constexpr int NUM_48 = 48;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_384 = 384;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;
constexpr int NUM_1536 = 1536;
constexpr int NUM_1792 = 1792;
constexpr int NUM_4096 = 4096;
constexpr int NUM_6144 = 6144;
constexpr int NUM_8192 = 8192;
constexpr int NUM_7168 = 7168;
constexpr float F_1 = 1.0;
constexpr float F_0 = 0.0;
constexpr float F_NEGA_1 = -1.0;
constexpr double DF_1E_20 = 1e-20;

static std::map<std::string, std::variant<bool, int, float, std::string>> g_deepseekConfig = {
    {          "architectures", "DeepseekForCausalLM"},
    {         "attention_bias",                 false},
    {      "attentionDropout",                     0},
    {             "AutoConfig",      "DeepseekConfig"},
    {              "AutoModel",       "DeepseekModel"},
    {   "AutoModelForCausalLM", "DeepseekForCausalLM"},
    {         "auxLossAlpha",                0.001f},
    {           "bosTokenId",                100000},
    {           "eosTokenId",                100001},
    {                "epSize",                     1},
    {  "firstKDenseReplace",                     3},
    {             "hiddenAct",                "silu"},
    {            "hiddenSize",                  7168},
    {      "initializerRange",                 0.02f},
    {      "intermediateSize",                 18432},
    {           "kvLoraRank",                   512},
    {                "lmHead",                 false},
    {"maxPositionEmbeddings",                  4096},
    {             "modelType",         "deepseek_v3"},
    {  "moeIntermediateSize",                  2048},
    {         "moeLayerFreq",                     1},
    {                "nGroup",                     8},
    {       "nRoutedExperts",                   256},
    {       "nSharedExperts",                     1},
    {         "normTopkProb",                  true},
    {    "numAttentionHeads",                   128},
    {    "numExpertsPerTok",                     8},
    {      "numHiddenLayers",                    61},
    {    "numKeyValueHeads",                   128},
    {         "pretrainingTp",                     1},
    {            "qLoraRank",                  1536},
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

struct AttenTilingData {
    std::vector<int> bmmVec;
    std::vector<int> commonVec;
    int kvLoraRank;
};

struct AttentionW {
    Tensor qAProjW;
    Tensor qBProjW;
    Tensor qBProjWScale;
    Tensor kvAProjWithMqaW;
    Tensor kvBProjWK;
    Tensor kvBProjWV;
    Tensor oProjW;
};

class DeepseekAttention {
public:
    DeepseekAttention(std::map<std::string, std::variant<bool, int, float, std::string>> config,
        AttentionW aw, const int inLayerIdx);
    Tensor Attention(Tensor q, Tensor kv, Tensor attenMask);
    Tensor AttentionPost(Tensor attenRes);
    Tensor AttentionPost2(Tensor attenRes);
    std::tuple<Tensor, Tensor> QkvPre(Tensor hiddenStates);
    std::tuple<Tensor, Tensor> QkvPreCv(Tensor hiddenStates);
    std::vector<Tensor> QkvPre2(Tensor hiddenStates, bool isQuant = false);
    std::tuple<Tensor, Tensor> QkvPreFp32(Tensor hiddenStates);
    Tensor Forward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor>  AtentionPreForward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor>  AtentionPreForwardCv(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor> MlaPrologAbForward(Tensor hiddenStates, Tensor qPeRope, bool isQuant = false);
    std::vector<Tensor> MlaPrologFoward(Tensor hiddenStates, Tensor positionIds,
        Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates,
        const RoPETileShapeConfig &ropeTileShapeConfig, bool isQuant = false);

private:
    int layerIdx = 0;
    int attentionDropout = 0;
    int hiddenSize = 0;
    int numHeads = 0;
    int maxPositionEmbeddings = 0;
    int ropeTheta = 0;
    int qLoraRank = 0;
    int qkRopeHeadDim = 0;
    int kvLoraRank = 0;
    int vHeadDim = 0;
    int qkNopeHeadDim = 0;
    int qHeadDim = 0;
    bool isCausal = true;
    Tensor qAProjW;
    Tensor qBProjW;
    Tensor qBProjWScale;
    Tensor kvAProjWithMqaW;
    Tensor kvBProjWK;
    Tensor kvBProjWV;
    Tensor oProjW;
    float softmaxScale = 0.0f;
};

class DeepseekV2MLP {
public:
    explicit DeepseekV2MLP(std::map<std::string, std::variant<bool, int, float, std::string>> config) {
        hiddenSize = std::get<int>(config["hiddenSize"]);
        intermediateSize = std::get<int>(config["intermediateSize"]);
        gateProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        upProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        downProjW = Tensor(DataType::DT_FP32, {intermediateSize, hiddenSize});
    }

    DeepseekV2MLP(int hs, int is) : hiddenSize(hs), intermediateSize(is) {
        gateProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        upProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        downProjW = Tensor(DataType::DT_FP32, {intermediateSize, hiddenSize});
    }

    Tensor Forward(Tensor x);
    Tensor Forward(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3);
    Tensor ForwardWithQuant(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale);

private:
    int hiddenSize = 0;
    int intermediateSize = 0;
    Tensor gateProjW;
    Tensor upProjW;
    Tensor downProjW;
};

class MoEGate {
public:
    explicit MoEGate(std::map<std::string, std::variant<bool, int, float, std::string>> config) {
        nRoutedExperts = std::get<int>(config["nRoutedExperts"]);
        nGroup = std::get<int>(config["nGroup"]);
        topkGroup = std::get<int>(config["topkGroup"]);
        numExpertsPerTok = std::get<int>(config["numExpertsPerTok"]);

        int hiddenSize = std::get<int>(config["hiddenSize"]);
        std::vector<int64_t> biasShape = {1, nRoutedExperts};
        weight = Tensor(DataType::DT_FP32, {nRoutedExperts, hiddenSize});
        eScoreCorrectionBias = Tensor(DataType::DT_FP32, biasShape, "eScoreCorrectionBias");
    }

    std::tuple<Tensor, Tensor> Forward(const Tensor &hiddenStates);

private:
    int nRoutedExperts = 0;
    int nGroup = 0;
    int topkGroup = 0;
    int numExpertsPerTok = 0;

    Tensor weight;                  // [nRoutedExperts, hiddenSize]
    Tensor eScoreCorrectionBias; // [nRoutedExperts]
};

class DeepseekV2MoE {
public:
    explicit DeepseekV2MoE(std::map<std::string, std::variant<bool, int, float, std::string>> config)
        : expert(std::get<int>(config["hiddenSize"]), std::get<int>(config["moeIntermediateSize"])),
          moeGate(config),
          sharedExpert(std::get<int>(config["hiddenSize"]),
              std::get<int>(config["moeIntermediateSize"]) * std::get<int>(config["nSharedExperts"])) {
        numExpertsPerTok = std::get<int>(config["numExpertsPerTok"]);
        epSize = 1;
        expertsPerRank = std::get<int>(config["nRoutedExperts"]);
        epRank = 0;
    }

    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3, int nRoutedExperts);
    Tensor MoeInferSingleMlp(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3, int nRoutedExperts);
    Tensor MoeInferSingleMlpQuant(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
        Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale, int nRoutedExperts);
    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3,
                        Tensor &idxs, Tensor &sortedTokens, Tensor &outs, int nRoutedExperts);
    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, int nRoutedExperts = 256);
    Tensor Forward(Tensor hiddenStates);

private:
    int numExpertsPerTok = 0;
    int epSize = 0;
    int expertsPerRank = 0;
    int epRank = 0;;
    DeepseekV2MLP expert;
    MoEGate moeGate;
    DeepseekV2MLP sharedExpert;
};

} // namespace npu::tile_fwk

#endif // DEEPSEEK_MLA_H
