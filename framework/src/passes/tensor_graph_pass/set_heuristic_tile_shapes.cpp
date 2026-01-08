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
 * \file set_heuristic_tile_shapes.cpp
 * \brief
 */

#include <climits>
#include <queue>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "set_heuristic_tile_shapes.h"
 
using namespace npu::tile_fwk;
 
namespace npu::tile_fwk {
Status SetHeuristicTileShapes::RunOnFunction(Function &function) {
    SetHeuristicTileShapesFunc(function);
    return SUCCESS;
}
 
const std::unordered_map<DataType, int64_t> Latency{
    {DataType::DT_FP16, 200},
    {DataType::DT_FP32, 200},
    {DataType::DT_INT32, 0},
    {DataType::DT_INT16, 0}
};
 
const std::unordered_map<DataType, int64_t> Parallelism{
    {DataType::DT_FP16, 64}, // 128B/cycle
    {DataType::DT_FP32, 32},
    {DataType::DT_INT32, 32},
    {DataType::DT_INT16, 64}
};

const std::set<Opcode> uniqueOps = { // Ops with possible InputShape != OutputShape
    // Unary ops
    Opcode::OP_INDEX_PUT,
    Opcode::OP_TRANSPOSE_MOVEIN,
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_VNCHWCONV,
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWSUM,
    Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWSUMLINE,
    Opcode::OP_ROWMAXLINE,
    Opcode::OP_ROWMINLINE,
    // Binary ops
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWSUM_SINGLE,
    // Move ops
    Opcode::OP_INDEX_OUTCAST,
    // Logic ops
    Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_RESHAPE
};

const std::set<Opcode> wholeLastDimOps = { // Ops with Tile[lastDim] = Shape[lastDim]
    // Unary ops
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWSUM,
    Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWSUMLINE,
    Opcode::OP_ROWMAXLINE,
    Opcode::OP_ROWMINLINE,
    Opcode::OP_TRANSPOSE_MOVEIN,
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_VNCHWCONV,
    Opcode::OP_INDEX_PUT,
    // Binary ops
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWSUM_SINGLE,
    // Move ops
    Opcode::OP_INDEX_OUTCAST
};

const std::set<Opcode> reduceOps = {
    // Unary ops
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWSUM,
    Opcode::OP_ROWSUMLINE,
    Opcode::OP_ROWMAXLINE,
    Opcode::OP_ROWMINLINE,
    Opcode::OP_ROWEXPMAX,
    Opcode::OP_ROWEXPSUM,
    // Binary ops
    Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,
    Opcode::OP_ROWMAX_SINGLE,
    Opcode::OP_ROWMIN_SINGLE,
    Opcode::OP_ROWSUM_SINGLE
};

const std::set<Opcode> transposeOps = {
    Opcode::OP_TRANSPOSE_MOVEIN,
    Opcode::OP_TRANSPOSE_MOVEOUT,
    Opcode::OP_TRANSPOSE_VNCHWCONV
};

uint64_t GetLatency(DataType dtype) {
    auto iterDtype = Latency.find(dtype);
    if (iterDtype == Latency.end()) {
        return DEFAULT_LATENCY;
    }
    return iterDtype->second;
}
 
uint64_t GetParallelism(DataType dtype) {
    auto iterDtype = Parallelism.find(dtype);
    if (iterDtype == Parallelism.end()) {
        return DEFAULT_MAX_PARALLELISM;
    }
    return iterDtype->second;
}
 
bool IsFloat(const std::shared_ptr<LogicalTensor> tensor) {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_FP16) || (dataType == DT_FP32) || (dataType == DT_BF16)) {
        return true;
    }
    return false;
}
 
double CalculateGeometricMean(const std::vector<double>& vectorRatio) {
    double product = 1.0;
    for (double num : vectorRatio) {
        product *= num;
    }
    return std::pow(product, 1.0 / vectorRatio.size());
}
 
std::map<int, int> FordBellman(const std::vector<std::pair<int, int>> &edges, Function &function) {
    std::map<int, int> subgrDepthMap;
    for (auto &op : function.Operations()) {
        subgrDepthMap[op.GetOpMagic()] = INT_MAX;
    }
    subgrDepthMap[-1] = 0;
    
    bool any = true;
    while (any == true) {
        any = false;
        for (auto elem: edges) {
            if (subgrDepthMap[elem.second] > subgrDepthMap[elem.first] - 1) {
                subgrDepthMap[elem.second] = subgrDepthMap[elem.first] - 1;
                any = true;
            }
        }
    }
    return subgrDepthMap;
}

void FindCubeTilesCombinations(std::map<std::vector<int64_t>, double>& setOfCubeTiles, int64_t m, int64_t k, int64_t n, int64_t inputTypeSize, int64_t outputTypeSize) {
    // Platform params
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    std::vector<int64_t> tmpTile = {0,0,0}; // m,k,n
    if (((m * k * inputTypeSize) <= (L0A_MAX_SIZE / DOUBLE_BUFFER)) && ((k * n * inputTypeSize) <= (L0B_MAX_SIZE / DOUBLE_BUFFER)) && ((m * n * outputTypeSize) <= (L0C_MAX_SIZE / DOUBLE_BUFFER))) {
        tmpTile[M_DIM] = m;
        tmpTile[K_DIM] = k;
        tmpTile[N_DIM] = n;
        setOfCubeTiles[tmpTile] = 0.f; // set initial score = 0
    }
}

void FindScoreForCubeTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles,
                           int64_t cubeL1ReuseMode, int64_t cubeNBuffer, int64_t numOfMatmuls) {
    // Platform params
    const int64_t L0A_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    const int64_t L0B_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    const int64_t L0C_MAX_SIZE = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    const int64_t CUBE_CORES = Platform::Instance().GetSoc().GetAICoreNum();

    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
 
    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
 
    uint64_t inputMKN = std::max(M, MIN_TILE) * std::max(K, MIN_TILE) * std::max(N, MIN_TILE);
    std::vector<double> vectorRatio = {0.f, 0.f, 0.f};
    for (auto & [tile, score] : setOfCubeTiles) {
        uint64_t mkn = tile[M_DIM] * tile[K_DIM] * tile[N_DIM];
 
        // If the tiling size = shape size -> the preferred option
        score = (tile[M_DIM] == std::max(M, MIN_TILE)) ? (score + WHOLE_M_SCORE) : score;
        score = (tile[K_DIM] == std::max(K, MIN_TILE)) ? (score + WHOLE_K_SCORE) : score;
        score = (tile[N_DIM] == std::max(N, MIN_TILE)) ? (score + WHOLE_N_SCORE) : score;
 
        // The more filled L0A, L0B, L0C is better
        double utilizationL0A = static_cast<double>((tile[M_DIM] * tile[K_DIM] * inputTypeSize)) / (L0A_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0B = static_cast<double>((tile[K_DIM] * tile[N_DIM] * inputTypeSize)) / (L0B_MAX_SIZE / DOUBLE_BUFFER);
        double utilizationL0C = static_cast<double>((tile[M_DIM] * tile[N_DIM] * outputTypeSize)) / (L0C_MAX_SIZE / DOUBLE_BUFFER);
        vectorRatio = {utilizationL0A, utilizationL0B, utilizationL0C};
        double geomeanUtilizationL0 = CalculateGeometricMean(vectorRatio);
        score += WEIGHT_L0 * geomeanUtilizationL0;
 
        // The closer the tasksRatio is to 1, the better
        double tasks = numOfMatmuls * (std::max(M, MIN_TILE) / static_cast<double>(tile[M_DIM])) * (std::max(N, MIN_TILE) / static_cast<double>(tile[N_DIM])) / (cubeL1ReuseMode * cubeNBuffer);
        double tasksRatioLess = (tasks < CUBE_CORES) ? (CUBE_CORES / tasks  - 1) : 0;
        double tasksRatioMore = (tasks > 2 * CUBE_CORES) ? (tasks / (2 * CUBE_CORES) - 1) : 0;
 
        // Penalty for tasks < CUBE_CORES & tasks > 2 * CUBE_CORES
        score -= TASKS_CUBE_WEIGHT * (tasksRatioLess + tasksRatioMore);
 
        // The more residualTasks the better
        int64_t residualTasks = (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES) == 0) ? static_cast<int64_t>(CUBE_CORES) : (static_cast<int64_t>(std::ceil(tasks)) % static_cast<int64_t>(CUBE_CORES));
        score += RESIDUAL_CUBE_TASKS_WEIGHT * residualTasks;
 
        //The closer the ratio's is to 1, the better
        double ratioMK = (tile[M_DIM] > tile[K_DIM]) ? static_cast<double>((tile[M_DIM] / tile[K_DIM])) : static_cast<double>((tile[K_DIM] / tile[M_DIM]));
        double ratioKN = (tile[K_DIM] > tile[N_DIM]) ? static_cast<double>((tile[K_DIM] / tile[N_DIM])) : static_cast<double>((tile[N_DIM] / tile[K_DIM]));
        double ratioMN = (tile[M_DIM] > tile[N_DIM]) ? static_cast<double>((tile[M_DIM] / tile[N_DIM])) : static_cast<double>((tile[N_DIM] / tile[M_DIM]));
        vectorRatio = {ratioMK, ratioKN, ratioMN};
        double ratioMKN = CalculateGeometricMean(vectorRatio);
 
        // Penalty for bad balance
        score -= BALANCE_WEIGHT * ratioMKN;
 
        // Consider num of L1CopyIn cycles
        uint64_t numL1CopyInL1A = inputMKN / (mkn * cubeL1ReuseMode * cubeNBuffer); // Num of L1CopyIn instructions for A
        uint64_t numL1CopyInL1B = inputMKN / mkn; // Num of L1CopyIn instructions for B
 
        uint64_t elePerRepeat = BYTES_PER_REPEAT / BytesOf(inputType);
        uint64_t parallelism = GetParallelism(inputType) == 0 ? 1 : GetParallelism(inputType);
        uint64_t cyclePerRepeat = elePerRepeat / parallelism;
        uint64_t latency = GetLatency(inputType);
 
        uint64_t repeatCountL1A = (tile[M_DIM] * tile[K_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;
        uint64_t repeatCountL1B = (tile[K_DIM] * tile[N_DIM] * inputTypeSize - BYTES_PER_REPEAT) / BYTES_PER_REPEAT + 1;
 
        uint64_t cyclesL1A = numL1CopyInL1A * (latency + cyclePerRepeat * repeatCountL1A - 1);
        uint64_t cyclesL1B = numL1CopyInL1B * (latency + cyclePerRepeat * repeatCountL1B - 1);
 
        // Overall cycles of L1CopyIn for {m, k, n} tiles (the less the better)
        double cyclesLog = std::log2(cyclesL1A + cyclesL1B);
 
        // Penalty for large num of cycles
        score -= CYCLES_WEIGHT * cyclesLog;
    }
}
 
void SetPossibleCubeTiles(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, std::map<std::vector<int64_t>, double>& setOfCubeTiles) {
    // Input shapes
    int64_t M = shapeAndTypeInfo.first[M_DIM];
    int64_t K = shapeAndTypeInfo.first[K_DIM];
    int64_t N = shapeAndTypeInfo.first[N_DIM];
 
    // Input types
    DataType inputType = shapeAndTypeInfo.second[M_DIM];
    DataType outputType = shapeAndTypeInfo.second[K_DIM];
    int64_t inputTypeSize = BytesOf(inputType);
    int64_t outputTypeSize = BytesOf(outputType);
 
    int64_t newM = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(M, MIN_TILE))))));
    int64_t newK = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(K, MIN_TILE))))));
    int64_t newN = static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::ceil(std::log2(std::max(N, MIN_TILE))))));
 
    for (int64_t m = MIN_TILE; m <= newM; m *= FACTOR) {
        m = m > std::max(M, MIN_TILE) ? std::max(M, MIN_TILE) : m;
        for (int64_t k = MIN_TILE; k <= newK; k *= FACTOR) {
            k = k > std::max(K, MIN_TILE) ? std::max(K, MIN_TILE) : k;
            for (int64_t n = MIN_TILE; n <= newN; n *= FACTOR) {
                n = n > std::max(N, MIN_TILE) ? std::max(N, MIN_TILE) : n;
                FindCubeTilesCombinations(setOfCubeTiles, m, k, n, inputTypeSize, outputTypeSize);
            }
        }
    }
}
 
std::vector<int64_t> FindAndSetCubeTileShapes(std::pair<std::vector<int64_t>, std::vector<DataType>> shapeAndTypeInfo, int64_t numOfMatmuls, int64_t cubeL1ReuseMode, int64_t cubeNBuffer) {
    // Define set of possible cube tiles
    std::map<std::vector<int64_t>, double> setOfCubeTiles;
    SetPossibleCubeTiles(shapeAndTypeInfo, setOfCubeTiles);
 
    // Find score for each set of cube tiles
    FindScoreForCubeTiles(shapeAndTypeInfo, setOfCubeTiles, cubeL1ReuseMode, cubeNBuffer, numOfMatmuls);
 
    // Find set of tiles with max Score
    double maxScore = -std::numeric_limits<double>::max();
    int64_t mFinal = 0;
    int64_t kFinal = 0;
    int64_t nFinal = 0;
    for (auto & [tile, score] : setOfCubeTiles) {
        if (maxScore < score) {
            mFinal = tile[M_DIM];
            kFinal = tile[K_DIM];
            nFinal = tile[N_DIM];
            maxScore = score;
        }
    }
 
    std::vector<int64_t> resultCubeTiles;
    resultCubeTiles.push_back(mFinal);
    resultCubeTiles.push_back(kFinal);
    resultCubeTiles.push_back(nFinal);
    return resultCubeTiles;
}

size_t DimsCalculation(Operation *op, size_t tensorsNum, bool isInput) {
    size_t tensorDims = 0;
    for (size_t tensor = 0; tensor < tensorsNum; tensor++) {
        if (isInput) {
            tensorDims = std::max(tensorDims, op->GetIOperands()[tensor]->shape.size());
        } else {
            tensorDims = std::max(tensorDims, op->GetOOperands()[tensor]->shape.size());
        }
    }
    return tensorDims;
}

std::vector<int64_t> MaxInputShapeCalculation(Operation *op, size_t inputsNum, size_t inputDims, int64_t maxTypeSize) {
    // Find the maximum values of the shape dimensions among the inputs, to find the maximum boundary of the tile values
    if (maxTypeSize == 0) {
        ALOG_ERROR_F("maxTypeSize = 0, division by zero");
    }
    std::vector<int64_t> maxInputShape(inputDims, LLONG_MIN);
    for (size_t input = 0; input < inputsNum; input++) {
        for (size_t inputDim = 0; inputDim < op->GetIOperands()[input]->shape.size(); inputDim++) {
            maxInputShape[inputDim] = std::max(maxInputShape[inputDim], op->GetIOperands()[input]->shape[inputDim]);
        }
    }
    maxInputShape[inputDims - 1] = (maxInputShape[inputDims - 1] == -1) ? -1 : maxInputShape[inputDims - 1];
    return maxInputShape;
}

void VcnhwconvProcessing(Operation *opInit, Operation *opNew, std::vector<int64_t> &vectorTilesNew, const std::vector<int64_t> maxInputShape,
                         std::vector<int> perm, int64_t maxTypeSize, int64_t tileSize, bool isForward, bool isTranspose) {
    bool backwardTranspose = isTranspose ? (opNew->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV && !isForward) : (opInit->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV && !isForward);
    bool forwardTranspose = (opNew->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV && isForward);
    if (backwardTranspose || forwardTranspose) {
        if (tileSize == 0) {
            ALOG_ERROR_F("tileSize = 0, division by zero");
        }
        int64_t smallDim = (maxInputShape[perm[0]] < maxInputShape[perm[1]]) ? perm[0] : perm[1];
        int64_t bigDim = (maxInputShape[perm[0]] > maxInputShape[perm[1]]) ? perm[0] : perm[1];
        vectorTilesNew[bigDim] = std::max(VNCHWCONV_POINTERS, vectorTilesNew[bigDim]);
        int64_t smallDimSize = BLOCK_SIZE / (maxTypeSize * vectorTilesNew[smallDim]);
        int64_t newTileSize = std::accumulate(vectorTilesNew.begin(), vectorTilesNew.end(), 1, std::multiplies<int64_t>());
        newTileSize *= std::max(smallDimSize, static_cast<int64_t>(1));
        int64_t dimRatio = newTileSize / tileSize;
        if (dimRatio > 1) {
            for (size_t dim = 0; dim < vectorTilesNew.size(); dim++) {
                int64_t curRatio = std::min(dimRatio, vectorTilesNew[dim]);
                vectorTilesNew[dim] = (curRatio > 1) ? vectorTilesNew[dim] / curRatio : vectorTilesNew[dim];
                dimRatio /= curRatio;
            }
        }
    }
}

void TransposeTileSetting(Operation *opInit, Operation *opBase, Operation *opNew, const std::vector<int64_t> vectorTilesOld, const std::vector<int64_t> maxInputShape,
                          std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS, int64_t maxTypeSize, int64_t tileSize, bool isForward) {
    auto perm = opBase->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    std::vector<int64_t> vectorTilesNew = vectorTilesOld;
    if (opBase->GetOpcode() != Opcode::OP_TRANSPOSE_VNCHWCONV) {
        std::swap(vectorTilesNew[perm[0]], vectorTilesNew[perm[1]]);
    } else {
        std::swap(vectorTilesNew[perm[0]], vectorTilesNew[perm[1]]);
        VcnhwconvProcessing(opInit, opNew, vectorTilesNew, maxInputShape, perm, maxTypeSize, tileSize, isForward, true);
    }
    opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
    if (!visitedBFS[opNew->GetOpMagic()]) {
        queueBFS.push(opNew);
    }
    visitedBFS[opNew->GetOpMagic()] = true;
}

void ReshapeInfoFilling(std::vector<int64_t> &reshapeInfo, std::vector<int64_t> opBaseInputShape, std::vector<int64_t> opBaseOutputShape) {
    for (int64_t i = 0; i < static_cast<int64_t>(reshapeInfo.size()); i++) {
        if (!((static_cast<int64_t>(opBaseInputShape.size() - i - 1) >= 0) && (static_cast<int64_t>(opBaseOutputShape.size() - i - 1) >= 0) && (opBaseInputShape[opBaseInputShape.size() - i - 1] == opBaseOutputShape[opBaseOutputShape.size() - i - 1]))) {
            reshapeInfo[reshapeInfo.size() - i - 1] = 0;
        }
    }
    bool isZero = std::all_of(reshapeInfo.begin(), reshapeInfo.end(), [](int64_t i){ return i == 0; });
    if (isZero) {
        std::fill(reshapeInfo.begin(), reshapeInfo.end(), 1);
        for (int64_t i = 0; i < static_cast<int64_t>(reshapeInfo.size()); i++) {
            if (!((static_cast<int64_t>(opBaseInputShape.size() - i - 1) >= 0) && (static_cast<int64_t>(opBaseOutputShape.size() - i - 1) >= 0) && (opBaseInputShape[i] == opBaseOutputShape[i]))) {
                reshapeInfo[i] = 0;
            }
        }
    }
}

void ReshapeTileSetting(Operation *opInit, Operation *opBase, Operation *opNew, std::vector<int64_t> vectorTilesOld, const std::vector<int64_t> maxInputShape,
                        std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS, size_t inputDims, int64_t maxTypeSize, int64_t tileSize, bool isForward) {
    // Fill reshapeInfo
    int64_t initTileSize = tileSize;
    std::vector<int64_t> opBaseInputShape = opBase->GetIOperands()[0]->shape;
    std::vector<int64_t> opBaseOutputShape = opBase->GetOOperands()[0]->shape;
    std::vector<int64_t> reshapeInfo(std::max(opBaseInputShape.size(), opBaseOutputShape.size()), 1);
    ReshapeInfoFilling(reshapeInfo, opBaseInputShape, opBaseOutputShape);
    
    std::vector<int64_t> vectorTilesNew(inputDims, -1);
    if (reshapeInfo[reshapeInfo.size() - 1] == 1) {
        for (size_t i = 0; i < reshapeInfo.size(); i++) {
            if ((reshapeInfo[reshapeInfo.size() - i - 1] == 1) && (static_cast<int64_t>(vectorTilesOld.size() - i - 1) >= 0) && (static_cast<int64_t>(vectorTilesNew.size() - i - 1) >= 0)) {
                vectorTilesNew[vectorTilesNew.size() - i - 1] = vectorTilesOld[vectorTilesOld.size() - i - 1];
                tileSize /= vectorTilesOld[vectorTilesOld.size() - i - 1];
            }
        }
    } else {
        for (size_t i = 0; i < reshapeInfo.size(); i++) {
            if ((reshapeInfo[i] == 1) && (static_cast<int64_t>(vectorTilesOld.size() - i - 1) >= 0) && (static_cast<int64_t>(vectorTilesNew.size() - i - 1) >= 0)) {
                vectorTilesNew[i] = vectorTilesOld[i];
                tileSize /= vectorTilesOld[i];
            }
        }
    }
    
    for (size_t dim = 0; dim < inputDims; dim++) {
        if (vectorTilesNew[inputDims - 1 - dim] != -1) {
            continue;
        }
        int64_t curTile = std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(tileSize)))), static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(maxInputShape[inputDims - 1 - dim])))));
        curTile = ((maxInputShape[inputDims - 1 - dim] > curTile) && (dim == 0)) ? std::gcd(curTile, maxInputShape[inputDims - 1 - dim]) : curTile;
        curTile = (curTile != 0) ? curTile : 1;
        vectorTilesNew[inputDims - 1 - dim] = curTile;
        tileSize /= curTile;
    }
    
    auto perm = isForward ? opNew->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape") : opInit->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    VcnhwconvProcessing(opInit, opNew, vectorTilesNew, maxInputShape, perm, maxTypeSize, initTileSize, isForward, false);
    opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);

    if (!visitedBFS[opNew->GetOpMagic()]) {      
        queueBFS.push(opNew);
    }  
    visitedBFS[opNew->GetOpMagic()] = true;
}

int64_t TileSizeCalculation(Operation *op, std::vector<int64_t> vectorTilesOld, int64_t maxTypeSize) {
    if (maxTypeSize == 0) {
        ALOG_ERROR_F("maxTypeSize = 0, division by zero");
    }
    int64_t tileSize = std::accumulate(vectorTilesOld.begin(), vectorTilesOld.end(), 1, std::multiplies<int64_t>());
    tileSize = (op->GetOpcode() == Opcode::OP_TRANSPOSE_VNCHWCONV) ? tileSize * (BLOCK_SIZE / maxTypeSize) : tileSize;
    while (tileSize < MIN_TILE_SIZE) {
        tileSize *= NUM2;
    }
    while (tileSize > (MAX_TILE_SIZE / maxTypeSize)) {
        tileSize /= NUM2;
    }
    return tileSize;
}

void TileShapeSetting(Operation *opBase, Operation *opNew, std::vector<int64_t> vectorTilesOld, std::vector<int64_t> maxInputShape, int64_t tileSize, int64_t inputTypeSize, size_t inputDims, size_t outputDims) {
    std::vector<int64_t> vectorTilesNew;
    auto iterLastDim = wholeLastDimOps.find(opBase->GetOpcode());
    auto iterRowDim = reduceOps.find(opBase->GetOpcode());
    auto iterUniqueDim = uniqueOps.find(opBase->GetOpcode());
    if (inputTypeSize == 0) {
        ALOG_ERROR_F("inputTypeSize = 0, division by zero");
    }

    // Last Dim processing
    int64_t curTile = (iterUniqueDim != uniqueOps.end()) ? std::max(std::max(vectorTilesOld[outputDims - 1], BLOCK_SIZE / inputTypeSize), static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(std::min(maxInputShape[inputDims - 1], BYTES_PER_REPEAT / inputTypeSize))))))
                                                         : vectorTilesOld[outputDims - 1]; // For ops with InputShape = OutputShape just set previous tileShape
    curTile = (maxInputShape[inputDims - 1] > curTile) ? std::max(std::gcd(curTile, maxInputShape[inputDims - 1]), BLOCK_SIZE / inputTypeSize) : curTile; // GCD for VIEW and ASSEMBLE ops 
    curTile = (iterLastDim != wholeLastDimOps.end()) ? std::max(std::max(curTile, maxInputShape[inputDims - 1]), BLOCK_SIZE / inputTypeSize) : curTile; // TileShape[lastDim] = InputShape[lastDim]
    curTile = ((iterRowDim != reduceOps.end()) && (maxInputShape[inputDims - 1] >= (UINT8MAX * BLOCK_SIZE / inputTypeSize))) ? std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(UINT8MAX * BLOCK_SIZE / inputTypeSize)))), curTile) : curTile; // Consider additional restriction for REDUCE ops
    curTile = std::min(tileSize, curTile); // UB overflow condition
    curTile = (curTile != 0) ? curTile : 1;
    vectorTilesNew.push_back(curTile);
    tileSize /= curTile;

    // Other Dims processing
    for (size_t dim = 1; dim < inputDims; dim++) {
        curTile = (iterUniqueDim != uniqueOps.end()) ? ((maxInputShape[inputDims - 1 - dim] != -1) ? std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(tileSize)))), static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(maxInputShape[inputDims - 1 - dim])))))
                                                     : static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(tileSize))))) : std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(tileSize)))), vectorTilesOld[outputDims - 1 - dim]); // For ops with InputShape = OutputShape just set previous tileShape
        curTile = (curTile != 0) ? curTile : 1;
        vectorTilesNew.push_back(curTile);
        tileSize /= curTile;
    }
    if (wholeLastDimOps.find(opNew->GetOpcode()) == wholeLastDimOps.end()) {
        vectorTilesNew[0] *= tileSize;
    } else if (wholeLastDimOps.find(opNew->GetOpcode()) != wholeLastDimOps.end()){
        if (inputDims != 1) {
            vectorTilesNew[inputDims - 1] *= tileSize;
        }
    }
    std::reverse(vectorTilesNew.begin(), vectorTilesNew.end());
    opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
}

void CubeDepsProcessing(Operation *cubeOp, Operation *opInit, Operation *opBase, Operation *opNew, bool isFirst, bool isSecond) {
    std::vector<int64_t> vectorTilesCube;
    auto &cubeTile = cubeOp->GetTileShape().GetCubeTile();
    int magicA = cubeOp->GetIOperands()[0]->magic;
    int magicB = cubeOp->GetIOperands()[1]->magic;
    int magicC = cubeOp->GetOOperands()[0]->magic;
    
    std::vector<int64_t> vectorTilesA = {cubeTile.m[0], cubeTile.k[0]};
    std::vector<int64_t> vectorTilesB = {cubeTile.k[0], cubeTile.n[0]};
    std::vector<int64_t> vectorTilesC = {cubeTile.m[0], cubeTile.n[0]};
    
    if (isFirst) {
        if ((opBase->GetOOperands()[0]->magic == magicA) || (opBase->GetOOperands()[0]->magic == magicB)) {
            if (opInit->GetOpcodeStr() == "At_MUL_B" || opInit->GetOpcodeStr() == "At_MUL_Bt") {
                std::reverse(vectorTilesA.begin(), vectorTilesA.end());
            }
            if (opInit->GetOpcodeStr() == "A_MUL_Bt" || opInit->GetOpcodeStr() == "At_MUL_Bt") {
                std::reverse(vectorTilesB.begin(), vectorTilesB.end());
            }
        
            vectorTilesCube = (vectorTilesA[1] > vectorTilesB[1]) ? vectorTilesA : vectorTilesB;
            opInit->GetTileShapeForSetting().SetVecTile(vectorTilesCube); // Set vector tiles for Matmuls
        }
    }
    if (isSecond) {
        if (opNew->GetIOperands()[0]->magic == magicC) {
            vectorTilesCube = vectorTilesC;
            opInit->GetTileShapeForSetting().SetVecTile(vectorTilesCube); // Set vector tiles for Matmuls
        }         
    }
}

bool Propagation(Operation *cubeOp, Operation *opInit, Operation *opBase, Operation *opNew, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS, bool isFirst, bool isSecond, bool isReduce, bool isForward) {
    // Direct cube dependencies proccesing
    if (isFirst || isSecond) {
        CubeDepsProcessing(cubeOp, opInit, opBase, opNew, isFirst, isSecond);
    }

    size_t inputsNum = opNew->GetIOperands().size();
    size_t outputsNum = (isForward) ? opBase->GetIOperands().size() : opBase->GetOOperands().size(); 
    size_t inputDims = DimsCalculation(opNew, inputsNum, true);
    size_t outputDims = DimsCalculation(opBase, outputsNum, isForward);             
    int64_t inputTypeSize = BytesOf(opNew->GetIOperands()[0]->tensor->GetDataType());
    int64_t outputTypeSize = BytesOf(opNew->GetOOperands()[0]->tensor->GetDataType());
    int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
    std::vector<int64_t> vectorTilesOld = opInit->GetTileShape().GetVecTile().tile;
    if (inputTypeSize == 0 || maxTypeSize == 0) {
        ALOG_ERROR_F("typeSize = 0, division by zero");
    }
    if (isReduce) {
        if (reduceOps.find(opNew->GetOpcode()) != reduceOps.end()) {
            return true;
        }
    }

    std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(opNew, inputsNum, inputDims, maxTypeSize);
    int64_t tileSize = TileSizeCalculation(opNew, vectorTilesOld, maxTypeSize);
    auto iterTransposeDim = transposeOps.find(opBase->GetOpcode());
    if (iterTransposeDim != transposeOps.end()) {
        TransposeTileSetting(opInit, opBase, opNew, vectorTilesOld, maxInputShape, queueBFS, visitedBFS, maxTypeSize, tileSize, isForward);
        return true;
    }
    
    if (opBase->GetOpcode() == Opcode::OP_RESHAPE) {
        ReshapeTileSetting(opInit, opBase, opNew, vectorTilesOld, maxInputShape, queueBFS, visitedBFS, inputDims, maxTypeSize, tileSize, isForward);
        return true;
    }
    
    if (isReduce) {
        // Broadcast tiles from Reduce op directly
        auto iterRowDim = reduceOps.find(opBase->GetOpcode());
        if (iterRowDim != reduceOps.end()) {
            opNew->GetTileShapeForSetting().SetVecTile(vectorTilesOld);
            if (!visitedBFS[opNew->GetOpMagic()]) {      
                queueBFS.push(opNew);
            }  
            visitedBFS[opNew->GetOpMagic()] = true;
            return true;
        }
    }

    // Calculate and set tiles
    TileShapeSetting(opBase, opNew, vectorTilesOld, maxInputShape, tileSize, inputTypeSize, inputDims, outputDims);
    return false;
}

std::pair<std::vector<int64_t>, std::vector<DataType>> ShapeAndTypeSetting(Operation *op, int64_t &shapeM, int64_t &shapeK, int64_t &shapeN) {
    if (op->GetOpcodeStr() == "A_MUL_B") { // [m, k] * [k, n]
        shapeM = op->GetIOperands()[0]->shape[0];
        shapeK = op->GetIOperands()[0]->shape[1];
        shapeN = op->GetIOperands()[1]->shape[1];
    }
    if (op->GetOpcodeStr() == "A_MUL_Bt") { // [m, k] * [n, k]
        shapeM = op->GetIOperands()[0]->shape[0];
        shapeK = op->GetIOperands()[0]->shape[1];
        shapeN = op->GetIOperands()[1]->shape[0];
    }
    if (op->GetOpcodeStr() == "At_MUL_B") { // [k, m] * [k, n]
        shapeM = op->GetIOperands()[0]->shape[1];
        shapeK = op->GetIOperands()[0]->shape[0];
        shapeN = op->GetIOperands()[1]->shape[1];
    }
    if (op->GetOpcodeStr() == "At_MUL_Bt") { // [k, m] * [n, k]
        shapeM = op->GetIOperands()[0]->shape[1];
        shapeK = op->GetIOperands()[0]->shape[0];
        shapeN = op->GetIOperands()[1]->shape[0];
    }
    DataType inputType = op->GetIOperands()[0]->tensor->GetDataType();
    DataType outputType = (IsFloat(op->GetOOperands()[0])) ? DataType::DT_FP32 : DataType::DT_INT32;
    return {{shapeM, shapeK, shapeN}, {inputType, outputType}};
}

void UniqueTilesFilling(Function &function, std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, int64_t> &uniqueTiles,
                        std::pair<std::vector<int64_t>, std::vector<DataType>> &curShapeAndType, int64_t &shapeM, int64_t &shapeK, int64_t &shapeN) {
    for (auto &op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC") {
            // Calculate ShapeAndType for each operation
            curShapeAndType = ShapeAndTypeSetting(&op, shapeM, shapeN, shapeK);

            // Find set of tiles by key
            auto it = uniqueTiles.find(curShapeAndType);
            if (it != uniqueTiles.end()) {
                uniqueTiles[curShapeAndType]++;
            } else {
                uniqueTiles[curShapeAndType] = 1;
            }
        }
    }
}

void SetHeuristicCubeTiles(Function &function, std::unordered_set<Operation *> cubeOperations) {
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, int64_t> uniqueTiles;
    std::pair<std::vector<int64_t>, std::vector<DataType>> curShapeAndType = {{0, 0, 0}, {DataType::DT_FP16, DataType::DT_FP16}}; // shapeM, shapeK, shapeN, InputType, OutputType
    
    int64_t cubeL1ReuseMode = (function.paramConfigs_.l1ReuseNum == 0) ? 1 : function.paramConfigs_.l1ReuseNum;
    int64_t cubeNBuffer = (function.paramConfigs_.cubeNBufferSetting.size() == 1 && function.paramConfigs_.cubeNBufferSetting.begin()->first == -1) ? 1 : function.paramConfigs_.cubeNBufferSetting.begin()->second;
    int64_t shapeM = 0, shapeK = 0, shapeN = 0;
    UniqueTilesFilling(function, uniqueTiles, curShapeAndType, shapeM, shapeK, shapeN);
 
    // Find and set heuristic cube tile shapes
    std::map<std::pair<std::vector<int64_t>, std::vector<DataType>>, std::vector<int64_t>> resultCubeTilesAndInfo;
    for (auto & [shapeAndTypeInfo, numOfMatmuls] : uniqueTiles) {
        std::vector<int64_t> resultCubeTiles = FindAndSetCubeTileShapes(shapeAndTypeInfo, numOfMatmuls, cubeL1ReuseMode, cubeNBuffer);
        resultCubeTilesAndInfo[shapeAndTypeInfo] = resultCubeTiles;
    }
 
    std::array<int64_t, MAX_MDIM> m = {0, 0};
    std::array<int64_t, MAX_KDIM> k = {0, 0, 0};
    std::array<int64_t, MAX_NDIM> n = {0, 0};
 
    for (auto &op : cubeOperations) {
        // Calculate ShapeAndType for each operation
        curShapeAndType = ShapeAndTypeSetting(op, shapeM, shapeN, shapeK);
 
        m[0] = resultCubeTilesAndInfo[curShapeAndType][M_DIM];
        k[0] = resultCubeTilesAndInfo[curShapeAndType][K_DIM];
        n[0] = resultCubeTilesAndInfo[curShapeAndType][N_DIM];
 
        // The algorithm calculates tiles for L0, let's assume that tiles for L1 are the same
        m[1] = m[0];
        k[1] = k[0];
        k[MAX_KDIM - 1] = k[0];
        n[1] = n[0];

        // Set new tiles for each operation (M = m, N = n, K = k, enableMultiDataLoad = true)
        op->GetTileShapeForSetting().SetCubeTile(m, k, n, true);
    }
}

std::vector<Operation *> FordBellman(Function &function, std::unordered_set<Operation *> cubeOperations, std::map<int, int> &subgrDepthMap) {
    // Create edges
    std::vector<std::pair<int, int>> edges;
    for (auto &op : function.Operations()) {
        for (auto consumerOp : op.ConsumerOps()) {
            edges.push_back({consumerOp->GetOpMagic(), op.GetOpMagic()});
        }
    }
    
    // Create lastVertices
    std::vector<int> lastVertices;
    for (auto &op : function.Operations()) {
        if (op.ConsumerOps().size() == 0 && op.ProducerOps().size() != 0) {
            lastVertices.push_back(op.GetOpMagic());
        }
    }
    
    // Define depth for each node using FordBellman algorithm
    for (size_t idx = 0; idx < lastVertices.size(); idx++) {
        edges.push_back({-1, lastVertices[idx]});
    }
    auto d = FordBellman(edges, function);
    for (auto& [magic, depth] : d) {
        subgrDepthMap[magic] = std::max(subgrDepthMap[magic], -depth - 1);
    }
    subgrDepthMap.erase(-1);
    edges.clear();
    lastVertices.clear();
    
    // Sort cube operations by depth
    std::vector<std::pair<Operation *, int>> cubeTmpOperations;
    for (auto cubeOp : cubeOperations) {
        cubeTmpOperations.push_back(std::make_pair(cubeOp, subgrDepthMap[cubeOp->GetOpMagic()]));
    }
    std::sort(cubeTmpOperations.begin(), cubeTmpOperations.end(), [](const std::pair<Operation *, int> &x, const std::pair<Operation *, int> &y) {return x.second < y.second;});
    
    std::vector<Operation *> cubeOrderedOperations;
    for (auto op : cubeTmpOperations) {
        cubeOrderedOperations.push_back(op.first);
    }
    return cubeOrderedOperations;
}

bool DuplicateTileSetting(Operation *opInit, Operation *opNew, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    std::vector<int64_t> vectorTilesNew;
    if (opNew->GetIOperands().size() == 0) {
        vectorTilesNew = opInit->GetTileShape().GetVecTile().tile;
        opNew->GetTileShapeForSetting().SetVecTile(vectorTilesNew);
        if (!visitedBFS[opNew->GetOpMagic()]) {      
            queueBFS.push(opNew);
        }  
        visitedBFS[opNew->GetOpMagic()] = true;
        return true;
    }
    return false;
}

void UpdateBFS(Operation *op, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    if (!visitedBFS[op->GetOpMagic()]) {      
        queueBFS.push(op);
    }  
    visitedBFS[op->GetOpMagic()] = true;
}

void BackwardCubePropagation(std::vector<Operation *> cubeOrderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto cubeOp : cubeOrderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedBFS);
                if (isContinue) {
                    continue;
                }

                // Call propagation
                isContinue = Propagation(cubeOp, op, producerOp, producerOp, queueBFS, visitedBFS, true, false, false, false);
                if (isContinue) {
                    continue;
                }
                
                // Update queueBFS and visitedBFS
                UpdateBFS(producerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void ForwardCubePropagation(std::vector<Operation *> cubeOrderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto cubeOp : cubeOrderedOperations) {
        queueBFS.push(cubeOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto consumerOp : op->ConsumerOps()) {
                bool isVisitedNode = (consumerOp->GetTileShape().GetVecTile()[0] != -1);
                if (isVisitedNode) {
                    // Update queueBFS and visitedBFS
                    UpdateBFS(consumerOp, queueBFS, visitedBFS);
                    continue;
                }

                // Call propagation
                bool isContinue = Propagation(cubeOp, op, op, consumerOp, queueBFS, visitedBFS, false, true, false, true);
                if (isContinue) {
                    continue;
                }
                
                // Update queueBFS and visitedBFS
                UpdateBFS(consumerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

std::vector<Operation *> FillNoConsumersOperations(Function &function) {
    std::vector<Operation *> noConsumersOperations;
    std::vector<int64_t> vectorTilesNew;
    for (auto &op : function.Operations()) {
        if (op.ConsumerOps().size() == 0) {
            vectorTilesNew.clear();
            size_t inputDims = op.GetIOperands()[0]->shape.size();
            int64_t inputTypeSize = BytesOf(op.GetIOperands()[0]->tensor->GetDataType());
            int64_t defaultTileSize = DEFAULT_TILE_SIZE;
            if (inputTypeSize == 0) {
                ALOG_ERROR_F("inputTypeSize = 0, division by zero");
            }
            if (op.GetTileShape().GetVecTile()[0] == -1) {
                int64_t curTile = std::min(defaultTileSize, static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(std::max(op.GetIOperands()[0]->shape[inputDims - 1], BLOCK_SIZE / inputTypeSize))))));
                curTile = (curTile != 0) ? curTile : 1;
                vectorTilesNew.push_back(curTile);
                defaultTileSize /= curTile;
                for (size_t j = 1; j < inputDims; j++) {
                    curTile = std::min(defaultTileSize,  static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(op.GetIOperands()[0]->shape[inputDims - 1 - j])))));
                    curTile = (curTile != 0) ? curTile : 1;
                    vectorTilesNew.push_back(curTile);
                    defaultTileSize /= curTile;
                }
                std::reverse(vectorTilesNew.begin(), vectorTilesNew.end());
                op.GetTileShapeForSetting().SetVecTile(vectorTilesNew);
            }
            noConsumersOperations.push_back(&op);
        }
    }
    return noConsumersOperations;
}

void BackwardNoConsumersPropagation(std::vector<Operation *> noConsumersOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto noConsumerOp : noConsumersOperations) {
        queueBFS.push(noConsumerOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                bool isVisitedNode = (producerOp->GetTileShape().GetVecTile()[0] != -1);
                if (isVisitedNode) {
                    // Update queueBFS and visitedBFS
                    UpdateBFS(producerOp, queueBFS, visitedBFS);
                    continue;
                }
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedBFS);
                if (isContinue) {
                    continue;
                }
                
                // Call propagation
                isContinue = Propagation(noConsumerOp, op, producerOp, producerOp, queueBFS, visitedBFS, false, false, false, false);
                if (isContinue) {
                    continue;
                }
                // Update queueBFS and visitedBFS
                UpdateBFS(producerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void SetReduceTiles(std::vector<Operation *> reduceOrderedOperations) {
    std::vector<int64_t> vectorTilesReduce;
    for (auto op : reduceOrderedOperations) {
        size_t inputsNum = op->GetIOperands().size();
        size_t outputsNum = op->GetOOperands().size();
        size_t inputDims = DimsCalculation(op, inputsNum, true);
        size_t outputDims = DimsCalculation(op, outputsNum, false);
        int64_t inputTypeSize = BytesOf(op->GetIOperands()[0]->tensor->GetDataType());
        int64_t outputTypeSize = BytesOf(op->GetOOperands()[0]->tensor->GetDataType());
        int64_t maxTypeSize = std::max(inputTypeSize, outputTypeSize);
        if (inputTypeSize == 0 || maxTypeSize == 0) {
            ALOG_ERROR_F("typeSize = 0, division by zero");
        }
        ASSERT(inputDims == outputDims) << "Input dims should be equal output dims";
        ASSERT(outputsNum == 1) << "ReduceOp must have 1 output";

        std::vector<int64_t> maxInputShape = MaxInputShapeCalculation(op, inputsNum, inputDims, maxTypeSize);
        int64_t reducedDim = -1;
        for (size_t dim = 0; dim < inputDims; dim++) {
            if ((maxInputShape[dim] != op->GetOOperands()[0]->shape[dim]) && (op->GetOOperands()[0]->shape[dim] == 1) && (reducedDim == -1)) {
                reducedDim = dim;
            } else {
                ASSERT(!((maxInputShape[dim] != op->GetOOperands()[0]->shape[dim]) && (op->GetOOperands()[0]->shape[dim] == 1) && (reducedDim != -1))) << "Several reduced dims \n";
            }
        }
        ASSERT(reducedDim >= 0) << "Not found reduced dims";
        int64_t tileSize = MAX_TILE_SIZE / maxTypeSize;
        vectorTilesReduce.resize(inputDims);
        
        // Reduce Dim processing
        int64_t curTile = std::max(maxInputShape[reducedDim], BLOCK_SIZE / inputTypeSize);
        curTile = (maxInputShape[reducedDim] >= (UINT8MAX * BLOCK_SIZE / inputTypeSize)) ? std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(UINT8MAX * BLOCK_SIZE / inputTypeSize)))), curTile) : curTile; // Consider additional restriction for REDUCE ops
        curTile = std::min(tileSize, curTile); // UB overflow condition
        curTile = (curTile != 0) ? curTile : 1;
        vectorTilesReduce[reducedDim] = curTile;
        tileSize /= curTile;

        // Other Dims processing
        for (size_t dim = inputDims; dim > 0; dim--) {
            if ((dim - 1) == static_cast<size_t>(reducedDim)) {
                continue;
            }
            curTile = std::min(static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(tileSize)))), static_cast<int64_t>(std::pow(NUM2, static_cast<int64_t>(std::log2(maxInputShape[dim - 1])))));
            curTile = (curTile != 0) ? curTile : 1;
            vectorTilesReduce[dim - 1] = curTile;
            tileSize /= curTile;
        }
        vectorTilesReduce[0] = (inputDims != 1) ? vectorTilesReduce[0] * tileSize : vectorTilesReduce[0];
        op->GetTileShapeForSetting().SetVecTile(vectorTilesReduce);
    }
}

void ForwardReducePropagation(std::vector<Operation *> reduceOrderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto reduceOp : reduceOrderedOperations) {
        queueBFS.push(reduceOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto consumerOp : op->ConsumerOps()) {
                // Call propagation
                bool isContinue = Propagation(reduceOp, op, op, consumerOp, queueBFS, visitedBFS, false, false, true, true);
                if (isContinue) {
                    continue;
                }
                    
                // Update queueBFS and visitedBFS
                UpdateBFS(consumerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void BackwardReducePropagation(std::vector<Operation *> reduceOrderedOperations, std::queue<Operation *> &queueBFS, std::map<int, bool> &visitedBFS) {
    for (auto reduceOp : reduceOrderedOperations) {
        queueBFS.push(reduceOp);
        while (!queueBFS.empty()) {
            auto op = queueBFS.front();
            queueBFS.pop();
            for (auto producerOp : op->ProducerOps()) {
                bool isContinue = DuplicateTileSetting(op, producerOp, queueBFS, visitedBFS);
                if (isContinue) {
                    continue;
                }
                
                // Call propagation
                isContinue = Propagation(reduceOp, op, producerOp, producerOp, queueBFS, visitedBFS, false, false, true, false);
                if (isContinue) {
                    continue;
                }
                
                // Update queueBFS and visitedBFS
                UpdateBFS(producerOp, queueBFS, visitedBFS);
            }
        }
        visitedBFS.clear();
    }
}

void SetHeuristicVectorTiles(Function &function, std::unordered_set<Operation *> cubeOperations) {
    // Define cube operations oredered by depth
    std::map<int, int> subgrDepthMap;
    std::vector<Operation *> cubeOrderedOperations = FordBellman(function, cubeOperations, subgrDepthMap);
    
    // Define auxiliary data structures
    std::queue<Operation *> queueBFS;
    std::map<int, bool> visitedBFS;
    
    // 1. Backward propagation from CubeOps
    BackwardCubePropagation(cubeOrderedOperations, queueBFS, visitedBFS);

    // 2. Forward propagation from CubeOps
    ForwardCubePropagation(cubeOrderedOperations, queueBFS, visitedBFS);
    
    // Need to set initial vector tiles for nodes without consumers
    std::vector<Operation *> noConsumersOperations = FillNoConsumersOperations(function);
    
    // 3. Backward propagation from nodes without consumers
    BackwardNoConsumersPropagation(noConsumersOperations, queueBFS, visitedBFS);
    
    // Sort reduce operations by depth
    std::unordered_set<Operation *> reduceOperations;
    std::vector<std::pair<Operation *, int>> reduceTmpOperations;
    for (auto &op : function.Operations()) {
        auto iterRowDim = reduceOps.find(op.GetOpcode());
        if (iterRowDim != reduceOps.end()) {
            reduceOperations.insert(&op);
        }
    }
    
    for (auto reduceOp : reduceOperations) {
        reduceTmpOperations.push_back(std::make_pair(reduceOp, subgrDepthMap[reduceOp->GetOpMagic()]));
    }
    std::sort(reduceTmpOperations.begin(), reduceTmpOperations.end(), [](const std::pair<Operation *, int> &x, const std::pair<Operation *, int> &y) {return x.second < y.second;});
    
    std::vector<Operation *> reduceOrderedOperations;
    for (auto op : reduceTmpOperations) {
        reduceOrderedOperations.push_back(op.first);
    }
    reduceTmpOperations.clear();
    
    // Need to set initial vector tiles for ReduceOps
    SetReduceTiles(reduceOrderedOperations);
    
    // 4. Forward propagation from ReduceOps
    ForwardReducePropagation(reduceOrderedOperations, queueBFS, visitedBFS);
    
    // 5. Backward propagation from ReduceOps
    BackwardReducePropagation(reduceOrderedOperations, queueBFS, visitedBFS);
}

void SetHeuristicTileShapes::SetHeuristicTileShapesFunc(Function &function) const {
    (void)function;
    
    // Find all cube operations from all operations
    std::unordered_set<Operation *> cubeOperations;
    for (auto &op : function.Operations()) {
        if (op.GetCoreTypeStr() == "AIC") {
            cubeOperations.insert(&op);
        }
    }
    
    #ifdef CUBE_TILES
    // Set heuristic cube tiles
    SetHeuristicCubeTiles(function, cubeOperations);
    #endif
    
    #ifdef VECTOR_TILES
    // Define -1 tile shapes for non-cubes operations
    std::vector<int64_t> defTile = {-1};
    for (auto &op : function.Operations()) {
        op.GetTileShapeForSetting().SetVecTile(defTile);
    }

    // Set heuristic vector tiles
    SetHeuristicVectorTiles(function, cubeOperations);
    
    // Check that all tiles was defined by algorithm
    for (auto &op : function.Operations()) {
        ASSERT(op.GetTileShape().GetVecTile()[0] != -1) << "Not all tiles was set";
    }
    #endif
}
} // namespace npu::tile_fwk