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
 * \file tile_shape.h
 * \brief
 */

#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <sstream>

#define MAX_DIST_DIM_SIZE 3

#define MAX_M_DIM_SIZE 2
#define MAX_K_DIM_SIZE 3
#define MAX_N_DIM_SIZE 2

/**
 * @brief VecTile tile for elementwise operation
 *
 */
struct VecTile {
    std::vector<int64_t> tile;

    bool valid() const {
        return std::all_of(tile.begin(), tile.end(), [](int x) { return x > 0; }) && tile.size() > 0;
    }

    int64_t operator[](int i) const { return tile[i]; }
    int64_t &operator[](int i) { return tile[i]; }

    size_t size() const { return tile.size(); }
};

/**
 * @brief CubeTile tile for matmul operation, m[0], k[0], n[0] for L0 Cache, m[1], k[1], n[1] for L1 Cache
 *
 */
struct CubeTile {
    std::array<int64_t, MAX_M_DIM_SIZE> m;
    std::array<int64_t, MAX_K_DIM_SIZE> k;
    std::array<int64_t, MAX_N_DIM_SIZE> n;
    bool setL1Tile {false};
    bool enableSplitK {false};

    bool valid() const {
        return std::all_of(m.begin(), m.end(), [](int64_t x) { return x > 0; }) &&
               std::all_of(k.begin(), k.end(), [](int64_t x) { return x > 0; }) &&
               std::all_of(n.begin(), n.end(), [](int64_t x) { return x > 0; });
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "CubeTile: " << '{'
           << "m: {" << m[0] << ", " << m[1] << '}' << ", "
           << "k: {" << k[0] << ", " << k[1] << ", " << k[0x2] << '}' << ", "
           << "n: {" << n[0] << ", " << n[1] << '}' << ", "
           << "setL1Tile: " << setL1Tile
           << "enableSplitK: " << enableSplitK
           << "}";
        return ss.str();
    }
};

/**
 * \brief DistTile tile for distributed operation
 *
 */
struct DistTile {
    std::array<int, MAX_DIST_DIM_SIZE> row;
    std::array<int, MAX_DIST_DIM_SIZE> col;
    std::array<int, MAX_DIST_DIM_SIZE> rank;
    int rankId {INT16_MAX};

    bool valid() const {
        return std::all_of(row.begin(), row.end(), [](int x) { return x > 0; }) &&
               std::all_of(col.begin(), col.end(), [](int x) { return x > 0; }) &&
               std::all_of(rank.begin(), rank.end(), [](int x) { return x > 0; }) && rankId >= 0;
    }
};

enum class TileType {
    VEC,
    CUBE,
    DIST,
    MAX,
};

/**
 * @brief TileShape tile shape for operation
 *
 */
struct TileShape {
    /**
     * \brief Set the Vec Tile
     *
     * \param tile
     */
    void SetVecTile(const std::vector<int64_t> &tile);
    void SetVecTile(const VecTile &tile);

    template <typename... Args, typename = std::enable_if_t<std::conjunction_v<std::is_integral<Args>...>>>
    inline void SetVecTile(Args... args) {
        SetVecTile(std::vector<int64_t>{args...});
    }

    /**
     * \brief Get the Vec Tile
     *
     * \return const std::vector<int64_t>&
     */
    const VecTile &GetVecTile() const { return vecTile; }
    VecTile &GetVecTile() { return vecTile; }

    /**
     * \brief Set the Cube Tile
     *
     * \param m
     * \param k
     * \param n
     */
    void SetCubeTile(const std::array<int64_t, MAX_M_DIM_SIZE> &m, const std::array<int64_t, MAX_K_DIM_SIZE> &k,
        const std::array<int64_t, MAX_N_DIM_SIZE> &n, bool setL1Tile = false, bool enableSplitK = false) ;

    /**
     * \brief Get the Cube Tile
     *
     * \return const std::vector<int64_t>&
     */
    const CubeTile &GetCubeTile() const { return cubeTile; }
    CubeTile &GetCubeTile() { return cubeTile; }

    /**
     * \brief Set the Dist Tile
     *
     * \param row
     * \param col
     * \param rank
     */
    void SetDistTile(
        const std::array<int, MAX_DIST_DIM_SIZE> &row, const std::array<int, MAX_DIST_DIM_SIZE> &col, const std::array<int, MAX_DIST_DIM_SIZE> &rank) {
        SetDistTileRow(row);
        SetDistTileCol(col);
        SetDistTileRank(rank);
    }

    /**
     * \brief Get the Dist Tile
     *
     * \return const DistTile&
     */
    const DistTile &GetDistTile() const { return distTile; }
    DistTile &GetDistTile() { return distTile; }

    /**
     * @brief Set the Dist Rank Id
     *
     * @param rankId
     */
    void SetDistRankId(int64_t rankId) { distTile.rankId = rankId; }

    /**
     * @brief Get the Dist Rank Id
     *
     * @return int64_t
     */
    int64_t GetDistRankId() const { return distTile.rankId; }

    /**
     * @brief Set the Dist Col
     *
     * @param col
     */
    void SetDistTileCol(const std::array<int, MAX_DIST_DIM_SIZE> &col) {
        distTile.col = col;
    }

    /**
     * @brief Get the Dist Col
     *
     * @return const std::vector<int64_t>&
     */
    const std::array<int, MAX_DIST_DIM_SIZE> &GetDistTileCol() const { return distTile.col; }

    /**
     * @brief Set the Dist Row
     *
     * @param row
     */
    void SetDistTileRow(const std::array<int, MAX_DIST_DIM_SIZE> &row) {
        distTile.row = row;
    }

    /**
     * @brief Get the Dist Row
     *
     * @return const std::vector<int64_t>&
     */
    const std::array<int, MAX_DIST_DIM_SIZE> &GetDistTileRow() const { return distTile.row; }

    /**
     * @brief Set the Dist Rank
     *
     * @param rank
     */
    void SetDistTileRank(const std::array<int, MAX_DIST_DIM_SIZE> &rank) {
        distTile.rank = rank;
    }

    /**
     * @brief Get the Dist Rank
     *
     * @return const std::vector<int64_t>&
     */
    const std::array<int, MAX_DIST_DIM_SIZE> &GetDistTileRank() const { return distTile.rank; }

    /**
     * @brief Global tile shape
     *
     * @return TileShape&
     */
    static TileShape &Current();

    /**
     * @brief Reset the tile shape
     *
     */
    void Reset() {
        vecTile = {};
        cubeTile = {{0, 0}, {0, 0, 0}, {0, 0}};
        distTile = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, INT16_MAX};
    }

    void SetMatrixSize(const std::vector<int64_t> &size) ;

    const std::vector<int64_t> &GetMatrixSize() const {
        return matrixSize;
    }

    std::string toString(TileType type = TileType::MAX) const {
        std::stringstream ss;
        if (type == TileType::VEC || type == TileType::MAX) {
            ss << "VecTile: " << '{';
            for (size_t i = 0; i < vecTile.tile.size(); ++i) {
                if (i != 0) ss << ", ";
                ss << vecTile.tile[i];
            }
            ss << '}';
        } else if (type == TileType::CUBE || type == TileType::MAX) {
            ss << "CubeTile: " << '{'
               << "m: {" << cubeTile.m[0] << ", " << cubeTile.m[1] << '}' << ", "
               << "k: {" << cubeTile.k[0] << ", " << cubeTile.k[1] << ", " << cubeTile.k[0x2] << '}' << ", "
               << "n: {" << cubeTile.n[0] << ", " << cubeTile.n[1] << '}'
               << "}";
        } else if (type == TileType::DIST || type == TileType::MAX) {
            ss << "DistTile: " << '{'
               << "row: {" << distTile.row[0] << ", " << distTile.row[1] << ", " << distTile.row[0x2] << '}' << ", "
               << "col: {" << distTile.col[0] << ", " << distTile.col[1] << ", " << distTile.col[0x2] << '}' << ", "
               << "rank: {" << distTile.rank[0] << ", " << distTile.rank[1] << ", " << distTile.rank[0x2] << '}' << ", "
               << "rankId: " << distTile.rankId << '}';
        }
        return ss.str();
    }
private:
    VecTile vecTile;
    CubeTile cubeTile;
    DistTile distTile;
    std::vector<int64_t> matrixSize;
};