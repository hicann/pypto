/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <acl/acl_rt.h>
#include <cstdint>

namespace pypto {

// The ctypes boundary cannot propagate C++ exceptions. A positive int64_t is the
// actual block count; a negative result encodes -(kind << 32 | detail), reported
// unchanged in Python's launch error. The detail holds either the insufficient
// budget or all 32 bits of the ACL query's error code.
enum class LaunchError : uint32_t {
    CubeLimit = 1,
    VectorLimit = 2,
    CubeQuery = 3,
    VectorQuery = 4,
};

inline int64_t EncodeLaunchError(LaunchError kind, uint32_t detail)
{
    return -static_cast<int64_t>((static_cast<uint64_t>(kind) << 32) | detail);
}

template <aclrtDevResLimitType Resource, uint32_t CoresPerBlock>
inline int64_t LimitLaunchBlocks(uint32_t blockDim, [[maybe_unused]] aclrtStream stream)
{
    if constexpr (CoresPerBlock == 0) {
        // A kernel using only one engine must not query or depend on the other.
        return blockDim;
    } else {
        uint32_t limit = 0;
        const aclError status = aclrtGetStreamResLimit(stream, Resource, &limit);
        if (status != ACL_SUCCESS) {
            constexpr auto kind = Resource == ACL_RT_DEV_RES_CUBE_CORE ? LaunchError::CubeQuery :
                                                                         LaunchError::VectorQuery;
            return EncodeLaunchError(kind, static_cast<uint32_t>(status));
        }
        if (limit < CoresPerBlock) {
            constexpr auto kind = Resource == ACL_RT_DEV_RES_CUBE_CORE ? LaunchError::CubeLimit :
                                                                         LaunchError::VectorLimit;
            return EncodeLaunchError(kind, limit);
        }
        const uint32_t maxBlocks = limit / CoresPerBlock;
        return blockDim < maxBlocks ? blockDim : maxBlocks;
    }
}

template <uint32_t CubeCores, uint32_t VectorCores>
inline int64_t ResolveLaunchBlockDim(uint32_t blockDim, aclrtStream stream)
{
    static_assert(CubeCores != 0 || VectorCores != 0, "A kernel must use cube or vector cores");
    // Query this launch's stream every time, including capture and cached calls.
    // ACL resolves stream > device > hardware. Caching a budget would retain a
    // previous scope's limit; the template arguments cache only the binary's ABI.
    const int64_t cubeDim = LimitLaunchBlocks<ACL_RT_DEV_RES_CUBE_CORE, CubeCores>(blockDim, stream);
    if (cubeDim < 0) {
        return cubeDim;
    }
    // Both engines of a mixed block must fit together, otherwise cross-core waits
    // can lose a participant. Never round a budget up to force a single block.
    return LimitLaunchBlocks<ACL_RT_DEV_RES_VECTOR_CORE, VectorCores>(static_cast<uint32_t>(cubeDim), stream);
}

} // namespace pypto
