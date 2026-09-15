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

#include <acl/acl_prof.h>
#include <acl/acl_rt.h>
#include <profiling/aprof_pub.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace pypto {

inline uint64_t BeginCaptureTensorReport(bool profiling, aclrtStream stream)
{
    if (!profiling) {
        return 0;
    }
    aclrtStreamAttrValue attr{};
    if (aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_CACHE_OP_INFO, &attr) != ACL_SUCCESS ||
        !attr.cacheOpInfoSwitch) {
        return 0;
    }
    return MsprofSysCycleTime();
}

inline void ReportCaptureTensorInfo(const aclprofTensorInfo& info, uint64_t begin)
{
    if (begin == 0) {
        return;
    }
    const uint64_t end = MsprofSysCycleTime();
    const uint64_t timestamp = begin + (end - begin) / 2;
    const auto threadId = static_cast<uint32_t>(syscall(SYS_gettid));

    // During graph capture aclprofRangePop caches tensors in RuntimeOpInfo. CANN
    // can instead select the compiler's existing kernel node, whose tensor fields
    // are empty. Also emit node tensor information, as the eager range does.
    // A timestamp inside the launch range associates it with that kernel node;
    // keep the range push/pop as well so the graph's cached metadata survives.
    for (uint32_t offset = 0; offset < info.tensorNum; offset += MSPROF_GE_TENSOR_DATA_NUM) {
        MsprofAdditionalInfo additional{};
        additional.level = MSPROF_REPORT_NODE_LEVEL;
        additional.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
        additional.threadId = threadId;
        additional.timeStamp = timestamp;
        additional.dataLen = sizeof(MsprofTensorInfo);
        auto& tensors = *reinterpret_cast<MsprofTensorInfo*>(additional.data);
        tensors.opName = info.opNameId;
        const uint32_t remaining = info.tensorNum - offset;
        tensors.tensorNum = remaining < MSPROF_GE_TENSOR_DATA_NUM ? remaining : MSPROF_GE_TENSOR_DATA_NUM;
        for (uint32_t index = 0; index < tensors.tensorNum; ++index) {
            const aclprofTensor& source = info.tensors[offset + index];
            MsrofTensorData& dest = tensors.tensorData[index];
            dest.tensorType = source.type;
            dest.format = source.format;
            dest.dataType = source.dataType;
            for (uint32_t dim = 0; dim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++dim) {
                dest.shape[dim] = source.shape[dim];
            }
        }
        // Metadata collection must never prevent or fail a kernel launch.
        (void)MsprofReportAdditionalInfo(true, &additional, sizeof(additional));
    }
}

} // namespace pypto
