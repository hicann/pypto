/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

namespace npu::tile_fwk {
/*************************************************************************************************************
* 功能说明: 执行编译的指令图
* 输入参数:
    handle：compile返回值
    workspace： 前端申请workspace地址
    stream：前端申请的流
    args：前端申请的args地址，按照 TileFwkBeginFunction 的顺序传入
    prefetchSizes: 指定需要L2Cache prefetch大小的参数，与args一一对应，填0的arg表示不做预取，最多支持预取4个
* 输出参数: 无
* 返 回 值: 0表示成功，非零表示失败。
  **********************************************************************************************************/
int32_t TileFwkRunAsync(void *handle, const void *workspace, const void *stream, const std::vector<void *> &opArgs,
    const std::vector<std::size_t> &prefetchSizes = {});
}
