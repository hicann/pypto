#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# run_with_timeout.sh - 带超时控制的命令执行脚本
# =============================================================================
# 功能：
#   1. 执行指定命令
#   2. 监控执行时间
#   3. 超时后发送 SIGINT (Ctrl+C)
#   4. 等待进程优雅退出
#   5. 返回合适的退出码
#
# 参数：
#   $1: 超时时间（秒），默认300秒（5分钟）
#   $2...: 要执行的命令及其参数
#
# 退出码：
#   0   - 成功完成或超时中断（允许继续后续步骤）
#   非0 - 执行失败（停止后续步骤）
#
# 使用示例：
#   bash run_with_timeout.sh 300 python3 test.py
#   bash run_with_timeout.sh 600 python3 test.py --run-mode npu
# =============================================================================

set -e

# 参数处理
TIMEOUT_SECONDS=${1:-300}  # 默认5分钟
SCRIPT_CMD="${@:2}"        # 要执行的命令

echo "========================================="
echo "执行命令: $SCRIPT_CMD"
echo "超时设置: ${TIMEOUT_SECONDS} 秒"
echo "========================================="

# 后台执行命令
$SCRIPT_CMD &
CMD_PID=$!

# 监控进程（后台）
(
    sleep $TIMEOUT_SECONDS
    if ps -p $CMD_PID > /dev/null 2>&1; then
        echo ""
        echo "⚠️  算子执行已超过 ${TIMEOUT_SECONDS} 秒"
        echo "⚠️  自动发送 Ctrl+C 信号中断执行..."
        echo "✓  Pass 编译日志已保存，将继续性能分析"
        kill -INT $CMD_PID 2>/dev/null
        wait $CMD_PID 2>/dev/null
    fi
) &
TIMEOUT_PID=$!

# 等待命令完成
wait $CMD_PID
EXIT_CODE=$?

# 清理监控进程
kill $TIMEOUT_PID 2>/dev/null
wait $TIMEOUT_PID 2>/dev/null

# 处理退出码
if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ]; then
    # 130 = SIGINT (Ctrl+C), 137 = SIGKILL
    echo ""
    echo "✓ 算子执行被中断（退出码: $EXIT_CODE）"
    echo "✓ Pass 编译日志已保存，可以继续性能分析"
    exit 0  # 正常退出，允许继续后续步骤
elif [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ 算子执行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo ""
    echo "✓ 算子执行完成"
    exit 0
fi
