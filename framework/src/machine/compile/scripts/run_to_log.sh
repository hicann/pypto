#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Run a command string and write its stdout to a log file; stderr stays on the terminal.
# Invoked as: bash run_to_log.sh <log_path> <cmd_string>
# so the '>' redirect never appears in Checkinject()'d system() arguments.
if [ "$#" -ne 2 ]; then
    echo "usage: $0 <log_path> <cmd_string>" >&2
    exit 2
fi
eval "$2" >"$1"
