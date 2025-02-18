#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 1、Run in batches for the specified directory
python3 examples/validate_examples.py -t examples/02_intermediate --device-id 0

# 2、Run specified script
python3 examples/validate_examples.py -t examples/01_beginner/compute/elementwise_ops.py --device-id 0

# 3、Run the specified case of the specified script
python3 examples/validate_examples.py -t examples/01_beginner/compute/elementwise_ops.py add::test_add_basic --device-id 0

# 4、Set the timeout duration for each use case execution (Unit: seconds)
python3 examples/validate_examples.py -t examples/02_intermediate --device-id 0 --timeout 120

# 5、Run all scripts in the specified directory in batches in simulation mode (NPU mode by default)
python3 examples/validate_examples.py -t examples/02_intermediate --run-mode sim --device-id 0