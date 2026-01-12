#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 1. Execute directory on single NPU device
python3 examples/validate_examples.py -t examples/02_intermediate -d 0

# 2. Multi-device parallel execution
python3 examples/validate_examples.py -t examples -d 0,1,2,3

# 3. Execute specific script on device 0
python3 examples/validate_examples.py -t examples/01_beginner/basic/basic_ops.py -d 0

# 4. Simulation mode (single virtual worker)
python3 examples/validate_examples.py -t examples --run_mode sim -w 1

# 5. Concurrent execution in simulation mode (16 virtual workers)
python3 examples/validate_examples.py -t examples --run_mode sim -w 16

# 6. Custom timeout per script
python3 examples/validate_examples.py -t examples/02_intermediate -d 0 --timeout 120

# 7. Show failure diagnostics in summary
python3 examples/validate_examples.py -t examples -d 0 --show-fail-details

# 8. Include scripts marked with @pytest.mark.skip (override default behavior)
python3 examples/validate_examples.py -t examples -d 0 --no-skip-pytest-mark-skip

# 9. Full configuration
python3 examples/validate_examples.py -t examples -d 0,1,2,3 
    --parallel-retries 2 --serial-retries 5 --timeout 300 
    --show-fail-details --allow-pytest-auto-detect --skip-pytest-mark-skip