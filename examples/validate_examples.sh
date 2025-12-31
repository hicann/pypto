#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 1. Execute all Python scripts in a directory using single device
python3 examples/validate_examples.py -t examples/02_intermediate --device_ids 0

# 2. Single-device mode with up to 3 serial retry rounds for unstable scripts
python3 examples/validate_examples.py -t examples --device_ids 0 --serial-retries 3

# 3. Run a single script in default NPU mode on device 0
python3 examples/validate_examples.py -t examples/01_beginner/basic/basic_ops.py --device_ids 0

# 4. Execute all Python scripts in a directory using multiple devices (parallel execution)
python3 examples/validate_examples.py -t examples --device_ids 0,1,2,3

# 5. Execute in simulation mode (only scripts supporting --run_mode sim are run)
python3 examples/validate_examples.py -t examples --run_mode sim --device_ids 0

# 6. Concurrent execution in simulation mode
python3 examples/validate_examples.py -t examples --run_mode sim --device_ids 0,1,2,3,4,5,6,7

# 7. Set custom timeout (in seconds) for each script execution
python3 examples/validate_examples.py -t examples/02_intermediate --device_ids 0 --timeout 120

# 8. Enable parallel retries (initial run + 1 retry rounds) for flaky scripts
python3 examples/validate_examples.py -t examples --device_ids 0,1 --parallel-retries 1

# 9. Show last 5 lines of output for failed scripts in final summary (for debugging)
python3 examples/validate_examples.py -t examples --device_ids 0 --show-fail-details

# 10. Full production-grade validation: multi-device, retries, timeout, and failure details
python3 examples/validate_examples.py -t examples --device_ids 0,1,2,3,4,5,6,7 --parallel-retries 2 --serial-retries 5 --timeout 300 --show-fail-details