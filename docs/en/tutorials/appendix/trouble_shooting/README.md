# Overview

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:13:29.705Z pushedAt=2026-08-31T11:19:33.437Z -->

This document describes the error codes of the PyPTO framework and provides troubleshooting guidance.

- F0XXXX indicates external usage errors. Refer directly to the on-screen error messages. If the error message is unclear, visit the community and submit an [issue](https://gitcode.com/cann/pypto/issues).
- Non-F0XXXX indicates internal framework errors. Refer to the troubleshooting guidance of the corresponding component based on the error code range. If no matching error code is found or the issue remains unresolved, visit the community and submit an [issue](https://gitcode.com/cann/pypto/issues).
- Before troubleshooting, configure the following common log environment variables to quickly obtain key logs. For details, see [*Environment Variable Reference*](https://www.hiascend.com/document/redirect/CannCommunityEnvRef).

    ```bash
    #Example
    export ASCEND_GLOBAL_LOG_LEVEL=0
    export ASCEND_HOST_LOG_FILE_NUM=1000
    export ASCEND_WORK_PATH=./wk
    ```
