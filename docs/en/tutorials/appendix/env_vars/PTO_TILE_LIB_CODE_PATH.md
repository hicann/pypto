# PTO_TILE_LIB_CODE_PATH

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:55:31.764Z pushedAt=2026-08-20T10:38:57.903Z -->

## Function Description
Specifies the pto-isa source code path for compiling and running PyPTO operators. CANN toolkit includes pto-isa by default after being installed, so manual setting is not required. When the built-in version does not meet requirements, you can use this environment variable to point to a standalone pto-isa source code directory.

- Type: string (absolute path)

## Configuration Example
```bash
# Use the built-in pto-isa of CANN (recommended).
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/$(uname -m)-linux

# Use standalone pto-isa source code.
git clone https://gitcode.com/cann/pto-isa.git
export PTO_TILE_LIB_CODE_PATH="$PWD/pto-isa"
```

## Constraints
- The path must contain the `include/pto/` directory.
- If the built-in PyPTO in the CANN package is used, pto-isa is installed with the CANN package and does not need to be set separately.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products
