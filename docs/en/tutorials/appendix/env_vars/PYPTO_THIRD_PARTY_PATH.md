# PYPTO_THIRD_PARTY_PATH

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T08:57:50.944Z pushedAt=2026-08-20T10:39:04.955Z -->

## Function Description
Specifies the path of the third-party open-source software source package required for PyPTO compilation. When the compilation environment cannot access the `cann-src-third-party` repository for automatic download, you need to manually prepare the source package and specify its path using this variable.

- Type: string (absolute path)

## Configuration Example
```bash
# Set after manually preparing the third-party source package.
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# Then compile.
python3 -m pip install . --verbose
```

## Constraints
- The path must contain the source packages of `json-3.11.3` and `libboundscheck-v1.1.16`.
- Can take effect only during compilation and installation based on source code. It is not required for PyPI installation.

## Supported Models
- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products
