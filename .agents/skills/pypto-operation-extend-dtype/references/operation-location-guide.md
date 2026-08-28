# Operation 文件定位指南

本文件描述如何从 operation 名称定位到 pypto 源码、API 文档、测试用例、pto-isa 头文件。

## 文件路径总表

对于 operation 名称 `{op}`（如 `add`、`compare`），相关文件分布在以下位置：

| 文件类型 | 路径模式 | 示例 |
|---------|---------|------|
| C++ 源码 | `framework/src/interface/operation/vector/{file}.cpp` | `vector/binary.cpp` |
| C++ 头文件 | `framework/src/interface/operation/vector/{file}.h` | `vector/binary.h` |
| API 文档 | `docs/zh/api/operation/pypto-{op}.md` | `pypto-add.md` |
| C++ ST 测试 | `framework/tests/st/operation/src/test_{op}_operation.cpp` | `test_add_operation.cpp` |
| CSV 测试用例 | `framework/tests/st/operation/test_case/{Op}_st_test_cases.csv` | `Add_st_test_cases.csv` |
| JSON 测试用例 | `framework/tests/st/operation/test_case/{Op}_st_test_cases.json` | `Add_st_test_cases.json` |
| Python ST 测试 | `python/tests/st/operation/vector/test_vector_operation_{op}.py` | `test_vector_operation_add.py` |
| pto-isa A2A3 | `include/pto/npu/a2a3/T{Op}.hpp` | `TAdd.hpp` |
| pto-isa A5 | `include/pto/npu/a5/T{Op}.hpp` | `TAdd.hpp` |
| pto-isa CPU | `include/pto/cpu/T{Op}.hpp` | `TAdd.hpp` |

## Operation 名称到源码文件映射

多个 operation 可能共享同一个 `.cpp` 源码文件。以下是已知的映射关系：

| 源码文件 | 包含的 operations | dtype 集合前缀 |
|---------|------------------|---------------|
| `binary.cpp` | add, sub, mul, div, max, min, bitwise_and, bitwise_or, bitwise_xor | `ADD_`, `SUB_`, `MUL_`, `DIV_`, `MAX_`, `MIN_`, `BITWISE_` |
| `unary.cpp` | abs, exp, log, sqrt, rsqrt, ceil, floor, neg, relu, sigmoid, tanh, bitwise_not 等 | `ABS_`, `EXP_`, `LOG_`, `SQRT_`, `RELU_`, `BITWISE_` 等 |
| `compare.cpp` | eq, ne, gt, ge, lt, le | `CMP_` |
| `where.cpp` | where | （局部变量，无前缀） |
| `indexing.cpp` | gather, scatter, index_add, index_select | （局部变量） |
| `tensor_transformation.cpp` | concat, transpose, cast, reshape, permute, expand, squeeze, unsqueeze | `TRANSPOSE_`, `CAST_` 等 |
| `reduction.cpp` | sum, mean, max, min, prod, norm | `REDUCTION_` 等 |
| `math.cpp` | atan2, hypot, fmod, fmod, remainder 等 | （局部变量） |
| `bitwise_shift.cpp` | bitwise_left_shift, bitwise_right_shift | `BITWISESHIFT_` |
| `quantization.cpp` | quantize, dequantize | （局部变量） |
| `interleave.cpp` | interleave, deinterleave | （局部变量） |
| `pad.cpp` | pad | （局部变量） |
| `sort.cpp` | sort, argsort | （局部变量） |
| `uniform.cpp` | uniform | （局部变量） |

## Operation 名称到 pto-isa 头文件映射

pto-isa 头文件命名规则：`T{PascalCaseOp}.hpp`

| pypto operation | pto-isa 头文件 | 说明 |
|----------------|---------------|------|
| add | `TAdd.hpp` | |
| sub | `TSub.hpp` | |
| mul | `TMul.hpp` | |
| div | `TDiv.hpp` 或 `TDivs.hpp` | |
| max | `TMax.hpp` 或在 `TColMax.hpp`/`TRowMax.hpp` | |
| min | `TMin.hpp` | |
| abs | `TAbs.hpp` | |
| exp | `TExp.hpp` | |
| log | `TLog.hpp` | |
| sqrt | `TSqrt.hpp` | |
| rsqrt | `TRsqrt.hpp` | |
| compare (eq/gt/ge/lt/le/ne) | `TCmp.h` 或 `TCompare.hpp` | |
| where | `TSelect.hpp` | |
| gather | `TGather.hpp` | |
| scatter | `TScatter.hpp` | |
| concat | `TConcat.hpp` | |
| transpose | `TTranspose.hpp` 或 `TPermute.hpp` | |
| cast | `TCvt.hpp` | |
| bitwise_and | `TAnd.hpp` | |
| bitwise_or | `TOr.hpp` | |
| bitwise_xor | `TXor.hpp` | |
| bitwise_not | `TNot.hpp` | |
| bitwise_left_shift | `TShl.hpp` | |
| bitwise_right_shift | `TShr.hpp` | |

> 注意：部分 operation 的 pto-isa 头文件名称可能与上述模式不完全一致。如果按规则找不到文件，请在 `include/pto/npu/{a2a3,a5}/` 目录下搜索 operation 关键字。

## 定位策略

### 策略 1：通过 dtype 集合名称定位

如果已知 operation 的 dtype 集合前缀（如 `ADD_`），可以直接在源码目录中搜索：

```bash
# 在 pypto 源码中搜索 dtype 集合定义
grep -rn "ADD_A2A3_TYPES\|ADD_A5_TYPES" framework/src/interface/operation/
```

### 策略 2：通过 operation 名称定位

```bash
# 在 pypto 源码中搜索 operation 名称
grep -rn "\"Add\"\|GetBinaryOpName\|ADD_A2A3" framework/src/interface/operation/vector/
```

### 策略 3：通过 pto-isa 头文件定位

```bash
# 在 pto-isa 中搜索 operation 的头文件
find /mnt/workspace/gitCode/cann/pto-isa/include/pto/npu -name "TAdd*"
```

### 策略 4：通过测试用例文件定位

测试用例文件名使用 PascalCase 的 operation 名称：

```bash
# 查找测试用例文件
ls framework/tests/st/operation/test_case/ | grep -i add
```

## 文档文件名映射

API 文档文件名格式：`pypto-{op}.md`，其中 `{op}` 为 operation 的小写名称。

| pypto operation | 文档文件名 |
|----------------|-----------|
| add | `pypto-add.md` |
| sub | `pypto-sub.md` |
| mul | `pypto-mul.md` |
| div | `pypto-div.md` |
| eq | `pypto-eq.md` |
| ge | `pypto-ge.md` |
| gt | `pypto-gt.md` |
| le | `pypto-le.md` |
| lt | `pypto-lt.md` |
| where | `pypto-where.md` |
| gather | `pypto-gather.md` |
| scatter | `pypto-scatter_.md` |
| concat | `pypto-concat.md` |
| transpose | `pypto-transpose.md` |
| relu | `pypto-relu.md` |

> 注意：部分 operation 的文档文件名可能有后缀（如 `scatter_` 带下划线）。如果直接找不到，在 `docs/zh/api/operation/` 目录下搜索。

## 多重载场景

部分 operation 有多个函数重载，每个重载都可能有独立的 supportedTypes 定义：

| operation | 重载数量 | 重载说明 |
|-----------|---------|---------|
| add | 2 | Tensor+Tensor, Tensor+Scalar (Adds) |
| compare | 3 | Tensor+Tensor, Tensor+Element, Element+Tensor |
| where | 3 | Tensor+Tensor+Tensor, 含 Scalar 变体 |
| mul | 2 | Tensor+Tensor, Tensor+Scalar |

在修改时，需要确保所有重载中的 supportedTypes 都一致更新。

## pto-isa 架构目录对照

| 架构名 | pto-isa 目录 | pypto 架构判定 | 对应产品 |
|--------|-------------|---------------|---------|
| A2/A3 | `include/pto/npu/a2a3/` | 非 `DAV_3510` | Atlas A2/A3 训练/推理系列 |
| A5 | `include/pto/npu/a5/` | `DAV_3510` | Ascend 950PR/950DT |
| A6 | `include/pto/npu/a6/` | - | A6 平台 |
| kirin9030 | `include/pto/npu/kirin9030/` | - | kirin9030 平台 |
| kirinX90 | `include/pto/npu/kirinX90/` | - | kirinX90 平台 |
