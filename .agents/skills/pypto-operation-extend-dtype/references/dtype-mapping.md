# DT_* 枚举与 C++ 类型映射表

本文件是 pypto DataType 枚举、C++ 类型名、pto-isa 类型名的唯一事实来源。

## 完整映射表

来源：`framework/include/tilefwk/data_type.h` 中的 `DATA_TYPE_ALL` 宏定义。

| DT_* 枚举 | C++ 类型 | pto-isa static_assert 类型名 | 字节数 | 是否浮点 | cann_type |
|-----------|---------|--------------------------|-------|---------|-----------|
| DT_INT4 | int4 | - | 1 | 否 | 29 |
| DT_INT8 | int8_t | int8_t | 1 | 否 | 2 |
| DT_INT16 | int16_t | int16_t | 2 | 否 | 6 |
| DT_INT32 | int32_t | int32_t | 4 | 否 | 3 |
| DT_INT64 | int64_t | int64_t | 8 | 否 | 9 |
| DT_FP8 | float8_t | - | 1 | 是 | 28 |
| DT_FP16 | half | half | 2 | 是 | 1 |
| DT_FP32 | float | float | 4 | 是 | 0 |
| DT_BF16 | bfloat16_t | bfloat16_t | 2 | 是 | 27 |
| DT_HF4 | hfloat4 | - | 1 | 是 | 28 |
| DT_HF8 | hifloat8_t | - | 1 | 是 | 34 |
| DT_UINT8 | uint8_t | uint8_t | 1 | 否 | 4 |
| DT_UINT16 | uint16_t | uint16_t | 2 | 否 | 7 |
| DT_UINT32 | uint32_t | uint32_t | 4 | 否 | 8 |
| DT_UINT64 | uint64_t | uint64_t | 8 | 否 | 10 |
| DT_BOOL | bool | - | 1 | 否 | 12 |
| DT_DOUBLE | double | - | 8 | 是 | 11 |
| DT_FP8E4M3 | float8_e4m3_t | - | 1 | 是 | 36 |
| DT_FP8E5M2 | float8_e5m2_t | - | 1 | 是 | 35 |
| DT_FP8E8M0 | float8_e8m0_t | - | 1 | 是 | 37 |

## 用户常用名到 DT_* 映射

用户通常使用小写名称（如 `int64`、`uint64`、`fp32`），需要映射到 DT_* 枚举。

来源：`python/tests/st/operation/vector/pto_test_case_runner.py` 中的 `get_pto_dtype_by_name` 函数。

| 用户常用名 | DT_* 枚举 | C++ 类型 |
|-----------|-----------|---------|
| int4 | DT_INT4 | int4 |
| int8 | DT_INT8 | int8_t |
| int16 | DT_INT16 | int16_t |
| int32 | DT_INT32 | int32_t |
| int64 | DT_INT64 | int64_t |
| uint8 | DT_UINT8 | uint8_t |
| uint16 | DT_UINT16 | uint16_t |
| uint32 | DT_UINT32 | uint32_t |
| uint64 | DT_UINT64 | uint64_t |
| fp8 | DT_FP8 | float8_t |
| fp16 | DT_FP16 | half |
| fp32 | DT_FP32 | float |
| bf16 | DT_BF16 | bfloat16_t |
| bool | DT_BOOL | bool |
| double | DT_DOUBLE | double |
| fp8e4m3 | DT_FP8E4M3 | float8_e4m3_t |
| fp8e5m2 | DT_FP8E5M2 | float8_e5m2_t |
| fp8e8m0 | DT_FP8E8M0 | float8_e8m0_t |
| hf4 | DT_HF4 | hfloat4 |
| hf8 | DT_HF8 | hifloat8_t |

## 架构区分说明

pypto 使用 `GetSupportedDataTypesByArch(a2a3Types, a5Types)` 函数根据 NPU 架构返回不同的 dtype 集合：

```cpp
const std::unordered_set<DataType>& GetSupportedDataTypesByArch(
    const std::unordered_set<DataType>& a2a3Types,
    const std::unordered_set<DataType>& a5Types)
{
    bool isA5Architecture = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    return isA5Architecture ? a5Types : a2a3Types;
}
```

- **A2/A3 架构**：`NPUArch::DAV_3510` 以外的架构，使用 `a2a3Types` 集合
- **A5 架构**：`NPUArch::DAV_3510`，使用 `a5Types` 集合

## pto-isa 中的架构目录

pto-isa 按架构组织头文件：
- `include/pto/npu/a2a3/` — A2/A3 平台的 NPU ISA 实现
- `include/pto/npu/a5/` — A5 平台的 NPU ISA 实现
- `include/pto/npu/a6/` — A6 平台的 NPU ISA 实现
- `include/pto/npu/kirin9030/` — kirin9030 平台
- `include/pto/npu/kirinX90/` — kirinX90 平台
- `include/pto/cpu/` — CPU 模拟实现（用于仿真和测试）

## pto-isa static_assert 中的类型检查模式

pto-isa 在每个操作的 `Check` 函数中使用 `static_assert` 列出支持的类型。例如 TAdd：

```cpp
// a5/TAdd.hpp 中的 TAddCheck
static_assert(
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
        std::is_same_v<T, uint32_t> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> ||
        std::is_same_v<T, uint16_t> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
        std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
    "Fix: TADD has invalid data type.");
```

```cpp
// a2a3/TAdd.hpp 中的 TAddCheck — 注意不支持 int64/uint64
static_assert(
    std::is_same<T, int32_t>::value || std::is_same<T, int>::value || std::is_same<T, int16_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, float16_t>::value || std::is_same<T, float>::value ||
        std::is_same<T, float32_t>::value,
    "Fix: TADD has invalid data type.");
```

检查 dtype 是否被 pto-isa 支持的方法：在对应架构的 `T{Op}.hpp` 文件中搜索 `static_assert` 和 `std::is_same` 行，确认目标 C++ 类型是否出现。

## 特殊实现模式

某些 dtype 在 pto-isa 中有专门的实现路径，通过 `if constexpr` 分支选择：

```cpp
// a5/TAdd.hpp 中的 TAdd 实现
if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
    Int64Binary<Int64Op::Add, T, ...>(dstPtr, src0Ptr, src1Ptr, validRows, validCols);
} else {
    BinaryInstr<AddOp<T>, ...>(dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
}
```

如果 pto-isa 中已有 `if constexpr` 分支处理目标 dtype，说明底层实现已就绪，pypto 层可以安全地添加该 dtype 到 supportedTypes 集合。
