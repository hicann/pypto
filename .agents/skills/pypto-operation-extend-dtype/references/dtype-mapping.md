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

pypto 通过 `ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode)` 查询某个 opcode 在当前 NPU 架构下支持的输入 dtype。支持的 dtype **不再硬编码在 operation 的 `.cpp` 源码中**，而是集中定义在 JSON 配置文件中：

- `framework/src/interface/configs/platform_op_supported_dtypes/a2a3_supported_op_dtypes.json`
- `framework/src/interface/configs/platform_op_supported_dtypes/a5_supported_op_dtypes.json`
- `framework/src/interface/configs/platform_op_supported_dtypes/kirin9030_supported_op_dtypes.json`
- `framework/src/interface/configs/platform_op_supported_dtypes/kirinx90_supported_op_dtypes.json`

每个 JSON 的结构：

```json
{
    "npu_arch": 2201,
    "ops": {
        "ADD":  { "input_dtypes": ["int16", "int32", "fp16", "float32", "bf16"] },
        "ADDS": { "input_dtypes": ["int16", "int32", "fp16", "float32", "bf16"] }
    }
}
```

- `npu_arch`：`NPUArch` 枚举的整数值（见下表）。
- `ops`：opcode 字符串 → `input_dtypes` 数组。opcode 字符串是 `OpcodeManager` 注册的 str（`Opcode::OP_ADD` → `"ADD"`，标量版 `Opcode::OP_ADDS` → `"ADDS"`，`Maximum` → `"MAXIMUM"` 等）。
- dtype 字符串使用 `STR_DATA_TYPE_MAP`（`framework/include/tilefwk/data_type.h:230`）的小写友好名。

C++ 层读取函数（`framework/src/interface/configs/config_manager.h/.cpp`）：

```cpp
const std::unordered_set<DataType>& ConfigManager::Instance().GetOpSupportedInputDtypes(const Opcode& opcode);
```

该函数根据 `Platform::Instance().GetSoc().GetNPUArch()` 查表，只读 `input_dtypes` 类别，返回当前架构下该 opcode 的 dtype 集合（未配置时返回空集，会导致 dtype 检查报错 `Data type DT_xxx is not in supported types`）。

### JSON dtype 字符串映射（STR_DATA_TYPE_MAP）

| DT_* 枚举 | JSON 字符串 | DT_* 枚举 | JSON 字符串 |
|-----------|------------|-----------|------------|
| DT_INT4 | int4 | DT_UINT8 | uint8 |
| DT_INT8 | int8 | DT_UINT16 | uint16 |
| DT_INT16 | int16 | DT_UINT32 | uint32 |
| DT_INT32 | int32 | DT_UINT64 | uint64 |
| DT_INT64 | int64 | DT_BOOL | bool |
| DT_FP8 | fp8 | DT_DOUBLE | double |
| DT_FP16 | fp16 | DT_FP8E4M3 | fp8e4m3 |
| DT_FP32 | float32（注意不是 fp32） | DT_FP8E5M2 | fp8e5m2 |
| DT_BF16 | bf16 | DT_FP8E8M0 | fp8e8m0 |
| DT_HF4 | hf4 | DT_FP4_E2M1 | fp4_e2m1 |
| DT_HF8 | hf8 | DT_FP4_E1M2 | fp4_e1m2 |

> 注意：DT_FP32 在 JSON 中的键是 `float32`（`fp32` 不在 `STR_DATA_TYPE_MAP` 中）。

### NPUArch 枚举与 JSON 文件对照

`framework/include/tilefwk/platform.h` 中定义：

```cpp
enum class NPUArch { DAV_1001 = 1001, DAV_2201 = 2201, DAV_3510 = 3510, DAV_3003 = 3003, DAV_3113 = 3113, DAV_UNKNOWN };
```

| JSON 文件 | npu_arch | NPUArch | 架构 |
|-----------|----------|---------|------|
| a2a3_supported_op_dtypes.json | 2201 | DAV_2201 | A2/A3（云侧 Atlas A2/A3 系列） |
| a5_supported_op_dtypes.json | 3510 | DAV_3510 | A5（Ascend 950PR/950DT） |
| kirin9030_supported_op_dtypes.json | 3113 | DAV_3113 | kirin9030（Lite 端侧） |
| kirinx90_supported_op_dtypes.json | 3003 | DAV_3003 | kirinX90（Lite 端侧） |

> 历史上 A2/A3 同时映射 `DAV_1001` 与 `DAV_2201`；当前 JSON 以单一 `npu_arch=2201`（`DAV_2201`）代表 A2/A3 平台。

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
