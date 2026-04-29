# 打印上板信息指南

本指南用于在已知目标Op后，打印上板数据（DDR/GM和UB）、shape/offset属性值，验证实际运行时结果。

> **前置条件**：已通过 `pypto-precision-compare` 二分前端定位到具体Op。

---

## 目录

1. [概述](#概述)
2. [打印方法概述](#打印方法概述)
3. [前置配置](#前置配置)
4. [执行流程](#执行流程)
5. [如何找到目标Op在CCE中的位置](#如何找到目标op在cce中的位置)
6. [打印不出来诊断](#打印不出来诊断)
7. [脚本工具使用](#脚本工具使用)
8. [注意事项](#注意事项)

---

## 概述

**主要用途**：
1. 打印上板tensor数据，验证前端计算与上板输出是否一致
2. 打印动态shape/validshape的实际运行时值
3. 打印动态offset信息，定位越界访问问题

**触发场景**：
- 已定位问题Op，需要打印其输出tensor数据
- Pass侧OP报错涉及动态shape/validshape
- 验证Op属性值是否正确传递

---

## 打印方法概述

### 六种打印方法总览

| 打印类型 | 函数名称 | 适用场景 |
|---------|---------|---------|
| GM数据打印 | `AiCorePrintGmTensor` | 打印DDR/GM上的tensor数据（最常用） |
| UB数据打印 | `AiCorePrintUbTensor` | 打印UB上的tensor数据 |
| Shape批量打印 | `AiCorePrintShape` | 批量打印多维度shape信息 |
| Offset批量打印 | `AiCorePrintShape(CoordXDim)` | 批量打印多个offset值 |
| 单值Shape打印 | `AicoreLogF` | 打印单个shape变量值（推荐动态场景） |
| 单值Offset打印 | `AicoreLogF` | 打印单个offset值（推荐动态场景） |

### GM数据打印

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__float*)gmTensor_output.GetAddr(), 63, 0);
// 参数：末尾偏移量(63)、起始偏移量(0) → 打印64个元素
```

### UB数据打印

```cpp
AiCorePrintUbTensor(param->ctx, (__ub__float*)ubTensor_temp.GetAddr(), 63, 0);
```

### Shape批量打印

```cpp
AiCorePrintShape(param->ctx, Shape2Dim(sym_15_dim_0, sym_15_dim_1));
AiCorePrintShape(param->ctx, Shape3Dim(sym_10_dim_0, sym_10_dim_1, sym_10_dim_2));
```

### Offset批量打印

```cpp
AiCorePrintShape(param->ctx, Coord2Dim(
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 19, 0)),
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 1, 1))
));
```

### 单值打印（推荐动态场景）

```cpp
// 打印单个shape变量
AicoreLogF(param->ctx, "sym_329_dim_0=%llu\n", sym_329_dim_0);

// 打印单个offset
AicoreLogF(param->ctx, "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0,0,2,10,0)=%llu\n",
           RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0,0,2,10,0));
```

### dtype类型对照

| C++类型 | PyPTO DataType |
|--------|----------------|
| `float` | DT_FP32 |
| `bfloat16_t` | DT_BF16 |
| `half` | DT_FP16 |
| `int32_t` | DT_INT32 |

### 打印注意事项

1. **元素数量限制**：末尾-起始+1 ≤ 80
2. **打印时机**：必须在Op执行**之后**
3. **头文件**：`#include "tilefwk/aicore_print.h"`
4. **日志位置**：`$ASCEND_WORK_PATH/log/debug/device-*/device-*.log`

---

## 前置配置

### 1. 配置 tile_fwk_config.json

修改 `framework/src/interface/configs/tile_fwk_config.json`：

```json
"codegen": {
    "fixed_output_path": true,
    "force_overwrite": false,
    "parallel_compile": 1
}
```

| 配置项 | 正确值 | 说明 |
|-------|-------|------|
| `fixed_output_path` | `true` | 固定CCE输出路径，生成在 `./kernel_aicore/` |
| `force_overwrite` | `false` | 不覆盖已修改的CCE文件 |
| `parallel_compile` | `1` | 单线程编译，便于调试 |

### 2. 配置打印开关

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：

```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

---

## 执行流程

### 标准执行步骤

| 步骤 | 操作 | 命令 |
|------|------|------|
| 1 | 配置固定输出路径 | 修改 `tile_fwk_config.json` |
| 2 | 编译安装 | `python3 -m pip install . -v` |
| 3 | 运行用例一次 | 生成 CCE 文件 |
| 4 | 在 CCE 中添加打印 | 修改 `./kernel_aicore/*.cpp` |
| 5 | 再次运行用例 | 获取打印数据 |
| 6 | 查看对应日志 | `$ASCEND_WORK_PATH/log/debug/device-*/device-*.log` |

### CCE 文件生成位置

配置 `fixed_output_path: true` 后，CCE 文件固定生成在：

```
./kernel_aicore/
├── TENSOR_xxx_xxx_0_aiv.cpp
├── TENSOR_xxx_xxx_1_aiv.cpp
└── ...
```

---

## 如何找到目标Op在CCE中的位置

已知Op信息（funcHash、Op名称、param索引等），在CCE文件中定位具体代码位置。

### 方法一：根据 funcHash 搜索

```bash
grep -l "funcHash: <hash值>" kernel_aicore/*.cpp
```

### 方法二：根据 Op 类型关键字搜索

| Op类型 | CCE关键字 | 搜索命令 |
|-------|---------|---------|
| MATMUL | Matmul | `grep -n "Matmul" *.cpp` |
| ReshapeCopyIn | DynReshapeCopyIn | `grep -n "DynReshapeCopyIn" *.cpp` |
| ReshapeCopyOut | DynReshapeCopyOut | `grep -n "DynReshapeCopyOut" *.cpp` |
| ADD | Add/TAdd | `grep -n "Add" *.cpp` |

### 方法三：根据 param 索引定位

```bash
# 错误日志中 %23@485#(41) 对应 GET_PARAM_ADDR(param, 41, 485)
grep -n "GET_PARAM_ADDR(param, 41" kernel_aicore/*.cpp
```

### 方法四：根据 tensor 名称定位

```bash
grep -n "gmTensor_<名称>" kernel_aicore/*.cpp
```

### 方法五：相邻Op辅助定位

当目标Op搜索不到时，通过相邻Op定位：

```bash
# 找到前后Op的位置，目标Op在其之间
grep -n "TLoad" kernel_aicore/*.cpp   # 前Op
grep -n "TStore" kernel_aicore/*.cpp  # 后Op
```

---

## 打印不出来诊断

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 日志无打印数据 | `ENABLE_AICORE_PRINT=0` | 改为 `1` |
| 找不到CCE文件 | `fixed_output_path=false` | 改为 `true` |
| 打印数据错误 | 偏移量超出限制 | 调整偏移量（末尾-起始+1≤80） |
| 打印时机错误 | 打印在Op执行前 | 移到Op执行后 |

---

## 脚本工具使用

脚本位置：`../scripts/print_npu_data.py`

### 初始化配置

```bash
python3 ../scripts/print_npu_data.py --init --work-path /path/to/work
```

### 列出CCE信息

```bash
python3 ../scripts/print_npu_data.py --work-path /path/to/work --list-cce
```

### 添加tensor打印

```bash
# 默认打印64个元素（偏移量0~63）
python3 ../scripts/print_npu_data.py \
    --work-path /path/to/work --print-idx 0 --tensor gmTensor_001

# 指定打印范围（例如打印前80个元素）
python3 ../scripts/print_npu_data.py \
    --work-path /path/to/work --print-idx 0 --tensor gmTensor_001 \
    --start-offset 0 --end-offset 79

# 指定数据类型
python3 ../scripts/print_npu_data.py \
    --work-path /path/to/work --print-idx 0 --tensor gmTensor_001 \
    --dtype bfloat16_t
```

**打印参数说明**：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--start-offset` | 0 | 打印起始偏移量 |
| `--end-offset` | 63 | 打印末尾偏移量 |
| 元素数量 | 64 | 默认打印64个元素（末尾-起始+1） |
| 元素数量限制 | ≤80 | 单次打印最多80个元素 |
| `--dtype` | float | 数据类型（float/bfloat16_t/half/int32_t） |

### 添加shape打印

```bash
# 批量打印（AiCorePrintShape）
python3 ../scripts/print_npu_data.py \
    --work-path /path/to/work --print-idx 0 \
    --print-shape sym_15_dim_0,sym_15_dim_1

# 单值打印（AicoreLogF）
python3 ../scripts/print_npu_data.py \
    --work-path /path/to/work --print-idx 0 \
    --print-shape sym_15_dim_0 --single-value
```

### 参数说明

| 参数 | 必填 | 说明 |
|-----|------|------|
| `--work-path` | 是 | ASCEND_WORK_PATH 工作目录 |
| `--init` | 否 | 初始化配置 |
| `--list-cce` | 否 | 列出CCE文件及tensor信息 |
| `--print-idx` | 否 | 指定CCE索引(0-based) |
| `--tensor` | 否 | 指定tensor名称 |
| `--print-type` | 否 | GM或UB（默认GM） |
| `--print-shape` | 否 | shape变量（逗号分隔） |
| `--single-value` | 否 | 使用单值打印（AicoreLogF） |

---

## 注意事项

1. **CCE文件格式**：每个cpp对应一张子图（kernel）
2. **打印后需重编译**：修改cpp后需重新运行测试
3. **日志位置**：`$ASCEND_WORK_PATH/log/debug/device-*/device-*.log`
4. **恢复原文件**：调试完成后删除打印语句

---

## 相关文档

- 主流程文档：[../SKILL.md](../SKILL.md)