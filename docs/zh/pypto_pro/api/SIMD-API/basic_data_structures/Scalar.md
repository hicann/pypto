# pypto_pro.language.Scalar

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

带数据类型的标量值类型标注，用于 kernel 标量参数声明或 kernel 内部局部变量类型标注。

`pypto_pro.language.Scalar` 不能使用普通 Python 数值直接构造。显式标量常量通过 [`pypto_pro.language.const`](../../Utils-API/python_syntax_sugar/const.md) 创建；`Scalar(expr)` 仅用于内部包装 IR 表达式。

`pypto_pro.language.DT_*` 类型常量主要用于：

1. kernel 函数签名中声明标量参数（如行数、列数、缩放系数等）
2. kernel 内部局部变量的类型标注（如 `pypto_pro.language.get_block_idx()` 返回值）
3. 通过 [`pypto_pro.language.const`](../../Utils-API/python_syntax_sugar/const.md) 创建编译期常量标量时指定数据类型

## 可用类型常量

| 类型常量 | 说明 |
|---|---|
| `pypto_pro.language.DT_BOOL` | 布尔标量类型 |
| `pypto_pro.language.DT_INT8` | 8 位有符号整数标量类型 |
| `pypto_pro.language.DT_INT16` | 16 位有符号整数标量类型 |
| `pypto_pro.language.DT_INT32` | 32 位有符号整数标量类型 |
| `pypto_pro.language.DT_INT64` | 64 位有符号整数标量类型，常用于坐标和偏移计算 |
| `pypto_pro.language.DT_UINT8` | 8 位无符号整数标量类型 |
| `pypto_pro.language.DT_UINT16` | 16 位无符号整数标量类型 |
| `pypto_pro.language.DT_UINT32` | 32 位无符号整数标量类型 |
| `pypto_pro.language.DT_UINT64` | 64 位无符号整数标量类型 |
| `pypto_pro.language.DT_FP16` | 16 位浮点标量类型 |
| `pypto_pro.language.DT_BF16` | 16 位 Brain 浮点标量类型 |
| `pypto_pro.language.DT_FP32` | 32 位浮点标量类型 |

## 约束说明

以下低精度类型为存储和张量计算专用类型，**不支持**用于标量表达式：

`DT_FP4`、`DT_FP8E4M3FN`、`DT_FP8E5M2`、`DT_INT4`、`DT_UINT4`、`DT_HF4`、`DT_HF8`

## 调用示例

```python
import pypto_pro.language as pl
# kernel 签名中声明标量参数
@pl.jit()
def kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    scale: pl.DT_FP32,
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ...

# kernel 内部局部变量类型标注
vidx = pl.get_block_idx()
offset: pl.DT_INT64 = vidx * 64

# kernel 内部显式创建标量常量
scale = pl.const(1.0, pl.DT_FP32)
zero_idx = pl.const(0, pl.DT_INT64)
```

显式创建标量值的完整说明见 [`pypto_pro.language.const`](../../Utils-API/python_syntax_sugar/const.md)。
