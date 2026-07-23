# int[N] / float[N] / bool[N]

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

在 tiling class 中声明定长同构数组字段。元素类型限定为 `int`、`float` 或 `bool`，分别映射为 IR 类型 `INDEX`、`FP32`、`BOOL`。

数组字段在 IR 中表示为包含 N 个标量元素的嵌套 Tuple。例如 `offsets: int[4]` 对应一个包含 4 个 `INDEX` 元素的 Tuple 字段。

## 函数原型

```python
int[N]     # N 个 INDEX 元素
float[N]   # N 个 FP32 元素
bool[N]    # N 个 BOOL 元素
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| 元素类型 | 输入 | 仅限 `int`、`float`、`bool` |
| `N` | 输入 | 数组长度，正整数字面量 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| 元素类型 | 输入 | `int` → `DataType.INDEX`<br>`float` → `DataType.FP32`<br>`bool` → `DataType.BOOL`<br>不支持其他元素类型 |
| `N` | 输入 | 取值范围为 1～2048，必须使用整数字面量<br>`N <= 0`、`N > 2048`、布尔值或非字面量均为非法配置 |

## 补充说明

使用 `int[N]`、`float[N]` 或 `bool[N]` 标注时，文件开头须包含 `from __future__ import annotations`，使字段标注以字符串形式保留并由 PyPTO 解析。

运行时使用普通 Python 序列为数组字段赋值，序列长度必须与声明的 `N` 一致：

```python
tiling = MyTiling(m=64, n=128, offsets=[0, 64, 128, 192])
```

在 kernel 中可通过下标访问数组元素，也可以先读取整个数组字段再访问。字面量下标越界时编译报错；数组元素类型相同时，也支持使用运行时下标。

```python
first_offset = tiling.offsets[0]
offsets = tiling.offsets
current_offset = offsets[index]
```

## 调用示例

以下代码展示 tiling class 声明和 kernel 内字段访问片段：

```python
from __future__ import annotations

from dataclasses import dataclass

import pypto_pro.language as pl


@dataclass
class MyTiling:
    m: int
    n: int
    offsets: int[4]


@pl.jit()
def kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
    tiling: MyTiling,
):
    m = tiling.m
    n = tiling.n
    first_offset = tiling.offsets[0]
    ...
```

运行时传入 tiling 实例：

```python
tiling = MyTiling(m=64, n=128, offsets=[0, 64, 128, 192])
kernel(x, out, tiling)
```
