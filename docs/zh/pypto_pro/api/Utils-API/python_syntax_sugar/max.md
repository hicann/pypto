# pypto_pro.language.max

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

在 kernel 代码中可直接使用 Python 内置 `max(lhs, rhs)`，前端会自动将其解析为 `pypto_pro.language.max(lhs, rhs)`，取两个标量中的较大值。

## 函数原型

```python
# 以下两种写法等价
result = max(lhs, rhs)
result = pypto_pro.language.max(lhs, rhs)
```

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `lhs` | 输入 | 左操作数（标量 Expr、Python `int` 或 `float`） |
| `rhs` | 输入 | 右操作数（标量 Expr、Python `int` 或 `float`） |

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    bottom_k = max(1, trunk_len - topk + 1)
    end = pl.max(start, min_end)

    offset = max(i - window, 0)
    c = max(a, b)
```

## 注意事项

- **仅用于标量**：用于循环边界、索引计算等场景
- **tile 逐元素取最大值**：使用 [`pl.maximum`](../../SIMD-API/operation/memory_vector_computation/elementwise/maximum.md)
- **不支持多参数**：仅接受恰好 2 个参数，`max(a, b, c)` 不支持
- **同类别约束**：两个操作数须同为整型或同为浮点型，混合 int/float 会报错

详见 [`pl.max` 接口文档](../../SIMD-API/operation/memory_vector_computation/math_functions/max.md)。
