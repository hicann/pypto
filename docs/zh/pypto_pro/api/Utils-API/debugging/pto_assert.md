# pypto_pro.language.pto_assert

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

运行时断言：条件为假时打印错误信息到设备日志。支持纯文本和 printf 风格的格式化消息。条件为真时静默通过，无输出。

> **注意**：当前 NPU 实现中，断言失败仅打印日志，不会中止 kernel 执行或抛出 host 侧异常。具体行为取决于后端实现。

## 函数原型

```python
pypto_pro.language.pto_assert(condition, format_str=None, *args, *, loc=False)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `condition` | 输入 | 断言条件，必须为标量布尔值 |
| `format_str` | 输入 | 可选，错误消息格式串（编译时常量） |
| `*args` | 输入 | 可选，格式串中的参数值 |
| `loc` | 输入 | 可选，是否在输出前打印源文件/行号 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `condition` | 输入 | 标量布尔值（dtype 为 BOOL）<br>非布尔标量（如 INT32）报 `TypeError`<br>接受 Python `True`/`False` 字面量<br>条件表达式的源码文本由编译器自动提取，用于默认输出 |
| `format_str` | 输入 | 编译时常量字符串字面量，不能是运行时变量<br>不提供时不允许传 `*args`<br>不提供时，输出固定为 `Assertion failed: <条件表达式源码>`<br>提供时，先输出 `Assertion failed: <条件表达式源码>`，再追加一行格式化消息<br>格式说明符规则同 [`pypto_pro.language.printf`](printf.md) |
| `*args` | 输入 | 标量值（`int`、`float`、`Expr`、`bool`）<br>数量和类型须与 `format_str` 中的格式说明符匹配 |
| `loc` | 输入 | `True` 或 `False`（默认） |

## 流水类型

S（标量流水）。

## 调用示例

```python
import pypto_pro.language as pl
# 仅条件
pl.pto_assert(flag)

# 条件 + 纯文本消息
pl.pto_assert(flag, "flag is false")

# 条件 + 格式化消息
pl.pto_assert(offset != 2, "offset=%d", offset)

# 带源码位置
pl.pto_assert(flag, "unexpected state", loc=True)
```

示例输出（条件为假时）：

| 调用 | 输出 |
|---|---|
| `pl.pto_assert(flag)` | `Assertion failed: flag` |
| `pl.pto_assert(flag, "flag is false")` | `Assertion failed: flag`<br>`flag is false` |
| `pl.pto_assert(offset != 2, "offset=%d", offset)` | `Assertion failed: offset != 2`<br>`offset=2`（假设 offset 值为 2） |
| `pl.pto_assert(flag, "unexpected state", loc=True)` | `Assertion failed: flag`<br>`unexpected state` |

> 条件为真时无输出，断言静默通过。
