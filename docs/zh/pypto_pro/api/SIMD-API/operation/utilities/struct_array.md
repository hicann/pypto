# pypto_pro.language.struct_array

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

结构体数组，N 个相同的 struct 按索引存取。用于流水线 / FIFO 场景中按槽位索引存取上下文。

## 函数原型

```python
pypto_pro.language.struct_array(size, "TypeName", field1=default1, ...)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `size` | 输入 | 数组长度（正整数） |
| `"TypeName"` | 输入 | 结构体类型名（字符串字面量） |
| `field=value` | 输入 | 字段名和初始值（关键字参数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `size` | 输入 | 必须是编译时常量正整数（`size >= 1`）<br>非常量或非正整数报 `ParserSyntaxError` |
| `"TypeName"` | 输入 | 必须是字符串字面量，作为第二个位置参数<br>缺失或非字符串报 `ParserSyntaxError` |
| `field=value` | 输入 | 至少一个关键字参数<br>字段名须为合法标识符<br>不支持 `**kwargs` 展开 |
| `index`（存取时） | 输入 | 整数常量或运行时 Expr（如 `task_id % size`、循环变量 `i`）<br>不支持负数索引、切片、`for slot in ctx_arr:` 遍历、`len(ctx_arr)`<br>越界访问编译期检查，越界报 `GetItemExpr index N out of bounds for tuple with M elements` |

## 调用示例

```python
import pypto_pro.language as pl

# 创建 3 槽的结构体数组
ctx_arr = pl.struct_array(3, "CubeCtx", sq_off=0, task_id=0, qi=0, ki=0)

# 按索引存取
for ki in pl.range(0, skv_tiles):
    ctx_curr = ctx_arr[task_id % 3]
    ctx_curr.sq_off = sq_off
    ctx_curr.task_id = task_id
    ctx_curr.qi = qi
    ctx_curr.ki = ki
    # ... 使用 ctx_curr 的字段 ...
```
