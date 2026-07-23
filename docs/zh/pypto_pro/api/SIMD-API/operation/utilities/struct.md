# pypto_pro.language.struct

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

编译期结构体，将多个标量值组合为一个具名字段的结构体变量。

PyPTO kernel 最终编译为 NPU 上的 C++ 代码，`pl.struct` 在编译期生成对应的 C++ struct 定义和实例。典型用途是通过 [`ssbuf_store`](../memory_data_movement/ssbuf_store.md)/[`ssbuf_load`](../memory_data_movement/ssbuf_load.md) 在 SSBUF（共享标量缓冲区）中传递少量元数据（如批次号、块号、地址偏移），实现不同流水线（pipe）或不同核之间的标量通信。

> **关于 pipe**：NPU 内部有多条并行流水线（MTE1/MTE2/MTE3/M/V/S/FIX），各自负责不同阶段的数据搬运和计算。详见 [`PipeType`](../../basic_data_structures/PipeType.md)。

## 函数原型

```python
pypto_pro.language.struct("TypeName", field1=default1, field2=default2, ...)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `"TypeName"` | 输入 | 结构体类型名（字符串字面量） |
| `field=value` | 输入 | 字段名和初始值（关键字参数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `"TypeName"` | 输入 | 必须是字符串字面量，不能是变量<br>须为合法的 C 标识符（字母/数字/下划线，不以数字开头）<br>编译后生成的 C++ struct 以此命名 |
| `field=value` | 输入 | 至少一个关键字参数<br>字段名须为合法的 C 标识符，不可重复<br>初始值可为整型、浮点型或运行时 Expr<br>不支持 `**kwargs` 展开<br>字段值可多次修改（通过 `message.field = new_value` 赋值）<br>可在 `for` 循环内创建 struct；`if/else` 分支内只能修改字段，不能创建新 struct（须在分支外创建）<br>字段数量已验证 8 个，SSBUF 容量有限（通常 256 字节），传递的 struct 不宜过大 |

## 调用示例

```python
import pypto_pro.language as pl

# 创建结构体并赋值字段
message = pl.struct("Message", batch=0, block=0, offset=0)
message.batch = 8
message.block = 1
message.offset = 32768

# 跨流水线传递：vector 侧写入 SSBUF，cube 侧读取
with pl.section_vector():
    if pl.get_subblock_idx() == 0:
        pl.ssbuf_store(message, 0)
        pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

with pl.section_cube():
    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.ssbuf_load(message, 0)
    pl.printf("batch=%d, block=%d, offset=%d",
              message.batch, message.block, message.offset)
```
