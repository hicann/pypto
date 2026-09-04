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

创建字段布局在编译期确定的具名结构体变量。字段声明顺序与关键字参数顺序一致。典型用途是通过[ssbuf_store](../memory_data_movement/ssbuf_store.md)/[ssbuf_load](../memory_data_movement/ssbuf_load.md)传递批次号、块号、地址偏移等少量元数据。

## 函数原型

```python
pypto_pro.language.struct(
    type_name: str,
    **fields,
) -> Struct
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| type_name | 输入 | 结构体类型名，必须是字符串常量，不能是变量。名称只能包含字母、数字和下划线，且不能以数字开头。 |
| **fields | 输入 | 字段名和初始值（关键字参数），至少包含一个字段，字段名不可重复。标量字段初始值支持整数、浮点数、布尔值或Scalar表达式；列表用于创建定长数组字段，例如offsets=[0, 0, 0, 0]。数组长度和元素类型必须在编译期确定，元素应为同一数据类型。不支持**kwargs展开，也不支持将Tensor、Tile、Ptr或嵌套struct作为字段。字段值可通过message.field = value修改，数组字段可通过message.field[index]读写。可在for循环内创建struct；if/else分支内只能修改已存在字段，不能只在某个分支创建新struct。 |

## 返回值说明

返回一个具名struct变量，字段可通过点号访问。若只需要编译期聚合，请使用[pypto_pro.language.make_tuple](make_tuple.md)。

## 约束说明

### 类型名和字段布局

1. 同一个Kernel中，同一type_name只能对应一种字段定义。重复使用相同类型名时，字段名、顺序、标量类型和数组长度必须完全一致。
2. 字段类型由创建时的初始表达式确定，后续赋值必须与该字段类型兼容，且不会改变字段布局。
3. 数组长度必须为正的编译期常量。运行时只能修改数组元素，不能改变数组长度。

### 与SSBUF配合使用

1. 发送端和接收端必须使用完全相同的type_name和字段定义。
2. 用于SSBUF传输时，结构体占用空间必须为4字节的整数倍，并确保offset与结构体占用空间之和不超过可用SSBUF范围。
3. pl.struct本身不提供跨流水或跨核同步；发布和接收顺序参见ssbuf_store/ssbuf_load文档。

## 调用示例

### 跨核传递struct

```python
import pypto_pro.language as pl


@pl.jit()
def struct_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    # 创建结构体：标量字段 + 数组字段
    message = pl.struct("Message", batch=0, block=0, offsets=[0, 0, 0, 0])

    # 修改标量字段
    message.batch = 8
    message.block = 1

    # 修改数组字段元素
    message.offsets[0] = 32768
    message.offsets[2] = 65536

    # Vector 侧写入 SSBUF，Cube 侧读取
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(
                pipe=pl.PipeType.S,
                event_id=15,
                sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
            )

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("batch=%d, block=%d, offset0=%d",
                  message.batch, message.block, message.offsets[0])
```

### 循环读写数组字段

```python
import pypto_pro.language as pl


@pl.jit()
def struct_field_kernel(out: pl.Tensor[[5], pl.DT_INT32]):
    # 创建带数组字段的结构体
    s = pl.struct("RunInfo", batch_id=0, offsets=[0, 0, 0, 0])

    with pl.section_vector():
        # 数组字段元素赋值（s.arr_field[idx] = val）
        s.offsets[0] = 10
        s.offsets[1] = 20
        s.offsets[2] = 30
        s.offsets[3] = 40

        # 数组字段元素读取（s.arr_field[idx]）
        total = 0
        for i in pl.range(0, 4):
            total = total + s.offsets[i]
        pl.setval(out, 0, s.offsets[0])
        pl.setval(out, 1, s.offsets[3])
        pl.setval(out, 2, total)
        pl.setval(out, 3, s.batch_id)
        pl.setval(out, 4, s.offsets[1] + s.offsets[2])
```
