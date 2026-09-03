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

结构体数组，N个相同的struct按索引存取。用于流水线 / FIFO场景中按槽位索引存取上下文。

## 函数原型

```python
pypto_pro.language.struct_array(
    size: int,
    type_name: str,
    **fields,
) -> StructArray
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| size | 输入 | 数组长度，必须是编译时常量正整数（size >= 1）。非常量或非正整数报ParserSyntaxError。 |
| type_name | 输入 | 结构体类型名，必须是字符串常量，作为第二个位置参数。名称只能包含字母、数字和下划线，且不能以数字开头。 |
| **fields | 输入 | 字段名和初始值（关键字参数），至少一个关键字参数。字段名须为合法标识符且不可重复。初始值可为整型、浮点型、标量表达式或列表（数组字段，如arr=[0, 0, 0, 0]）。不支持**kwargs展开。 |
| index（存取时） | 输入 | 整数常量或整数标量表达式（如task_id % size、循环变量i）。调用方须保证0 <= index < size；当前接口不对常量或动态索引执行越界检查，越界访问的行为未定义。不支持负数索引、切片、for slot in ctx_arr:遍历、len(ctx_arr)。 |

## 返回值说明

返回一个结构体数组对象。每个元素可通过arr[i].field访问标量字段、arr[i].field[j]访问数组字段元素。

## 约束说明

- 同一Kernel中，同一type_name只能对应一种字段定义。重复使用相同类型名时，字段名、顺序、标量类型和数组长度必须完全一致。
- 字段类型由创建时的初始表达式确定，后续赋值必须能转换到该成员类型；赋值不会改变struct布局。
- 数组长度必须为正的编译期常量。运行时只能修改数组元素，不能改变数组长度。
- 与SSBUF配合使用时，结构体占用空间必须为4字节的整数倍。
- 数组字段元素可通过arr[i].field[j] = val赋值，通过arr[i].field[j]读取。

## 调用示例

### 按索引读写字段

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 2 槽结构体数组：标量字段 + 数组字段
    run_infos = pl.struct_array(2, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    # 按索引修改标量字段
    run_infos[0].batch_id = 7
    run_infos[1].batch_id = 9

    # 按索引修改数组字段元素
    run_infos[0].innerS1Realsize[3] = 128
    run_infos[1].innerS1Realsize[1] = 64

    # 读回数组字段元素并输出
    with pl.section_vector():
        pl.setval(out, 0, run_infos[0].batch_id)
        pl.setval(out, 1, run_infos[1].batch_id)
        pl.setval(out, 2, run_infos[0].innerS1Realsize[3])
        pl.setval(out, 3, run_infos[1].innerS1Realsize[1])
```

### 循环读写数组字段

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_field_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 4 槽结构体数组，含数组字段
    run_infos = pl.struct_array(4, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    with pl.section_vector():
        # 数组字段元素赋值（arr[i].field[j] = val）
        for i in pl.range(0, 4):
            run_infos[i].innerS1Realsize[0] = i * 100

        # 数组字段元素读取（arr[i].field[j]）
        for i in pl.range(0, 4):
            pl.setval(out, i, run_infos[i].innerS1Realsize[0])
```
