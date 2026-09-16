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

创建字段布局在编译期确定的具名结构体变量。字段声明顺序与关键字参数顺序一致，可用于组织批次号、块号、地址偏移等少量元数据。

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
| type_name | 输入 | 结构体类型名。<br>- 名称必须是字符串常量，不能是变量。<br>- 名称只能包含字母、数字和下划线，不能以数字开头，且不能是C++关键字。 |
| fields | 输入 | 结构体成员变量名称和初始值（关键字参数）。<br>- 至少包含一个成员变量，成员变量名称不可重复。<br>- 成员变量名称只能包含字母、数字和下划线、不能以数字开头，且不能是C++关键字。<br>- 成员变量仅支持如下类型：<br>&nbsp;&nbsp;- **标量**：初始值支持整数、浮点数、布尔值或Scalar表达式。<br>&nbsp;&nbsp;- **一维标量数组**：一维非空标量数组，数组的长度和元素类型必须在编译期确定，元素应为同一数据类型。不支持将嵌套的具名结构体（通过make_tuple、struct创建）作为成员变量。<br>- 标量成员变量的值可通过`arr.field = value`修改，数组成员变量可通过`arr.field[index]`读写，仅支持相同类型的赋值操作。 |

## 约束说明

- 对结构体进行赋值时，必须保证等式两边的结构体类型名，成员变量的名称、顺序、标量类型和数组长度必须完全相同。例如，对同一个变量，在if/else分支中分别对其进行struct赋值，那么这两个struct必须满足上述条件。
- 在控制流（if/else/for/while）中，如果用值拷贝的方式创建struct，得到的是独立副本，修改它不会影响到原始struct。
- 用下标访问数组成员时，需要确保0 <= index < size。

## 返回值说明

返回一个具名struct变量。

## 调用示例

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
