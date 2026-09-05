# pypto_pro.language.make_tensor

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

用一个裸指针（pypto_pro.language.Ptr[dtype]）或已有Tensor，加上显式的shape和可选的stride，构造一个Tensor视图。Tensor视图本身不分配内存，只为同一段GM地址设置“形状 + 步长”的解释，之后即可像普通GM Tensor一样被[load](../memory_data_movement/load.md)/[store](../memory_data_movement/store.md)使用。省略stride时，接口根据shape自动生成连续的行主序stride。

常与[pypto_pro.language.addptr](addptr.md)配合：用addptr切出workspace的某一段地址，再用make_tensor把它包装成可读写的Tensor。

下图展示make_tensor的核心语义：它使用shape和stride为已有地址创建Tensor视图，不申请新内存，也不搬运数据。

![make_tensor从裸指针创建Tensor视图](../../../figures/make_tensor_view.jpg "make_tensor从裸指针创建Tensor视图")

## 函数原型

```python
pypto_pro.language.make_tensor(
    ptr: Union[Ptr, Tensor],
    shape: Sequence[Union[int, Scalar]],
    stride: Optional[Sequence[Union[int, Scalar]]] = None,
    dtype: Optional[DataType] = None,
) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| ptr | 输入 | 用于构造Tensor视图的地址对象，Ptr或Tensor类型。创建的Tensor视图与源对象共享底层数据地址。 |
| shape | 输入 | Tensor形状，Sequence[int或Scalar]类型，每一维可以是整型常量或运行时整型Scalar表达式。 |
| stride | 输入 | Tensor步长，Sequence[int或Scalar]类型，可选，单位为元素。显式传入时，长度必须与shape的维数相同，元素[i0, i1, ...]相对ptr的元素偏移为i0 × stride[0] + i1 × stride[1] + ...。省略时自动生成连续行主序stride。对于亚字节数据类型，最内层维必须连续，即最后一项stride必须为1。 |
| dtype | 输入 | Tensor元素类型，[DataType](../../basic_data_structures/DataType.md)类型，可选。传入时按该类型解释源地址；省略时沿用源指针或源Tensor的数据类型。 |

## 约束说明

- 创建的Tensor只是地址视图。调用方必须保证shape、stride和dtype描述的所有访问均落在源对象的合法内存范围内，并满足dtype的地址对齐要求。
- stride还用于load和store的起始地址换算及GM搬运描述，必须满足对应搬运接口的能力范围；通用load和store主要用于末维连续的二维搬运。

## 返回值说明

返回与输入ptr共享底层地址的Tensor视图。

## 调用示例

### 连续行主序Tensor视图

省略stride时，接口自动生成连续的行主序步长：最后一维步长为1，其余各维步长等于后续各维大小的乘积。

```python
normal = pl.make_tensor(ptr, [8, 16])
# 等价于：normal = pl.make_tensor(ptr, [8, 16], [16, 1])
```

### 行间不连续的Tensor视图

下面的Tensor每行包含16个连续元素，相邻两行的起始地址相隔32个元素，因此行间跳过16个元素：

```python
pitched = pl.make_tensor(ptr, [8, 16], [32, 1])
# pitched[i, j]的元素地址：ptr + i * 32 + j
```

### 交换shape和stride表达转置视图

```python
normal = pl.make_tensor(ptr, [8, 16], [16, 1])
transposed = pl.make_tensor(ptr, [16, 8], [1, 16])
```

两者共享同一个ptr，且normal[i, j]与transposed[j, i]指向相同地址；接口只改变逻辑索引到物理地址的映射，不转置底层数据。

![make_tensor的非连续与转置视图](../../../figures/make_tensor_stride_cases.jpg "make_tensor的非连续与转置视图")

### 在Kernel中访问行间不连续的Tensor

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [256, 1])

    tt = pl.TileType(shape=[32, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])

        pl.load(t, a, [32, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [32, 0])
        pl.load(t, ws_buf, [32, 0])
        pl.store(out, t, [32, 0])
```
