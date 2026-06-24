# pypto.experimental.shmem_store

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

以offsets指定的shared memory tensor索引位置为基准，将输入的Tensor赋值到shared memory tensor的对应区域。

## 函数原型

```python
shmem_store(
    src: Tensor,
    offsets: list[Union[int, SymbolicScalar]],
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 源操作数。 <br> 不支持空Tensor；Shape支持2 - 4维；Shape Size不大于2147483647（即INT32_MAX）。 <br> 支持的数据格式为ND。 |
| offsets   | 输入      | dst的偏移量。 <br> 支持int或SymbolicScalar类型的列表。 <br> offsets的维度应与dst的维度一致，且每个维度的偏移量值应小于dst对应维度的大小。 |
| dst   | 输入      | 目的操作数，一个shared memory tensor，其shape在各维度上均不小于src的shape。 |
| dst_pe   | 输入      | shared memory tensor所属的pe。<br> 支持的数据类型为int或SymbolicScalar类型。 <br> 0 <= pe < n_pes。 |
| put_op   | 输入      | 数据传输时应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET，AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空Tensor。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

1. pred是一个张量列表，表示控制依赖关系。建议不要把src放在这个列表中。
2. src的dtype必须和dst的dtype一致。
3. src支持的数据类型说明：
    - **Ascend 950PR**：DT_INT32、DT_FP16、DT_FP32、DT_BF16
    - **Atlas A3 训练系列产品/Atlas A3 推理系列产品**：DT_INT32、DT_FP16、DT_FP32、DT_BF16

## 调用示例

### TileShape设置示例

> [!NOTE]说明
> 调用该接口前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和src一致。

- 示例：输入的shape为 [m, n]，TileShape设置为 [m1, n1]，则m1，n1分别用于切分m，n轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例：先创建一个shared memory tensor。将输入数据赋值到pe = 2的shared memory tensor的指定区域，并与该视图原本的数据进行累加操作。注意，shared memory tensor的dtype和输入数据的dtype必须一致。

    ```python
    input_tensor = pypto.tensor([16, 64], pypto.DT_BF16, "input_tensor")
    shmem_shape = input_tensor.shape
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_BF16, shape=shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    store_out = pypto.experimental.shmem_store(
        src=input_tensor,
        offsets=[0, 0],
        dst=shmem_tensor,
        dst_pe=2,
        put_op=pypto.AtomicType.ADD,
    )
    ```

- 示例：先创建一个shared memory tensor。将输入数据赋值到pe = 3的shared memory tensor的指定区域，并覆盖该视图原本的数据。

    ```python
    input_tensor = pypto.tensor([16, 64], pypto.DT_BF16, "input_tensor")
    shmem_shape = input_tensor.shape
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_BF16, shape=shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    store_out = pypto.experimental.shmem_store(
        src=input_tensor,
        offsets=[0, 0],
        dst=shmem_tensor,
        dst_pe=3,
        put_op=pypto.AtomicType.SET,
    )
    ```
