# pypto.distributed.shmem_put

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

以 offsets 指定的 shared memory tensor 索引位置为基准，将输入的 Tensor 赋值到 shared memory tensor 的对应区域。

## 函数原型

```python
shmem_put(
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
| src   | 输入      | 源操作数。 <br> 支持的数据类型为：DT_INT32，DT_FP16，DT_FP32，DT_BF16。 <br> 不支持空 Tensor；Shape 仅支持 2 维；Shape Size 不大于 2147483647（即 INT32_MAX）。 <br> 支持的数据格式为 ND。 |
| offsets   | 输入      | dst 的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> offsets 的维度应与 dst 的维度一致，且每个维度的偏移量值应小于 dst 对应维度的大小。 |
| dst   | 输入      | 目的操作数，一个 shared memory tensor，其形状为 [1] + src.shape。 |
| dst_pe   | 输入      | shared memory tensor 所属的 pe。<br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= dst_pe < n_pes。 |
| put_op   | 输入      | 数据传输时应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET，AtomicType.ADD。 <br> 默认为 AtomicType.SET 类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回输出 Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

### TileShape 设置示例

说明：调用该接口前，应通过 set_vec_tile_shapes 设置 TileShape。TileShape 维度应和 src 一致。

- 示例 1：src 的 shape 为 [m, n]，TileShape 设置为 [m1, n1]，则 m1，n1 分别用于切分 m，n 轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例 1：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 1 的 shared memory tensor 的指定区域，并与该视图原本的数据进行累加操作。

    ```python
    input_tensor = pypto.tensor([16, 64], pypto.DT_BF16, "input_tensor")
    shmem_shape = [1] + input_tensor.shape
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP32, shape=shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    put_out = pypto.distributed.shmem_put(
        src=input_tensor ,
        offsets=[0, 0, 0],
        dst=shmem_tensor,
        dst_pe=1,
        put_op=pypto.AtomicType.ADD,
    )
    ```

- 示例 2：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 3 的 shared memory tensor 的指定区域，并覆盖该视图原本的数据。

    ```python
    input_tensor = pypto.tensor([16, 64], pypto.DT_BF16, "input_tensor")
    shmem_shape = [1] + input_tensor.shape
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP32, shape=shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    put_out = pypto.distributed.shmem_put(
        src=input_tensor,
        offsets=[0, 0, 0],
        dst=shmem_tensor,
        dst_pe=3,
        put_op=pypto.AtomicType.SET,
    )
    ```
