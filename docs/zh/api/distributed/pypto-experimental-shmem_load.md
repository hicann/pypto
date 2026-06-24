# pypto.experimental.shmem_load

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

从输入的shared memory tensor中取出部分视图到本地。

## 函数原型

```python
shmem_load(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 源操作数，一个shared memory tensor。 |
| src_pe   | 输入      | shared memory tensor所属的pe。 <br> 支持的数据类型为int或SymbolicScalar类型。 <br> 0 <= src_pe < n_pes。 |
| shape   | 输入      | 需要获取的视图大小。 <br> 参数类型为list[int] 类型。 |
| offsets   | 输入      | 需要获取的视图偏移量。 <br> 支持int或SymbolicScalar类型的列表。 <br> offsets的维度应与src的维度一致，且每个维度的偏移量值应小于src对应维度的大小。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空Tensor。 |
| valid_shape   | 输入      | 用于指定需要获取的有效数据大小。 <br> 需要保证valid_shape小于shape。 |

## 返回值说明

返回一个与src数据类型和形状相同的Tensor。

## 约束说明

1. shmem_load通常在shmem_wait_until之后执行，以保证要获取的数据已经写入到了目标地址上。在shmem_wait_until切块数据大于1的场景下，shmem_load需要与其保持相同的切块配置，以便两者能够形成更优的流水排布，并保证精度正常。

## 调用示例

### TileShape设置示例

> [!NOTE]说明
> 调用该接口前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和src一致。

- 示例：输入的shape为 [m, n]，输出的shape为 [m, n]，TileShape设置为 [m1, n1]，则m1，n1分别用于切分m，n轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例：从pe = 1的shared memory tensor的全部视图中获取数据并输出该数据，对应的输出数据shape为 [128, 256]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    load_out = pypto.experimental.shmem_load(
        src=shmem_tensor,
        src_pe=1,
    )
    ```

- 示例：从pe = 1的shared memory tensor的部分视图中获取数据并输出该数据。该部分视图的shape为 [128, 128]，offset为 [0, 0]，对应的输出数据shape为 [128, 128]，实际获取的数据有效大小为 [128, 64]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    load_out = pypto.experimental.shmem_load(
        src=shmem_tensor,
        src_pe=1,
        shape=[128, 128],
        offsets=[0, 0],
        valid_shape=[128, 64],
    )
    ```
