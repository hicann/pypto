# pypto.distributed.shmem_load

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

从输入的 shared memory tensor 中取出部分视图到本地。

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
| src   | 输入      | 源操作数，一个 shared memory tensor。 <br> 目前 shared memory tensor 的 shape 只支持 3 维。 |
| src_pe   | 输入      | shared memory tensor 所属的 pe。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= src_pe < n_pes。 |
| shape   | 输入      | 需要获取的视图大小。 <br> 目前最高维限制为 1。 <br> 参数类型为 list[int] 类型。 <br> Shape 仅支持 3 维。 |
| offsets   | 输入      | 需要获取的视图偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> offsets 的维度应与 src 的维度一致，且每个维度的偏移量值应小于 src 对应维度的大小。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |
| valid_shape   | 输入      | 用于指定需要获取的有效数据大小。 <br> 需要保证 valid_shape 小于 shape。 |

## 返回值说明

返回一个与 src 数据类型相同的 Tensor，形状为将 shape 参数前两维合并为一个维度，最后一维保持不变。如果指定了 valid_shape，则真实有效数据形状为 valid_shape 进行相同操作后得到的shape。
假设 shape=[1, 128, 256]，valid_shape=None，则返回的 Tensor 形状为 [1 * 128, 256]。
假设 shape=[1, 128, 256]，valid_shape=[1, 128, 128]，则返回的 Tensor 形状为 [1 * 128, 256]，真实有效数据形状为 [1 * 128， 128]。

## 约束说明

1. shmem_load 通常在 shmem_wait_until 之后执行，以保证要获取的数据已经写入到了目标地址上。在 shmem_wait_until 切块数据大于1的场景下，shmem_load 需要与其保持相同的切块配置，以便两者能够形成更优的流水排布，并保证精度正常。

## 调用示例

### TileShape 设置示例

说明：调用该接口前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和输出一致。

- 示例 1：输入的 shape 为 [1, m, n]，输出的 shape 为 [m, n]，TileShape设置为 [m1, n1]，则 m1，n1 分别用于切分 m，n 轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例 1：从  pe = 1 的 shared memory tensor 的全部视图中获取数据并输出该数据，对应的输出数据 shape 为 [128, 256]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    load_out = pypto.experimental.shmem_load(
        src=shmem_tensor,
        src_pe=1,
        pred=predToken,
    )
    ```

- 示例 2：从  pe = 1 的 shared memory tensor 的部分视图中获取数据并输出该数据。该部分视图的 shape 为 [1, 128, 128]，offset 为 [0, 0, 0]，对应的输出数据 shape 为 [128, 128]，实际获取的数据有效大小为 [128, 64]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    load_out = pypto.experimental.shmem_load(
        src=shmem_tensor,
        src_pe=1,
        shape=[1, 128, 128],
        offsets=[0, 0, 0],
        pred=predToken,
        valid_shape=[1, 128, 64],
    )
    ```
