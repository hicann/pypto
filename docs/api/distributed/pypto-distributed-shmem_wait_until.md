# pypto.distributed.shmem_wait_until

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

根据 offsets 指定的索引位置，从 src_pe 对应的 shared memory tensor 的部分视图中等待，直到该视图的值达到目标数值 cmp_value。当条件满足时，当前 pe 会接收到信号。

## 函数原型

```python
shmem_wait_until(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    cmp_value: int = 0,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    cmp: OpType = OpType.EQ,
    clear_signal: bool = False,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 触发信号的 shared memory tensor。 |
| src_pe  | 输入      | shared memory tensor 所属的 pe。 <br> 支持的数据类型为：int 或 SymbolicScalar。 <br> 0 <= src_pe < n_pes。 |
| cmp_value   | 输入      | 要等待的目标数值。 <br> 支持的数据类型为 int 类型。 |
| shape   | 输入      | 需要等待信号的 shared memory tensor 的视图大小。 <br> 参数类型为 list[int] 类型。 <br> 仅支持 3 维。 |
| offsets   | 输入      | 需要等待信号的 shared memory tensor 的视图的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> offsets 的维度应与 src 的维度一致，且每个维度的偏移量值应小于 src 对应维度的大小。 |
| cmp   | 输入      | 用于条件判断的比较操作类型。 <br> 目前仅支持 EQ（等于）类型。 |
| clear_signal   | 输入      | 是否在等待完成后重置信号（true/false）。 <br>支持的数据类型为: bool类型。 <br> 默认为 false。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回一个输出Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. 在设置 TileShape 时，确保切块总数不超过1024。
2. shmem_signal 和 shmem_wait_until必须配合使用，且设置 TileShape 时，切块大小保持一致。


## 调用示例

### TileShape 设置示例

说明：调用 shmem_wait_until 前，应通过set_vec_tile_shapes设置TileShape。TileShape 维度应和参数 shape 的后两维一致。

- 示例 1：参数 shape 为 [1, m, n]，TileShape设置为 [m1, n1]，则 m1，n1 分别用于切分 m，n 轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例 1：当前 pe = 1 在给定的 pe = 1 的 shared memory tensor 全部视图上等待，直到该视图的值达到目标值 cmp_value = 4。一旦条件满足，当前 pe 收到信号。等待完成后，不重置该视图的值。注意，shmem_signal 和 shmem_wait_until必须配合使用，且设置的切块大小保持一致。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=1,
        signal=2,
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
        pred=predToken,
    )
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=1,
        cmp_value=4,
        clear_signal=False,
        pred=[signal_out],
    )
    ```

- 示例 2：当前 pe = 1 在给定的 pe = 1 的 shared memory tensor 部分视图上等待，直到该视图的值达到目标值 cmp_value = 4。一旦条件满足，当前 pe 收到信号。等待完成后，不重置该视图的值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=1,
        signal=2,
        shape=[1, 64, 64],
        offsets=[0, 0, 0],
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
        pred=predToken,
    )
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=1,
        cmp_value=4,
        shape=[1, 64, 64],
        offsets=[0, 0, 0],
        cmp=pypto.OpType.EQ,
        clear_signal=False,
        pred=[signal_out],
    )
    ```

- 示例 3：当前 pe = 5 在给定的 pe = 3 的 shared memory tensor 部分视图上等待，直到该视图的值达到目标值 cmp_value = 4。一旦条件满足，当前 pe 收到信号。等待完成后，该视图的值重置为0。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=3,
        signal=4,
        shape=[1, 64, 64],
        offsets=[0, 0, 1],
        target_pe=5,
        sig_op=pypto.AtomicType.SET,
        pred=predToken,
    )
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=3,
        cmp_value=4,
        shape=[1, 64, 64],
        offstes=[0, 0, 1],
        clear_signal=True,
        pred=[signal_out],
    )
    ```
