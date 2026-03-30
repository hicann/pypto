# pypto.distributed.shmem_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

根据 offsets 指定的索引位置，将信号值 signal 写入 target_pe 对应的 shared memory tensor 的部分视图，从而通知 target_pe。

## 函数原型

```python
shmem_signal(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    signal: int,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    target_pe: Union[int, SymbolicScalar],
    sig_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src  | 输入      | 触发信号的 shared memory tensor。|
| src_pe   | 输入      | shared memory tensor 所属的 pe，0 <= pe < n_pes。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 |
| signal   | 输入      | 发送到 src 中的信号值。 <br> 支持的数据类型为：int类型。 |
| shape   | 输入      |  需要写入信号的 shared memory tensor 的视图大小。 <br> 参数类型为 list[int] 类型。 <br> 仅支持 3 维。 |
| offsets   | 输入      | 需要写入信号的 shared memory tensor 的视图的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> offsets 的维度应与 src 的维度一致，且每个维度的偏移量值应小于 src 对应维度的大小。 |
| target_pe   | 输入      | 接收信号的 pe。 <br> 如果 target_pe = -1，则广播信号给所有 pe。 <br> 支持 int 或 SymbolicScalar 类型的列表。 |
| sig_op   | 输入      | 数据传输时应用的原子操作类型。 <br>支持的数据类型为: AtomicType.SET，AtomicType.ADD。 <br> 默认为 AtomicType.SET 类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回一个输出 Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. shmem_signal 和 shmem_wait_until 必须配合使用，且设置 TileShape 时，切块大小保持一致。

## 调用示例

### TileShape 设置示例

说明：调用 shmem_signal 前，应通过set_vec_tile_shapes设置TileShape， TileShape 维度应和参数 shape 的后两维一致。

- 示例1：参数 shape 为 [1, m, n]，TileShape设置为 [m1, n1]，则 m1，n1 分别用于切分 m，n 轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例 1：将信号值 2 写入 pe = 1 的 shared memory tensor 的全部视图中，并与该视图原本的值进行累加操作，从而通知 pe = 1。

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
    ```

- 示例 2：将信号值 2 写入 pe = 1 的 shared memory tensor 的部分视图中，从而通知 pe = 1。该部分视图的 shape 为 [1, 64, 64]，offset 为 [0, 0, 0]， 并与该视图原本的值进行累加操作。

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
    ```

- 示例 3：将信号值 4 写入 pe = 3 的 shared memory tensor 的部分视图中，从而通知 pe = 5。该部分视图的 shape 为 [1, 64, 64]，offset 为 [0, 0, 1]， 并覆盖该视图原本的值。

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
    ```

- 示例 4：将信号值 4 写入 pe = 3 的 shared memory tensor 的部分视图中，从而通知所有 pe。该部分视图的 shape 为 [1, 64, 64]，offset 为 [0, 0, 1]， 并覆盖该视图原本的值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        sec_pe=3,
        signal=4,
        shape=[1, 64, 64],
        offsets=[0, 0, 1],
        target_pe=-1,
        pred=predToken,
    )
    ```
