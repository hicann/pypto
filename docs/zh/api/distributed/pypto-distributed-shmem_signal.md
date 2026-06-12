# pypto.distributed.shmem_signal

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

根据offsets指定的索引位置，将信号值signal写入target_pe对应的shared memory tensor的部分视图，从而通知target_pe。

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
| src  | 输入      | 触发信号的shared memory tensor。|
| src_pe   | 输入      | shared memory tensor所属的pe，0 <= pe < n_pes。 <br> 支持的数据类型为int或SymbolicScalar类型。 |
| signal   | 输入      | 发送到src中的信号值。 <br> 支持的数据类型为：int类型。 |
| shape   | 输入      |  需要写入信号的shared memory tensor的视图大小。 <br> 参数类型为list[int] 类型。 |
| offsets   | 输入      | 需要写入信号的shared memory tensor的视图的偏移量。 <br> 支持int或SymbolicScalar类型的列表。 <br> offsets的维度应与src的维度一致，且每个维度的偏移量值应小于src对应维度的大小。 |
| target_pe   | 输入      | 接收信号的pe。 <br> 如果target_pe = -1，则广播信号给所有pe。 <br> 支持int或SymbolicScalar类型。 |
| sig_op   | 输入      | 数据传输时应用的原子操作类型。 <br>支持的数据类型为: AtomicType.SET，AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空Tensor。 |

## 返回值说明

返回一个输出Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. shmem_signal和shmem_wait_until必须配合使用，且设置TileShape时，切块大小保持一致。

## 调用示例

### TileShape设置示例

> [!NOTE]说明
> 调用shmem_signal前，应通过set_vec_tile_shapes设置TileShape， TileShape维度应和参数shape保持一致。

- 示例：参数shape为 [m, n]，TileShape设置为 [m1, n1]，则m1，n1分别用于切分m，n轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例：将信号值2写入pe = 1的shared memory tensor的全部视图中，并与该视图原本的值进行累加操作，从而通知pe = 1。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=1,
        signal=2,
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
    )
    ```

- 示例：将信号值2写入pe = 1的shared memory tensor的部分视图中，从而通知pe = 1。该部分视图的shape为 [64, 64]，offset为 [0, 0]，并与该视图原本的值进行累加操作。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=1,
        signal=2,
        shape=[64, 64],
        offsets=[0, 0],
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
    )
    ```

- 示例：将信号值4写入pe = 3的shared memory tensor的部分视图中，从而通知pe = 5。该部分视图的shape为 [64, 64]，offset为 [0, 1]，并覆盖该视图原本的值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=3,
        signal=4,
        shape=[64, 64],
        offsets=[0, 1],
        target_pe=5,
        sig_op=pypto.AtomicType.SET,
    )
    ```

- 示例：将信号值4写入pe = 3的shared memory tensor的部分视图中，从而通知所有pe。该部分视图的shape为 [64, 64]，offset为 [0, 1]，并覆盖该视图原本的值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=3,
        signal=4,
        shape=[64, 64],
        offsets=[0, 1],
        target_pe=-1,
    )
    ```
