# pypto.distributed.shmem_wait_until

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

根据offsets指定的索引位置，从src_pe对应的shared memory tensor的部分视图中等待，直到该视图的值达到目标数值cmp_value。当条件满足时，当前pe会接收到信号。

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
| src   | 输入      | 触发信号的shared memory tensor。 |
| src_pe  | 输入      | shared memory tensor所属的pe。 <br> 支持的数据类型为：int或SymbolicScalar。 <br> 0 <= src_pe < n_pes。 |
| cmp_value   | 输入      | 要等待的目标数值。 <br> 支持的数据类型为int类型。 |
| shape   | 输入      | 需要等待信号的shared memory tensor的视图大小。 <br> 参数类型为list[int] 类型。 |
| offsets   | 输入      | 需要等待信号的shared memory tensor的视图的偏移量。 <br> 支持int或SymbolicScalar类型的列表。 <br> offsets的维度应与src的维度一致，且每个维度的偏移量值应小于src对应维度的大小。 |
| cmp   | 输入      | 用于条件判断的比较操作类型。 <br> 目前仅支持EQ（等于）类型。 |
| clear_signal   | 输入      | 是否在等待完成后重置信号（true/false）。 <br>支持的数据类型为: bool类型。 <br> 默认为false。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空Tensor。 |

## 返回值说明

返回一个输出Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. shmem_signal和shmem_wait_until必须配合使用，且设置TileShape时，切块大小保持一致。

## 调用示例

### TileShape设置示例

> [!NOTE]说明
> 调用shmem_wait_until前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和参数shape保持一致。

- 示例：参数shape为 [m, n]，TileShape设置为 [m1, n1]，则m1，n1分别用于切分m，n轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

- 示例：当前pe = 1在给定的pe = 1的shared memory tensor全部视图上等待，直到该视图的值达到目标值cmp_value = 4。一旦条件满足，当前pe收到信号。等待完成后，不重置该视图的值。注意，shmem_signal和shmem_wait_until必须配合使用，且设置的切块大小保持一致。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
    pypto.set_vec_tile_shapes(32, 64)
    signal_out = pypto.distributed.shmem_signal(
        src=shmem_tensor,
        src_pe=1,
        signal=2,
        shape=None,
        offsets=None,
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
    )
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=1,
        cmp_value=4,
        shape=None,
        offsets=None,
        clear_signal=False,
        pred=[signal_out],
    )
    ```

- 示例：当前pe = 1在给定的pe = 1的shared memory tensor部分视图上等待，直到该视图的值达到目标值cmp_value = 4。一旦条件满足，当前pe收到信号。等待完成后，不重置该视图的值。

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
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=1,
        cmp_value=4,
        shape=[64, 64],
        offsets=[0, 0],
        cmp=pypto.OpType.EQ,
        clear_signal=False,
        pred=[signal_out],
    )
    ```

- 示例：当前pe = 5在给定的pe = 3的shared memory tensor部分视图上等待，直到该视图的值达到目标值cmp_value = 4。一旦条件满足，当前pe收到信号。等待完成后，该视图的值重置为0。

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
    wait_until_out = pypto.distributed.shmem_wait_until(
        src=shmem_tensor,
        src_pe=3,
        cmp_value=4,
        shape=[64, 64],
        offsets=[0, 1],
        clear_signal=True,
        pred=[signal_out],
    )
    ```
