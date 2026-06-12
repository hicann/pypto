# pypto.distributed.shmem_clear_data

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

用于清除当前pe对应的shared memory tensor的部分视图

## 函数原型

```python
shmem_clear_data(
    src: ShmemTensor,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的shared memory tensor。|
| shape   | 输入      | 需要清除的视图大小。 <br> 参数类型为list[int] 类型。 |
| offsets   | 输入      | 需要清除的视图的偏移量。 <br> 支持int或SymbolicScalar类型的列表。 <br> offsets的维度应与src的维度一致，且每个维度的偏移量值应小于src对应维度的大小。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 |

## 返回值说明

返回一个Tensor，用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

- 示例：创建了一个shape = [128, 256] 的shared memory tensor，清除当前pe对应的shared memory tensor的部分视图的数据。该部分视图的shape为 [128, 128], offsets为 [0, 0]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        src=shmem_tensor,
        shape=[128, 128],
        offsets=[0, 0],
    )
    ```

- 示例：创建了一个shape = [128, 256] 的shared memory tensor，清除当前pe对应的shared memory tensor的全部视图的数据。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        src=shmem_tensor,
    )
    ```
